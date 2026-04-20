#!/bin/bash

# Configuration
CONFIG_FILE="config.json"
BLACKLIST_FILE="failed_nodes.txt"
export CONFIG_FILE
export BLACKLIST_FILE

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found!"
    exit 1
fi

# Ensure blacklist file exists
touch "$BLACKLIST_FILE"

# Helper to get config values
get_cfg() {
    python3 -c "import json,os; cfg=json.load(open('$CONFIG_FILE')); print(cfg.get('$1', {}).get('$2', '$3'))"
}

MAX_RETRIES=$(get_cfg slurm max_retries 1)
AUTO_EXCLUDE=$(get_cfg slurm auto_exclude true)
LOG_DIR=$(get_cfg slurm log_dir logs)
mkdir -p "$LOG_DIR"

# --- Manager Job Logic ---
# If not in SLURM and not forced to run locally, submit itself as a job
if [[ -z "$SLURM_JOB_ID" ]] && [[ "$1" != "--local" ]]; then
    echo "Detected login node. Submitting orchestrator as a SLURM manager job..."
    
    MGR_ACCOUNT=$(get_cfg slurm account "")
    MGR_PARTITION=$(get_cfg slurm partition "")
    MGR_QOS=$(get_cfg slurm qos "")
    MGR_NAME=$(get_cfg slurm job_name "quant_tests")
    
    # We use very few resources for the manager job itself
    # We pass all arguments to the manager job
    sbatch --job-name="mgr_$MGR_NAME" \
           --ntasks=1 \
           --cpus-per-task=1 \
           --mem=2G \
           --time=48:00:00 \
           ${MGR_ACCOUNT:+--account=$MGR_ACCOUNT} \
           ${MGR_PARTITION:+--partition=$MGR_PARTITION} \
           ${MGR_QOS:+--qos=$MGR_QOS} \
           --output="$LOG_DIR/manager_%j.out" \
           --error="$LOG_DIR/manager_%j.err" \
           "$0" "$@"
           
    echo "✓ Manager job submitted. Track progress in: $LOG_DIR/manager_JOBID.out"
    echo "  (Or run with --local to bypass this for debugging)"
    exit 0
fi
# --- End Manager Logic ---

# Initial tasks to run (all of them)
python3 <<'PYEOF' > .tasks_to_run
import json, os
cfg = json.load(open(os.environ['CONFIG_FILE']))
num_methods = len(cfg['quantization']['methods'])
num_tasks = len(cfg['tasks'])
num_custom = len(cfg.get('custom_tests', {}).get('tests', [])) if cfg.get('custom_tests', {}).get('enabled', False) else 0
print(f"0-{num_methods * num_tasks + num_methods * num_custom - 1}")
PYEOF
TASKS_TO_RUN=$(cat .tasks_to_run)
rm .tasks_to_run

RETRY_COUNT=0
SUCCESS=false

while [ $RETRY_COUNT -le $MAX_RETRIES ]; do
    echo "--- Attempt $((RETRY_COUNT + 1)) / $((MAX_RETRIES + 1)) ---"
    
    # Merge config exclude with persistent blacklist
    CONFIG_EXCLUDE=$(get_cfg slurm exclude_nodes "")
    PERSISTENT_EXCLUDE=$(paste -sd "," "$BLACKLIST_FILE")
    ALL_EXCLUDE=$(echo "$CONFIG_EXCLUDE,$PERSISTENT_EXCLUDE" | sed 's/^,//;s/,,/,/;s/,$//')
    
    echo "Tasks to run: $TASKS_TO_RUN"
    if [ ! -z "$ALL_EXCLUDE" ]; then
        echo "Excluding nodes: $ALL_EXCLUDE"
    fi

    # Generate the SLURM script for this attempt
    TEMP_SCRIPT=$(mktemp)
    export TASKS_TO_RUN
    export ALL_EXCLUDE
    python3 <<'PYEOF' > "$TEMP_SCRIPT"
import json, sys, os
cfg = json.load(open(os.environ['CONFIG_FILE']))
slurm = cfg['slurm']
tasks = os.environ.get('TASKS_TO_RUN', '0')
exclude = os.environ.get('ALL_EXCLUDE', '')
exclude_str = f"#SBATCH --exclude={exclude}" if exclude else ""

script = f"""#!/bin/bash
#SBATCH --job-name={slurm['job_name']}
#SBATCH --nodes={slurm['nodes']}
#SBATCH --ntasks={slurm['ntasks']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --partition={slurm['partition']}
{exclude_str}
#SBATCH --qos={slurm['qos']}
#SBATCH --account={slurm['account']}
#SBATCH --time={slurm['time']}
#SBATCH --array={tasks}
#SBATCH --output={slurm['log_dir']}/quant_%A_%a.out
#SBATCH --error={slurm['log_dir']}/quant_%A_%a.err
#SBATCH --gres={slurm['gres']}
set -euo pipefail

echo "NODE_HOSTNAME: $(hostname)"

module purge
module load cuda/12.6 
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /cta/users/fastinference2/workfolder/FIXEDvenv/bin/activate
# Unset memory variables to avoid conflict with srun (especially when running under a manager job)
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
srun python -u run_single_test.py --config {os.environ['CONFIG_FILE']} --array_id $SLURM_ARRAY_TASK_ID
deactivate
"""
print(script)
PYEOF

    # Submit
    echo "Submitting job array..."
    JOB_ID=$(sbatch --parsable "$TEMP_SCRIPT")
    EXIT_CODE=$?
    rm -f "$TEMP_SCRIPT"

    if [ $EXIT_CODE -ne 0 ] || [ -z "$JOB_ID" ]; then
        echo "✗ Failed to submit job."
        exit 1
    fi

    echo "Job ID $JOB_ID submitted. Monitoring progress..."

    # Polling loop
    while true; do
        # Check if any tasks of the job ID are still running or pending
        RUNNING_TASKS=$(squeue -j "$JOB_ID" -h | wc -l)
        if [ "$RUNNING_TASKS" -eq 0 ]; then
            # Double check with sacct to ensure it's actually finished.
            # We redirect stderr to /dev/null because on some nodes sacct may not 
            # have access to the Slurm database, causing redundant error messages.
            PENDING_SACCT=$(sacct -j "$JOB_ID" --format=State -n -P 2>/dev/null | grep -E "PENDING|RUNNING" | wc -l)
            if [ "x$PENDING_SACCT" == "x0" ]; then
                break
            fi
        fi
        echo -n "."
        sleep 30
    done
    echo ""
    echo "Job $JOB_ID finished."

    # Scan logs for failures and identify bad nodes
    export JOB_ID
    python3 <<'PYEOF' > .retry_info
import json, os, glob, re, sys

log_dir = os.environ.get('LOG_DIR', 'logs')
job_id = os.environ.get('JOB_ID')
blacklist_file = os.environ.get('BLACKLIST_FILE')

if not job_id:
    print("FAILED: JOB_ID not set")
    sys.exit(0)

out_files = glob.glob(f"{log_dir}/quant_{job_id}_*.out")
if not out_files:
    # If no logs yet, we might be too fast or srun failed completely
    # We should return a dummy failure to trigger log check again or retry
    print("ALL_FAILED_MISSING_LOGS")
    sys.exit(0)

failed_tasks = []
new_bad_nodes = set()

for out_path in out_files:
    task_id = out_path.split('_')[-1].split('.')[0]
    content = ""
    node = "unknown"
    
    with open(out_path, 'r') as f:
        content = f.read()
    
    # Extract hostname
    node_match = re.search(r"NODE_HOSTNAME: (.+)", content)
    if node_match:
        node = node_match.group(1).strip()
    
    if "TASK_STATUS: SUCCESS" not in content:
        failed_tasks.append(task_id)
        if node != "unknown":
            new_bad_nodes.add(node)

if not failed_tasks:
    print("SUCCESS")
else:
    # Update blacklist file
    if new_bad_nodes:
        with open(blacklist_file, 'a') as f:
            for n in new_bad_nodes:
                f.write(f"{n}\n")
    
    # Print failed tasks for Bash
    print(",".join(failed_tasks))
PYEOF
    
    RETRY_INFO=$(cat .retry_info)
    rm .retry_info

    if [ "$RETRY_INFO" == "SUCCESS" ]; then
        echo "✓ All tasks completed successfully!"
        SUCCESS=true
        break
    elif [ "$RETRY_INFO" == "ALL_FAILED_MISSING_LOGS" ]; then
        echo "✗ No log files found for job $JOB_ID. This usually means srun failed to start."
        # If no logs, we can't identify nodes, so we just marks everything as failed
        # to trigger a retry if possible. 
        # For simplicity, we just use the original TASKS_TO_RUN
        echo "Retrying job $JOB_ID entirely..."
        RETRY_COUNT=$((RETRY_COUNT + 1))
    else
        echo "✗ Tasks failed: $RETRY_INFO"
        TASKS_TO_RUN=$RETRY_INFO
        RETRY_COUNT=$((RETRY_COUNT + 1))
        
        if [ $RETRY_COUNT -le $MAX_RETRIES ]; then
            echo "Retrying failed tasks on different nodes..."
        else
            echo "Reached maximum retries. Some tasks still failed."
        fi
    fi
done

if [ "$SUCCESS" = true ]; then
    exit 0
else
    exit 1
fi
