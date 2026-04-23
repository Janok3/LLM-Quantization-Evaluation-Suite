#!/bin/bash
#SBATCH --job-name=LlamaTrain
#SBATCH --account=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=cuda
#SBATCH --partition=cuda
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

# --- Setup ---
MODEL_ID="Qwen/Qwen3.5-4B"

# PLEASE USE A NEW TOKEN HERE
export HF_TOKEN="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"

# Environment
module purge
source /cta/users/fastinference2/workfolder/FIXEDvenv/bin/activate

mkdir -p logs
mkdir -p results

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Starting Training for $MODEL_ID"
echo "=================================================="

# Run Training
srun python -u llama_train.py \
    --output_dir "./results/lama3b-$SLURM_JOB_ID" \
    --epochs 1 \
    --learning_rate 2e-4

echo "=================================================="
echo "Training Completed."