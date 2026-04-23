# LLM Quantization Evaluation Suite

A toolkit for evaluating, fine-tuning, and visualizing Large Language Models across different quantization methods.

---

## Project Components

### 1. EvaluationEngine
Benchmarks models (Base, INT8, INT4) on standard and custom tasks via SLURM.

### 2. FineTuning
LoRA fine-tuning with TRL (`finetune.py` uses `SFTConfig`, `llama_train.py` uses `TrainingArguments`).

### 3. ResultsDashboard
Flask web app for visualizing benchmark results. Start with `./run_dashboard.sh`.

---

## Quick Start

```bash
# 1. Configure
export HF_TOKEN="your_token"
vim EvaluationEngine/config.json

# 2. Run evaluations
cd EvaluationEngine && ./submit_quant_tests.sh

# 3. View results
cd ResultsDashboard && ./run_dashboard.sh
```

---

## Configuration

All settings are in `EvaluationEngine/config.json`.

### General Settings (portable)

These work on any SLURM cluster:

```json
{
  "model": {
    "id": "tiiuae/Falcon-H1-Tiny-90M-Base",  // Change model here
    "trust_remote_code": true,
    "device_map": "auto"
  },
  "quantization": {
    "methods": ["base", "int8", "int4"]         // Quantization levels
  },
  "tasks": [
    "wikitext", "arc_challenge", "truthfulqa_mc2",
    "gsm8k", "gsm8k_tr", "hendrycks_math"
  ],
  "custom_tests": {                              // Optional custom tests
    "enabled": true,
    "tests": [
      { "name": "perplexity_custom", "script": "perplexity_test.py" },
      { "name": "throughput_custom", "script": "throughput_test.py" },
      { "name": "coherence_custom", "script": "coherence_test.py" },
      { "name": "accuracy_custom", "script": "accuracy_test.py" },
      { "name": "ocr_custom", "script": "ocr_test.py" },
      { "name": "tool_calling", "script": "tool_calling_test.py" }
    ]
  },
  "evaluation": {
    "num_fewshot": 4,
    "batch_size": 8,
    "output_dir": "lm_eval_results"
  }
}
```

### Cluster-Specific Settings (Sabanci/Tosun)

These are pre-configured for the **Tosun cluster** at Sabanci University. External users must update them:

```json
{
  "slurm": {
    "job_name": "LLM_Eval_ENS492",
    "account": "cuda",           // Tosun: "cuda"
    "partition": "cuda",         // Tosun: "cuda"
    "qos": "cuda",               // Tosun: "cuda"
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 4,
    "time": "08:00:00",
    "gres": "gpu:1",
    "max_retries": 5,
    "auto_exclude": true,
    "exclude_nodes": "",
    "log_dir": "logs"
  }
}
```

The SLURM script (`submit_quant_tests.sh`) also contains **hardcoded environment setup** for Tosun (lines 126-149). Key things to change for other clusters:

```bash
# Line 131: CUDA module
module load cuda/12.6

# Line 133-136: CUDA paths
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Line 139: Virtualenv
CLUSTER_VENV="/cta/users/fastinference2/workfolder/FIXEDvenv/bin/activate"
```

Fine-tuning scripts (`FineTuning/train.sh`, `FineTuning/lama_train.sh`) also hardcode this venv path:

```bash
# train.sh line 24
source /cta/users/fastinference2/workfolder/FIXEDvenv/bin/activate
```

For the **FineTuning directory**, the model is set in the shell scripts:

```bash
# train.sh / lama_train.sh line 17
MODEL_ID="Qwen/Qwen3.5-4B"   # Change model here
```

---

## Custom Tests

| Test | What it measures | Dataset |
|------|-----------------|---------|
| `perplexity_test.py` | Perplexity on WikiText-2 | text |
| `throughput_test.py` | Tokens/sec and latency | text |
| `coherence_test.py` | Semantic coherence (cosine sim) | text |
| `accuracy_test.py` | Q&A accuracy | qa (question + answer) |
| `ocr_test.py` | Text extraction from images/PDFs | ocr (image + ground_truth) |
| `tool_calling_test.py` | Tool invocation accuracy | 8 hardcoded test cases |

Dataset files go in `EvaluationEngine/datasets/`. Supported formats: `.jsonl`, `.json`, `.csv`, `.txt`.

---

## Dashboard API

| Endpoint | Description |
|----------|-------------|
| `GET /api/data` | All benchmark results |
| `GET /api/models` | Evaluated models |
| `GET /api/checkpoints` | Found checkpoints/adapters |
| `GET /api/download_model?folder=...` | Download checkpoint as zip |
| `POST /api/parse` | Re-run results parser |

Results must be in `ResultsDashboard/lm_eval_results/` in the format `{model}_{quant}_{task}/` containing `results.json` files.

---

## Fine-Tuning

```bash
cd FineTuning
# Edit train.sh: set HF_TOKEN and MODEL_ID
sbatch train.sh
```

Output: LoRA adapters saved to `{output_dir}/final_adapter/`. Checkpoints are automatically discovered by the dashboard's `/api/checkpoints` endpoint.

---

## Adding a Custom Test

1. Create `EvaluationEngine/custom_tests/my_test.py` with `--model_id`, `--quant_method`, `--output_dir` arguments
2. Load model with quantization via `BitsAndBytesConfig(load_in_8bit=True)` or `load_in_4bit=True`
3. Save `results.json` to `--output_dir`
4. Add to `config.json` → `custom_tests` → `tests` array
5. Re-run `./submit_quant_tests.sh`