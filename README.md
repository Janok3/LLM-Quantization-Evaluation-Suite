# LLM Evaluation & Visualization Suite

A professional toolkit for quantifying, evaluating, and visualizing the performance of Large Language Models (LLMs) across different quantization methods.

## Features
- **Multi-Method Quantization Support**: Easily benchmark Base, INT8, and INT4 models.
- **Unified Evaluation Engine**: Standardized interface for `lm_eval` and custom domain-specific tests (OCR, Tool Calling, Coherence, etc.).
- **Smart SLURM Orchestration**: Automated "Manager Job" logic that handles node failures and blacklisting dynamically.
- **Interactive Dashboard**: Modular visualization tool for analyzing results and comparing quantization impacts.

## Project Structure

- **EvaluationEngine/**: The core testing framework.
  - `config.json`: Centralized configuration for models, tasks, and SLURM settings.
  - `submit_quant_tests.sh`: SLURM entry point with automated retry and node-blacklisting logic.
  - `custom_tests/`: Extensible directory for domain-specific evaluation scripts.
- **ResultsDashboard/**: Web-based analysis tool.
  - Provides a visual breakdown of benchmark performance.
  - Place your results in `lm_eval_results/` and run the dashboard to visualize.

## Setup & Configuration

### 1. Configuration (`config.json`)
The repository is **pre-configured for the ENS492 shared cluster**. 
- **Internal Users**: Standard SLURM settings (`account`, `partition`, `qos`) are already set.
- **External Users**: Please update the `slurm` object in `EvaluationEngine/config.json` with your specific credentials.

### 2. Environment Setup (`submit_quant_tests.sh`)
- **Shared Cluster**: The script is set to automatically use the pre-installed virtualenv at `/cta/users/fastinference2/...`.
- **Other Environments**: If the pre-configured venv is not found, the script will fallback to looking for a local `./venv` or `../venv`. 

### 3. Hugging Face Authentication
The engine requires a Hugging Face token to load models:
1.  **Environment Variable**: Set `export HF_TOKEN=your_token` (Recommended).
2.  **Config File**: Set the `token` field in `EvaluationEngine/config.json`.

### 3. Running Evaluations
On a SLURM-managed cluster:
```bash
cd EvaluationEngine
./submit_quant_tests.sh
```

### 4. Visualizing Results
1. Ensure your results are in `ResultsDashboard/lm_eval_results/`.
2. Start the dashboard:
```bash
cd ResultsDashboard
./run_dashboard.sh
```

## Contributing
This suite is designed to be extensible. Add new tests to `custom_tests/` and register them in the `config.json` to integrate them into the automated pipeline.
