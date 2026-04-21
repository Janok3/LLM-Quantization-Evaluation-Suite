#!/usr/bin/env python3
"""
LLM Evaluation Runner
---------------------
Orchestrates the evaluation of Large Language Models (LLMs) across different 
quantization methods using both standard benchmarks and custom test suites.

Supports:
- lm_eval benchmarks
- Custom domain-specific tests (OCR, Tool Detection, etc.)
- SLURM-based parallel execution
"""

import os
import subprocess
import sys
import torch
import datetime
import argparse
import json
from huggingface_hub import login


def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"✗ Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in configuration file: {e}")
        sys.exit(1)


def authenticate(token):
    """Authenticate with Hugging Face."""
    try:
        login(token=token, add_to_git_credential=False)
        print("✓ Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"✗ Hugging Face Login failed: {e}")
        sys.exit(1)


def get_custom_tests(config):
    """Get list of enabled custom tests from config, or empty list if disabled/missing."""
    custom_cfg = config.get('custom_tests', {})
    if not custom_cfg.get('enabled', False):
        return []
    return custom_cfg.get('tests', [])


def get_method_and_task(array_id, methods, tasks, custom_tests=None):
    """
    Calculate which method and task to run based on array ID.
    
    Array layout:
        IDs 0 .. (num_methods * num_tasks - 1)         → lm_eval tasks
        IDs (num_methods * num_tasks) .. end             → custom tests
    
    Returns:
        (method, task_name, is_custom, custom_test_info)
        - is_custom: bool — True if this is a custom test
        - custom_test_info: dict with 'name', 'script', etc. (None for lm_eval tasks)
    """
    if custom_tests is None:
        custom_tests = []
    
    num_lm_tasks = len(tasks)
    num_methods = len(methods)
    lm_total = num_methods * num_lm_tasks
    
    if array_id < lm_total:
        # Standard lm_eval task
        method_index = array_id // num_lm_tasks
        task_index = array_id % num_lm_tasks
        return methods[method_index], tasks[task_index], False, None
    else:
        # Custom test
        custom_id = array_id - lm_total
        num_custom = len(custom_tests)
        method_index = custom_id // num_custom
        custom_index = custom_id % num_custom
        return methods[method_index], custom_tests[custom_index]['name'], True, custom_tests[custom_index]


def run_eval_command(config, quant_method, task_name):
    """Run the lm_eval evaluation command with specified quantization method and task."""
    model_id = config['model']['id']
    model_short_name = model_id.split('/')[-1]
    output_dir = config['evaluation']['output_dir']
    
    # Unique directory for each specific Quant Method + Task combination
    output_dir_name = f"{output_dir}/{model_short_name}_{quant_method}_{task_name}"
    os.makedirs(output_dir_name, exist_ok=True)
    
    start_time = datetime.datetime.now()
    print("=" * 80)
    print(f"    [lm_eval] Task: {task_name} | Method: {quant_method}")
    print(f"    Output: {output_dir_name}")
    print(f"    Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Build model arguments
    base_model_args = (
        f"pretrained={model_id},"
        f"trust_remote_code={str(config['model']['trust_remote_code']).lower()},"
        f"device_map={config['model']['device_map']}"
    )
    
    if quant_method == "int8":
        model_args = f"{base_model_args},load_in_8bit=True"
    elif quant_method == "int4":
        model_args = f"{base_model_args},load_in_4bit=True"
    else:
        model_args = base_model_args
    
    # Build command
    command_list = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", task_name,
        "--num_fewshot", str(config['evaluation']['num_fewshot']),
        "--batch_size", str(config['evaluation']['batch_size']),
        "--output_path", output_dir_name
    ]
    
    try:
        subprocess.run(command_list, check=True)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"\n✓ Finished {task_name} with {quant_method}")
        print(f"  Duration: {duration}")
        print("TASK_STATUS: SUCCESS")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR during {task_name}: Exit code {e.returncode}")
        print("TASK_STATUS: FAILED")
        sys.exit(1)


def run_custom_test(config, quant_method, test_info):
    """Run a custom test script with the specified quantization method."""
    # Per-test model override: use test-specific model_id if provided
    model_id = test_info.get('model_id', config['model']['id'])
    model_short_name = model_id.split('/')[-1]
    output_dir = config['evaluation']['output_dir']
    test_dir = config.get('custom_tests', {}).get('test_dir', 'custom_tests')
    
    test_name = test_info['name']
    test_script = test_info['script']
    script_path = os.path.join(test_dir, test_script)
    
    # Unique directory for each specific Quant Method + Custom Test combination
    output_dir_name = f"{output_dir}/{model_short_name}_{quant_method}_{test_name}"
    os.makedirs(output_dir_name, exist_ok=True)
    
    start_time = datetime.datetime.now()
    print("=" * 80)
    print(f"    [Custom Test] {test_name} | Method: {quant_method}")
    print(f"    Model: {model_id}")
    print(f"    Script: {script_path}")
    print(f"    Output: {output_dir_name}")
    print(f"    Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Build command for custom test
    command_list = [
        sys.executable, script_path,
        "--model_id", model_id,
        "--quant_method", quant_method,
        "--output_dir", output_dir_name,
    ]
    
    if config['model'].get('trust_remote_code', False):
        command_list.append("--trust_remote_code")
    
    # Pass dataset path if configured for this test
    if test_info.get('dataset'):
        command_list.extend(["--dataset", test_info['dataset']])
    
    try:
        subprocess.run(command_list, check=True)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"\n✓ Finished custom test '{test_name}' with {quant_method}")
        print(f"  Duration: {duration}")
        print("TASK_STATUS: SUCCESS")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR during custom test '{test_name}': Exit code {e.returncode}")
        print("TASK_STATUS: FAILED")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run single quantization test")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--array_id",
        type=int,
        help="SLURM array task ID (for batch submission)"
    )
    parser.add_argument(
        "--quant_method",
        type=str,
        choices=["base", "int8", "int4"],
        help="Quantization method (alternative to array_id)"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="The specific lm_eval benchmark task to run (alternative to array_id)"
    )
    parser.add_argument(
        "--custom_test",
        type=str,
        help="Name of a custom test to run (alternative to array_id, use with --quant_method)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Authenticate with Hugging Face
    hf_token = os.environ.get("HF_TOKEN") or config.get('huggingface', {}).get('token')
    if hf_token and hf_token != "YOUR_HF_TOKEN_HERE":
        authenticate(hf_token)
    else:
        print("⚠ No Hugging Face token found in environment (HF_TOKEN) or config. Manual login might be required for private models.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ CUDA not available.")
        sys.exit(1)
    
    # Get custom tests from config
    custom_tests = get_custom_tests(config)
    
    # Determine method and task
    if args.array_id is not None:
        # Using SLURM array mode
        quant_method, task_name, is_custom, custom_test_info = get_method_and_task(
            args.array_id,
            config['quantization']['methods'],
            config['tasks'],
            custom_tests
        )
        prefix = "[Custom]" if is_custom else "[lm_eval]"
        print(f"NODE_HOSTNAME: {os.uname().nodename}")
        print(f"Array ID: {args.array_id} → {prefix} Method: {quant_method}, Task: {task_name}")
        
        if is_custom:
            run_custom_test(config, quant_method, custom_test_info)
        else:
            run_eval_command(config, quant_method, task_name)
    
    elif args.custom_test and args.quant_method:
        # Manual custom test mode
        quant_method = args.quant_method
        test_info = None
        for ct in custom_tests:
            if ct['name'] == args.custom_test:
                test_info = ct
                break
        
        if test_info is None:
            available = [ct['name'] for ct in custom_tests]
            print(f"✗ Custom test '{args.custom_test}' not found in config.")
            print(f"  Available custom tests: {available}")
            sys.exit(1)
        
        if quant_method not in config['quantization']['methods']:
            print(f"✗ Method '{quant_method}' not in config")
            sys.exit(1)
        
        run_custom_test(config, quant_method, test_info)
    
    elif args.quant_method and args.task:
        # Using manual lm_eval mode
        quant_method = args.quant_method
        task_name = args.task
        
        # Validate against config
        if quant_method not in config['quantization']['methods']:
            print(f"✗ Method '{quant_method}' not in config")
            sys.exit(1)
        if task_name not in config['tasks']:
            print(f"✗ Task '{task_name}' not in config")
            sys.exit(1)
        
        run_eval_command(config, quant_method, task_name)
    else:
        print("✗ Must provide one of:")
        print("  --array_id")
        print("  --quant_method and --task (for lm_eval benchmarks)")
        print("  --quant_method and --custom_test (for custom tests)")
        sys.exit(1)


if __name__ == "__main__":
    main()
