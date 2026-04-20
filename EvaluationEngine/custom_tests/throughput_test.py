"""
Custom Throughput Test
=====================
Measures text generation speed (tokens/second) and latency across
different prompt lengths and generation configs.

Usage (called automatically by run_single_test.py):
    python throughput_test.py \
        --model_id tiiuae/Falcon-H1-Tiny-90M-Base \
        --quant_method base \
        --output_dir lm_eval_results/custom_throughput \
        --trust_remote_code

Requirements:
    1. Accept --model_id, --quant_method, and --output_dir CLI arguments
    2. Write a JSON results file to --output_dir
"""

import argparse
import csv
import json
import os
import sys
import time
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Custom throughput evaluation")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--quant_method", type=str, required=True, choices=["base", "int8", "int4"],
                        help="Quantization method")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
                        help="Trust remote code when loading model")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of generation runs per prompt length")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to a local dataset file (.jsonl, .json, .csv, .txt). "
                             "If omitted, built-in prompts are used.")
    return parser.parse_args()


def load_prompts_from_file(dataset_path, max_samples=20):
    """Load prompts from a local file.

    Supported formats:
        .jsonl / .json  – each object must have a "text" field
        .csv            – must have a "text" column
        .txt            – each non-empty line is one sample

    Returns a dict of {"prompt_0": text, "prompt_1": text, ...}
    """
    print(f"\nLoading prompts from: {dataset_path}")
    ext = os.path.splitext(dataset_path)[1].lower()

    if ext == ".jsonl":
        texts = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(json.loads(line)["text"])
    elif ext == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            texts = [item["text"] for item in data]
        else:
            texts = [data["text"]]
    elif ext == ".csv":
        texts = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row["text"])
    elif ext == ".txt":
        with open(dataset_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print(f"✗ Unsupported file format: {ext}")
        print("  Supported: .jsonl, .json, .csv, .txt")
        sys.exit(1)

    texts = [t for t in texts if t.strip()][:max_samples]
    return {f"prompt_{i}": t for i, t in enumerate(texts)}


def load_model(model_id, quant_method, trust_remote_code):
    """Load model with the specified quantization method."""
    print(f"Loading model: {model_id} (method: {quant_method})")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
    }

    model_kwargs["torch_dtype"] = torch.bfloat16

    if quant_method == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quant_method == "int4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, tokenizer


# Diverse prompts at different lengths to stress-test generation
PROMPTS = {
    "short": "The capital of France is",
    "medium": (
        "Artificial intelligence has transformed many industries in recent years. "
        "One of the most significant areas of impact has been natural language processing, "
        "where large language models have demonstrated remarkable capabilities in"
    ),
    "long": (
        "In the field of quantum computing, researchers have been exploring various "
        "approaches to achieve quantum advantage over classical computers. The two "
        "leading paradigms are gate-based quantum computing and quantum annealing. "
        "Gate-based systems, such as those built by IBM and Google, use quantum gates "
        "to manipulate qubits and perform computations. Quantum annealing, on the other "
        "hand, is used by D-Wave Systems and focuses on solving optimization problems. "
        "Recent breakthroughs in error correction and qubit coherence times have brought "
        "us closer to practical quantum computing applications. The implications for "
        "cryptography, drug discovery, and materials science are"
    ),
}


def measure_generation(model, tokenizer, prompt, max_new_tokens, num_runs):
    """Measure generation throughput and latency for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)
    input_length = inputs.input_ids.shape[1]

    # Warmup run (not counted)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=8, do_sample=False)

    latencies = []
    tokens_generated_list = []

    for run in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        torch.cuda.synchronize()
        end = time.perf_counter()

        elapsed = end - start
        tokens_generated = output_ids.shape[1] - input_length

        latencies.append(elapsed)
        tokens_generated_list.append(tokens_generated)

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(tokens_generated_list) / len(tokens_generated_list)
    avg_throughput = avg_tokens / avg_latency if avg_latency > 0 else 0.0

    # Time to first token (approximate from shortest run)
    min_latency = min(latencies)
    ttft_approx = min_latency / (max(tokens_generated_list) or 1)

    return {
        "input_tokens": input_length,
        "avg_tokens_generated": round(avg_tokens, 1),
        "avg_latency_seconds": round(avg_latency, 4),
        "avg_throughput_tokens_per_sec": round(avg_throughput, 2),
        "min_latency_seconds": round(min_latency, 4),
        "max_latency_seconds": round(max(latencies), 4),
        "approx_time_per_token_ms": round((avg_latency / avg_tokens) * 1000, 2) if avg_tokens > 0 else None,
    }


def main():
    args = parse_args()
    start_time = datetime.datetime.now()

    print("=" * 60)
    print("  Custom Throughput Test")
    print(f"  Model: {args.model_id}")
    print(f"  Quant: {args.quant_method}")
    print(f"  Runs per prompt: {args.num_runs}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(args.model_id, args.quant_method, args.trust_remote_code)

    # Ensure pad token is set for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    if args.dataset:
        prompts = load_prompts_from_file(args.dataset)
    else:
        prompts = PROMPTS
    print(f"  Using {len(prompts)} prompts")

    # Benchmark each prompt
    prompt_results = {}
    for label, prompt in prompts.items():
        print(f"\n  Benchmarking '{label}' prompt ({len(prompt)} chars)...")
        result = measure_generation(model, tokenizer, prompt, args.max_new_tokens, args.num_runs)
        prompt_results[label] = result
        print(f"    → {result['avg_throughput_tokens_per_sec']} tokens/sec "
              f"(avg latency: {result['avg_latency_seconds']:.3f}s)")

    # Compute aggregate metrics
    all_throughputs = [r["avg_throughput_tokens_per_sec"] for r in prompt_results.values()]
    overall_avg_throughput = round(sum(all_throughputs) / len(all_throughputs), 2)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"  Results:")
    print(f"  Overall avg throughput: {overall_avg_throughput} tokens/sec")
    print(f"  Duration: {duration:.1f}s")
    print(f"{'=' * 60}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "test_name": "throughput_custom",
        "model_id": args.model_id,
        "quant_method": args.quant_method,
        "metrics": {
            "overall_avg_throughput_tokens_per_sec": overall_avg_throughput,
            "per_prompt": prompt_results,
        },
        "config": {
            "num_runs": args.num_runs,
            "max_new_tokens": args.max_new_tokens,
            "prompts": {k: v[:80] + "..." if len(v) > 80 else v for k, v in PROMPTS.items()},
        },
        "runtime": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
        },
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    main()
