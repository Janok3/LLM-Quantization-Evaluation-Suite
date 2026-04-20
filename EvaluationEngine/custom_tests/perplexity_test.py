"""
Custom Perplexity Test
======================
A sample custom test that computes perplexity on the WikiText-2 dataset.

Usage (called automatically by run_single_test.py):
    python perplexity_test.py \
        --model_id tiiuae/Falcon-H1-Tiny-90M-Base \
        --quant_method base \
        --output_dir lm_eval_results/custom_perplexity \
        --trust_remote_code

You can use this as a template for your own custom tests.
The only requirements are:
    1. Accept --model_id, --quant_method, and --output_dir CLI arguments
    2. Write a JSON results file to --output_dir
"""

import argparse
import csv
import json
import os
import sys
import math
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Custom perplexity evaluation")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--quant_method", type=str, required=True, choices=["base", "int8", "int4"],
                        help="Quantization method")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
                        help="Trust remote code when loading model")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for evaluation")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to a local dataset file (.jsonl, .json, .csv, .txt). "
                             "If omitted, WikiText-2 is used.")
    return parser.parse_args()


def load_texts(dataset_path, max_samples):
    """Load text samples from a local file or fall back to WikiText-2.

    Supported formats:
        .jsonl / .json  – each object must have a "text" field
        .csv            – must have a "text" column
        .txt            – each non-empty line is one sample
    """
    if dataset_path is None:
        # Default: WikiText-2 from HuggingFace
        from datasets import load_dataset
        print("\nLoading WikiText-2 test set (default)...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [s["text"] for s in dataset if s["text"].strip()]
        return texts[:max_samples]

    print(f"\nLoading dataset from: {dataset_path}")
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

    texts = [t for t in texts if t.strip()]
    return texts[:max_samples]


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


def compute_perplexity(model, tokenizer, texts, max_length):
    """Compute perplexity over a list of text samples."""
    total_loss = 0.0
    total_tokens = 0

    for i, text in enumerate(texts):
        if not text.strip():
            continue

        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encodings.input_ids.to(model.device)

        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids)
            # Cast logits to float32 to avoid NaN from float16 overflow
            logits = outputs.logits.float()

            # Shift so that token n predicts token n+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )

        num_tokens = input_ids.shape[1] - 1
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        if (i + 1) % 10 == 0:
            running_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
            print(f"  Processed {i + 1}/{len(texts)} samples (running PPL: {running_ppl:.2f})")

    if total_tokens == 0:
        return float("inf"), 0

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, total_tokens


def main():
    args = parse_args()
    start_time = datetime.datetime.now()

    print("=" * 60)
    print("  Custom Perplexity Test")
    print(f"  Model: {args.model_id}")
    print(f"  Quant: {args.quant_method}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Max length: {args.max_length}")
    print("=" * 60)

    # Load dataset
    texts = load_texts(args.dataset, args.max_samples)
    print(f"✓ Loaded {len(texts)} text samples")

    # Load model
    model, tokenizer = load_model(args.model_id, args.quant_method, args.trust_remote_code)

    # Compute perplexity
    print("\nComputing perplexity...")
    perplexity, total_tokens = compute_perplexity(model, tokenizer, texts, args.max_length)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"  Results:")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Total tokens evaluated: {total_tokens}")
    print(f"  Duration: {duration:.1f}s")
    print(f"{'=' * 60}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "test_name": "perplexity_custom",
        "model_id": args.model_id,
        "quant_method": args.quant_method,
        "metrics": {
            "perplexity": perplexity,
            "total_tokens": total_tokens,
        },
        "config": {
            "max_samples": args.max_samples,
            "max_length": args.max_length,
            "dataset": "wikitext-2-raw-v1",
            "split": "test",
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
