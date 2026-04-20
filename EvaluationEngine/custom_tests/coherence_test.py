"""
Custom Semantic Coherence Test
==============================
Evaluates how semantically coherent a model's generated continuations are
relative to the original prompt, using the model's own hidden-state embeddings
to compute cosine similarity.

Usage (called automatically by run_single_test.py):
    python coherence_test.py \
        --model_id tiiuae/Falcon-H1-Tiny-90M-Base \
        --quant_method base \
        --output_dir lm_eval_results/custom_coherence \
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
import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Custom semantic coherence evaluation")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--quant_method", type=str, required=True, choices=["base", "int8", "int4"],
                        help="Quantization method")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
                        help="Trust remote code when loading model")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Number of tokens to generate for each continuation")
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

    Returns a list of {"id": ..., "text": ..., "domain": "custom"} dicts.
    """
    print(f"\nLoading prompts from: {dataset_path}")
    ext = os.path.splitext(dataset_path)[1].lower()

    if ext == ".jsonl":
        entries = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    elif ext == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else [data]
    elif ext == ".csv":
        entries = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
    elif ext == ".txt":
        with open(dataset_path, "r", encoding="utf-8") as f:
            entries = [{"text": line.strip()} for line in f if line.strip()]
    else:
        print(f"✗ Unsupported file format: {ext}")
        print("  Supported: .jsonl, .json, .csv, .txt")
        sys.exit(1)

    entries = [e for e in entries if e.get("text", "").strip()][:max_samples]
    return [
        {
            "id": e.get("id", e.get("domain", f"custom_{i}")),
            "text": e["text"],
            "domain": e.get("domain", "custom"),
        }
        for i, e in enumerate(entries)
    ]


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


# Test prompts spanning different domains and difficulty levels
PROMPTS = [
    {
        "id": "science",
        "text": "Photosynthesis is the process by which plants convert sunlight into",
        "domain": "biology",
    },
    {
        "id": "history",
        "text": "The French Revolution, which began in 1789, fundamentally changed",
        "domain": "history",
    },
    {
        "id": "technology",
        "text": "Machine learning algorithms can be broadly categorized into supervised and unsupervised methods, where",
        "domain": "computer science",
    },
    {
        "id": "literature",
        "text": "Shakespeare's influence on the English language extends far beyond his plays, as he",
        "domain": "literature",
    },
    {
        "id": "mathematics",
        "text": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse",
        "domain": "mathematics",
    },
    {
        "id": "philosophy",
        "text": "Existentialism, as articulated by Sartre and Camus, argues that human existence",
        "domain": "philosophy",
    },
    {
        "id": "geography",
        "text": "The Amazon rainforest, spanning across nine countries in South America, is home to",
        "domain": "geography",
    },
    {
        "id": "economics",
        "text": "Supply and demand is a fundamental economic model that explains how prices are determined in",
        "domain": "economics",
    },
]


def get_mean_embedding(model, tokenizer, text):
    """Get the mean-pooled last hidden state embedding for a text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        # Use the last hidden state layer
        hidden_states = outputs.hidden_states[-1].float()  # cast to float32
        # Mean pool across sequence length
        embedding = hidden_states.mean(dim=1)  # shape: (1, hidden_dim)

    return embedding


def compute_coherence(model, tokenizer, prompt_text, max_new_tokens):
    """
    Compute semantic coherence between a prompt and its generated continuation.

    Returns:
        cosine_similarity: float in [-1, 1], higher = more coherent
        generated_text: the generated continuation string
    """
    # 1. Get embedding for the original prompt
    prompt_embedding = get_mean_embedding(model, tokenizer, prompt_text)

    # 2. Generate continuation
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the generated part
    generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if not generated_text.strip():
        return 0.0, "(empty generation)"

    # 3. Get embedding for prompt + continuation (full context)
    full_text = prompt_text + generated_text
    full_embedding = get_mean_embedding(model, tokenizer, full_text)

    # 4. Compute cosine similarity
    similarity = F.cosine_similarity(prompt_embedding, full_embedding, dim=1).item()

    return similarity, generated_text


def main():
    args = parse_args()
    start_time = datetime.datetime.now()

    print("=" * 60)
    print("  Custom Semantic Coherence Test")
    print(f"  Model: {args.model_id}")
    print(f"  Quant: {args.quant_method}")
    print(f"  Prompts: {len(PROMPTS)}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(args.model_id, args.quant_method, args.trust_remote_code)

    # Load prompts
    if args.dataset:
        prompts = load_prompts_from_file(args.dataset)
    else:
        prompts = PROMPTS
    print(f"  Using {len(prompts)} prompts")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate each prompt
    prompt_results = []
    total_similarity = 0.0

    for i, prompt_info in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] Testing '{prompt_info['id']}' ({prompt_info['domain']})...")

        similarity, generated_text = compute_coherence(
            model, tokenizer, prompt_info["text"], args.max_new_tokens
        )

        total_similarity += similarity
        preview = generated_text[:100] + "..." if len(generated_text) > 100 else generated_text

        prompt_results.append({
            "prompt_id": prompt_info["id"],
            "domain": prompt_info["domain"],
            "cosine_similarity": round(similarity, 6),
            "generated_preview": preview,
        })

        print(f"    → Coherence: {similarity:.4f}")
        print(f"    → Generated: {preview}")

    # Aggregate metrics
    avg_coherence = total_similarity / len(prompts)
    min_coherence = min(r["cosine_similarity"] for r in prompt_results)
    max_coherence = max(r["cosine_similarity"] for r in prompt_results)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"  Results:")
    print(f"  Avg coherence (cosine sim): {avg_coherence:.4f}")
    print(f"  Min coherence: {min_coherence:.4f}")
    print(f"  Max coherence: {max_coherence:.4f}")
    print(f"  Duration: {duration:.1f}s")
    print(f"{'=' * 60}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "test_name": "coherence_custom",
        "model_id": args.model_id,
        "quant_method": args.quant_method,
        "metrics": {
            "avg_coherence": round(avg_coherence, 6),
            "min_coherence": round(min_coherence, 6),
            "max_coherence": round(max_coherence, 6),
            "per_prompt": prompt_results,
        },
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": len(PROMPTS),
            "domains_tested": list(set(p["domain"] for p in PROMPTS)),
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
