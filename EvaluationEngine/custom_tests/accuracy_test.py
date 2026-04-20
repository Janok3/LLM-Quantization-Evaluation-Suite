"""
Custom Q&A Accuracy Test
========================
Evaluates a model's ability to answer questions correctly by comparing
generated responses against expected answers.

Usage (called automatically by run_single_test.py):
    python accuracy_test.py \
        --model_id tiiuae/Falcon-H1-Tiny-90M-Base \
        --quant_method base \
        --output_dir lm_eval_results/custom_accuracy \
        --dataset datasets/sample_qa.jsonl \
        --trust_remote_code

Requirements:
    1. Accept --model_id, --quant_method, --output_dir, and --dataset CLI arguments
    2. Dataset must have "question" and "answer" fields
    3. Write a JSON results file to --output_dir
"""

import argparse
import csv
import json
import os
import sys
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Custom Q&A accuracy evaluation")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--quant_method", type=str, required=True, choices=["base", "int8", "int4"],
                        help="Quantization method")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
                        help="Trust remote code when loading model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to a Q&A dataset file (.jsonl, .json, .csv). "
                             "Each entry must have 'question' and 'answer' fields.")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum number of tokens to generate per answer")
    return parser.parse_args()


def load_qa_pairs(dataset_path, max_samples=100):
    """Load question-answer pairs from a local file.

    Supported formats:
        .jsonl / .json  – each object must have "question" and "answer" fields
        .csv            – must have "question" and "answer" columns

    Returns a list of {"question": ..., "answer": ...} dicts.
    """
    print(f"\nLoading Q&A dataset from: {dataset_path}")
    ext = os.path.splitext(dataset_path)[1].lower()

    pairs = []
    if ext == ".jsonl":
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    pairs.append({"question": obj["question"], "answer": obj["answer"]})
    elif ext == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            pairs = [{"question": d["question"], "answer": d["answer"]} for d in data]
        else:
            pairs = [{"question": data["question"], "answer": data["answer"]}]
    elif ext == ".csv":
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append({"question": row["question"], "answer": row["answer"]})
    else:
        print(f"✗ Unsupported file format: {ext}")
        print("  Supported: .jsonl, .json, .csv")
        sys.exit(1)

    pairs = pairs[:max_samples]
    print(f"✓ Loaded {len(pairs)} Q&A pairs")
    return pairs


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


def check_answer(generated_text, expected_answer):
    """Check if the generated text contains the expected answer.

    Returns:
        exact_match: bool – True if the answer matches exactly (case-insensitive)
        contains_match: bool – True if the answer appears anywhere in the response
    """
    gen_clean = generated_text.strip().lower()
    exp_clean = expected_answer.strip().lower()

    # Exact match: the generated text starts with or equals the expected answer
    exact_match = gen_clean == exp_clean or gen_clean.startswith(exp_clean)

    # Contains match: the expected answer appears anywhere in the response
    contains_match = exp_clean in gen_clean

    return exact_match, contains_match


def main():
    args = parse_args()
    start_time = datetime.datetime.now()

    print("=" * 60)
    print("  Custom Q&A Accuracy Test")
    print(f"  Model: {args.model_id}")
    print(f"  Quant: {args.quant_method}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print("=" * 60)

    # Load Q&A pairs
    qa_pairs = load_qa_pairs(args.dataset)

    if not qa_pairs:
        print("✗ No Q&A pairs found in dataset!")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model(args.model_id, args.quant_method, args.trust_remote_code)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate each question
    question_results = []
    exact_correct = 0
    contains_correct = 0

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        expected = qa["answer"]

        # Format prompt to encourage short answers
        prompt = f"Question: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        # Decode only the generated part
        generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Take only the first line of the response (avoid rambling)
        first_line = generated_text.split("\n")[0].strip()

        exact_match, contains_match = check_answer(first_line, expected)

        if exact_match:
            exact_correct += 1
        if contains_match:
            contains_correct += 1

        status = "✓ EXACT" if exact_match else ("~ CONTAINS" if contains_match else "✗ WRONG")

        question_results.append({
            "question": question,
            "expected_answer": expected,
            "model_answer": first_line,
            "exact_match": exact_match,
            "contains_match": contains_match,
        })

        print(f"  [{i+1}/{len(qa_pairs)}] {status}")
        print(f"    Q: {question}")
        print(f"    Expected: {expected}")
        print(f"    Got: {first_line[:120]}")

    # Aggregate metrics
    total = len(qa_pairs)
    exact_accuracy = (exact_correct / total) * 100
    contains_accuracy = (contains_correct / total) * 100

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"  Results:")
    print(f"  Exact match accuracy:    {exact_accuracy:.1f}% ({exact_correct}/{total})")
    print(f"  Contains match accuracy: {contains_accuracy:.1f}% ({contains_correct}/{total})")
    print(f"  Duration: {duration:.1f}s")
    print(f"{'=' * 60}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "test_name": "accuracy_custom",
        "model_id": args.model_id,
        "quant_method": args.quant_method,
        "metrics": {
            "exact_match_accuracy_pct": round(exact_accuracy, 2),
            "contains_match_accuracy_pct": round(contains_accuracy, 2),
            "exact_correct": exact_correct,
            "contains_correct": contains_correct,
            "total_questions": total,
            "per_question": question_results,
        },
        "config": {
            "dataset": args.dataset,
            "max_new_tokens": args.max_new_tokens,
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
