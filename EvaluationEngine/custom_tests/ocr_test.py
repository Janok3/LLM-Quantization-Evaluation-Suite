"""
Custom OCR Test
===============
Evaluates a vision-language model's ability to understand text in images and PDFs.
Supports two modes:
  1. Text extraction: "Read all text in this image" → compare against ground_truth
  2. Visual Q&A: "What does the sign say?" → compare against expected answer

Usage (called automatically by run_single_test.py):
    python ocr_test.py \
        --model_id Qwen/Qwen2-VL-7B-Instruct \
        --quant_method base \
        --dataset datasets/ocr/default_ocr.jsonl \
        --output_dir lm_eval_results/custom_ocr \
        --trust_remote_code

Dataset formats:
    Text extraction:
        {"image": "doc.png", "ground_truth": "Hello World"}
    Visual Q&A:
        {"image": "sign.png", "question": "What does the sign say?", "ground_truth": "No Parking"}
    OCRBench style:
        {"image": "img.png", "question": "...", "ground_truth": "...", "category": "scene_text"}

Dependencies (pip install):
    - transformers, torch, Pillow
    - PyMuPDF (for PDF support): pip install PyMuPDF
    - qwen-vl-utils (for Qwen2-VL): pip install qwen-vl-utils
"""

import argparse
import csv
import json
import os
import sys
import datetime
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig


DEFAULT_EXTRACTION_PROMPT = (
    "Extract all the text from this image. "
    "Return only the extracted text, nothing else."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Custom OCR evaluation using a vision-language model")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace VLM model ID")
    parser.add_argument("--quant_method", type=str, required=True, choices=["base", "int8", "int4"],
                        help="Quantization method")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
                        help="Trust remote code when loading model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to OCR dataset file (.jsonl, .json, .csv). "
                             "Each entry must have 'image' and 'ground_truth'. "
                             "Optional 'question' field for VQA mode.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate per response")
    return parser.parse_args()


# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_ocr_dataset(dataset_path, max_samples=200):
    """Load OCR dataset entries.

    Each entry must have:
        - "image": path to image or PDF file
        - "ground_truth": expected answer / text
        - "question" (optional): custom question (VQA mode); if absent, uses extraction prompt
        - "page" (optional): page number for PDFs (0-indexed, default 0)
        - "category" (optional): for per-category reporting

    Returns a list of dicts.
    """
    print(f"\nLoading OCR dataset from: {dataset_path}")
    ext = os.path.splitext(dataset_path)[1].lower()

    entries = []
    if ext == ".jsonl":
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
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
    else:
        print(f"✗ Unsupported dataset format: {ext}")
        print("  Supported: .jsonl, .json, .csv")
        sys.exit(1)

    entries = entries[:max_samples]

    # Detect mode
    has_questions = sum(1 for e in entries if e.get("question"))
    mode = "vqa" if has_questions > len(entries) // 2 else "extraction"
    print(f"✓ Loaded {len(entries)} samples (mode: {mode}, {has_questions} with questions)")
    return entries, mode


def load_image_from_entry(entry, dataset_dir=None):
    """Load an image from a dataset entry, handling both images and PDFs.

    Returns a PIL Image.
    """
    image_path = entry["image"]

    # Resolve relative paths against dataset directory
    if dataset_dir and not os.path.isabs(image_path):
        image_path = os.path.join(dataset_dir, image_path)

    if not os.path.exists(image_path):
        print(f"  ⚠ Image not found: {image_path}")
        return None

    ext = os.path.splitext(image_path)[1].lower()

    if ext == ".pdf":
        return load_pdf_page(image_path, page=int(entry.get("page", 0)))
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
        return Image.open(image_path).convert("RGB")
    else:
        print(f"  ⚠ Unsupported image format: {ext}")
        return None


def load_pdf_page(pdf_path, page=0, dpi=200):
    """Convert a PDF page to a PIL Image using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("✗ PyMuPDF is required for PDF support: pip install PyMuPDF")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    if page >= len(doc):
        print(f"  ⚠ PDF has {len(doc)} pages, requested page {page}")
        return None

    pdf_page = doc[page]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = pdf_page.get_pixmap(matrix=mat)

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_id, quant_method, trust_remote_code):
    """Load a vision-language model with the specified quantization method."""
    print(f"Loading VLM: {model_id} (method: {quant_method})")

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }

    if quant_method == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quant_method == "int4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Try loading as Qwen2-VL first, then fall back to generic
    try:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    except (ImportError, Exception):
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    print(f"✓ VLM loaded successfully")
    return model, processor


# ── Inference ────────────────────────────────────────────────────────────────

def ask_vlm(model, processor, image, question, max_new_tokens):
    """Send an image + question to the VLM and return the response.

    Works for both text extraction and visual Q&A — the only difference is the question.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    try:
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        inputs = processor(
            text=question,
            images=image,
            return_tensors="pt",
        ).to(model.device)

    if hasattr(inputs, "pop"):
        inputs.pop("token_type_ids", None)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()

    return response


# ── Metrics ──────────────────────────────────────────────────────────────────

def edit_distance(s1, s2):
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(predicted, reference):
    """Character Error Rate: edit_distance(pred, ref) / len(ref)."""
    if not reference:
        return 0.0 if not predicted else 1.0
    return edit_distance(predicted, reference) / len(reference)


def compute_wer(predicted, reference):
    """Word Error Rate: edit_distance(pred_words, ref_words) / len(ref_words)."""
    pred_words = predicted.split()
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return edit_distance(pred_words, ref_words) / len(ref_words)


def compute_normalized_similarity(predicted, reference):
    """Normalized similarity: 1 - (edit_distance / max_length). Range [0, 1]."""
    if not reference and not predicted:
        return 1.0
    max_len = max(len(predicted), len(reference))
    if max_len == 0:
        return 1.0
    return 1.0 - (edit_distance(predicted, reference) / max_len)


def check_answer(predicted, reference):
    """Check answer for VQA mode (exact/contains match, case-insensitive)."""
    pred = predicted.strip().lower()
    ref = reference.strip().lower()
    exact = pred == ref or pred.startswith(ref)
    contains = ref in pred
    return exact, contains


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    start_time = datetime.datetime.now()

    print("=" * 60)
    print("  Custom OCR Test")
    print(f"  Model: {args.model_id}")
    print(f"  Quant: {args.quant_method}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print("=" * 60)

    # Load dataset
    ocr_data, mode = load_ocr_dataset(args.dataset)
    if not ocr_data:
        print("✗ No OCR samples found!")
        sys.exit(1)

    dataset_dir = os.path.dirname(os.path.abspath(args.dataset))

    # Load model
    model, processor = load_model(args.model_id, args.quant_method, args.trust_remote_code)

    # Evaluate
    sample_results = []
    total_cer = 0.0
    total_wer = 0.0
    total_similarity = 0.0
    exact_matches = 0
    contains_matches = 0
    skipped = 0

    for i, entry in enumerate(ocr_data):
        ground_truth = entry["ground_truth"].strip()
        category = entry.get("category", "general")
        question = entry.get("question", DEFAULT_EXTRACTION_PROMPT)

        image = load_image_from_entry(entry, dataset_dir)
        if image is None:
            skipped += 1
            continue

        is_vqa = "question" in entry
        label = f"Q: {question[:50]}" if is_vqa else entry["image"]
        print(f"\n  [{i+1}/{len(ocr_data)}] {label}")

        # Ask the VLM
        response = ask_vlm(model, processor, image, question, args.max_new_tokens)

        # Compute metrics
        cer = compute_cer(response, ground_truth)
        wer = compute_wer(response, ground_truth)
        similarity = compute_normalized_similarity(response, ground_truth)
        exact, contains = check_answer(response, ground_truth)

        total_cer += cer
        total_wer += wer
        total_similarity += similarity
        if exact:
            exact_matches += 1
        if contains:
            contains_matches += 1

        result_entry = {
            "image": entry["image"],
            "category": category,
            "ground_truth": ground_truth[:200],
            "model_response": response[:200],
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "similarity": round(similarity, 4),
            "exact_match": exact,
            "contains_match": contains,
        }
        if is_vqa:
            result_entry["question"] = question

        sample_results.append(result_entry)

        status = "✓" if exact else ("~" if contains else ("○" if similarity > 0.8 else "✗"))
        print(f"    {status} CER: {cer:.3f}  WER: {wer:.3f}  Sim: {similarity:.3f}")
        print(f"    Expected: {ground_truth[:80]}")
        print(f"    Got:      {response[:80]}")

    # Aggregate
    evaluated = len(sample_results)
    if evaluated == 0:
        print("✗ No samples could be evaluated!")
        sys.exit(1)

    avg_cer = total_cer / evaluated
    avg_wer = total_wer / evaluated
    avg_similarity = total_similarity / evaluated
    exact_pct = (exact_matches / evaluated) * 100
    contains_pct = (contains_matches / evaluated) * 100

    # Per-category breakdown
    categories = {}
    for r in sample_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"cer": [], "wer": [], "similarity": [], "exact": [], "contains": []}
        categories[cat]["cer"].append(r["cer"])
        categories[cat]["wer"].append(r["wer"])
        categories[cat]["similarity"].append(r["similarity"])
        categories[cat]["exact"].append(r["exact_match"])
        categories[cat]["contains"].append(r["contains_match"])

    category_summary = {}
    for cat, data in categories.items():
        n = len(data["cer"])
        category_summary[cat] = {
            "count": n,
            "avg_cer": round(sum(data["cer"]) / n, 4),
            "avg_wer": round(sum(data["wer"]) / n, 4),
            "avg_similarity": round(sum(data["similarity"]) / n, 4),
            "exact_match_pct": round(sum(data["exact"]) / n * 100, 1),
            "contains_match_pct": round(sum(data["contains"]) / n * 100, 1),
        }

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"  Results ({evaluated} samples, {skipped} skipped, mode: {mode}):")
    print(f"  Avg CER:              {avg_cer:.4f} (lower is better)")
    print(f"  Avg WER:              {avg_wer:.4f} (lower is better)")
    print(f"  Avg Similarity:       {avg_similarity:.4f} (higher is better)")
    print(f"  Exact Match:          {exact_pct:.1f}% ({exact_matches}/{evaluated})")
    print(f"  Contains Match:       {contains_pct:.1f}% ({contains_matches}/{evaluated})")
    print(f"  Duration:             {duration:.1f}s")
    if category_summary:
        print(f"\n  Per-category:")
        for cat, stats in category_summary.items():
            print(f"    {cat}: CER={stats['avg_cer']:.3f} WER={stats['avg_wer']:.3f} "
                  f"EM={stats['exact_match_pct']:.0f}% CM={stats['contains_match_pct']:.0f}% "
                  f"(n={stats['count']})")
    print(f"{'=' * 60}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "test_name": "ocr_custom",
        "model_id": args.model_id,
        "quant_method": args.quant_method,
        "mode": mode,
        "metrics": {
            "avg_cer": round(avg_cer, 6),
            "avg_wer": round(avg_wer, 6),
            "avg_similarity": round(avg_similarity, 6),
            "exact_match_pct": round(exact_pct, 2),
            "contains_match_pct": round(contains_pct, 2),
            "exact_matches": exact_matches,
            "contains_matches": contains_matches,
            "total_evaluated": evaluated,
            "total_skipped": skipped,
            "per_category": category_summary,
            "per_sample": sample_results,
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
