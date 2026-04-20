"""
Download OCRBench Dataset
=========================
Downloads the OCRBench dataset from HuggingFace and converts it to our
JSONL format for use with ocr_test.py.

Usage:
    python datasets/ocr/download_ocrbench.py

This will create:
    datasets/ocr/ocrbench.jsonl     — the dataset file
    datasets/ocr/ocrbench_images/   — downloaded images

Dependencies:
    pip install datasets Pillow
"""

import json
import os

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("✗ 'datasets' package required: pip install datasets")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "ocrbench_images")
    os.makedirs(images_dir, exist_ok=True)

    print("Downloading OCRBench from HuggingFace...")
    print("(This may take a few minutes on first run)")

    try:
        ds = load_dataset("echo840/OCRBench", split="test")
    except Exception as e:
        print(f"✗ Failed to download OCRBench: {e}")
        print("  Try: pip install datasets && huggingface-cli login")
        return

    print(f"✓ Downloaded {len(ds)} samples")

    entries = []
    skipped = 0

    for idx, sample in enumerate(ds):
        # OCRBench has: image, question, answer, dataset_name, etc.
        image = sample.get("image")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        category = sample.get("dataset_name", "general")

        if not image or not question or not answer:
            skipped += 1
            continue

        # Save image to disk
        image_filename = f"ocrbench_{idx:04d}.png"
        image_path = os.path.join(images_dir, image_filename)
        image.save(image_path, "PNG")

        entries.append({
            "image": f"ocrbench_images/{image_filename}",
            "question": question,
            "ground_truth": str(answer),
            "category": category,
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(ds)}...")

    # Write JSONL
    output_path = os.path.join(script_dir, "ocrbench.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved {len(entries)} samples to {output_path} (skipped {skipped})")
    print(f"✓ Images saved to {images_dir}/")
    print(f"\nTo use with our pipeline:")
    print(f'  Set "dataset": "datasets/ocr/ocrbench.jsonl" in config.json')


if __name__ == "__main__":
    main()
