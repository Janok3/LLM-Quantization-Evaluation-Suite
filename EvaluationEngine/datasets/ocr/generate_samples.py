"""
Generate Sample OCR Test Images
================================
Creates simple test images with text rendered on them for testing the OCR pipeline.
Run this script once to populate the datasets/ocr/ directory with sample images.

Usage:
    python datasets/ocr/generate_samples.py

Dependencies:
    pip install Pillow
"""

import json
import os
from PIL import Image, ImageDraw, ImageFont


# Sample texts for different categories
SAMPLES = [
    {
        "text": "Hello, World!",
        "category": "simple",
        "filename": "simple_hello.png",
    },
    {
        "text": "The quick brown fox jumps over the lazy dog.",
        "category": "simple",
        "filename": "simple_pangram.png",
    },
    {
        "text": "Machine learning is a subset of artificial intelligence.",
        "category": "sentence",
        "filename": "sentence_ml.png",
    },
    {
        "text": "E = mc^2",
        "category": "formula",
        "filename": "formula_einstein.png",
    },
    {
        "text": "Total: $1,234.56\nDate: 2026-01-15\nVendor: Acme Corp",
        "category": "document",
        "filename": "document_invoice.png",
    },
    {
        "text": "Python 3.12\nRelease Date: October 2023\nPEP 695: Type Parameter Syntax",
        "category": "document",
        "filename": "document_release.png",
    },
    {
        "text": "Item        Qty    Price\nWidget A     10    $5.00\nWidget B      5   $12.50\nTotal:       15   $112.50",
        "category": "table",
        "filename": "table_items.png",
    },
    {
        "text": "ABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz\n0123456789",
        "category": "characters",
        "filename": "characters_alphabet.png",
    },
    {
        "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "category": "paragraph",
        "filename": "paragraph_lorem.png",
    },
    {
        "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "category": "code",
        "filename": "code_fibonacci.png",
    },
]


def create_text_image(text, output_path, width=800, padding=40, font_size=24, bg_color="white", text_color="black"):
    """Render text onto a clean image."""
    # Try to use a monospace font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Calculate image dimensions
    lines = text.split("\n")
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    line_heights = []
    for line in lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])

    line_height = max(line_heights) if line_heights else font_size
    total_height = len(lines) * (line_height + 8) + 2 * padding

    # Create image
    img = Image.new("RGB", (width, total_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=text_color, font=font)
        y += line_height + 8

    img.save(output_path)
    return img


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    dataset_entries = []

    print("Generating sample OCR test images...")
    for sample in SAMPLES:
        output_path = os.path.join(images_dir, sample["filename"])
        create_text_image(sample["text"], output_path)

        dataset_entries.append({
            "image": f"images/{sample['filename']}",
            "ground_truth": sample["text"],
            "category": sample["category"],
        })

        print(f"  ✓ {sample['filename']} ({sample['category']})")

    # Write default dataset JSONL
    dataset_path = os.path.join(script_dir, "default_ocr.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✓ Generated {len(SAMPLES)} images in {images_dir}/")
    print(f"✓ Dataset written to {dataset_path}")
    print(f"\nTo use: python ocr_test.py --dataset {dataset_path} ...")


if __name__ == "__main__":
    main()
