# Datasets Directory

Drop your dataset files here. The custom tests support three types:

## Dataset Types

### 📄 Text Type
**Used by:** perplexity, throughput, coherence tests

Each entry has a `"text"` field. Coherence also supports an optional `"domain"` field.

```jsonl
{"text": "Your text passage goes here."}
{"text": "Another passage for evaluation.", "domain": "science"}
```

---

### ❓ Q&A Type
**Used by:** accuracy test

Each entry has a `"question"` and `"answer"` field.

```jsonl
{"question": "What is the capital of France?", "answer": "Paris"}
{"question": "What is 2+2?", "answer": "4"}
```

---

### 🖼️ OCR Type
**Used by:** ocr test

Each entry has an `"image"` path and `"ground_truth"` text. Optional `"page"` for PDFs and `"category"` for reporting.

```jsonl
{"image": "images/document.png", "ground_truth": "Hello World", "category": "simple"}
{"image": "scans/invoice.pdf", "ground_truth": "Total: $100", "page": 0, "category": "document"}
```

**Supported image formats:** `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`, `.pdf`

**Setup:** Run `python datasets/ocr/generate_samples.py` on the HPC to create sample test images.

---

## Included Datasets

| File | Type | Description |
|------|------|-------------|
| `default_throughput.jsonl` | text | 10 prompts (short → long) for throughput benchmarking |
| `default_coherence.jsonl` | text | 12 domain-tagged prompts for coherence evaluation |
| `default_qa.jsonl` | qa | 25 Q&A pairs across science, history, math |
| `ocr/default_ocr.jsonl` | ocr | 10 images across multiple categories (generated) |
| `sample_text.jsonl` | text | 8 general text passages |
| `sample_qa.jsonl` | qa | 10 Q&A pairs |

## Supported File Formats

| Extension | Format |
|-----------|--------|
| `.jsonl` | One JSON object per line (recommended) |
| `.json` | Array of JSON objects |
| `.csv` | CSV with appropriate column headers |
| `.txt` | One entry per line (text type only) |

## Connecting to Tests

Add a `"dataset"` field to any test entry in `config.json`:

```json
{
  "name": "perplexity_custom",
  "script": "perplexity_test.py",
  "dataset": "datasets/my_data.jsonl"
}
```

For the OCR test, you can also override the vision model per-test:

```json
{
  "name": "ocr_custom",
  "script": "ocr_test.py",
  "model_id": "Qwen/Qwen2-VL-7B-Instruct",
  "dataset": "datasets/ocr/default_ocr.jsonl"
}
```

If `"dataset"` is omitted, perplexity falls back to WikiText-2 from HuggingFace.
Throughput and coherence fall back to their built-in hardcoded prompts.
Accuracy and OCR always require a dataset.
