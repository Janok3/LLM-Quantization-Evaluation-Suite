#!/usr/bin/env python3
"""
parse_lm_eval_results.py

Parses an lm_eval_results folder structured like:
  <model>_<quant>_<task>/results_*.json

Example subdirs:
  gpt2_base_gsm8k
  gpt2_int8_mmlu_tr
  Llama-3.1-8B-Instruct_int4_arc_challenge
  ...

Outputs:
  - One CSV per model: <out_dir>/<model>_wide.csv
  - One CSV per model: <out_dir>/<model>_long.csv
  - (Optional) a combined CSV across all models

Wide format: one row per (quant_method, task_key-from-json) with primary metric + all metrics as columns.
Long format: one row per metric.

Usage:
  python parse_lm_eval_results.py --root lm_eval_results --out_dir parsed

Notes:
- Metadata (model/quant/task) is taken from the DIRECTORY name, not the json filename.
- If a directory contains multiple results_*.json files, all are read (rows include source_file).
- If a JSON "results" contains multiple task keys (e.g., mmlu_* subtasks), those are kept as separate tasks.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Directory format: <model>_<quant>_<task>
# quant allowed: base | int4 | int8 (extend if needed)
DIR_RE = re.compile(r"^(?P<model>.+)_(?P<quant>base|int4|int8)_(?P<task>.+)$")


def split_metric_key(k: str) -> Tuple[str, Optional[str]]:
    # "acc,none" -> ("acc", "none")
    if "," in k:
        a, b = k.split(",", 1)
        return a.strip(), b.strip()
    return k.strip(), None


def is_stderr_metric(metric_name: str) -> bool:
    return metric_name.endswith("_stderr")


def pick_primary_metric(metrics: Dict[Tuple[str, Optional[str]], float]) -> Optional[Tuple[str, Optional[str]]]:
    """
    Heuristic for a primary score per task:
      - prefer acc_norm
      - else acc
      - else exact_match (prefer strict-match)
      - else first metric
    Excludes *_stderr.
    """
    keys = [k for k in metrics.keys() if not is_stderr_metric(k[0])]
    if not keys:
        return None

    accn = [k for k in keys if k[0] == "acc_norm"]
    if accn:
        return sorted(accn, key=lambda x: "" if x[1] is None else x[1])[0]

    acc = [k for k in keys if k[0] == "acc"]
    if acc:
        return sorted(acc, key=lambda x: "" if x[1] is None else x[1])[0]

    em_strict = [k for k in keys if k[0] == "exact_match" and k[1] == "strict-match"]
    if em_strict:
        return em_strict[0]

    em = [k for k in keys if k[0] == "exact_match"]
    if em:
        return em[0]

    return sorted(keys, key=lambda x: (x[0], "" if x[1] is None else x[1]))[0]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_dir_metadata(dirname: str) -> Optional[Tuple[str, str, str]]:
    m = DIR_RE.match(dirname)
    if not m:
        return None
    return m.group("model"), m.group("quant"), m.group("task")


def find_result_jsons(subdir: str) -> List[str]:
    # Search recursively under each <model>_<quant>_<task> folder
    preferred = sorted(glob.glob(os.path.join(subdir, "**", "results_*.json"), recursive=True))
    if preferred:
        return preferred
    return sorted(glob.glob(os.path.join(subdir, "**", "*.json"), recursive=True))


@dataclass
class LongRow:
    model: str
    quant_method: str
    # task_from_dir is the label from folder name; task_key is the actual JSON task key
    task_from_dir: str
    task_key: str
    parent_task: Optional[str]
    source_dir: str
    source_file: str
    metric: str
    variant: Optional[str]
    value: float
    stderr: Optional[float]
    n_shot: Optional[int]
    n_samples_effective: Optional[int]


def parse_one_results_json(
    json_path: str,
    model: str,
    quant: str,
    task_from_dir: str,
    source_dir: str,
) -> List[LongRow]:
    data = load_json(json_path)

    results: Dict[str, Dict[str, Any]] = data.get("results", {}) or {}
    nshot_map = data.get("n-shot", {}) or {}
    nsamples_map = data.get("n-samples", {}) or {}

    rows: List[LongRow] = []
    source_file = os.path.basename(json_path)

    for task_key, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue

        parent_task = "mmlu" if task_key.startswith("mmlu_") else None
        if task_key == "mmlu":
            parent_task = None

        numeric: Dict[Tuple[str, Optional[str]], float] = {}
        for k, v in task_metrics.items():
            if k == "alias":
                continue
            if isinstance(v, (int, float)):
                mn, var = split_metric_key(k)
                numeric[(mn, var)] = float(v)

        stderr_map: Dict[Tuple[str, Optional[str]], float] = {}
        for (mn, var), v in numeric.items():
            if is_stderr_metric(mn):
                base = (mn.replace("_stderr", ""), var)
                stderr_map[base] = v

        n_shot = None
        if isinstance(nshot_map, dict) and task_key in nshot_map:
            try:
                n_shot = int(nshot_map[task_key])
            except Exception:
                n_shot = None

        n_eff = None
        if isinstance(nsamples_map, dict) and task_key in nsamples_map:
            eff = nsamples_map[task_key].get("effective")
            if isinstance(eff, int):
                n_eff = eff

        for (mn, var), v in numeric.items():
            if is_stderr_metric(mn):
                continue
            rows.append(
                LongRow(
                    model=model,
                    quant_method=quant,
                    task_from_dir=task_from_dir,
                    task_key=task_key,
                    parent_task=parent_task,
                    source_dir=source_dir,
                    source_file=source_file,
                    metric=mn,
                    variant=var,
                    value=v,
                    stderr=stderr_map.get((mn, var)),
                    n_shot=n_shot,
                    n_samples_effective=n_eff,
                )
            )

    return rows


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def write_long_csv(rows: List[LongRow], path: str) -> None:
    fields = list(LongRow.__annotations__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def write_wide_csv(rows: List[LongRow], path: str) -> None:
    """
    Wide rows keyed by (quant_method, task_key).
    Columns include:
      - primary_metric, primary_variant, primary_value, primary_stderr
      - all observed metric::variant columns + stderr companions
    """
    grouped: Dict[Tuple[str, str], List[LongRow]] = {}
    for r in rows:
        grouped.setdefault((r.quant_method, r.task_key), []).append(r)

    metric_cols = set((r.metric, r.variant) for r in rows if not is_stderr_metric(r.metric))
    metric_cols_sorted = sorted(metric_cols, key=lambda x: (x[0], "" if x[1] is None else x[1]))

    fieldnames = [
        "model",
        "quant_method",
        "task_from_dir",
        "task_key",
        "parent_task",
        "n_shot",
        "n_samples_effective",
        "primary_metric",
        "primary_variant",
        "primary_value",
        "primary_stderr",
        "source_dir",
        "source_file",
    ]
    for m, v in metric_cols_sorted:
        col = m if v is None else f"{m}::{v}"
        fieldnames.append(col)
        fieldnames.append(col + "_stderr")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for (quant, task_key), rs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            # Prefer a single row context; if multiple source_files exist, we still keep them separate
            # by writing one wide row per source_file.
            by_file: Dict[str, List[LongRow]] = {}
            for r in rs:
                by_file.setdefault(r.source_file, []).append(r)

            for source_file, frs in sorted(by_file.items(), key=lambda x: x[0]):
                lookup = {(r.metric, r.variant): r for r in frs}
                numeric = {(r.metric, r.variant): r.value for r in frs}
                pk = pick_primary_metric(numeric)
                pr = lookup.get(pk) if pk else None

                base = frs[0]
                row = {
                    "model": base.model,
                    "quant_method": base.quant_method,
                    "task_from_dir": base.task_from_dir,
                    "task_key": base.task_key,
                    "parent_task": base.parent_task,
                    "n_shot": base.n_shot,
                    "n_samples_effective": base.n_samples_effective,
                    "primary_metric": pk[0] if pk else None,
                    "primary_variant": pk[1] if pk else None,
                    "primary_value": pr.value if pr else None,
                    "primary_stderr": pr.stderr if pr else None,
                    "source_dir": base.source_dir,
                    "source_file": base.source_file,
                }

                for m, v in metric_cols_sorted:
                    col = m if v is None else f"{m}::{v}"
                    r = lookup.get((m, v))
                    row[col] = r.value if r else None
                    row[col + "_stderr"] = r.stderr if r else None

                w.writerow(row)


def sanitize_model_name(name: str) -> str:
    # Keep filenames safe
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to lm_eval_results folder.")
    ap.add_argument("--out_dir", default="parsed", help="Output directory for CSVs.")
    ap.add_argument(
        "--write_combined",
        action="store_true",
        help="Also write combined_all_long.csv and combined_all_wide.csv",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"Not a directory: {root}")

    ensure_dir(args.out_dir)

    # Collect rows per model
    per_model: Dict[str, List[LongRow]] = {}

    for entry in sorted(os.listdir(root)):
        subdir = os.path.join(root, entry)
        if not os.path.isdir(subdir):
            continue

        meta = parse_dir_metadata(entry)
        if meta is None:
            # Skip dirs that don't match <model>_<quant>_<task>
            continue

        model, quant, task_from_dir = meta

        jsons = find_result_jsons(subdir)
        if not jsons:
            continue

        for jp in jsons:
            rows = parse_one_results_json(jp, model, quant, task_from_dir, source_dir=entry)
            if rows:
                per_model.setdefault(model, []).extend(rows)

    if not per_model:
        raise SystemExit("No parsable result directories found under --root.")

    # Write per-model files
    combined_rows: List[LongRow] = []

    for model, rows in sorted(per_model.items(), key=lambda x: x[0].lower()):
        safe = sanitize_model_name(model)
        long_path = os.path.join(args.out_dir, f"{safe}_long.csv")
        wide_path = os.path.join(args.out_dir, f"{safe}_wide.csv")

        write_long_csv(rows, long_path)
        write_wide_csv(rows, wide_path)

        combined_rows.extend(rows)
        print(f"[OK] {model}: {len(rows)} metric rows -> {long_path}, {wide_path}")

    if args.write_combined:
        write_long_csv(combined_rows, os.path.join(args.out_dir, "combined_all_long.csv"))
        write_wide_csv(combined_rows, os.path.join(args.out_dir, "combined_all_wide.csv"))
        print(f"[OK] combined: {len(combined_rows)} metric rows")

    print("Done.")


if __name__ == "__main__":
    main()
