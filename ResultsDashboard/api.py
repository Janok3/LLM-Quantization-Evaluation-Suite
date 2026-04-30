#!/usr/bin/env python3
"""
Results Dashboard API
---------------------
Flask-based REST API for serving LLM evaluation results and fine-tuning logs.
Provides endpoints for data parsing, visualization data, and model downloads.
"""

from flask import Flask, jsonify, send_from_directory, send_file, request
from flask_cors import CORS
import os
import sys
import subprocess
import csv
import ast
import re
import shutil
import tempfile
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / "lm_eval_results"
PARSED_DIR = BASE_DIR / "parsed"
PARSER_SCRIPT = BASE_DIR / "parse_lm_eval_results.py"

# Directory for multiple fine-tuning logs
FINETUNES_DIR = BASE_DIR / "finetunes"
os.makedirs(FINETUNES_DIR, exist_ok=True) 

def run_parser():
    if not RESULTS_DIR.exists():
        return False, "lm_eval_results directory not found"
    try:
        result = subprocess.run(
            [sys.executable, str(PARSER_SCRIPT), "--root", str(RESULTS_DIR), "--out_dir", str(PARSED_DIR), "--write_combined"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            return True, "Parser completed successfully"
        return False, f"Parser failed: {result.stderr}"
    except Exception as e:
        return False, f"Error running parser: {str(e)}"

def read_csv_as_json(csv_path):
    data = []
    fallback_model = csv_path.name.replace('_wide.csv', '').replace('combined_all', 'Unknown_Model')
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_row = {}
            for key, value in row.items():
                if value == '' or value == 'None':
                    processed_row[key] = None
                else:
                    try: processed_row[key] = float(value)
                    except (ValueError, TypeError): processed_row[key] = value
            
            if 'model' not in processed_row:
                processed_row['model'] = processed_row.get('Model', fallback_model)
            if 'task_key' not in processed_row:
                processed_row['task_key'] = processed_row.get('Task', processed_row.get('task', 'unknown_task'))
            if 'quant_method' not in processed_row:
                m_name = str(processed_row['model']).lower()
                if 'int8' in m_name: processed_row['quant_method'] = 'int8'
                elif 'int4' in m_name: processed_row['quant_method'] = 'int4'
                else: processed_row['quant_method'] = 'base'
                
            if 'primary_value' not in processed_row:
                for k, v in processed_row.items():
                    k_lower = str(k).lower()
                    if isinstance(v, float) and any(m in k_lower for m in ['acc', 'f1', 'rouge', 'score', 'value', 'metric']):
                        processed_row['primary_value'] = v
                        break
                if 'primary_value' not in processed_row:
                    processed_row['primary_value'] = 0.0

            data.append(processed_row)
    return data

@app.route('/api/parse', methods=['POST'])
def parse_data():
    success, message = run_parser()
    return jsonify({'success': success, 'message': message})

@app.route('/api/data')
def get_all_data():
    combined_wide = PARSED_DIR / "combined_all_wide.csv"
    if not combined_wide.exists():
        run_parser()
        
    data = []
    if combined_wide.exists():
        data.extend(read_csv_as_json(combined_wide))
        
    if not data:
        for directory in [PARSED_DIR, BASE_DIR, RESULTS_DIR]:
            if directory.exists():
                for file in directory.glob("*wide.csv"):
                    if file.name != "combined_all_wide.csv":
                        data.extend(read_csv_as_json(file))
                        
    if not data:
        return jsonify({'success': False, 'message': 'No CSV data found.'}), 404
        
    return jsonify({'success': True, 'count': len(data), 'data': data})

@app.route('/api/models')
def get_models():
    response = get_all_data()
    if response[1] != 200 if isinstance(response, tuple) else response.status_code != 200:
        return jsonify({'success': False, 'models': []})
    
    data = response[0].json['data'] if isinstance(response, tuple) else response.json['data']
    models = sorted(list(set(row['model'] for row in data if row.get('model'))))
    return jsonify({'success': True, 'models': models})

@app.route('/api/finetunes')
def get_finetunes():
    if not FINETUNES_DIR.exists():
        return jsonify({'success': True, 'files': []})
        
    files = []
    for f in FINETUNES_DIR.iterdir():
        if f.is_file() and f.suffix in ['.out', '.log', '.txt']:
            files.append(f.name)
            
    files.sort()
    return jsonify({'success': True, 'files': files})

@app.route('/api/train_logs')
def get_train_logs():
    file_name = request.args.get('file')
    
    if not file_name or '/' in file_name or '\\' in file_name:
        return jsonify({"success": False, "message": "Invalid file request."}), 400
        
    log_file_path = FINETUNES_DIR / file_name
    if not log_file_path.exists():
        return jsonify({"success": False, "message": f"Log file {file_name} not found in finetunes folder."}), 404
        
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        lines = raw_text.split('\n')
        train_metrics, eval_metrics = [], []
        dict_pattern = re.compile(r'^\{.*\}$')
        
        for line in lines:
            if dict_pattern.match(line.strip()):
                try:
                    metric_dict = ast.literal_eval(line.strip())
                    if 'eval_loss' in metric_dict: eval_metrics.append(metric_dict)
                    elif 'loss' in metric_dict: train_metrics.append(metric_dict)
                except (ValueError, SyntaxError):
                    continue
                    
        return jsonify({"success": True, "raw_text": raw_text, "train_metrics": train_metrics, "eval_metrics": eval_metrics})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

CUSTOM_TEST_TYPES = {
    "accuracy_custom": {"file": "results.json", "items_key": "per_question", "item_fields": ["question", "expected_answer", "model_answer", "exact_match", "contains_match"]},
    "coherence_custom": {"file": "results.json", "items_key": "per_prompt", "item_fields": ["prompt_id", "domain", "cosine_similarity", "generated_preview"]},
    "tool_calling": {"file": "tool_results.json", "items_key": "results", "item_fields": ["prompt", "expected", "generated", "is_tool_correct", "params_present", "fully_correct"]},
    "ocr_custom": {"file": "results.json", "items_key": "per_sample", "item_fields": ["image", "category", "question", "ground_truth", "model_response", "cer", "wer", "similarity", "exact_match", "contains_match"]},
}

DIR_RE = re.compile(r"^(?P<model>.+)_(?P<quant>base|int4|int8)_(?P<task>.+)$")
DIR_RE_CUSTOM = re.compile(r"^(?P<model>.+)_(?P<quant>base|int4|int8)_(?P<test>accuracy_custom|coherence_custom|tool_calling|ocr_custom)$")

LM_EVAL_TASK_FIELD_MAP = {
    "gsm8k": {"question": "question", "answer": "answer", "model_answer_src": "filtered_resps"},
    "gsm8k_tr": {"question": "question", "answer": "answer", "model_answer_src": "filtered_resps"},
    "hendrycks_math": {"question": "problem", "answer": "solution", "model_answer_src": "filtered_resps"},
    "arc_challenge": {"question": "question", "answer": "choices_text", "model_answer_src": "filtered_resps", "choices_key": "choices"},
    "truthfulqa_mc2": {"question": "question", "answer": "target", "model_answer_src": "filtered_resps"},
    "wikitext": {"question": "page", "answer": None, "model_answer_src": "filtered_resps"},
}


def _extract_lm_eval_sample(doc, task_name, filtered_resps, acc_value):
    """Extract standardized fields from an lm_eval sample line."""
    field_map = LM_EVAL_TASK_FIELD_MAP.get(task_name)
    if field_map is None:
        doc_keys = list(doc.keys()) if isinstance(doc, dict) else []
        q_key = doc_keys[0] if doc_keys else None
        return {
            "question": str(doc.get(q_key, doc))[:500] if q_key else str(doc)[:500],
            "expected_answer": "",
            "model_answer": filtered_resps[0] if filtered_resps else "",
            "acc": acc_value,
        }

    q_key = field_map["question"]
    a_key = field_map["answer"]
    question = doc.get(q_key, "")

    if a_key and a_key == "choices_text" and field_map.get("choices_key"):
        choices = doc.get(field_map["choices_key"], {})
        if isinstance(choices, dict) and "text" in choices:
            answer = choices["text"][doc.get("answer", 0)] if "answer" in doc else ""
        else:
            answer = ""
    elif a_key:
        answer = doc.get(a_key, "")
    else:
        answer = ""

    model_answer = filtered_resps[0] if filtered_resps else ""

    return {
        "question": str(question)[:500],
        "expected_answer": str(answer)[:500],
        "model_answer": str(model_answer)[:500],
        "acc": acc_value,
    }


def parse_lm_eval_sample_files(subdir, task_name, quant):
    """Find and parse {task}_eval_samples.jsonl files in an lm_eval directory."""
    results = []
    for root, dirs, files in os.walk(subdir):
        for fname in files:
            if fname.endswith("_eval_samples.jsonl"):
                file_path = os.path.join(root, fname)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            sample = json.loads(line)
                            doc = sample.get("doc", {})
                            filtered_resps = sample.get("filtered_resps", [])
                            target = sample.get("target", "")
                            acc_value = sample.get("acc", 0)
                            doc_id = sample.get("doc_id", "")

                            extracted = _extract_lm_eval_sample(doc, task_name, filtered_resps, acc_value)
                            results.append({
                                "item_key": str(doc_id),
                                "quant": quant,
                                "data": extracted,
                            })
                except Exception:
                    continue
    return results


def parse_detailed_results():
    """Parse per-question/per-item data from custom test results and lm_eval sample files."""
    if not RESULTS_DIR.exists():
        return {}

    aggregated = {}

    for entry in sorted(os.listdir(RESULTS_DIR)):
        subdir = os.path.join(RESULTS_DIR, entry)
        if not os.path.isdir(subdir):
            continue

        m = DIR_RE.match(entry)
        if not m:
            continue

        model = m.group("model")
        quant = m.group("quant")
        task_name = m.group("test") if "test" in m.groupdict() else m.group("task")

        m_custom = DIR_RE_CUSTOM.match(entry)
        if m_custom:
            test_type = m_custom.group("test")
            if test_type not in CUSTOM_TEST_TYPES:
                continue
            cfg = CUSTOM_TEST_TYPES[test_type]
            json_file = os.path.join(subdir, cfg["file"])
            if not os.path.exists(json_file):
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            metrics = data.get("metrics", data)
            items = metrics.get(cfg["items_key"])
            if not items:
                continue

            aggregated.setdefault(model, {})
            aggregated[model].setdefault(test_type, {})

            for item in items:
                item_key = item.get("question", item.get("prompt", item.get("prompt_id", item.get("image", ""))))
                aggregated[model][test_type].setdefault(item_key, {})
                aggregated[model][test_type][item_key][quant] = {
                    field: item.get(field) for field in cfg["item_fields"]
                }
        else:
            samples = parse_lm_eval_sample_files(subdir, task_name, quant)
            if samples:
                aggregated.setdefault(model, {})
                aggregated[model].setdefault(task_name, {})
                for sample in samples:
                    ik = sample["item_key"]
                    aggregated[model][task_name].setdefault(ik, {})
                    aggregated[model][task_name][ik][quant] = sample["data"]

    return aggregated


@app.route('/api/detailed-results')
def get_detailed_results():
    """Return per-question/per-item results grouped by model, test type, and item."""
    try:
        detailed = parse_detailed_results()
        if not detailed:
            return jsonify({'success': False, 'message': 'No detailed custom test results found.', 'data': {}})
        return jsonify({'success': True, 'data': detailed})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/checkpoints')
def get_checkpoints():
    """Recursively scan for nested checkpoint and adapter folders."""
    items = []
    # Ignore these folders so the scan happens instantly
    ignore_dirs = {'venv', '.git', '__pycache__', 'node_modules', 'parsed', 'lm_eval_results', 'finetunes'}
    
    for root, dirs, files in os.walk(BASE_DIR):
        # Exclude ignored directories in-place to prevent hanging
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        
        for d in dirs:
            if d.startswith('checkpoint-') or d == 'final_adapter':
                full_path = Path(root) / d
                rel_path = full_path.relative_to(BASE_DIR)
                # Convert to standard forward slashes for cross-platform UI
                items.append(str(rel_path).replace('\\', '/'))
                
    # Sort them by parent folder, then numerically by step
    def sort_key(path_str):
        parts = path_str.split('/')
        parent = '/'.join(parts[:-1])
        name = parts[-1]
        if name == 'final_adapter': return (parent, float('inf'))
        try: return (parent, int(name.split('-')[1]))
        except: return (parent, 0)
            
    items.sort(key=sort_key)
    return jsonify({'success': True, 'checkpoints': items})

@app.route('/api/download_model')
def download_model():
    """Zip and download ANY requested checkpoint, even nested ones."""
    # Get the relative path from the URL
    folder_path_raw = request.args.get('folder')
    
    if not folder_path_raw:
        return jsonify({"success": False, "message": "No folder specified."}), 400

    # Clean the path: remove leading slashes and handle Windows/Linux separator differences
    folder_path_clean = folder_path_raw.lstrip('/\\')
    
    # Strict security check
    if '..' in folder_path_clean:
        return jsonify({"success": False, "message": "Security violation: Invalid path."}), 400
        
    # Combine with BASE_DIR to get the absolute path on your PC
    adapter_dir = (BASE_DIR / folder_path_clean).resolve()
    
    # Debug print so you can see exactly where Python is looking in your terminal
    print(f"DEBUG: Looking for folder at: {adapter_dir}")

    if not adapter_dir.exists() or not adapter_dir.is_dir():
        return jsonify({
            "success": False, 
            "message": f"Directory not found: {folder_path_clean}",
            "checked_path": str(adapter_dir)
        }), 404
        
    try:
        temp_dir = tempfile.gettempdir()
        # Create a safe filename for the zip
        safe_name = folder_path_clean.replace('/', '_').replace('\\', '_')
        zip_base_path = os.path.join(temp_dir, safe_name)
        
        shutil.make_archive(zip_base_path, 'zip', str(adapter_dir))
        return send_file(f"{zip_base_path}.zip", as_attachment=True, download_name=f'{safe_name}.zip')
    except Exception as e:
        return jsonify({"success": False, "message": f"Zipping failed: {str(e)}"}), 500

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(BASE_DIR, path)

if __name__ == '__main__':
    print("Starting Server on http://localhost:5000")
    print(f"Tracking finetunes in: {FINETUNES_DIR}")
    app.run(debug=True, port=5000, host='0.0.0.0')