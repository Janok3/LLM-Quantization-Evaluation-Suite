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