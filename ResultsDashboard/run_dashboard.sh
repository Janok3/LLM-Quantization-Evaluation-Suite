#!/bin/bash
# Startup script for LM Evaluation Quantization Impact Dashboard

echo "🚀 Starting Quantization Impact Dashboard..."
echo ""

# Navigate to the script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install flask flask-cors
fi

# Start the Flask API server
echo "📊 Starting Flask API server..."
echo "   - Auto-parsing LM evaluation results"
echo "   - Server will be available at http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

./venv/bin/python api.py
