#!/usr/bin/env bash
set -e

# Change directory to the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "  Indic-Speak & Transcribe OpenAI-Compatible API Server"
echo "=========================================================="

# 1. Create virtual environment if not already available
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' not found. Creating a new one..."

    # Find suitable Python 3
    if command -v python3.12 &>/dev/null; then
        PYTHON_CMD=python3.12
    elif command -v python3 &>/dev/null; then
        PYTHON_CMD=python3
    else
        echo "Error: Python 3 is not installed or not in PATH."
        exit 1
    fi

    echo "Using $($PYTHON_CMD --version) to create virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "Virtual environment created at $SCRIPT_DIR/$VENV_DIR"
else
    echo "Found existing virtual environment at $SCRIPT_DIR/$VENV_DIR"
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Install packages needed from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Checking and installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found."
fi

# 4. Prepare required directories
mkdir -p data/generated data/transcriptions data/uploads data/out static

# 5. Start the API server
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "=========================================================="
echo "  🚀 Server is starting up!"
echo "  🌐 Web UI Test Interface:  http://localhost:${PORT}/"
echo "  📖 Interactive Docs:       http://localhost:${PORT}/docs"
echo "  🔊 OpenAI TTS Endpoint:    http://localhost:${PORT}/v1/audio/speech"
echo "  🎙️ OpenAI ASR Endpoint:    http://localhost:${PORT}/v1/audio/transcriptions"
echo "  🤖 OpenAI Models Endpoint: http://localhost:${PORT}/v1/models"
echo "=========================================================="

exec uvicorn server:app --host "$HOST" --port "$PORT" "$@"
