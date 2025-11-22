#!/bin/bash
set -e

echo "========================================="
echo "RAG System Setup Script"
echo "========================================="
echo ""

echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python $PYTHON_VERSION"

echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker Desktop."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi
echo "Docker is running"
echo ""

echo "Creating virtual environment..."
python3 -m venv .venv
echo "Virtual environment created"
echo ""

echo "Installing dependencies..."
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "Dependencies installed"
echo ""

if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo ".env file created. Please edit it to add your DEEPSEEK_API_KEY"
else
    echo ".env file already exists"
fi
echo ""

echo "Starting Docker services..."
docker compose up -d
echo "Waiting for services to be ready..."
sleep 10
echo ""

echo "Checking system status..."
python3 ./rag status
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Add your DeepSeek API key to .env file:"
echo "   DEEPSEEK_API_KEY=your_key_here"
echo ""
echo "3. Test document ingestion:"
echo "   python3 ./rag ingest ./data/corpus"
echo ""
echo "4. Test queries:"
echo "   python3 ./rag ask \"What is RAG?\" --strict-local  # No API key needed"
echo "   python3 ./rag ask \"What is RAG?\"                 # Requires API key"
echo ""
echo "For more commands, see: make help"
echo ""
