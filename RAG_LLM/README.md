# RAG System - Advanced Retrieval Augmented Generation with LLM

A production-ready RAG (Retrieval Augmented Generation) system with hybrid retrieval, intelligent routing, multimodal support, and domain-specific tools.

## Features

- **Hybrid Retrieval**: Combines dense (Qdrant) and sparse (Elasticsearch BM25) retrieval with Reciprocal Rank Fusion (RRF)
- **Advanced Reranking**: Cross-encoder reranking with freshness and credibility scoring
- **Intelligent Routing**: LLM-based source selection using DeepSeek with parallel web search
- **Parallel Web Search**: Web search runs alongside domain-specific tools (not just as fallback) to provide comprehensive answers combining specialized data with current web information
- **Domain-Specific Tools**: Weather (Open-Meteo), finance (yfinance + Alpha Vantage with intraday support), transport, and web search integrations
- **Multimodal Support**: PDF, HTML, Office documents, images (OCR), audio (Whisper transcription)
- **Document Attachments**: Attach files directly to queries without indexing them into the knowledge base
- **Audio Transcription**: Support for audio files (MP3, WAV, M4A, OGG, FLAC, WEBM) using Whisper
- **Fast-Path Routing**: Simple questions (math, trivia, translations) bypass retrieval for <10s latency
- **Progress Bar**: Real-time CLI progress showing stages during answer generation
- **Caching**: Redis-based caching for queries, reranking, and answers
- **CLI Interface**: Simple command-line interface for all operations

## Architecture

### Core Services
- **Vector DB**: Qdrant (HNSW + quantization)
- **Keyword Search**: Elasticsearch (BM25)
- **Cache**: Redis
- **LLM**: DeepSeek (routing, tool calls, synthesis)
- **Embeddings**: bge-m3 (multilingual, hybrid-friendly)
- **Reranker**: BAAI/bge-reranker-large (swappable)

### Workflow
1. **Router** → Intelligent source selection (can select multiple sources)
2. **Fetch** → Parallel tool execution (local RAG, domain tools, web search)
3. **Parallel Web Search** → Web search runs alongside domain-specific tools to combine specialized data with current web information
4. **Normalize** → Parse and tabularize results
5. **Rerank & Filter** → Cross-encoder + freshness + credibility scoring
6. **Synthesis** → LLM generates answer combining all sources with citations
7. **Citations** → Inline citations for each fact

## Installation

### Prerequisites

**Required:**
- Python 3.10+ (3.12 recommended)
- Docker Desktop (running)
- DeepSeek API key (get from https://platform.deepseek.com/)
- ffmpeg (required for audio transcription)

**Optional:**
- Tesseract OCR for image processing

**Verify Prerequisites:**

Check Python version (must be 3.10+):
```bash
python3 --version
```

Check Docker is installed and running:
```bash
docker --version
docker compose version  # or: docker-compose -v
docker info  # Should show server info if Docker is running
```

Check ffmpeg (required for audio transcription):
```bash
ffmpeg -version
```

**Install Missing Prerequisites:**

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv docker.io docker-compose ffmpeg tesseract-ocr
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER  # Add yourself to docker group, then logout/login
```

macOS:
```bash
# Install Homebrew if not already installed: https://brew.sh
brew install python@3.12 ffmpeg tesseract
# Install Docker Desktop from: https://www.docker.com/products/docker-desktop
# Start Docker Desktop application
```

Windows:
```bash
# Use WSL2 (Ubuntu) - recommended for best compatibility
# Install Docker Desktop for Windows with WSL2 backend
# Inside WSL2 Ubuntu terminal:
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv ffmpeg tesseract-ocr
```

### Quick Start (Automated Setup)

**Option 1: Using setup script (recommended)**
```bash
git clone https://github.com/Viictte/RAG_LLM.git
cd RAG_LLM
chmod +x rag setup.sh  # Make scripts executable
./setup.sh
```

The setup script will:
- Check prerequisites (Python, Docker, ffmpeg)
- Create a virtual environment
- Install all dependencies
- Create .env file from template
- Start Docker services
- Verify system status

**After setup completes:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Edit .env file to add your DeepSeek API key
nano .env  # or use your preferred editor
# Set: DEEPSEEK_API_KEY=sk-your-actual-key-here
```

**Option 2: Using Makefile**
```bash
git clone https://github.com/Viictte/RAG_LLM.git
cd RAG_LLM
make setup          # Create venv and install dependencies
make up             # Start Docker services
source .venv/bin/activate
cp .env.example .env  # Create .env file
nano .env           # Add your DEEPSEEK_API_KEY
make status         # Verify system
```

### Manual Setup (Step-by-Step)

**1. Clone the repository:**
```bash
git clone https://github.com/Viictte/RAG_LLM.git
cd RAG_LLM
```

**2. Create virtual environment and install dependencies:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# On Windows WSL: source .venv/bin/activate
# On Windows CMD: .venv\Scripts\activate.bat
# On Windows PowerShell: .venv\Scripts\Activate.ps1

# Upgrade pip
python3 -m pip install --upgrade pip

# Install dependencies (this may take 5-10 minutes)
python3 -m pip install -r requirements.txt
```

**Note:** The first time you run the system, it will download ML models (embeddings and reranker) which may take several minutes. Models are cached in `~/.cache/huggingface/` for future use.

**3. Set up environment variables:**
```bash
# Copy example .env file
cp .env.example .env

# Edit .env file and add your API keys
nano .env  # or use your preferred editor (vim, code, etc.)
```

**Required in .env:**
```bash
DEEPSEEK_API_KEY=sk-your-actual-deepseek-key-here
```

**Optional in .env (for enhanced features):**
```bash
# Web Search (recommended)
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-google-cse-id
TAVILY_API_KEY=your-tavily-api-key  # Fallback if Google not configured

# Finance (for real-time stock data)
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key

# Transport (for routing)
OPENROUTESERVICE_API_KEY=your-openrouteservice-key
```

**4. Start infrastructure services:**
```bash
# Start Docker services in background
docker compose up -d

# Wait for services to be ready (takes ~10 seconds)
sleep 10
```

**Special note for Linux users:** If Elasticsearch fails to start, you may need to increase vm.max_map_count:
```bash
sudo sysctl -w vm.max_map_count=262144
echo "vm.max_map_count=262144" | sudo tee /etc/sysctl.d/99-elasticsearch.conf
sudo sysctl --system
docker compose restart elasticsearch
```

**5. Verify system status:**
```bash
# Make rag script executable
chmod +x rag

# Check system status
python3 ./rag status
```

You should see:
- ✓ Qdrant: Connected
- ✓ Elasticsearch: Connected  
- ✓ Redis: Connected

**Platform-Specific Notes:**

**macOS:**
- Ensure Docker Desktop is running before starting services
- On Apple Silicon (M1/M2/M3), PyTorch will use CPU mode by default
- If you get permission errors: `chmod +x rag setup.sh`

**Windows (WSL2):**
- Use WSL2 with Ubuntu for best compatibility
- Install Docker Desktop with WSL2 backend enabled
- Run all commands inside WSL2 terminal
- If `./rag` doesn't work, use: `python3 ./rag`

**Ubuntu/Debian:**
- Add your user to docker group: `sudo usermod -aG docker $USER` (then logout/login)
- If Elasticsearch fails, apply vm.max_map_count fix above

## Usage

### Ingest Documents

Ingest a single file:
```bash
./rag ingest ./data/corpus/document.pdf
```

Ingest a directory:
```bash
./rag ingest ./data/corpus
```

Ingest a URL:
```bash
./rag ingest https://example.com/article
```

### Ask Questions

Basic query:
```bash
./rag ask "What is the impact of NVIDIA earnings on stock price?"
```

Strict local mode (no web search):
```bash
./rag ask --strict-local "Summarize the internal design memo"
```

Fast mode (skip web search):
```bash
./rag ask --fast "Give me a quick answer"
```

JSON output:
```bash
./rag ask --json "What is the weather in Tokyo?"
```

Disable progress bar:
```bash
./rag ask --no-progress "What is RAG?"
```

### Attach Documents to Queries

Attach files directly to your query without indexing them into the knowledge base:

```bash
# Attach a single file
./rag ask "Summarize this document" --file ./report.pdf

# Attach multiple files
./rag ask "Compare these documents" --file ./doc1.pdf --file ./doc2.docx --file ./data.xlsx

# Attach audio file (automatically transcribed)
./rag ask "What does this audio say?" --file ./recording.mp3
```

**Supported file types:**
- **Documents**: PDF, Word (.doc, .docx), Excel (.xls, .xlsx), CSV, text, markdown, HTML, JSON, XML
- **Images**: PNG, JPG, JPEG, GIF (with OCR)
- **Audio**: MP3, WAV, M4A, OGG, FLAC, WEBM (with Whisper transcription)

**Features:**
- Files are parsed and sent directly to the LLM without indexing
- Audio files are automatically transcribed using Whisper (base model)
- Token budget management (8k tokens per attachment by default)
- Progress bar shows parsing and transcription status
- Supports multiple file attachments in a single query

### Configuration

View configuration:
```bash
./rag config show
```

Get a specific value:
```bash
./rag config get reranker.model
```

Set a value:
```bash
./rag config set reranker.model cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Configuration

Edit `config/config.yaml` to customize:

- LLM settings (model, temperature, max_tokens)
- Embeddings model
- Reranker model and top_k
- Retrieval parameters (top_k, RRF k)
- Scoring weights (cross-encoder, freshness, credibility)
- Chunking parameters
- Domain credibility priors
- Tool enablement

## Domain-Specific Tools

### Weather
Uses Open-Meteo API (no key required) with improved geocoding for accurate location matching:
```bash
./rag ask "What's the weather in Taipei on 2025-11-05?"
```

**Features:**
- Accurate geocoding with country validation (e.g., Taipei → Taiwan, not Toronto)
- Automatic fallback to web search if weather data is unavailable

### Finance
Uses yfinance with automatic Alpha Vantage fallback and intraday support:
```bash
./rag ask "Compare NVDA and AMD stock performance"
./rag ask "What is the current price of NVDA stock?"
```

**Features:**
- Primary: yfinance (free, no key required)
- Fallback: Alpha Vantage API (requires `ALPHA_VANTAGE_API_KEY`)
- Intraday data: Automatically uses Alpha Vantage intraday (5-minute intervals) for queries containing "current", "now", "today", "latest", or "real-time"
- Parallel web search: Always runs alongside finance tools to provide current market information

### Transport
Uses OpenRouteService with automatic web search fallback:
```bash
./rag ask "Route from San Francisco to Los Angeles"
```

**Features:**
- Primary: OpenRouteService API (requires `OPENROUTESERVICE_API_KEY`)
- Fallback: Web search for route information

### Web Search
Uses Google Custom Search API or Tavily API:
```bash
./rag ask "Latest news about AI developments"
```

**Features:**
- Primary: Google Custom Search (requires `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`)
- Fallback: Tavily API (requires `TAVILY_API_KEY`)
- Parallel execution: Runs alongside domain-specific tools (weather, finance, transport) to combine specialized data with current web information
- Domain-aware query enhancement: Automatically rewrites queries for better web search results based on the domain

## Performance

### Speed Optimizations
- **Fast-Path Routing**: Simple questions (arithmetic, trivia, translations) skip retrieval/reranking/web search for <10s latency
- **Progress Bar**: Real-time CLI progress showing stages (routing, retrieval, tools, web search, reranking, generation)
- HNSW indexing with quantization
- Parallel tool execution
- Redis caching (queries, reranking, answers)
- Batch reranking
- Configurable top_k cutoffs

**Latency Examples:**
- Simple questions (math, trivia): ~5-8 seconds (fast-path)
- Complex questions (finance, weather): ~15-20 seconds (full pipeline)
- Document attachments: ~7-10 seconds (direct LLM path)

### Accuracy Optimizations
- RRF hybrid retrieval
- Cross-encoder reranking
- Freshness scoring (exponential decay)
- Credibility scoring (domain priors)
- Deduplication (cosine similarity)
- Mandatory citations

### Reliability Features
- **Automatic Fallbacks**: Domain tools automatically fall back to web search when they fail
- **Multi-Source Synthesis**: Combines information from multiple sources for comprehensive answers
- **Graceful Degradation**: System provides partial information when complete data is unavailable
- **Minimum Context Threshold**: Ensures sufficient context (3+ documents) before synthesis

## Development

### Project Structure
```
RAG_LLM/
├── config/              # Configuration files
├── data/                # Data directories
│   ├── corpus/          # Documents to ingest
│   └── uploads/         # Uploaded files
├── rag_system/          # Main package
│   ├── core/            # Configuration management
│   ├── services/        # Vector DB, search, caching
│   ├── parsers/         # Document parsers
│   ├── tools/           # Domain-specific tools
│   ├── workflows/       # LangGraph orchestration
│   └── cli/             # CLI interface
├── tests/               # Test files
├── docker-compose.yml   # Infrastructure services
├── requirements.txt     # Python dependencies
└── rag                  # CLI entry point
```

### Testing

Start infrastructure:
```bash
docker compose up -d
```

Run tests:
```bash
python -m pytest tests/
```

## Quick Test

After setup, test the system with these commands:

**1. Check system status:**
```bash
source .venv/bin/activate  # If not already activated
python3 ./rag status
```

**2. Ingest sample documents:**
```bash
python3 ./rag ingest ./data/corpus
```

**3. Test simple question (fast-path, ~5-8 seconds):**
```bash
python3 ./rag ask "What is 15 multiplied by 24?"
```

**4. Test with local knowledge base:**
```bash
python3 ./rag ask "What is RAG?" --strict-local
```

**5. Test with web search (requires API key):**
```bash
python3 ./rag ask "What is the weather in Tokyo today?"
```

**6. Test document attachment:**
```bash
python3 ./rag ask "Summarize this document" --file ./data/corpus/sample_doc1.txt
```

**7. Test audio transcription (requires ffmpeg):**
```bash
# Create a test audio file or use your own .mp3/.wav file
python3 ./rag ask "What does this audio say?" --file ./your_audio.mp3
```

## Troubleshooting

### Common Issues and Solutions

**Issue: `ModuleNotFoundError: No module named 'click'`**
- **Cause:** Virtual environment not activated or dependencies not installed
- **Solution:**
  ```bash
  source .venv/bin/activate  # Activate venv
  python3 -m pip install -r requirements.txt  # Reinstall dependencies
  ```

**Issue: `Permission denied: ./rag` or `./setup.sh`**
- **Cause:** Scripts don't have execute permission
- **Solution:**
  ```bash
  chmod +x rag setup.sh
  # Or run with python directly:
  python3 ./rag status
  ```

**Issue: `docker compose: command not found`**
- **Cause:** Docker Compose not installed or using older version
- **Solution:**
  ```bash
  # Try with hyphen (older Docker versions):
  docker-compose up -d
  
  # Or install Docker Compose plugin:
  # Ubuntu: sudo apt-get install docker-compose-plugin
  # macOS: Update Docker Desktop
  ```

**Issue: Elasticsearch fails to start (Linux)**
- **Cause:** Insufficient vm.max_map_count
- **Solution:**
  ```bash
  sudo sysctl -w vm.max_map_count=262144
  echo "vm.max_map_count=262144" | sudo tee /etc/sysctl.d/99-elasticsearch.conf
  sudo sysctl --system
  docker compose restart elasticsearch
  ```

**Issue: Port already in use (9200, 6333, 6379)**
- **Cause:** Another service using the same port
- **Solution:**
  ```bash
  # Find process using port (example for 9200):
  lsof -i :9200  # Linux/macOS
  netstat -ano | findstr :9200  # Windows
  
  # Stop the conflicting service or change ports in docker-compose.yml
  ```

**Issue: `faster-whisper` installation fails**
- **Cause:** Missing build dependencies or incompatible system
- **Solution:**
  ```bash
  # Option 1: Install build dependencies
  sudo apt-get install -y build-essential python3-dev
  
  # Option 2: Temporarily disable audio support
  # Edit requirements.txt and comment out: # faster-whisper==1.0.0
  # Note: Audio transcription features will be disabled
  ```

**Issue: `ffmpeg: command not found`**
- **Cause:** ffmpeg not installed
- **Solution:**
  ```bash
  # Ubuntu/Debian:
  sudo apt-get install -y ffmpeg
  
  # macOS:
  brew install ffmpeg
  
  # Verify:
  ffmpeg -version
  ```

**Issue: First query is very slow (30+ seconds)**
- **Cause:** ML models downloading on first use
- **Solution:** This is normal. Models are cached after first download. To pre-download:
  ```bash
  python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
  python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
  ```

**Issue: Docker permission denied (Linux)**
- **Cause:** User not in docker group
- **Solution:**
  ```bash
  sudo usermod -aG docker $USER
  # Logout and login again for changes to take effect
  # Or: newgrp docker
  ```

**Issue: Virtual environment activation doesn't work**
- **Cause:** Wrong shell or path
- **Solution:**
  ```bash
  # Bash/Zsh (Linux/macOS):
  source .venv/bin/activate
  
  # Fish shell:
  source .venv/bin/activate.fish
  
  # Windows CMD:
  .venv\Scripts\activate.bat
  
  # Windows PowerShell:
  .venv\Scripts\Activate.ps1
  ```

**Issue: `DEEPSEEK_API_KEY` not found**
- **Cause:** .env file not created or API key not set
- **Solution:**
  ```bash
  cp .env.example .env
  nano .env  # Add: DEEPSEEK_API_KEY=sk-your-actual-key
  ```

**Issue: Models download to wrong location**
- **Cause:** Custom HuggingFace cache location needed
- **Solution:**
  ```bash
  # Set custom cache directory (optional):
  export HF_HOME=/path/to/your/cache
  # Add to .bashrc or .zshrc to make permanent
  ```

## API Keys

**Required:**
- `DEEPSEEK_API_KEY`: For LLM routing and synthesis (get your key from https://platform.deepseek.com/)

**Optional:**
- `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`: For Google Custom Search (recommended for web search)
- `TAVILY_API_KEY`: For Tavily web search (fallback if Google CSE not configured)
- `OPENROUTESERVICE_API_KEY`: For transport routing
- `ALPHA_VANTAGE_API_KEY`: For advanced finance data and intraday stock prices

**Note:** Without a valid DeepSeek API key, the system will still perform document ingestion, retrieval, and reranking, but answer synthesis will fail. The retrieval system (hybrid search, reranking, caching) works independently and can be tested without an API key.

**Model Downloads:**
- Embeddings model (BAAI/bge-m3) and reranker model are downloaded automatically on first use
- Models are cached in `~/.cache/huggingface/` (typically 1-2 GB total)
- First query may take 30-60 seconds while models download
- Subsequent queries will be much faster (5-20 seconds depending on complexity)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

For issues or questions, please open a GitHub issue.
