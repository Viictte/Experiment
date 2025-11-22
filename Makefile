.PHONY: setup up down status ingest-samples ask-local ask clean help

help:
	@echo "RAG System - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo "  make up             - Start Docker services (Qdrant, Elasticsearch, Redis)"
	@echo "  make down           - Stop Docker services"
	@echo ""
	@echo "Testing:"
	@echo "  make status         - Check system status"
	@echo "  make ingest-samples - Ingest sample documents"
	@echo "  make ask-local      - Test query with local knowledge base (no API key needed)"
	@echo "  make ask            - Test query with LLM synthesis (requires API key)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove virtual environment and Docker volumes"

setup:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Installing dependencies..."
	.venv/bin/python3 -m pip install --upgrade pip
	.venv/bin/python3 -m pip install -r requirements.txt
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "1. Activate the virtual environment: source .venv/bin/activate"
	@echo "2. Copy .env.example to .env and add your DEEPSEEK_API_KEY"
	@echo "3. Start Docker services: make up"
	@echo "4. Check status: make status"

up:
	@echo "Starting Docker services..."
	docker compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "Services started!"

down:
	@echo "Stopping Docker services..."
	docker compose down

status:
	@echo "Checking system status..."
	python3 ./rag status

ingest-samples:
	@echo "Ingesting sample documents..."
	python3 ./rag ingest ./data/corpus

ask-local:
	@echo "Testing query with local knowledge base (no API key needed)..."
	python3 ./rag ask "What is RAG and what are its key components?" --strict-local

ask:
	@echo "Testing query with LLM synthesis..."
	python3 ./rag ask "What is RAG and what are its key components?"

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	docker compose down -v
	@echo "Cleanup complete!"
