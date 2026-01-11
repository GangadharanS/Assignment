# Makefile for RAG Application
# =============================

.PHONY: help build up down logs shell clean test dev

# Default target
help:
	@echo "RAG Application - Docker Commands"
	@echo "=================================="
	@echo ""
	@echo "Quick Start:"
	@echo "  make up          - Build and start all services (app + Ollama)"
	@echo "  make down        - Stop all services"
	@echo ""
	@echo "Development:"
	@echo "  make dev         - Run locally with host Ollama"
	@echo "  make build       - Build Docker images"
	@echo "  make logs        - View container logs"
	@echo "  make shell       - Open shell in app container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       - Remove containers and volumes"
	@echo "  make clean-all   - Remove everything including images"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Run tests"
	@echo "  make health      - Check service health"

# Build Docker images
build:
	docker compose build

# Start all services (full stack with Ollama)
up:
	@echo "Starting RAG application with Ollama..."
	@echo "This may take a few minutes on first run (downloading LLM model)..."
	docker compose up -d
	@echo ""
	@echo "Services starting..."
	@echo "  - App:    http://localhost:8000"
	@echo "  - Ollama: http://localhost:11434"
	@echo "  - Docs:   http://localhost:8000/docs"
	@echo ""
	@echo "Run 'make logs' to see progress"

# Start with host Ollama (simpler, requires Ollama running on host)
dev:
	@echo "Starting app with host Ollama..."
	@echo "Make sure Ollama is running: ollama serve"
	docker compose -f docker-compose.simple.yml up -d
	@echo "App available at: http://localhost:8000"

# Stop all services
down:
	docker compose down
	docker compose -f docker-compose.simple.yml down 2>/dev/null || true

# View logs
logs:
	docker compose logs -f

# View app logs only
logs-app:
	docker compose logs -f app

# View Ollama logs only
logs-ollama:
	docker compose logs -f ollama

# Open shell in app container
shell:
	docker compose exec app /bin/bash

# Check health of services
health:
	@echo "Checking services..."
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "App not responding"
	@echo ""
	@curl -s http://localhost:11434/api/tags | python3 -m json.tool || echo "Ollama not responding"

# Run tests
test:
	docker compose exec app pytest tests/ -v

# Clean up containers and volumes
clean:
	docker compose down -v
	docker compose -f docker-compose.simple.yml down -v 2>/dev/null || true

# Clean everything including images
clean-all: clean
	docker rmi rag-app 2>/dev/null || true
	docker volume rm rag-app-storage rag-vector-db ollama-data 2>/dev/null || true

# Pull Ollama model manually
pull-model:
	docker compose exec ollama ollama pull llama3.2

# Show running containers
ps:
	docker compose ps


