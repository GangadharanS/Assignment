# RAG Document Assistant

A conversational AI that answers questions about your documents using Retrieval-Augmented Generation.

---

## Quick Setup Instructions

```bash
# 1. Start Ollama (required for LLM)
ollama serve
ollama pull llama3.2

# 2. Run the application
cd assignment
source venv/bin/activate
docker compose -f docker-compose.simple.yml up -d

# 3. Upload a document
curl -X POST http://localhost:8080/upload -F 'file=@your-document.pdf'

# 4. Ask questions
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'
```

**API Docs:** http://localhost:8080/docs

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     FastAPI Server                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   /upload              /chat                /search      │
│      │                   │                     │         │
│      ▼                   ▼                     │         │
│  ┌────────┐      ┌─────────────┐               │         │
│  │Document│      │ RAG Pipeline│               │         │
│  │Processor      │             │               │         │
│  └───┬────┘      │ 1. Search   │◀──────────────┘         │
│      │           │ 2. Rerank   │                         │
│      ▼           │ 3. Guard    │                         │
│  ┌────────┐      │ 4. Generate │                         │
│  │Semantic│      └──────┬──────┘                         │
│  │Chunker │             │                                │
│  └───┬────┘             ▼                                │
│      │           ┌─────────────┐                         │
│      ▼           │   Ollama    │                         │
│  ┌────────┐      │  llama3.2   │                         │
│  │Embedding      └─────────────┘                         │
│  │Generator│                                             │
│  └───┬────┘                                              │
│      │                                                   │
│      ▼                                                   │
│  ┌────────┐                                              │
│  │ChromaDB│                                              │
│  │(Vector)│                                              │
│  └────────┘                                              │
└──────────────────────────────────────────────────────────┘
```

---

## RAG/LLM Design Decisions

### Chunking Strategy

**Choice:** Semantic chunking with sliding window

**Why:** Fixed-size chunking breaks sentences mid-thought. Semantic chunking detects topic shifts using embedding similarity and creates natural boundaries.

```python
# Sliding window compares embeddings before/after each sentence
similarity = cosine_similarity(before_window_avg, after_window_avg)
if similarity < threshold:  # Topic shift
    create_chunk_boundary()
```

**Config:** `max_chunk_size=5000`, `similarity_threshold=0.8`

---

### Embedding Choice

**Model:** `all-MiniLM-L6-v2` (Sentence Transformers)

| Reason | Detail |
|--------|--------|
| Speed | ~1ms per embedding |
| Size | 80MB (container-friendly) |
| Quality | 384 dims, 1B+ training pairs |
| License | Apache 2.0 |

---

### Prompt Structure

```
SYSTEM: Answer questions using ONLY the provided context.
        Cite sources as [Chunk N]. Say "I don't know" if unsure.

USER:   CONTEXT:
        [Chunk 1] Source: doc.pdf (relevance: 85%)
        <chunk content>
        
        QUESTION: {query}
```

This grounds the LLM in retrieved context and encourages citations.

---

### Guardrails Implemented

| Guardrail | What It Does |
|-----------|--------------|
| Query Validation | Rejects empty/too-short queries |
| Relevance Check | Ensures context meets min threshold (0.10) |
| Confidence Score | Rates answer as High/Medium/Low/None |
| Fallback Response | Graceful "I don't know" when no context |
| Source Attribution | Auto-adds citations to answers |
| Context Truncation | Limits context to prevent LLM timeout |

---

## Key Technical Decisions

| What | Choice | Why |
|------|--------|-----|
| Vector DB | ChromaDB | Embedded, no server, persistent storage |
| LLM | Ollama + llama3.2 | Free, local, fast on Mac (Metal) |
| Framework | FastAPI | Auto-docs, async, type validation |
| Deploy | Docker Compose | Simple, reproducible |
| Re-ranking | Cross-Encoder | Improves retrieval precision |

---

## How I Used AI Tools

**Tool:** Cursor + Claude

**AI helped with:**
- FastAPI boilerplate generation
- Semantic chunking algorithm implementation
- Docker networking debugging
- Error handling patterns
- Documentation writing

**My workflow:**
1. Describe feature requirements
2. AI generates initial code
3. Test and identify issues
4. Iterate with AI assistance

**Human judgment needed for:**
- Threshold tuning (relevance, confidence)
- Chunk size optimization

---

## What I'd Do Differently

1. **Hybrid Search** - Add BM25 for exact keyword matching
2. **Streaming** - SSE for real-time token output
3. **Query Expansion** - LLM-powered synonym expansion
4. **Spell Check** - Pre-process queries for typos
5. **Caching** - Redis for embeddings and frequent queries
6. **Evaluation** - RAGAs metrics for quality testing
7. **UI** - Admin dashboard for monitoring

---

## Running Instructions

### Prerequisites
- Docker
- Ollama: https://ollama.ai

### Start

```bash
# Terminal 1: Start Ollama
ollama serve
ollama pull llama3.2

# Terminal 2: Start app
cd assignment
docker compose -f docker-compose.simple.yml up -d
```

### Test

```bash
# Health check
curl http://localhost:8080/

# Upload document
curl -X POST http://localhost:8080/upload -F 'file=@document.pdf'

# Check index
curl http://localhost:8080/index/stats

# Ask question
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What cheeses are made from goat milk?"}'

# View logs
docker logs rag-app -f
```

### Stop

```bash
docker compose -f docker-compose.simple.yml down
```

### Clear Index

```bash
curl -X DELETE http://localhost:8080/index/clear
```

---

## Sample Data

A sample dataset is included for testing: `sample_data/cheese.csv`

**Dataset:** 1,187 cheeses from around the world with attributes like milk type, country, texture, and flavor.

### Upload Sample Data

```bash
curl -X POST http://localhost:8080/upload -F 'file=@sample_data/cheese.csv'
```

### Example Queries

Once the cheese dataset is uploaded, try these questions:

| Query | What It Tests |
|-------|---------------|
| `"What cheeses are made from goat milk?"` | Filtering by attribute |
| `"Which French cheeses have a soft texture?"` | Multi-attribute query |
| `"What cheese pairs well with wine?"` | Subjective/recommendation |
| `"Tell me about Brie"` | Specific entity lookup |
| `"What's the difference between cheddar and gouda?"` | Comparison query |
| `"List blue cheeses from Italy"` | Country + type filter |

### Example Chat Request

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What French cheeses are made from sheep milk?",
    "n_results": 5,
    "use_reranking": true
  }'
```

### Multi-turn Conversation

```bash
curl -X POST http://localhost:8080/chat/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is Roquefort?"},
      {"role": "assistant", "content": "Roquefort is a French blue cheese made from sheep milk..."},
      {"role": "user", "content": "What other cheeses are similar to it?"}
    ]
  }'
```

---

## Testing Strategy

### Approach

**Unit tests** for each module with **mocking** for external dependencies:

| Module | Test Strategy |
|--------|---------------|
| `chunker.py` | Test chunk boundaries, size limits, metadata |
| `embeddings.py` | Test dimension, similarity properties |
| `vector_store.py` | Test CRUD with temp directories |
| `guardrails.py` | Test validation rules, confidence scoring |
| `rag.py` | Mock LLM and vector store, test pipeline logic |
| `reranker.py` | Test score combination, ordering |
| `api.py` | Integration tests with TestClient |

### Key Testing Patterns

1. **Fixtures** - Shared setup (temp dirs, mock objects)
2. **Mocking** - Isolate LLM/DB calls for fast tests
3. **Parametrization** - Test multiple inputs
4. **Edge cases** - Empty inputs, Unicode, large data

### Running Tests

```bash
source venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=blob_storage --cov-report=html

# Run specific test file
pytest tests/test_guardrails.py -v

# Run specific test
pytest tests/test_chunker.py::TestSemanticChunker -v
```

---

## Project Structure

```
assignment/
├── src/blob_storage/
│   ├── api.py            # FastAPI endpoints
│   ├── chunker.py        # Semantic chunking
│   ├── embeddings.py     # Sentence embeddings
│   ├── vector_store.py   # ChromaDB
│   ├── rag.py            # RAG pipeline
│   ├── llm.py            # Ollama client
│   ├── reranker.py       # Cross-encoder
│   └── guardrails.py     # Safety checks
├── tests/
│   ├── test_api.py       # API endpoint tests
│   ├── test_storage.py   # Storage tests
│   ├── test_chunker.py   # Chunking tests
│   ├── test_embeddings.py # Embedding tests
│   ├── test_vector_store.py # Vector store tests
│   ├── test_rag.py       # RAG pipeline tests
│   ├── test_guardrails.py # Guardrails tests
│   ├── test_reranker.py  # Reranker tests
│   └── test_llm.py       # LLM tests
├── sample_data/
│   └── cheese.csv        # Sample dataset (1,187 cheeses)
├── docker-compose.simple.yml
├── Dockerfile
└── README.md
```
