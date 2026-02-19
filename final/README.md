# ContextForge API Service

A production-ready conversational RAG service built with FastAPI, supporting Groq (LLM) and SPLADE (sparse retrieval).

## Features
- **LLM**: Groq API integration (supports `qwen/qwen3-32b`, `llama-3.3-70b`, etc.).
- **Retrieval**: Hybrid (BM25 + Vector + Reranker) or SPLADE (GPU-accelerated sparse retrieval).
- **API**: High-performance FastAPI service with LRU caching and 8-worker thread pool.
- **Observability**: Real-time metrics for latency, throughput, memory, cost, and errors.
- **Docker**: Containerized for easy deployment.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/contextforge.git
    cd contextforge
    ```

2.  **Install dependencies:**
    ```bash
    pip install -e .
    ```
    *Note: Requires Python 3.10+ and PyTorch with CUDA 12.4 for GPU support.*

3.  **Configure Environment:**
    Copy `.env.example` to `.env` and set your API keys:
    ```bash
    cp .env.example .env
    # Edit .env with your GROQ_API_KEY
    ```

## Usage

### Start the Server
```bash
python -m uvicorn contextforge.api_server:app --host 0.0.0.0 --port 8000
```

### Run Load Tests
```bash
python scripts/load_test.py 10 4  # 10 requests, concurrency=4
```

### Metrics
Visit `http://localhost:8000/metrics` to see live performance data.

## Project Structure
- `src/contextforge/`: Source code package.
- `data/`: Local storage for documents, logs, and vector DBs.
- `scripts/`: Utility scripts (load testing, etc.).
