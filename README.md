# ContextForge

**ContextForge** is a production-grade, local-first RAG (Retrieval-Augmented Generation) system designed for deep document understanding and interactive Q&A. It combines **Hybrid Search** (Dense + Sparse), **Hierarchical Memory**, and **Adaptive Retrieval** to deliver grounded, context-aware answers from your local documents.

## 🚀 Key Features

*   **Hybrid Retrieval Engine**:
    *   **Dense Retrieval**: Meaning-based search using `sentence-transformers`.
    *   **Sparse Retrieval (SPLADE)**: Precise lexical matching for improved keyword handling.
    *   **BM25**: Traditional term-matching fallback.
    *   **Reciprocal Rank Fusion (RRF)**: Smartly combines results from all retrievers.
*   **Hierarchical Memory System**:
    *   **Short-term**: Remembers recent conversation turns.
    *   **Episodic**: Recalls relevant past turns using FTS5 full-text search.
    *   **Semantic Profile**: Extracts and persists fast facts about user preferences.
*   **Smart Document Processing**:
    *   **Semantic Chunking**: Intelligently splits text based on meaning (optional) or recursive splitting.
    *   **Chapter Awareness**: Detects and navigates document chapters for scoped queries.
    *   **Logical Page Mapping**: Maps PDF physical pages to logical printed page numbers.
*   **Production Ready**:
    *   **FastAPI Server**: Built-in API with metrics and caching.
    *   **Dockerized**: Ready-to-run container support.
    *   **Observability**: Structured metrics logging (latency, tokens, cost) and request tracing.
*   **Local & Privacy-Focused**:
    *   Runs 100% locally with **Ollama** or connects to **Groq API** for cloud inference.
    *   GPU acceleration (CUDA) detected and used automatically.

---

## 🛠️ Installation

### Prerequisites
*   **Python 3.10+**
*   **Ollama** (for local LLMs) or **Groq API Key** (for cloud LLMs).
*   *(Optional)* **Build Tools**: for installing `hnswlib` or `chromadb` dependencies on some OSs.

### Quick Start

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/vijayabhaskar78/-ContextForge.git
    cd contextforge-main
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r final/requirements.txt
    ```

3.  **Set Up Environment**
    Copy the example environment file and configure it:
    ```bash
    cp .env.example .env
    ```
    *Edit `.env` to set `USE_API_LLM=True` and `GROQ_API_KEY=...` if using cloud models, or keep defaults for local Ollama.*

4.  **Run the CLI**
    ```bash
    python final/src/contextforge/app.py
    ```

---

## ⚙️ Configuration

Control the system via `.env` or environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| **LLM Settings** | | |
| `USE_API_LLM` | `False` | Toggle between Local (Ollama) and Cloud (Groq). |
| `LOCAL_MODEL_NAME` | `granite3.3:2b` | Ollama model tag (e.g., `llama3`, `mistral`). |
| `API_MODEL_NAME` | `gemma2-9b-it` | Groq model ID. |
| **Retrieval** | | |
| `Use_SEMANTIC_CHUNKING` | `False` | Enable experimental semantic chunking (slower but better). |
| `CHUNK_SIZE` | `1000` | Token size for document chunks. |
| `SPLADE_CPU_BATCH_SIZE` | `8` | Batch size for sparse embedding generation. |
| **System** | | |
| `DB_PATH` | `./data/vector_store_db` | Location for ChromaDB vector store. |
| `CACHE_DIR` | `./data/runtime_cache` | Location for SQLite memory and caches. |

---

## 🖥️ Usage

### CLI Commands
The CLI (`app.py`) provides an interactive shell.

*   **`upload <path>`**: Ingest a PDF or text file.
    *   *Example*: `upload docs/manual.pdf`
*   **`clear_history`**: Reset the conversation memory for the current session.
*   **`exit` / `quit`**: Close the application.
*   **Just type a question**: Ask anything about your uploaded documents!

### API Server
Run the REST API for integration with other apps:

```bash
uvicorn final.src.contextforge.api_server:app --host 0.0.0.0 --port 8000
```

*   **POST `/query`**: Ask a question.
    ```json
    {
      "query": "What is the battery life?",
      "session_id": "user-123"
    }
    ```
*   **GET `/metrics`**: View system performance stats (latency, tokens, etc.).

### Docker
Run the entire system in a container:

1.  **Build**:
    ```bash
    docker build -t contextforge-api -f final/Dockerfile .
    ```
2.  **Run**:
    ```bash
    docker run -d -p 8000:8000 --env-file .env contextforge-api
    ```

---

## 🏗️ Architecture Deep Dive

### 1. RAG Pipeline (`rag_pipeline.py`)
Incoming queries go through a multi-stage process:
1.  **Query Analysis**: Determines if the user is asking for specific chapters or general facts.
2.  **Hybrid Retrieval**:
    *   **Vector Search**: Finds conceptually similar chunks.
    *   **BM25**: Finds exact keyword matches.
    *   **SPLADE**: Uses sparse representations to find "rare word" matches that dense vectors might miss.
3.  **Rank Fusion**: Merges results from all retrievers using Reciprocal Rank Fusion (RRF) to bubble up the best candidates.
4.  **Reranking**: An optional cross-encoder pass re-scores the top candidates for high precision.

### 2. Memory System (`memory_manager.py`)
ContextForge doesn't just "chat"; it *remembers*.
*   **SQLite Backend**: All memory is persisted in `conversation_memory.sqlite`.
*   **FTS5 Search**: Episodic memory uses full-text search to find relevant past turns even from days ago.
*   **Adaptive Context**: It dynamically mixes recent turns, retrieved past turns, and document chunks into the prompt context window based on a token budget.

### 3. Document Management (`document_manager.py`)
*   **Chunk Registry**: Tracks every text chunk and its "neighbors" in `chunk_registry.sqlite`. This allows the system to fetch surrounding context (previous/next chunks) during generation for better coherence.
*   **Deduplication**: Files are hashed; re-uploading the same file works intelligently without duplication.

---

## 🤝 Contributing

We welcome contributions!
1.  **Fork & Clone**: Fork the repo and clone locally.
2.  **Branch**: Create a feature branch (`git checkout -b feature-cool-thing`).
3.  **Hack**: Make your changes.
    *   *Tip*: Run `final/tests` to ensure no regressions.
4.  **Submit**: Open a Pull Request on GitHub.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

**ContextForge** — *Recall everything. Hallucinate nothing.*
