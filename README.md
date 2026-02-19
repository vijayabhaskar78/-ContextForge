# ContextForge

ContextForge is a local-first, CLI-based Retrieval-Augmented Generation (RAG) system designed for grounded document Q&A. It allows users to upload documents, index them, and engage in interactive Q&A sessions with source attribution.

## Features

- **Hybrid Retrieval**: Combines BM25 (keyword) and Vector search (semantic) for robust retrieval.
- **Cross-Encoder Reranking**: Re-ranks retrieved documents to ensure high relevance.
- **Hierarchical Memory**: Maintains conversational context, handling follow-up queries and ambigious references.
- **Chapter-Aware Workflows**: Deterministic routing for chapter-specific intents (count, list, lookup, explain).
- **Grounded Generation**: Answers are strictly grounded in provided context, with source page citations.
- **Local & API Support**: Run entirely locally with Ollama or use Groq API for inference.
- **Docker Support**: Containerized deployment available (see [Docker Instructions](#docker-usage)).

## Architecture

### System Modules
- **`final/app.py`**: Main entry point, CLI orchestration, query routing, and chapter logic.
- **`final/rag_pipeline.py`**: Handles document chunking, retriever construction, reranking, and LLM chain creation.
- **`final/memory_manager.py`**: Manages conversational memory (short-term, episodic, semantic) using SQLite.
- **`final/document_manager.py`**: Handles document upload, metadata tracking, and indexing status.
- **`final/config.py`**: Centralized configuration for models, paths, and environment variables.

### Data Persistence
- **`workspace_documents/`**: Stores uploaded source documents and `uploaded_docs.json` metadata.
- **`vector_store_db/`**: Persisted Chroma vector database.
- **`runtime_cache/`**: Stores BM25 index cache and SQLite conversation memory.

### Tech Stack
- **Languages**: Python 3.10+
- **CLI**: Rich
- **Orchestration**: LangChain
- **Vector DB**: Chroma
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: Ollama (Local) or Groq (API)

## End-to-End Workflows

### 1. Document Upload
Documents are uploaded via the CLI. They are copied to `workspace_documents/`, assigned a unique ID, and tracked in `uploaded_docs.json`.

### 2. RAG Pipeline Initialization
When a session starts:
1.  **Index**: New documents are chunked and added to Chroma.
2.  **Retrieve**: Vector and BM25 retrievers are initialized.
3.  **Ensemble**: Retrievers are combined (weighted 50/50).
4.  **Rerank**: Results are refined using a Cross-Encoder.

### 3. Query Processing
1.  **Resolution**: User query is checked against memory to resolve pronouns/follow-ups (e.g., "Explain *this*").
2.  **Routing**: Query is classified.
    *   **Chapter Intents**: Routed to deterministic logic (e.g., "List chapters").
    *   **General QA**: Routed to RAG pipeline.
3.  **Retrieval**: Hybrid retrieval + Reranking fetches top context.
4.  **Generation**: LLM generates an answer based *only* on retrieved context.
5.  **Sanitization**: Unsupported acronym expansions are removed.
6.  **Response**: Answer is displayed with source citations.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/vijayabhaskar78/-ContextForge.git
    cd -ContextForge
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r final/requirements.txt
    ```

3.  **Setup LLM (Local)**:
    Install [Ollama](https://ollama.com/) and pull the default model:
    ```bash
    ollama pull granite3.3:2b
    ```
    *Note: Model can be changed in `.env` or `config.py`.*

## Usage

### Run CLI Application
```bash
python final/src/contextforge/app.py
# or
cd final/src && python -m contextforge.app
```

### Run API Server
The project includes a production-ready FastAPI server.
```bash
cd final/src
python -m uvicorn contextforge.api_server:app --host 0.0.0.0 --port 8000
```
- **Swagger UI**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics

### Docker Usage
Build and run the containerized API server.
```bash
docker build -t contextforge-api:latest final/
docker run -p 8000:8000 contextforge-api:latest
```

## Contributing
See `CONTRIBUTING.md` for guidelines on how to contribute to this project.
