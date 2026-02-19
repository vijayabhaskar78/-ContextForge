# ContextForge

ContextForge is a local-first, CLI-based document Q&A system built on RAG.
It lets you upload your own documents, index them, and ask grounded questions with source pages.

## Full System Documentation

- `SYSTEM_DOCUMENTATION.md`

## Features

- Document upload and metadata tracking
- Hybrid retrieval (BM25 + vector retrieval)
- Cross-encoder reranking
- Conversational memory with follow-up resolution
- Chapter-aware deterministic QA workflows
- Grounded answer generation with source attribution

## Project Layout

- `final/app.py`: CLI and query orchestration
- `final/rag_pipeline.py`: retriever and generation chain setup
- `final/memory_manager.py`: conversational memory and session state
- `final/document_manager.py`: document lifecycle management
- `final/config.py`: runtime configuration
- `final/tests/`: regression tests
- `workspace_documents/`: uploaded files and metadata
- `vector_store_db/`: Chroma persistence
- `runtime_cache/`: BM25 and memory cache

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r final/requirements.txt
```

3. Install and run Ollama, then pull the configured model:

```bash
ollama pull granite3.3:2b
```

## Run

```bash
python final/app.py
```

## Notes

- The app auto-migrates legacy storage folders to current names on startup.
- PDF support depends on `PyMuPDF`.
- If local model inference is unavailable, verify Ollama is installed and running.
