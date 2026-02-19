# ContextForge

ContextForge Is a production-grade, local-first RAG (Retrieval-Augmented Generation) system designed for deep document understanding and interactive Q&A. It features a dual-mode retrieval engine, hierarchical conversational memory, and intelligent document processing to deliver grounded, context-aware answers from local documents.

## Key Architecture

ContextForge utilizes a modular architecture with two distinct, selectable retrieval strategies and a multi-layered memory system.

### 1. Dual-Mode Retrieval Engine

The system supports two mutually exclusive retrieval modes, selectable at runtime (CLI) or via configuration (API).

*   **Hybrid Mode (Default)**
    *   **Mechanism**: Combines results from **Dense Vector Search** (mediated by ChromaDB) and **Sparse Keyword Search** (BM25).
    *   **Fusion**: Uses Reciprocal Rank Fusion (RRF) to merge and normalize scores from both retrievers.
    *   **Refinement**: Top candidates are passed through a **Cross-Encoder Reranker** (using sentence-transformers) to maximize relevance before generation.
    *   **Use Case**: Best for general-purpose queries requiring both semantic understanding and exact keyword matching.

*   **SPLADE Mode (Experimental)**
    *   **Mechanism**: Uses **SPLADE** (Sparse Lexical and Expansion Model) to generate learned sparse representations.
    *   **Efficiency**: Performs retrieval in a single pass using inverted indices, capturing both semantic meaning and specific terms without requiring a separate heavy reranking step.
    *   **Use Case**: High-efficiency environments or domains where learned sparse representations outperform traditional dense vectors.

### 2. Hierarchical Memory System

ContextForge maintains state across three levels to support long-running, context-heavy conversations:

*   **Short-Term Memory**: Buffers the immediate conversation history (recent turns) for direct context.
*   **Episodic Memory**: Archives past conversation turns in SQLite. Retrieves relevant historical interactions using FTS5 (Full-Text Search) based on the current query.
*   **Semantic Profile**: Extracts and persists enduring user facts and preferences to personalize responses over time.

### 3. Smart Document Processing

*   **Logical Page Mapping**: Automatically detects physical-to-logical page offsets in PDFs (e.g., matching the printed "Page 1" to physical page 12) for accurate citations.
*   **Chapter Awareness**: Scans and indexes document structures to support scoped queries like "summarize chapter 4" or "what is in the conclusion?".
*   **Chunk Registry**: Tracks the linear relationship between text chunks, allowing the system to expand context windows dynamically (retrieving previous/next chunks) during generation.

---

## Installation

### Prerequisites

*   **Python 3.10+**
*   **Ollama** (for local LLM inference) or a **Groq API Key** (for cloud inference).

### Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/vijayabhaskar78/-ContextForge.git
    cd contextforge-main
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r final/requirements.txt
    ```

3.  **Configure Environment**
    Copy the example configuration and edit it:
    ```bash
    cp .env.example .env
    ```
    To use cloud models, set `USE_API_LLM=True` and provide `GROQ_API_KEY`.

---

## Configuration

The system is configured via environment variables or the `.env` file.

| Variable | Default | Description |
| :--- | :--- | :--- |
| **Retrieval** | | |
| `RETRIEVAL_MODE` | `hybrid` | Selects the retrieval engine: `hybrid` or `splade`. |
| `Use_SEMANTIC_CHUNKING`| `False` | Toggles experimental semantic chunking (requires `langchain_experimental`). |
| `CHUNK_SIZE` | `1000` | Token size for document chunks. |
| **Inference** | | |
| `USE_API_LLM` | `False` | `True` for Groq API, `False` for local Ollama. |
| `LOCAL_MODEL_NAME` | `granite3.3:2b` | Model tag for local Ollama instance. |
| `API_MODEL_NAME` | `gemma2-9b-it` | Model ID for Groq API. |
| **System** | | |
| `DB_PATH` | `./data/vector_store_db` | Storage path for ChromaDB. |

---

## Usage

### CLI Application

The CLI provides an interactive shell for document ingestion and Q&A.

1.  **Start the Application**:
    ```bash
    python final/src/contextforge/app.py
    ```

2.  **Workflow**:
    *   **Upload**: Use option `1` to ingest PDF or text files.
    *   **Chat**: Use option `3` to start a session.
    *   **Select Mode**: You will be prompted to choose between **Standard Hybrid** or **Experimental SPLADE** mode for the session.

### API Server

The FastAPI server exposes the RAG pipeline for external integrations.

1.  **Start the Server**:
    ```bash
    uvicorn final.src.contextforge.api_server:app --host 0.0.0.0 --port 8000
    ```
    *Note: The server uses the `RETRIEVAL_MODE` environment variable to determine the engine.*

2.  **Endpoints**:
    *   `POST /query`: Submit a question.
        ```json
        {
          "query": "Summarize the safety protocols",
          "session_id": "optional-client-session-id"
        }
        ```
    *   `GET /metrics`: Retrieve system performance telemetry (latency, token usage, cost).

### Docker Deployment

Run the complete system in a containerized environment.

1.  **Build Image**:
    ```bash
    docker build -t contextforge-api -f final/Dockerfile .
    ```

2.  **Run Container**:
    ```bash
    docker run -d -p 8000:8000 --env-file .env contextforge-api
    ```

---

## Observability

The system logs comprehensive metrics to `logs/metrics.jsonl`, including:
*   **Latency**: Request processing time (ms).
*   **Token Usage**: Input and output token counts.
*   **Cost Estimation**: USD cost based on configured model pricing.
*   **Error Rates**: Tracking of failed requests.

These logs can be ingested by monitoring tools or analyzed directly for performance tuning.
