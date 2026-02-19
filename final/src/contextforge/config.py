# /contextforge_cli/config.py
"""
Centralized configuration for the RAG application.
Includes model names, paths, feature toggles, and hardware detection.
"""
import functools
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from .observability import configure_logging

# ==============================================================================
# CONSOLE & ENVIRONMENT
# ==============================================================================
console = Console()
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return max(float(minimum), value)

# ==============================================================================
# GPU DETECTION & SETUP
# ==============================================================================
def _load_torch():
    try:
        import torch  # Imported lazily to avoid heavy startup for non-LLM test paths.
        return torch
    except ImportError:
        return None


@functools.cache
def detect_gpu_setup():
    """Detects and prints GPU information on first access only."""
    torch = _load_torch()
    if torch is not None and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        console.print(Panel(
            f"[bold green]GPU Detected![/bold green]\n"
            f"Device: {device_name}\n"
            f"Memory: {memory_gb:.1f} GB",
            title="GPU Configuration",
            border_style="green"
        ))
        return {'device': 'cuda', 'name': device_name}
    else:
        console.print("[yellow]No GPU detected. Using CPU instead.[/yellow]")
        return {'device': 'cpu', 'name': 'cpu'}


@functools.cache
def get_model_kwargs() -> dict[str, str]:
    return {'device': detect_gpu_setup()['device']}


# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
# --- Application Toggles ---
USE_API_LLM = _env_bool("USE_API_LLM", False)              # True for Groq API, False for local Ollama
USE_SEMANTIC_CHUNKING = _env_bool("USE_SEMANTIC_CHUNKING", False)    # True for semantic chunking, False for faster recursive chunking

# --- Model Names ---
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "granite3.3:2b") # Local model to use with Ollama
API_MODEL_NAME = os.getenv("API_MODEL_NAME", "gemma2-9b-it")  # API model to use with Groq
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Path Configuration ---
# Data directory is at ../../data relative to this file (src/contextforge/config.py)
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _BASE_DIR / "data"

DB_PATH = os.getenv("DB_PATH", str(_DATA_DIR / "vector_store_db"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(_DATA_DIR / "workspace_documents")))
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(_DATA_DIR / "runtime_cache")))
BM25_CACHE_FILE = CACHE_DIR / "bm25_retriever.json"

# --- Chunking Configuration ---
CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000, minimum=128)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 150, minimum=0)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    CHUNK_OVERLAP = max(0, CHUNK_SIZE // 4)

# --- Retrieval / Ingestion Tuning ---
ASYNC_RETRIEVER_MAX_WORKERS = _env_int("ASYNC_RETRIEVER_MAX_WORKERS", 2, minimum=1)
INGEST_MAX_WORKERS = _env_int("INGEST_MAX_WORKERS", 4, minimum=1)
SPLADE_CPU_BATCH_SIZE = _env_int("SPLADE_CPU_BATCH_SIZE", 8, minimum=1)
SPLADE_GPU_BATCH_PER_GB = _env_int("SPLADE_GPU_BATCH_PER_GB", 4, minimum=1)
SPLADE_BATCH_MIN = _env_int("SPLADE_BATCH_MIN", 2, minimum=1)
SPLADE_BATCH_MAX = _env_int("SPLADE_BATCH_MAX", 128, minimum=1)
SPLADE_LOW_VRAM_THRESHOLD_GB = _env_float("SPLADE_LOW_VRAM_THRESHOLD_GB", 6.0, minimum=1.0)
SPLADE_LOW_VRAM_BATCH_CAP = _env_int("SPLADE_LOW_VRAM_BATCH_CAP", 4, minimum=1)
SPLADE_RETRIEVER_K = _env_int("SPLADE_RETRIEVER_K", 30, minimum=1)
# Alias for compatibility if needed, though SPLADE_CPU_BATCH_SIZE is preferred.
SPLADE_BATCH_SIZE = SPLADE_CPU_BATCH_SIZE

# --- Memory Tuning ---
MEMORY_CONTEXT_RECENT_TURNS = _env_int("MEMORY_CONTEXT_RECENT_TURNS", 4, minimum=1)
MEMORY_CONTEXT_EPISODIC_TURNS = _env_int("MEMORY_CONTEXT_EPISODIC_TURNS", 3, minimum=1)
MEMORY_PROFILE_FACT_LIMIT = _env_int("MEMORY_PROFILE_FACT_LIMIT", 5, minimum=1)
MEMORY_RECALL_LOOKBACK_ASSISTANT_TURNS = _env_int("MEMORY_RECALL_LOOKBACK_ASSISTANT_TURNS", 8, minimum=1)
MEMORY_SUMMARY_REFRESH_EVERY_N_TURNS = _env_int("MEMORY_SUMMARY_REFRESH_EVERY_N_TURNS", 6, minimum=2)
MEMORY_EPISODIC_TOKEN_FILTER_LIMIT = _env_int("MEMORY_EPISODIC_TOKEN_FILTER_LIMIT", 8, minimum=1)
MEMORY_EPISODIC_SCAN_LIMIT = _env_int("MEMORY_EPISODIC_SCAN_LIMIT", 220, minimum=20)
PROMPT_TOTAL_TOKEN_BUDGET = _env_int("PROMPT_TOTAL_TOKEN_BUDGET", 4096, minimum=512)
PROMPT_MEMORY_RATIO = _env_float("PROMPT_MEMORY_RATIO", 0.30, minimum=0.05)
PROMPT_DOCS_RATIO = _env_float("PROMPT_DOCS_RATIO", 0.60, minimum=0.10)
PROMPT_QUERY_RATIO = _env_float("PROMPT_QUERY_RATIO", 0.10, minimum=0.05)
ratio_total = PROMPT_MEMORY_RATIO + PROMPT_DOCS_RATIO + PROMPT_QUERY_RATIO
if ratio_total > 1.0:
    PROMPT_MEMORY_RATIO = PROMPT_MEMORY_RATIO / ratio_total
    PROMPT_DOCS_RATIO = PROMPT_DOCS_RATIO / ratio_total
    PROMPT_QUERY_RATIO = PROMPT_QUERY_RATIO / ratio_total

# --- Streaming Tuning ---
STREAM_POLL_INTERVAL_S = _env_float("STREAM_POLL_INTERVAL_S", 0.20, minimum=0.01)
# Set to 0 to disable stall timeout and prefer continuous streaming.
STREAM_IDLE_TIMEOUT_S = _env_float("STREAM_IDLE_TIMEOUT_S", 0.0, minimum=0.0)

# --- Create necessary directories ---
Path(DB_PATH).mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
LOG_PATH = Path(os.getenv("LOG_PATH", str(CACHE_DIR / "app.log")))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
configure_logging(LOG_PATH)

# --- Dependency Availability Flags ---
# These flags prevent crashes if optional libraries are not installed.
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import langchain_groq
    GROQ_API_AVAILABLE = True
except ImportError:
    GROQ_API_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
