# /contextforge_cli/rag_pipeline.py
"""
Handles the creation of the RAG pipeline, including document processing,
retriever setup (Chroma, BM25, Reranker), and LLM chain initialization.
"""
import os
import time
import asyncio
import re
import json
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from pydantic import ConfigDict

# LangChain and related imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

try:
    # LangChain v1.x compatibility path
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    # LangChain <=0.x compatibility path
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
from langchain_ollama import OllamaLLM
try:
    import fitz
except ImportError:
    fitz = None
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    from scipy import sparse
except ImportError:
    sparse = None
try:
    import torch
except ImportError:
    torch = None
try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError:
    AutoModelForMaskedLM = None
    AutoTokenizer = None

# Local Imports
from .config import (
    API_MODEL_NAME,
    ASYNC_RETRIEVER_MAX_WORKERS,
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    EMBEDDING_MODEL_NAME,
    GROQ_API_AVAILABLE,
    INGEST_MAX_WORKERS,
    LOCAL_MODEL_NAME,
    OLLAMA_AVAILABLE,
    PDF_SUPPORT,
    RERANKER_MODEL_NAME,
    SEMANTIC_CHUNKING_AVAILABLE, # Kept this as it's used below
    SPLADE_BATCH_MAX,
    SPLADE_BATCH_MIN,
    SPLADE_CPU_BATCH_SIZE,
    SPLADE_GPU_BATCH_PER_GB,
    SPLADE_LOW_VRAM_BATCH_CAP,
    SPLADE_LOW_VRAM_THRESHOLD_GB,
    SPLADE_RETRIEVER_K,
    STREAM_IDLE_TIMEOUT_S,
    STREAM_POLL_INTERVAL_S,
    USE_API_LLM,
    USE_SEMANTIC_CHUNKING,
    console,
    detect_gpu_setup,
    get_model_kwargs,
)
from .document_manager import DocumentUploader
from .observability import get_logger
from .tokenization import tokenize_for_matching
if SEMANTIC_CHUNKING_AVAILABLE:
    from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()
logger = get_logger(__name__)

# Retrieval depth settings tuned for richer grounded answers.
VECTOR_RETRIEVER_K = 20
BM25_RETRIEVER_K = 20
RERANK_TOP_N = 5
RRF_K = 60
SPLADE_RETRIEVER_K = 5
SPLADE_BATCH_SIZE = SPLADE_CPU_BATCH_SIZE
SPLADE_MAX_LENGTH = 256
SPLADE_DOC_TOP_TERMS = 256
SPLADE_QUERY_TOP_TERMS = 256
SPLADE_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
SPLADE_CACHE_VERSION = 1
FOOTER_NUMBER_RE = re.compile(r"\b(\d{1,4})\b")
_EMBEDDING_MODEL = None
_RERANKER_MODEL = None
_BM25_RETRIEVER = None
_BM25_SIGNATURE: tuple[str, ...] | None = None
_SPLADE_MODEL = None
_SPLADE_TOKENIZER = None
_SPLADE_DEVICE = "cpu"
_SPLADE_SERVICE = None


def get_embeddings():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=get_model_kwargs(),
        )
    return _EMBEDDING_MODEL


def get_reranker():
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        _RERANKER_MODEL = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL_NAME,
            model_kwargs=get_model_kwargs(),
        )
    return _RERANKER_MODEL


def _release_torch_model(model_obj):
    if model_obj is None or torch is None:
        return
    move_fn = getattr(model_obj, "to", None)
    if callable(move_fn):
        try:
            move_fn("cpu")
        except Exception:
            pass


def unload_hybrid_components():
    """Explicitly unload embedding/reranker model references to reduce memory pressure."""
    global _EMBEDDING_MODEL, _RERANKER_MODEL
    _release_torch_model(_EMBEDDING_MODEL)
    _release_torch_model(_RERANKER_MODEL)
    _EMBEDDING_MODEL = None
    _RERANKER_MODEL = None
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


class SpladeService:
    """Lifecycle-managed SPLADE service with explicit load/unload semantics."""

    def __init__(self, model_name: str = SPLADE_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

    def is_loaded(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    def load(self):
        """Loads tokenizer/model on the configured device."""
        if AutoTokenizer is None or AutoModelForMaskedLM is None or torch is None:
            console.print("[bold red]SPLADE dependencies missing. Install transformers + torch.[/bold red]")
            return None, None, "cpu"
        if sparse is None or np is None:
            console.print("[bold red]SPLADE dependencies missing. Install scipy + numpy.[/bold red]")
            return None, None, "cpu"

        if self.is_loaded():
            return self.tokenizer, self.model, self.device

        configured_device = str(get_model_kwargs().get("device", "cpu")).lower()
        use_cuda = configured_device == "cuda" and torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            console.print(f"[bold red]Failed to load SPLADE model '{self.model_name}': {exc}[/bold red]")
            self.tokenizer = None
            self.model = None
            return None, None, self.device
        return self.tokenizer, self.model, self.device

    def unload(self):
        """Releases model/tokenizer references and clears CUDA cache when available."""
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self.tokenizer = None
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        self.device = "cpu"


def get_splade_service() -> SpladeService:
    global _SPLADE_SERVICE
    if _SPLADE_SERVICE is None:
        _SPLADE_SERVICE = SpladeService(model_name=SPLADE_MODEL_NAME)
    return _SPLADE_SERVICE


def get_splade_components():
    """Loads and caches SPLADE tokenizer/model for sparse lexical retrieval."""
    global _SPLADE_MODEL, _SPLADE_TOKENIZER, _SPLADE_DEVICE

    service = get_splade_service()
    tokenizer, model, device = service.load()
    _SPLADE_TOKENIZER = tokenizer
    _SPLADE_MODEL = model
    _SPLADE_DEVICE = device
    return _SPLADE_TOKENIZER, _SPLADE_MODEL, _SPLADE_DEVICE


def unload_splade_components():
    """Explicitly unloads SPLADE resources when mode is not in use."""
    global _SPLADE_MODEL, _SPLADE_TOKENIZER, _SPLADE_DEVICE
    service = get_splade_service()
    service.unload()
    _SPLADE_MODEL = None
    _SPLADE_TOKENIZER = None
    _SPLADE_DEVICE = "cpu"


def _get_available_vram_gb() -> float:
    """Best-effort CUDA VRAM detection for dynamic SPLADE batching."""
    if torch is None or not torch.cuda.is_available():
        return 0.0
    try:
        total_bytes = float(torch.cuda.get_device_properties(0).total_memory)
        return max(0.0, total_bytes / (1024 ** 3))
    except Exception:
        return 0.0


def get_optimal_splade_batch_size() -> int:
    """
    Chooses SPLADE batch size from hardware capacity.
    - CPU: conservative fixed batch
    - GPU: scales with VRAM, with low-VRAM safety cap
    """
    cpu_default = max(1, int(SPLADE_CPU_BATCH_SIZE))
    configured_device = str(get_model_kwargs().get("device", "cpu")).lower()
    if configured_device != "cuda":
        return cpu_default

    vram_gb = _get_available_vram_gb()
    if vram_gb <= 0.0:
        return cpu_default

    dynamic_batch = int(max(1.0, vram_gb) * max(1, int(SPLADE_GPU_BATCH_PER_GB)))
    dynamic_batch = max(int(SPLADE_BATCH_MIN), min(int(SPLADE_BATCH_MAX), dynamic_batch))
    if vram_gb <= float(SPLADE_LOW_VRAM_THRESHOLD_GB):
        dynamic_batch = min(dynamic_batch, int(SPLADE_LOW_VRAM_BATCH_CAP))
    return max(1, int(dynamic_batch))


def _doc_signature(file_paths: list[str]) -> tuple[str, ...]:
    parts = []
    for file_path in sorted(file_paths):
        path = Path(file_path)
        try:
            mtime = int(path.stat().st_mtime)
        except OSError:
            mtime = 0
        parts.append(f"{path}:{mtime}")
    return tuple(parts)


def _splade_cache_path(doc_signature: tuple[str, ...], *, batch_size: int | None = None) -> Path:
    """Builds a stable cache filename for SPLADE index artifacts."""
    effective_batch = int(batch_size) if batch_size is not None else int(SPLADE_BATCH_SIZE)
    key_parts = [
        f"v={SPLADE_CACHE_VERSION}",
        f"model={SPLADE_MODEL_NAME}",
        f"chunking={'semantic' if USE_SEMANTIC_CHUNKING else 'recursive'}",
        f"batch={effective_batch}",
        f"max_len={SPLADE_MAX_LENGTH}",
        f"doc_top={SPLADE_DOC_TOP_TERMS}",
        f"query_top={SPLADE_QUERY_TOP_TERMS}",
        *doc_signature,
    ]
    digest = hashlib.sha256("||".join(key_parts).encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"splade_index_{digest}"


def _splade_cache_meta_path(cache_base_path: Path) -> Path:
    return Path(cache_base_path).with_suffix(".json")


def _splade_cache_matrix_path(cache_base_path: Path) -> Path:
    return Path(cache_base_path).with_suffix(".npz")

def _is_plausible_logical_page(value: int | None) -> bool:
    try:
        page_num = int(value)
    except (TypeError, ValueError):
        return False
    return 0 < page_num < 2000


def _parse_numeric_page_label(label: Any) -> int | None:
    """Parses numeric page labels from structured PDF page-label strings."""
    raw = str(label or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    match = re.search(r"(?i)(?:page\s+)?(\d{1,4})\s*$", raw)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _read_structured_pdf_page_labels(path: Path) -> dict[int, int]:
    """
    Reads page labels using a structured PDF parser (pypdf) and returns
    {physical_page(1-based): logical_page}.
    """
    if PdfReader is None:
        return {}
    try:
        reader = PdfReader(str(path))
        labels = getattr(reader, "page_labels", None)
    except Exception:
        return {}
    if not labels:
        return {}

    label_map: dict[int, int] = {}
    for page_idx, label in enumerate(labels, start=1):
        numeric = _parse_numeric_page_label(label)
        if _is_plausible_logical_page(numeric):
            label_map[page_idx] = int(numeric)
    return label_map


class AsyncHybridRetriever(BaseRetriever):
    """Runs BM25 and vector retrieval concurrently, then rank-fuses results."""

    bm25_retriever: Any
    vector_retriever: Any
    weights: tuple[float, float] = (0.5, 0.5)
    execution_mode: str = "auto"  # auto | serial | thread
    rrf_k: int = RRF_K
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _invoke(retriever, query):
        try:
            return retriever.invoke(query)
        except TypeError:
            return retriever.invoke(input=query)

    @staticmethod
    def _doc_key(doc):
        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source", ""))
        page = metadata.get("page", -1)
        prefix = " ".join(str(getattr(doc, "page_content", "")).split())[:160]
        return (source, page, prefix)

    def _rank_fuse(self, bm25_docs, vector_docs):
        scores = {}
        dedup = {}
        weights = self.weights if len(self.weights) == 2 else (0.5, 0.5)
        smooth_k = max(1, int(self.rrf_k))

        for rank, doc in enumerate(bm25_docs or [], start=1):
            key = self._doc_key(doc)
            dedup[key] = doc
            scores[key] = scores.get(key, 0.0) + (weights[0] / (smooth_k + rank))

        for rank, doc in enumerate(vector_docs or [], start=1):
            key = self._doc_key(doc)
            dedup[key] = doc
            scores[key] = scores.get(key, 0.0) + (weights[1] / (smooth_k + rank))

        ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [dedup[key] for key in ranked_keys]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        mode = str(self.execution_mode or "auto").lower()
        use_parallel = mode == "thread"
        if mode == "auto":
            # CPU-bound lexical scoring often gains little from thread pooling under GIL.
            use_parallel = str(get_model_kwargs().get("device", "cpu")).lower() == "cuda"

        if use_parallel:
            with ThreadPoolExecutor(max_workers=ASYNC_RETRIEVER_MAX_WORKERS) as pool:
                future_bm25 = pool.submit(self._invoke, self.bm25_retriever, query)
                future_vector = pool.submit(self._invoke, self.vector_retriever, query)
                bm25_docs = future_bm25.result()
                vector_docs = future_vector.result()
        else:
            bm25_docs = self._invoke(self.bm25_retriever, query)
            vector_docs = self._invoke(self.vector_retriever, query)
        return self._rank_fuse(bm25_docs, vector_docs)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        mode = str(self.execution_mode or "auto").lower()
        use_parallel = mode == "thread"
        if mode == "auto":
            use_parallel = str(get_model_kwargs().get("device", "cpu")).lower() == "cuda"
        if use_parallel:
            loop = asyncio.get_running_loop()
            bm25_future = loop.run_in_executor(None, self._invoke, self.bm25_retriever, query)
            vector_future = loop.run_in_executor(None, self._invoke, self.vector_retriever, query)
            bm25_docs, vector_docs = await asyncio.gather(bm25_future, vector_future)
        else:
            bm25_docs = self._invoke(self.bm25_retriever, query)
            vector_docs = self._invoke(self.vector_retriever, query)
        return self._rank_fuse(bm25_docs, vector_docs)


class SpladeRetriever(BaseRetriever):
    """Sparse lexical retriever backed by an in-memory SPLADE CSR index."""

    documents: list[Document]
    tokenizer: Any
    model: Any
    doc_matrix: Any
    device: str = "cpu"
    k: int = SPLADE_RETRIEVER_K
    batch_size: int = SPLADE_BATCH_SIZE
    max_length: int = SPLADE_MAX_LENGTH
    doc_top_terms: int = SPLADE_DOC_TOP_TERMS
    query_top_terms: int = SPLADE_QUERY_TOP_TERMS
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(str(text or "").split())

    @classmethod
    def _encode_texts(
        cls,
        *,
        tokenizer,
        model,
        device: str,
        texts: list[str],
        batch_size: int,
        max_length: int,
        top_terms: int,
    ):
        if sparse is None or np is None or torch is None:
            return None
        vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            return None

        if not texts:
            return sparse.csr_matrix((0, vocab_size), dtype=np.float32)

        row_ids: list[int] = []
        col_ids: list[int] = []
        values: list[float] = []

        row_offset = 0
        with torch.inference_mode():
            for start_idx in range(0, len(texts), max(1, int(batch_size))):
                batch_texts = [
                    cls._clean_text(text)
                    for text in texts[start_idx:start_idx + max(1, int(batch_size))]
                ]
                if not batch_texts:
                    continue

                tokenized = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max(8, int(max_length)),
                    return_tensors="pt",
                )
                tokenized = {key: value.to(device) for key, value in tokenized.items()}
                logits = model(**tokenized).logits
                attn_mask = tokenized.get("attention_mask")
                if attn_mask is None:
                    attn_mask = torch.ones(logits.shape[:2], device=logits.device)
                splade_scores = torch.log1p(torch.relu(logits))
                splade_scores = splade_scores * attn_mask.unsqueeze(-1)
                pooled = torch.amax(splade_scores, dim=1).detach().cpu().numpy().astype(np.float32, copy=False)

                for local_row_idx, row_vector in enumerate(pooled):
                    if top_terms > 0 and top_terms < row_vector.size:
                        selected_cols = np.argpartition(row_vector, -top_terms)[-top_terms:]
                        selected_vals = row_vector[selected_cols]
                        nz_mask = selected_vals > 0.0
                        selected_cols = selected_cols[nz_mask]
                        selected_vals = selected_vals[nz_mask]
                    else:
                        selected_cols = np.flatnonzero(row_vector > 0.0)
                        selected_vals = row_vector[selected_cols]

                    if selected_cols.size == 0:
                        continue
                    row_id = row_offset + local_row_idx
                    row_ids.extend([row_id] * int(selected_cols.size))
                    col_ids.extend(selected_cols.astype(np.int32, copy=False).tolist())
                    values.extend(selected_vals.astype(np.float32, copy=False).tolist())

                row_offset += len(batch_texts)

        if not row_ids:
            return sparse.csr_matrix((len(texts), vocab_size), dtype=np.float32)

        return sparse.csr_matrix(
            (
                np.asarray(values, dtype=np.float32),
                (
                    np.asarray(row_ids, dtype=np.int32),
                    np.asarray(col_ids, dtype=np.int32),
                ),
            ),
            shape=(len(texts), vocab_size),
            dtype=np.float32,
        )

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        *,
        k: int = SPLADE_RETRIEVER_K,
        batch_size: int = SPLADE_BATCH_SIZE,
        max_length: int = SPLADE_MAX_LENGTH,
        doc_top_terms: int = SPLADE_DOC_TOP_TERMS,
        query_top_terms: int = SPLADE_QUERY_TOP_TERMS,
    ):
        tokenizer, model, device = get_splade_components()
        if tokenizer is None or model is None:
            return None
        texts = [str(getattr(doc, "page_content", "")) for doc in documents or []]
        doc_matrix = cls._encode_texts(
            tokenizer=tokenizer,
            model=model,
            device=device,
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            top_terms=doc_top_terms,
        )
        if doc_matrix is None:
            return None
        return cls(
            documents=documents or [],
            tokenizer=tokenizer,
            model=model,
            doc_matrix=doc_matrix,
            device=device,
            k=max(1, int(k)),
            batch_size=max(1, int(batch_size)),
            max_length=max(8, int(max_length)),
            doc_top_terms=max(1, int(doc_top_terms)),
            query_top_terms=max(1, int(query_top_terms)),
        )

    def save(self, path: Path) -> bool:
        """Serializes SPLADE documents + sparse matrix using JSON + NPZ (no pickle)."""
        try:
            cache_base = Path(path)
            cache_base.parent.mkdir(parents=True, exist_ok=True)
            meta_path = _splade_cache_meta_path(cache_base)
            matrix_path = _splade_cache_matrix_path(cache_base)
            docs_payload = []
            for doc in self.documents or []:
                metadata = dict(getattr(doc, "metadata", {}) or {})
                safe_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        safe_metadata[str(key)] = value
                    else:
                        safe_metadata[str(key)] = str(value)
                docs_payload.append(
                    {
                        "page_content": str(getattr(doc, "page_content", "") or ""),
                        "metadata": safe_metadata,
                    }
                )
            payload = {
                "cache_version": SPLADE_CACHE_VERSION,
                "model_name": SPLADE_MODEL_NAME,
                "documents": docs_payload,
                "batch_size": int(self.batch_size),
                "max_length": int(self.max_length),
                "doc_top_terms": int(self.doc_top_terms),
                "query_top_terms": int(self.query_top_terms),
            }
            meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
            matrix_tmp = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
            meta_tmp.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
            with matrix_tmp.open("wb") as matrix_handle:
                sparse.save_npz(matrix_handle, self.doc_matrix)
            meta_tmp.replace(meta_path)
            matrix_tmp.replace(matrix_path)
            return True
        except Exception as exc:
            console.print(f"[yellow]Failed to save SPLADE cache: {exc}[/yellow]")
            return False

    @classmethod
    def load(cls, path: Path, k: int = SPLADE_RETRIEVER_K):
        """Loads cached SPLADE sparse index from JSON + NPZ files."""
        try:
            cache_base = Path(path)
            meta_path = _splade_cache_meta_path(cache_base)
            matrix_path = _splade_cache_matrix_path(cache_base)
            if not meta_path.exists() or not matrix_path.exists():
                console.print("[yellow]SPLADE cache files missing. Rebuilding index.[/yellow]")
                return None
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            doc_matrix = sparse.load_npz(str(matrix_path))
        except Exception as exc:
            console.print(f"[yellow]Failed to load cached SPLADE index: {exc}[/yellow]")
            return None

        if not isinstance(payload, dict):
            console.print("[yellow]Invalid SPLADE cache payload. Rebuilding index.[/yellow]")
            return None

        cache_version = int(payload.get("cache_version", -1))
        if cache_version != SPLADE_CACHE_VERSION:
            console.print(
                f"[yellow]SPLADE cache version mismatch ({cache_version} != {SPLADE_CACHE_VERSION}). Rebuilding index.[/yellow]"
            )
            return None

        cached_model = str(payload.get("model_name", ""))
        if cached_model and cached_model != SPLADE_MODEL_NAME:
            console.print(
                f"[yellow]SPLADE cache model mismatch ({cached_model} != {SPLADE_MODEL_NAME}). Rebuilding index.[/yellow]"
            )
            return None

        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list) or doc_matrix is None:
            console.print("[yellow]SPLADE cache missing required fields. Rebuilding index.[/yellow]")
            return None
        documents = [
            Document(
                page_content=str(item.get("page_content", "")),
                metadata=dict(item.get("metadata", {}) or {}),
            )
            for item in raw_documents
            if isinstance(item, dict)
        ]

        matrix_shape = getattr(doc_matrix, "shape", None)
        if not matrix_shape or int(matrix_shape[0]) != len(documents):
            console.print("[yellow]SPLADE cache row-count mismatch. Rebuilding index.[/yellow]")
            return None

        tokenizer, model, device = get_splade_components()
        if tokenizer is None or model is None:
            return None

        batch_size = int(payload.get("batch_size", SPLADE_BATCH_SIZE))
        max_length = int(payload.get("max_length", SPLADE_MAX_LENGTH))
        doc_top_terms = int(payload.get("doc_top_terms", SPLADE_DOC_TOP_TERMS))
        query_top_terms = int(payload.get("query_top_terms", SPLADE_QUERY_TOP_TERMS))

        return cls(
            documents=documents,
            tokenizer=tokenizer,
            model=model,
            doc_matrix=doc_matrix,
            device=device,
            k=max(1, int(k)),
            batch_size=max(1, batch_size),
            max_length=max(8, max_length),
            doc_top_terms=max(1, doc_top_terms),
            query_top_terms=max(1, query_top_terms),
        )

    def _encode_query(self, query: str):
        matrix = self._encode_texts(
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            texts=[str(query or "")],
            batch_size=1,
            max_length=self.max_length,
            top_terms=self.query_top_terms,
        )
        if matrix is None:
            return None
        return matrix

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        if sparse is None or np is None:
            return []
        if self.doc_matrix is None or getattr(self.doc_matrix, "shape", (0, 0))[0] == 0:
            return []

        query_vector = self._encode_query(query)
        if query_vector is None or getattr(query_vector, "nnz", 0) == 0:
            return []

        score_matrix = self.doc_matrix.dot(query_vector.transpose())
        score_values = np.asarray(score_matrix.toarray()).reshape(-1)
        if score_values.size == 0:
            return []

        k = min(max(1, int(self.k)), int(score_values.size))
        if k == score_values.size:
            top_indices = np.argsort(score_values)[::-1]
        else:
            top_indices = np.argpartition(score_values, -k)[-k:]
            top_indices = top_indices[np.argsort(score_values[top_indices])[::-1]]

        top_docs: list[Document] = []
        for index in top_indices:
            if float(score_values[index]) <= 0.0:
                continue
            top_docs.append(self.documents[int(index)])
        return top_docs

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_relevant_documents, query)


class TopicBoundedRetriever(BaseRetriever):
    """
    Wraps a retriever and constrains results to a tight logical-page window
    around the dominant topical anchor page for the current query.
    """

    base_retriever: Any
    forward_window_pages: int = 3
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _is_toc_like_text(text: str) -> bool:
        """Heuristic detector for table-of-contents/listing chunks."""
        compact = " ".join(str(text or "").split())
        if not compact:
            return False
        lower = compact.lower()
        if "table of contents" in lower:
            return True
        if re.search(r"\bcontents\b", lower) and len(compact) < 1400:
            return True

        numeric_tokens = re.findall(r"\b\d{1,3}\b", compact)
        chapter_style_lines = re.findall(r"\b\d{1,2}\.\s+[A-Z][A-Z0-9 &'()/:\-]{2,}", compact)
        if len(chapter_style_lines) >= 2 and len(numeric_tokens) >= 4 and len(compact) < 2200:
            return True

        numbered_heading_hits = len(re.findall(r"\b\d{1,2}\.\s+[A-Z][A-Z0-9 &'()/:\-]{3,}\s+\d{1,3}\b", compact))
        chapter_label_hits = len(re.findall(r"\bchapter\s+\d{1,2}\b", lower))
        trailing_page_hits = len(re.findall(r"\b[A-Za-z][A-Za-z0-9 &'()/:\-]{3,}\s+\d{1,3}\b", compact))
        if numbered_heading_hits >= 2:
            return True
        if chapter_label_hits >= 3 and trailing_page_hits >= 5 and len(compact) < 1700:
            return True
        return False

    @staticmethod
    def _is_chapter_like_query(query: str) -> bool:
        lowered = str(query or "").lower()
        return bool(
            re.search(r"\bchapter\b", lowered)
            or re.search(r"\btable of contents\b", lowered)
            or re.search(r"\bcontents\b", lowered)
            or re.search(r"\btoc\b", lowered)
        )

    @staticmethod
    def _invoke(retriever, query):
        try:
            return retriever.invoke(query)
        except TypeError:
            return retriever.invoke(input=query)

    @staticmethod
    def _display_page(doc: Document) -> int | None:
        metadata = getattr(doc, "metadata", {}) or {}
        logical_page = metadata.get("logical_page")
        try:
            if logical_page not in (None, ""):
                return int(logical_page)
        except (TypeError, ValueError):
            pass

        page = metadata.get("page")
        try:
            page = int(page)
        except (TypeError, ValueError):
            return None

        logical_offset = metadata.get("logical_page_offset")
        try:
            if logical_offset not in (None, ""):
                predicted = (page + 1) - int(logical_offset)
                if predicted > 0:
                    return int(predicted)
        except (TypeError, ValueError):
            pass
        return page + 1

    @staticmethod
    def _query_tokens(query: str) -> set[str]:
        tokens = tokenize_for_matching(query, min_len=3)
        stopwords = {
            "what", "which", "when", "where", "who", "why", "how", "the", "this", "that",
            "is", "are", "was", "were", "about", "explain", "define", "tell", "me", "in",
            "simple", "terms", "and", "for", "with", "from", "into", "does",
        }
        return {t for t in tokens if t not in stopwords}

    def _select_anchor_page(self, query: str, docs: list[Document], pages: list[int | None]) -> int | None:
        query_tokens = self._query_tokens(query)
        scored_pages: list[tuple[int, int, int, bool]] = []
        chapter_like_query = self._is_chapter_like_query(query)
        for rank, (doc, page) in enumerate(zip(docs, pages)):
            if page is None:
                continue
            text_tokens = set(tokenize_for_matching(str(getattr(doc, "page_content", "")), min_len=1))
            overlap = len(query_tokens.intersection(text_tokens)) if query_tokens else 0
            is_toc = self._is_toc_like_text(str(getattr(doc, "page_content", "")))
            scored_pages.append((overlap, -rank, page, is_toc))

        if scored_pages:
            primary = scored_pages
            if not chapter_like_query:
                non_toc = [item for item in scored_pages if not item[3]]
                if non_toc:
                    primary = non_toc
            primary.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best_overlap = primary[0][0]
            if best_overlap <= 0:
                return None
            anchor_candidates = [page for overlap, _, page, _ in primary if overlap == best_overlap]
            if anchor_candidates:
                # Prefer the most common candidate page among top lexical matches.
                return Counter(anchor_candidates).most_common(1)[0][0]

        # Fallback: first valid page in reranked order.
        for page in pages:
            if page is not None:
                return page
        return None

    def _filter_docs(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs

        chapter_like_query = self._is_chapter_like_query(query)
        working_docs = list(docs)
        if not chapter_like_query:
            non_toc_docs = [doc for doc in working_docs if not self._is_toc_like_text(str(getattr(doc, "page_content", "")))]
            if non_toc_docs:
                working_docs = non_toc_docs

        pages = [self._display_page(doc) for doc in working_docs]
        anchor_page = self._select_anchor_page(query, working_docs, pages)
        if anchor_page is None:
            return working_docs or docs

        lower = int(anchor_page)
        upper = int(anchor_page) + max(1, int(self.forward_window_pages))
        filtered = []
        for doc, page in zip(working_docs, pages):
            if page is None:
                continue
            if lower <= int(page) <= upper:
                filtered.append(doc)
        return filtered or working_docs or docs

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        docs = self._invoke(self.base_retriever, query)
        return self._filter_docs(query, docs or [])

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, self._invoke, self.base_retriever, query)
        return self._filter_docs(query, docs or [])

# --- Document Processing ---

def _extract_logical_page_number(pdf_page) -> int:
    """Extracts best-effort logical page number for a PDF page."""
    physical_page = int(pdf_page.number) + 1

    # 1) PDF internal page labels.
    label = ""
    try:
        label = str(pdf_page.get_label() or "").strip()
    except Exception:
        label = ""
    if label.isdigit():
        label_num = int(label)
        if label_num != physical_page:
            return label_num

    # 2) Footer heuristic: scrape bottom 15% and pick last number.
    footer_candidate_numbers = []
    try:
        blocks = pdf_page.get_text("blocks") or []
    except Exception:
        blocks = []
    footer_threshold = float(pdf_page.rect.height) * 0.85
    for block in blocks:
        if len(block) >= 5 and float(block[1]) >= footer_threshold:
            block_text = " ".join(str(block[4]).split())
            if not block_text:
                continue
            if re.search(r"[A-Za-z]", block_text):
                if not re.fullmatch(r"(?i)(?:page\s+)?\d{1,4}", block_text):
                    continue
            block_numbers = FOOTER_NUMBER_RE.findall(block_text)
            if not block_numbers:
                continue
            # Avoid TOC/list blocks with many embedded numbers.
            if len(block_numbers) > 2 or len(block_text) > 80:
                continue
            trailing_number = re.search(r"\b(\d{1,4})\s*$", block_text)
            if trailing_number:
                footer_candidate_numbers.append(int(trailing_number.group(1)))
            else:
                footer_candidate_numbers.append(int(block_numbers[-1]))
    if footer_candidate_numbers:
        return footer_candidate_numbers[-1]

    # 3) Header heuristic for books that print page numbers in running headers.
    try:
        header_blocks = [
            block for block in blocks
            if len(block) >= 5 and float(block[1]) <= float(pdf_page.rect.height) * 0.08
        ]
    except Exception:
        header_blocks = []
    for block in sorted(header_blocks, key=lambda item: (float(item[1]), float(item[0]))):
        block_text = str(block[4] or "")
        raw_lines = [line.strip() for line in block_text.splitlines() if line.strip()]
        if raw_lines:
            first_line_num = re.fullmatch(r"(\d{1,4})", raw_lines[0])
            if first_line_num:
                return int(first_line_num.group(1))
            last_line_num = re.fullmatch(r"(\d{1,4})", raw_lines[-1])
            if last_line_num:
                return int(last_line_num.group(1))

        for line in raw_lines:
            compact = " ".join(line.split())
            if not compact:
                continue
            leading_match = re.match(r"^\s*(\d{1,4})\s*[|¦]\s*[A-Za-z]", compact)
            if leading_match:
                return int(leading_match.group(1))
            trailing_match = re.search(r"[|¦]\s*(\d{1,4})\s*$", compact)
            if trailing_match:
                return int(trailing_match.group(1))

    if label.isdigit():
        return int(label)
    return physical_page

def _load_pdf_docs_with_logical_pages(path: Path) -> list[Document]:
    """Loads PDF pages and attaches logical_page metadata per page."""
    if fitz is None:
        return []

    docs = []
    detected_offset = None
    structured_label_map = _read_structured_pdf_page_labels(path)
    with closing(fitz.open(str(path))) as pdf_doc:
        for pdf_page in pdf_doc:
            physical_page = int(pdf_page.number) + 1
            structured_label = structured_label_map.get(physical_page)
            if _is_plausible_logical_page(structured_label):
                extracted_logical = int(structured_label)
                logical_page_source = "structured_pdf_label"
            else:
                extracted_raw = int(_extract_logical_page_number(pdf_page))
                extracted_logical = extracted_raw if _is_plausible_logical_page(extracted_raw) else physical_page
                logical_page_source = "heuristic"
            logical_page = extracted_logical

            if detected_offset is None and extracted_logical != physical_page:
                proposed_offset = physical_page - extracted_logical
                if 0 <= proposed_offset <= 150:
                    detected_offset = proposed_offset
                else:
                    logical_page = physical_page
            elif detected_offset is not None:
                predicted_logical = physical_page - detected_offset
                if not _is_plausible_logical_page(predicted_logical):
                    predicted_logical = physical_page
                if abs(extracted_logical - predicted_logical) > 2:
                    logical_page = predicted_logical
            if not _is_plausible_logical_page(logical_page):
                logical_page = physical_page

            docs.append(
                Document(
                    page_content=pdf_page.get_text("text"),
                    metadata={
                        "source": str(path),
                        "page": int(pdf_page.number),
                        "logical_page": logical_page,
                        "logical_page_offset": detected_offset,
                        "logical_page_source": logical_page_source,
                    },
                )
            )

    if detected_offset is not None:
        console.print(f"[dim][ingest] PDF logical page offset locked: {detected_offset}[/dim]")
        for doc in docs:
            doc.metadata["logical_page_offset"] = detected_offset

    return docs


def _load_docs_for_path(path: Path) -> tuple[list[Document], str | None]:
    """Loads a single source path and returns docs + optional warning message."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if not PDF_SUPPORT or fitz is None:
            return [], f"[yellow]Skipping unsupported file: {path.name}[/yellow]"
        try:
            return _load_pdf_docs_with_logical_pages(path), None
        except Exception as exc:
            return [], f"[yellow]Skipping unreadable PDF {path.name}: {exc}[/yellow]"

    if suffix in [".txt", ".md", ".py", ".js", ".html", ".css"]:
        try:
            return TextLoader(str(path), encoding="utf-8").load(), None
        except Exception as exc:
            return [], f"[yellow]Skipping unreadable text file {path.name}: {exc}[/yellow]"

    return [], f"[yellow]Skipping unsupported file: {path.name}[/yellow]"

def _load_and_split_docs(file_paths, status, uploader=None):
    """Loads and splits document content into chunks for retrieval."""
    all_docs = []
    source_paths = [Path(file_path) for file_path in file_paths]
    if not source_paths:
        return []

    max_workers = min(int(INGEST_MAX_WORKERS), max(1, len(source_paths)))
    if len(source_paths) == 1:
        path = source_paths[0]
        status.update(f"[bold cyan]Loading: {path.name}...[/bold cyan]")
        docs, warning = _load_docs_for_path(path)
        all_docs.extend(docs)
        if warning:
            console.print(warning)
    else:
        status.update(
            f"[bold cyan]Loading {len(source_paths)} documents in parallel (workers={max_workers})...[/bold cyan]"
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(_load_docs_for_path, path): path for path in source_paths}
            for future in as_completed(future_map):
                path = future_map[future]
                status.update(f"[bold cyan]Loaded: {path.name}[/bold cyan]")
                try:
                    docs, warning = future.result()
                except Exception as exc:
                    docs, warning = [], f"[yellow]Skipping unreadable file {path.name}: {exc}[/yellow]"
                all_docs.extend(docs)
                if warning:
                    console.print(warning)

    if not all_docs:
        return []

    if USE_SEMANTIC_CHUNKING and SEMANTIC_CHUNKING_AVAILABLE:
        status.update("[bold cyan]Applying semantic chunking... (This can be slow)[/bold cyan]")
        text_splitter = SemanticChunker(embeddings=get_embeddings())
    else:
        status.update("[bold cyan]Applying fast recursive chunking...[/bold cyan]")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    split_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        split_docs = pool.submit(text_splitter.split_documents, all_docs).result()
    split_elapsed_ms = (time.perf_counter() - split_start) * 1000.0
    console.print(f"[dim][perf] _load_and_split_docs split time: {split_elapsed_ms:.1f} ms[/dim]")

    if uploader and hasattr(uploader, "register_chunks"):
        uploader.register_chunks(split_docs)
    console.print(f"[green]OK Created {len(split_docs)} chunks from {len(file_paths)} document(s).[/green]")
    return split_docs

# --- LLM & Chain Initialization ---

def _initialize_llm():
    """Initializes the LLM based on global configuration."""
    ollama_options = {
        "temperature": 0.4,
        "top_p": 0.95,
        "num_predict": 1200,
        "repeat_penalty": 1.15,
    }
    if USE_API_LLM:
        if not GROQ_API_AVAILABLE or ChatGroq is None or not os.getenv("GROQ_API_KEY"):
            console.print("[bold red]Groq API key or library not found. LLM disabled.[/bold red]")
            return None
        console.print(f"[green]Using API Model: {API_MODEL_NAME}[/green]")
        return ChatGroq(
            model_name=API_MODEL_NAME,
            temperature=ollama_options["temperature"],
            top_p=ollama_options["top_p"],
            max_tokens=ollama_options["num_predict"],
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    else:
        if not OLLAMA_AVAILABLE:
            console.print("[bold red]'ollama' library not installed. Local LLM disabled.[/bold red]")
            return None
        console.print(f"[green]Using Local Model: {LOCAL_MODEL_NAME}[/green]")
        return OllamaLLM(
            model=LOCAL_MODEL_NAME,
            temperature=ollama_options["temperature"],
            top_p=ollama_options["top_p"],
            num_predict=ollama_options["num_predict"],
            repeat_penalty=ollama_options["repeat_penalty"],
        )

def _create_document_chain(llm):
    """Creates the LangChain chain for answering questions based on context."""
    if not llm: return None
    qa_prompt = ChatPromptTemplate.from_template(
        """You are a grounded AI assistant explaining concepts from document context.
Use only the provided context. Do not invent facts.
Answer ONLY using the provided context.
If the context contains relevant information, provide a detailed explanation.
Never say the answer is not available if any part of the question can be addressed using the context.
If the context is truly insufficient, ask for clarification on the specific topic.
Do not use bullet points, numbered lists, or section headers.
Include all relevant named components, processors, steps, or lists found in context.
Do not expand acronyms unless the expansion appears in context.

CONTEXT:
{context}

{input}
"""
    )
    return create_stuff_documents_chain(llm, qa_prompt)

# --- Retriever Creation ---

def _needs_chunk_registry_refresh(uploader, all_doc_paths: list[str]) -> bool:
    return bool(
        all_doc_paths
        and hasattr(uploader, "has_chunk_registry_for_paths")
        and not uploader.has_chunk_registry_for_paths(all_doc_paths, require_logical_pages=True)
    )


def check_dependencies(retrieval_mode: str, file_paths: list[str] | None = None) -> tuple[bool, list[str]]:
    """
    Strict dependency preflight for selected pipeline mode.
    Returns (ok, error_messages).
    """
    errors: list[str] = []
    mode = str(retrieval_mode or "hybrid").strip().lower()
    paths = file_paths or []

    if USE_SEMANTIC_CHUNKING and not SEMANTIC_CHUNKING_AVAILABLE:
        errors.append("Semantic chunking is enabled but 'langchain_experimental' is not installed.")

    has_pdf = any(str(path).lower().endswith(".pdf") for path in paths)
    if has_pdf and (fitz is None or not PDF_SUPPORT):
        errors.append("PDF input detected but PyMuPDF ('fitz') is not installed.")

    if mode == "splade":
        if torch is None:
            errors.append("SPLADE mode requires 'torch'.")
        if np is None:
            errors.append("SPLADE mode requires 'numpy'.")
        if sparse is None:
            errors.append("SPLADE mode requires 'scipy'.")
        if AutoTokenizer is None or AutoModelForMaskedLM is None:
            errors.append("SPLADE mode requires 'transformers'.")

    return len(errors) == 0, errors


def _load_or_create_bm25(split_docs: list[Document] | None, doc_signature: tuple[str, ...], status):
    """Creates or reuses an in-memory BM25 retriever keyed by doc signature."""
    global _BM25_RETRIEVER, _BM25_SIGNATURE

    if split_docs is not None:
        status.update("[bold cyan]Building BM25 index (keyword search)...[/bold cyan]")
        if not split_docs:
            return None
        _BM25_RETRIEVER = BM25Retriever.from_documents(split_docs)
        _BM25_SIGNATURE = doc_signature
        console.print("[green]OK BM25 retriever built.[/green]")
    elif _BM25_RETRIEVER is None or _BM25_SIGNATURE != doc_signature:
        return None

    bm25_retriever = _BM25_RETRIEVER
    bm25_retriever.k = BM25_RETRIEVER_K
    return bm25_retriever

def build_rag_pipeline(uploader, retrieval_mode: str = "hybrid"):
    """
    The main function to build the entire RAG pipeline.
    retrieval_mode: "hybrid" (BM25 + vector + reranker) or "splade" (sparse lexical).
    Returns the retriever and the document chain.
    """
    mode = str(retrieval_mode or "hybrid").strip().lower()
    if mode not in {"hybrid", "splade"}:
        console.print(f"[yellow]Unknown retrieval mode '{retrieval_mode}'. Falling back to hybrid.[/yellow]")
        mode = "hybrid"

    with console.status("[bold cyan]Initializing RAG pipeline...[/bold cyan]", spinner="dots") as status:
        # Step 1: Load paths and initialize embeddings.
        all_doc_paths = uploader.get_all_paths()
        logger.info(
            "build_rag_pipeline_start",
            retrieval_mode=mode,
            document_count=len(all_doc_paths),
        )
        if not all_doc_paths:
            console.print("[bold red]Error: No uploaded documents found. Please upload a document first.[/bold red]")
            logger.warning("build_rag_pipeline_no_documents", retrieval_mode=mode)
            return None, None
        deps_ok, dep_errors = check_dependencies(mode, all_doc_paths)
        if not deps_ok:
            for error in dep_errors:
                console.print(f"[bold red]Dependency Error:[/bold red] {error}")
            logger.error("build_rag_pipeline_dependency_error", retrieval_mode=mode, errors=dep_errors)
            return None, None
        if mode == "splade":
            unload_hybrid_components()
            doc_signature = _doc_signature(all_doc_paths)
            splade_batch_size = get_optimal_splade_batch_size()
            splade_cache_path = _splade_cache_path(doc_signature, batch_size=splade_batch_size)
            splade_retriever = None
            all_docs_split: list[Document] | None = None
            logger.info(
                "splade_batch_selected",
                batch_size=int(splade_batch_size),
                device=str(get_model_kwargs().get("device", "cpu")),
                vram_gb=round(_get_available_vram_gb(), 2),
            )

            if _splade_cache_meta_path(splade_cache_path).exists() and _splade_cache_matrix_path(splade_cache_path).exists():
                status.update("[bold cyan]Loading cached SPLADE index...[/bold cyan]")
                cache_load_start = time.perf_counter()
                splade_retriever = SpladeRetriever.load(splade_cache_path, k=SPLADE_RETRIEVER_K)
                cache_load_elapsed_ms = (time.perf_counter() - cache_load_start) * 1000.0
                if splade_retriever is not None:
                    console.print(
                        f"[green]OK Loaded SPLADE index from cache ({cache_load_elapsed_ms:.1f} ms).[/green]"
                    )
                    logger.info(
                        "splade_cache_loaded",
                        elapsed_ms=round(cache_load_elapsed_ms, 2),
                        cache_path=str(splade_cache_path),
                    )
                else:
                    console.print("[yellow]SPLADE cache unavailable/invalid. Rebuilding index.[/yellow]")
                    logger.warning("splade_cache_invalid", cache_path=str(splade_cache_path))

            needs_registry_refresh = _needs_chunk_registry_refresh(uploader, all_doc_paths)
            if splade_retriever is None or needs_registry_refresh:
                status.update("[bold cyan]Preparing chunks for SPLADE sparse index...[/bold cyan]")
                all_docs_split = _load_and_split_docs(all_doc_paths, status, uploader=uploader)
                if not all_docs_split:
                    console.print("[bold red]Error: Could not load any supported document content.[/bold red]")
                    return None, None

            if splade_retriever is None:
                status.update(
                    "[bold cyan]Building in-memory SPLADE index (CPU inference can be slow on first build)...[/bold cyan]"
                )
                console.print(f"[dim][perf] SPLADE batch size: {int(splade_batch_size)}[/dim]")
                splade_start = time.perf_counter()
                splade_retriever = SpladeRetriever.from_documents(
                    all_docs_split,
                    k=SPLADE_RETRIEVER_K,
                    batch_size=int(splade_batch_size),
                )
                splade_elapsed_ms = (time.perf_counter() - splade_start) * 1000.0
                if splade_retriever is None:
                    console.print("[bold red]Error: SPLADE retriever initialization failed.[/bold red]")
                    return None, None
                cache_saved = splade_retriever.save(splade_cache_path)
                if cache_saved:
                    console.print(f"[green]OK SPLADE index cached at {splade_cache_path}[/green]")
                    logger.info("splade_cache_saved", cache_path=str(splade_cache_path))
                chunks_built = len(all_docs_split or [])
                console.print(
                    f"[dim][perf] build_rag_pipeline SPLADE index: {splade_elapsed_ms:.1f} ms "
                    f"(chunks={chunks_built})[/dim]"
                )
                logger.info(
                    "splade_index_built",
                    elapsed_ms=round(splade_elapsed_ms, 2),
                    chunks=chunks_built,
                )

            bounded_retriever = TopicBoundedRetriever(
                base_retriever=splade_retriever,
                forward_window_pages=3,
            )
            llm = _initialize_llm()
            doc_chain = _create_document_chain(llm)
            console.print("[green]OK RAG pipeline is ready. (mode=splade)[/green]")
            logger.info("build_rag_pipeline_ready", retrieval_mode="splade")
            return bounded_retriever, doc_chain

        # Free SPLADE model memory when running non-SPLADE mode.
        unload_splade_components()

        embeddings = get_embeddings()

        # Step 2: Determine shared ingest work.
        unindexed_paths = uploader.get_unindexed_paths()
        db_path_obj = Path(DB_PATH)
        doc_signature = _doc_signature(all_doc_paths)
        needs_bm25_rebuild = (
            _BM25_RETRIEVER is None
            or _BM25_SIGNATURE != doc_signature
            or bool(unindexed_paths)
            or _needs_chunk_registry_refresh(uploader, all_doc_paths)
        )

        all_docs_split: list[Document] | None = None
        new_docs_split: list[Document] | None = None
        if needs_bm25_rebuild:
            status.update("[bold cyan]Preparing shared chunks for BM25/vector indexing...[/bold cyan]")
            all_docs_split = _load_and_split_docs(all_doc_paths, status, uploader=uploader)
            if not all_docs_split:
                console.print("[bold red]Error: Could not load any supported document content.[/bold red]")
                return None, None
            if unindexed_paths:
                unindexed_sources = {str(path) for path in unindexed_paths}
                new_docs_split = [
                    doc
                    for doc in all_docs_split
                    if str(getattr(doc, "metadata", {}).get("source", "")) in unindexed_sources
                ]
        elif unindexed_paths:
            status.update("[bold cyan]Processing new documents for vector store...[/bold cyan]")
            new_docs_split = _load_and_split_docs(unindexed_paths, status, uploader=uploader)

        # Step 3: Process new documents and update ChromaDB.
        if unindexed_paths:
            if new_docs_split:
                if db_path_obj.exists() and any(db_path_obj.iterdir()):
                    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
                    vector_store.add_documents(new_docs_split)
                else:
                    vector_store = Chroma.from_documents(new_docs_split, embeddings, persist_directory=DB_PATH)
                console.print("[green]OK New documents indexed in Chroma DB.[/green]")
            else:
                console.print("[yellow]No chunkable content found in new documents.[/yellow]")
            for path in unindexed_paths:
                uploader.mark_as_indexed(path)

        # Step 4: Ensure vector DB exists and create vector retriever.
        if not db_path_obj.exists() or not any(db_path_obj.iterdir()):
            if all_docs_split is None:
                status.update("[bold cyan]Rebuilding vector store from all documents...[/bold cyan]")
                all_docs_split = _load_and_split_docs(all_doc_paths, status, uploader=uploader)
            if not all_docs_split:
                console.print("[bold red]Error: No documents processed. Please upload a document first.[/bold red]")
                return None, None
            Chroma.from_documents(all_docs_split, embeddings, persist_directory=DB_PATH)
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": VECTOR_RETRIEVER_K})

        # Step 5: Create BM25 retriever.
        bm25_seed_docs = all_docs_split if needs_bm25_rebuild else None
        bm25_retriever = _load_or_create_bm25(bm25_seed_docs, doc_signature, status)
        if bm25_retriever is None:
            if all_docs_split is None:
                status.update("[bold cyan]Preparing chunks for BM25 index...[/bold cyan]")
                all_docs_split = _load_and_split_docs(all_doc_paths, status, uploader=uploader)
            bm25_retriever = _load_or_create_bm25(all_docs_split, doc_signature, status)
        if not bm25_retriever:
            return None, None

        # Step 6: Create asynchronous hybrid retriever.
        ensemble_retriever = AsyncHybridRetriever(
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
            weights=(0.5, 0.5),
            execution_mode="auto",
        )

        # Step 7: Create reranker.
        rerank_start = time.perf_counter()
        cross_encoder = get_reranker()
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=RERANK_TOP_N)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
        rerank_elapsed_ms = (time.perf_counter() - rerank_start) * 1000.0
        console.print(f"[dim][perf] build_rag_pipeline reranker init: {rerank_elapsed_ms:.1f} ms (top_n={RERANK_TOP_N})[/dim]")
        bounded_retriever = TopicBoundedRetriever(
            base_retriever=compression_retriever,
            forward_window_pages=3,
        )

        # Step 8: Initialize LLM and chain.
        llm = _initialize_llm()
        doc_chain = _create_document_chain(llm)

        console.print("[green]OK RAG pipeline is ready. (mode=hybrid)[/green]")
        logger.info("build_rag_pipeline_ready", retrieval_mode="hybrid")
        return bounded_retriever, doc_chain

