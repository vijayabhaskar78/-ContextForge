# /contextforge_cli/app.py
"""
Main application file for the ContextForge Document Q&A CLI.
Handles the Command-Line Interface (CLI), user interactions, and orchestrates
the document management and RAG pipeline modules.
"""
import sys
import os
import json
import hashlib
import re
import random
import time
import queue
import threading
import unicodedata
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any


# Rich UI Components
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

# Local module imports
from .config import (
    API_MODEL_NAME,
    BM25_CACHE_FILE,
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    LOCAL_MODEL_NAME,
    MEMORY_CONTEXT_RECENT_TURNS,
    MEMORY_RECALL_LOOKBACK_ASSISTANT_TURNS,
    OLLAMA_AVAILABLE,
    PDF_SUPPORT,
    SPLADE_BATCH_SIZE,
    SPLADE_GPU_BATCH_PER_GB,
    SPLADE_LOW_VRAM_BATCH_CAP,
    SPLADE_LOW_VRAM_THRESHOLD_GB,
    STREAM_IDLE_TIMEOUT_S as CONFIG_STREAM_IDLE_TIMEOUT_S,
    STREAM_POLL_INTERVAL_S as CONFIG_STREAM_POLL_INTERVAL_S,
    USE_API_LLM,
    USE_SEMANTIC_CHUNKING,
    console,
    detect_gpu_setup,
    get_model_kwargs,
)
# from .db_migrations import apply_migrations
from .document_manager import CHUNK_REGISTRY, DocumentUploader
from .memory_manager import HierarchicalMemoryManager, is_acknowledgment_query
from .observability import get_logger
from .rag_pipeline import (
    AsyncHybridRetriever,
    _get_available_vram_gb,
    _initialize_llm,
    build_rag_pipeline,
    check_dependencies,
    get_optimal_splade_batch_size,
    unload_hybrid_components,
    unload_splade_components,
)
from .tokenization import tokenize_for_matching

# LangChain components for context expansion
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHAPTER_QUERY_TERMS = (
    "chapter",
    "chapters",
    "table of contents",
    "contents",
    "toc",
)
CHAPTER_ENTRY_CACHE_MAXSIZE = 16
CHAPTER_ENTRY_CACHE: OrderedDict[tuple[str, ...], tuple[list[dict], list[Any]]] = OrderedDict()
CHAPTER_DISK_CACHE_VERSION = 1
COMMON_CHAPTER_TYPO_TOKENS = {
    "chaper", "chaptor", "chaptr", "chpatr", "chpater",
    "chaopter", "chptr", "chpter", "chpaters", "chaptors",
    "chaptrs", "chpatr", "chpters",
}
GROUNDING_FALLBACK_ANSWER = "The answer is not available in the provided document."
ACK_RESPONSES = ("Understood.", "No problem.", "No response needed.")
PRONOUN_REFERENCE_RE = re.compile(r"\b(this|that|it|them|these|those)\b", re.IGNORECASE)
MEMORY_RECALL_PATTERNS = (
    r"\bwhat did i ask\b",
    r"\bwhat did i say\b",
    r"\bwhat was my question\b",
    r"\bwhat is my previous question\b",
    r"\bwhat is my last question\b",
    r"\bprevious question\b",
    r"\bearlier question\b",
    r"\blast question\b",
    r"\bwhat was the previous question\b",
)
STREAM_POLL_INTERVAL_S = CONFIG_STREAM_POLL_INTERVAL_S
STREAM_IDLE_TIMEOUT_S = CONFIG_STREAM_IDLE_TIMEOUT_S
logger = get_logger(__name__)


# --- UI & Formatting Functions ---

def display_welcome_banner():
    """Displays the application's welcome banner."""
    chunking_type = "Semantic" if USE_SEMANTIC_CHUNKING else "Recursive"
    console.print(Panel(
        "[bold magenta]ContextForge - Document Q&A CLI[/bold magenta]",
        subtitle="[cyan]Powered by Hybrid Search & Reranking[/cyan]",
        expand=False
    ))
    console.print(f"[green]Chunking Mode: {chunking_type}[/green]")


def _get_display_page_number(doc) -> int | None:
    """Returns display page using logical page, offset mapping, then physical fallback."""
    metadata = getattr(doc, "metadata", {}) or {}
    logical_page = metadata.get("logical_page")
    try:
        if logical_page not in (None, ""):
            return int(logical_page)
    except (TypeError, ValueError):
        pass

    physical_page = metadata.get("page", -1)
    try:
        physical_page = int(physical_page)
    except (TypeError, ValueError):
        return None

    logical_offset = metadata.get("logical_page_offset")
    try:
        if logical_offset not in (None, ""):
            predicted = (physical_page + 1) - int(logical_offset)
            if predicted > 0:
                return predicted
    except (TypeError, ValueError):
        pass

    return physical_page + 1 if physical_page != -1 else None


def format_sources(docs):
    """Formats source documents for display, grouping page numbers."""
    if not docs:
        return "No sources found."
    
    sources = {}
    for doc in docs:
        if doc.metadata.get("is_memory"):
            continue
        source_name = Path(doc.metadata.get('source', 'Unknown')).name
        page = _get_display_page_number(doc)
        if source_name not in sources:
            sources[source_name] = set()
        if page is not None:
            sources[source_name].add(page)

    if not sources:
        return "No sources found."

    lines = []
    for name, pages in sorted(sources.items()):
        if pages:
            lines.append(f"- {name} (Pages: {', '.join(map(str, sorted(pages)))})")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def _collect_source_refs(docs) -> list[dict[str, Any]]:
    """Builds compact source references for memory metadata."""
    refs = []
    seen = set()
    for doc in docs or []:
        source_path = str(doc.metadata.get("source", "Unknown"))
        source = Path(source_path).name
        page = _get_display_page_number(doc)
        doc_page = doc.metadata.get("page")
        try:
            doc_page = int(doc_page)
        except (TypeError, ValueError):
            doc_page = None
        key = (source, page, doc_page)
        if key in seen:
            continue
        seen.add(key)
        refs.append({"source": source, "source_path": source_path, "page": page, "doc_page": doc_page})
        if len(refs) >= 20:
            break
    return refs


def _resolve_source_refs_to_docs(source_refs: list[dict[str, Any]], file_paths: list[str]) -> list:
    """Rehydrates document pages from compact source refs stored in memory metadata."""
    if not source_refs or not file_paths:
        return []

    path_lookup: dict[str, list[str]] = {}
    for path in file_paths:
        path_lookup.setdefault(Path(path).name, []).append(path)

    docs = []
    seen = set()
    for ref in source_refs:
        if not isinstance(ref, dict):
            continue
        source_name = Path(str(ref.get("source", ""))).name
        source_path_hint = str(ref.get("source_path", "")).strip()
        if not source_name:
            continue
        ref_page = ref.get("page")
        try:
            ref_page_int = int(ref_page) if ref_page not in (None, "") else None
        except (TypeError, ValueError):
            ref_page_int = None
        doc_page = ref.get("doc_page")
        try:
            doc_page = int(doc_page)
        except (TypeError, ValueError):
            doc_page = None
        if source_path_hint and source_path_hint in file_paths:
            candidate_paths = [source_path_hint]
        else:
            candidate_paths = path_lookup.get(source_name, [])
        if not candidate_paths:
            continue

        for source_path in candidate_paths:
            pages = _load_document_pages(source_path)
            if not pages:
                continue

            matches = []
            if doc_page is not None:
                matches = [doc for doc in pages if int(doc.metadata.get("page", -1)) == doc_page]
                if not matches and 0 <= doc_page < len(pages):
                    matches = [pages[doc_page]]
            elif ref_page in (None, ""):
                matches = pages[:1]
            else:
                matches = [doc for doc in pages if _get_display_page_number(doc) == ref_page]
                if not matches:
                    try:
                        fallback_idx = int(ref_page) - 1
                    except (TypeError, ValueError):
                        fallback_idx = -1
                    if 0 <= fallback_idx < len(pages):
                        matches = [pages[fallback_idx]]

            for doc in matches:
                if ref_page_int is not None:
                    doc.metadata["logical_page"] = ref_page_int
                    if doc_page is not None:
                        doc.metadata["logical_page_offset"] = (doc_page + 1) - ref_page_int
                key = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    " ".join(str(getattr(doc, "page_content", "")).split())[:120],
                )
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
                if len(docs) >= 20:
                    return docs
    return docs


def _recover_followup_docs_from_memory(memory_manager, session_id: str, uploader) -> list:
    """Recovers grounded docs for follow-up turns without re-running chapter resolution."""
    if not (memory_manager and session_id and uploader):
        return []

    state = memory_manager.get_state(session_id)
    source_refs = state.get("active_source_refs") if isinstance(state.get("active_source_refs"), list) else []
    if not source_refs:
        for turn in memory_manager.get_recent_assistant_turns(
            session_id,
            limit=MEMORY_RECALL_LOOKBACK_ASSISTANT_TURNS,
        ):
            metadata = turn.get("metadata") if isinstance(turn.get("metadata"), dict) else {}
            turn_refs = metadata.get("sources") if isinstance(metadata.get("sources"), list) else []
            if turn_refs:
                source_refs = turn_refs
                break

    if not source_refs:
        return []
    return _resolve_source_refs_to_docs(source_refs, uploader.get_all_paths())


def _build_grounded_question(resolved_query: str) -> str:
    """Builds grounded chain input with only the resolved user question."""
    effective_question = (resolved_query or "").strip()
    followup_style_note = ""
    lower_question = effective_question.lower()
    if re.search(r"\b(simplify|simple|in simple terms?)\b", lower_question):
        followup_style_note = (
            "\n\nAnswer style:\n"
            "- Use plain everyday language for a beginner.\n"
            "- Keep the response concise (about 80-120 words).\n"
            "- Avoid technical jargon when possible.\n"
            "- Include one short concrete example."
        )
    elif re.search(
        r"\b(what is|what are|who is|define|explain|summarize|summary|describe|elaborate|clarify|more details?)\b",
        lower_question,
    ):
        followup_style_note = (
            "\n\nAnswer style:\n"
            "- Provide a fuller explanation in 2-3 short paragraphs.\n"
            "- Include one concrete example from context when possible."
        )
    return f"Question:\n{effective_question}{followup_style_note}"


def _needs_fuller_answer(query_text: str) -> bool:
    q = (query_text or "").strip().lower()
    if not q:
        return False
    if re.search(r"\b(simplify|simple|in simple terms?)\b", q):
        return False
    return bool(
        re.search(
            r"\b(what is|what are|who is|define|explain|simplify|simple|summarize|summary|describe|clarify|elaborate)\b",
            q,
        )
    )


def _answer_is_too_brief(answer_text: str) -> bool:
    answer = str(answer_text or "").strip()
    if not answer:
        return True
    # Skip auto-expansion for code-like answers.
    if re.search(r"\b(class|def|function|import)\b", answer) and re.search(r"[{}():]", answer):
        return False
    word_count = len(re.findall(r"\b\w+\b", answer))
    return word_count < 85


def _normalize_recall_text(text: str) -> str:
    return " ".join(tokenize_for_matching(text, min_len=1))


def detect_memory_recall_query(query: str) -> bool:
    """Detects explicit user requests to recall prior conversation turns."""
    q = (query or "").strip().lower()
    if not q:
        return False
    if any(re.search(pattern, q) for pattern in MEMORY_RECALL_PATTERNS):
        return True
    if re.search(r"\bwhat did i ask before\b", q):
        return True
    if re.search(r"\bwhat did i say before\b", q):
        return True
    if re.search(r"\bbefore\s+(?:this|that|it)\b", q) and ("what did i" in q or "what was my question" in q):
        return True
    return False


def answer_from_conversation_history(query: str, history: list[dict[str, Any]]) -> str:
    """Answers explicit recall questions using chat history only (no document retrieval)."""
    no_history_msg = "I do not have previous conversation in this session."
    if not history:
        return no_history_msg

    normalized_query = _normalize_recall_text(query)
    user_questions = [
        str(turn.get("text", "")).strip()
        for turn in history
        if str(turn.get("role", "")).lower() == "user" and str(turn.get("text", "")).strip()
    ]
    if not user_questions:
        return no_history_msg

    # Exclude the current recall question itself.
    if _normalize_recall_text(user_questions[-1]) == normalized_query:
        user_questions = user_questions[:-1]
    if not user_questions:
        return no_history_msg

    # Ignore prior recall questions and keep true content-bearing user prompts.
    content_questions = [q for q in user_questions if not detect_memory_recall_query(q)]
    if not content_questions:
        return no_history_msg

    lowered_query = (query or "").lower()
    anchor_match = re.search(r"\bbefore\s+(.+?)(?:[?.!]\s*)?$", lowered_query)
    anchor_phrase = anchor_match.group(1).strip() if anchor_match else ""
    anchor_phrase = re.sub(r"^(?:the\s+)?(?:question|one)\s+", "", anchor_phrase).strip()
    if anchor_phrase in {"", "this", "that", "it", "then", "now", "previous", "earlier", "last"}:
        anchor_phrase = ""

    if anchor_phrase:
        anchor_norm = _normalize_recall_text(anchor_phrase)
        for idx in range(len(content_questions) - 1, -1, -1):
            q_norm = _normalize_recall_text(content_questions[idx])
            if anchor_norm and anchor_norm in q_norm:
                if idx - 1 >= 0:
                    return f"You asked: {content_questions[idx - 1]}"
                return no_history_msg

    # Default: immediate previous meaningful question.
    return f"You asked: {content_questions[-1]}"


def _is_pronoun_reference_query(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    return bool(PRONOUN_REFERENCE_RE.search(q))


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


def _filter_toc_like_docs(query: str, docs: list):
    """Drops TOC-like chunks for general QA to improve answerable context density."""
    if not docs:
        return docs
    if is_chapter_query(query) or is_chapter_list_query(query):
        return docs

    kept = [doc for doc in docs if not _is_toc_like_text(getattr(doc, "page_content", ""))]
    return kept or docs


def _sanitize_answer_with_context(answer_text: str, context_docs: list) -> str:
    """Removes acronym expansions not explicitly present in the context."""
    if not answer_text:
        return answer_text
    context_text = " ".join(
        " ".join(str(getattr(doc, "page_content", "")).split()) for doc in (context_docs or [])
    ).lower()
    if not context_text:
        return answer_text

    sanitized = str(answer_text)

    def replace_acronym_expansion(match):
        acronym = match.group(1)
        expansion = match.group(2).strip()
        if expansion and expansion.lower() not in context_text:
            return acronym
        return match.group(0)

    def replace_expansion_acronym(match):
        expansion = match.group(1).strip()
        acronym = match.group(2)
        if expansion and expansion.lower() not in context_text:
            return acronym
        return match.group(0)

    def replace_appositive_acronym(match):
        acronym = match.group(1)
        expansion = match.group(2).strip()
        if expansion and expansion.lower() not in context_text:
            return acronym
        return match.group(0)

    # Pattern: RAG (Relevance Assessment Graph)
    sanitized = re.sub(r"\b([A-Z]{2,})\s*\(([^)]+)\)", replace_acronym_expansion, sanitized)
    # Pattern: Relevance Assessment Graph (RAG)
    sanitized = re.sub(r"\b([A-Za-z][A-Za-z0-9 \-]{3,})\s*\(([A-Z]{2,})\)", replace_expansion_acronym, sanitized)
    # Pattern: RAG, a Relevance Assessment Graph
    sanitized = re.sub(r"\b([A-Z]{2,})\s*,\s*(?:an?|the)\s+([A-Za-z][A-Za-z0-9 \-]{3,})", replace_appositive_acronym, sanitized)

    # Preserve paragraph structure while normalizing noisy spacing.
    sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
    sanitized = re.sub(r" *\n *", "\n", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    sanitized = sanitized.strip()
    return sanitized

# --- Query Processing ---

def _levenshtein_distance(a: str, b: str) -> int:
    """Computes Levenshtein distance for short tokens."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost  # substitution
            ))
        prev = curr
    return prev[-1]


def _has_chapter_like_word(query: str) -> bool:
    """Detects 'chapter/chapters' with tolerance for common typos."""
    words = tokenize_for_matching(query, min_len=1)
    targets = ("chapter", "chapters")
    for word in words:
        if word in targets:
            return True
        if word in COMMON_CHAPTER_TYPO_TOKENS:
            return True
        # Keep fuzzy matching scoped to likely chapter-like tokens to avoid false positives.
        if not word.startswith("ch"):
            continue
        if len(word) >= 5 and min(_levenshtein_distance(word, t) for t in targets) <= 3:
            return True
    return False

def is_chapter_query(query: str) -> bool:
    """Returns True when a query is asking for chapter names/list."""
    lower_query = query.lower()
    if re.search(r"\bwithout\s+(?:any\s+)?ch[a-z]*t(?:e)?r", lower_query):
        return False
    if re.search(r"\btable\s+of\s+contents\b", lower_query):
        return True
    if re.search(r"\bcontents\b", lower_query):
        return True
    if re.search(r"\btoc\b", lower_query):
        return True
    # Handles common misspellings like "chaper".
    if bool(re.search(r"\bch[a]?[p]?[a-z]*ter[s]?\b", lower_query)):
        return True
    return _has_chapter_like_word(lower_query)


def is_chapter_list_query(query: str) -> bool:
    """Returns True when user asks for a chapter list/table of contents."""
    q = query.lower()
    chapter_list_markers = (
        "what chapters",
        "which chapters",
        "list all chapters",
        "list chapters",
        "list chapter names",
        "list chapter titles",
        "show chapters",
        "show me all chapters",
        "give me chapters",
        "give all chapter names",
        "chapters are there",
        "all chapters in this book",
        "chapter list",
        "chapter titles list",
        "display all chapter titles",
        "table of contents",
        "toc",
        "contents",
    )
    return any(marker in q for marker in chapter_list_markers)


def is_chapter_explanation_query(query: str) -> bool:
    """Returns True when user asks to explain/summarize a chapter."""
    q = query.lower()
    explain_markers = ("explain", "summarize", "summary", "describe", "details")
    if not is_chapter_query(query):
        return False
    if any(marker in q for marker in explain_markers):
        return True
    if re.search(r"\babout\s+(?:the\s+)?ch[a-z]*t(?:e)?r(?:s)?(?:\s+\d{1,2}(?:st|nd|rd|th)?)?", q):
        return True
    question_style_markers = (
        "what is in chapter",
        "what does chapter",
        "tell me chapter",
    )
    return any(marker in q for marker in question_style_markers)


def is_chapter_count_query(query: str) -> bool:
    """Returns True when user asks for chapter count only."""
    q = query.lower()
    count_markers = (
        "how many chapters",
        "chapter count",
        "count of chapters",
        "number of chapters",
        "total chapters",
        "chapter total",
        "just tell me the count",
        "tell me the count",
        "count chapters",
    )
    if any(marker in q for marker in count_markers):
        return True
    if _has_chapter_like_word(q) and ("how many" in q or "number of" in q):
        return True
    return "count" in q and is_chapter_query(q)


def is_chapter_lookup_query(query: str) -> bool:
    """Returns True when user asks which chapter matches a topic."""
    q = query.lower()
    lookup_markers = (
        "which chapter",
        "what chapter is",
        "what is the chapter",
        "name of chapter",
        "title of chapter",
        "chapter number",
        "find chapter",
        "which chapter covers",
        "which one is chapter",
    )
    if not is_chapter_query(query):
        return False
    if any(marker in q for marker in lookup_markers):
        return True

    has_name_intent = any(token in q for token in ("name", "title", "called"))
    has_question_intent = any(token in q for token in ("which", "what"))
    has_numbered_chapter_pattern = bool(re.search(r"\b([1-9]|[1-2][0-9]|3[0-9])(?:st|nd|rd|th)?\s+ch[a-z]*ter[s]?\b", q))
    if has_numbered_chapter_pattern and (has_name_intent or has_question_intent):
        return True
    return False


def classify_chapter_intent(query: str) -> str:
    """
    Classifies chapter-related intent.
    Returns one of: none, count, list, number_lookup, lookup, explain, chapter_other
    """
    if not is_chapter_query(query):
        return "none"
    if is_chapter_count_query(query):
        return "count"
    if is_chapter_list_query(query):
        return "list"
    if _extract_chapter_number_from_query(query) is not None and not is_chapter_explanation_query(query):
        return "number_lookup"
    if is_chapter_lookup_query(query):
        return "lookup"
    if is_chapter_explanation_query(query):
        return "explain"
    return "chapter_other"


def _extract_chapter_number_from_query(query: str) -> int | None:
    """Extracts chapter number from flexible query forms like 'chapter 20' or '20th chapter'."""
    q = query.lower()
    patterns = (
        r"\bchapter\s+([1-9]|[1-2][0-9]|3[0-9])(?:st|nd|rd|th)?\b",
        r"\b([1-9]|[1-2][0-9]|3[0-9])(?:st|nd|rd|th)?\s+ch[a-z]*ter[s]?\b",
        r"\b(?:name|title)\s+of\s+(?:the\s+)?([1-9]|[1-2][0-9]|3[0-9])(?:st|nd|rd|th)?\s+ch[a-z]*ter[s]?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))
    return None


def _extract_chapter_heading(line: str) -> str | None:
    """Extracts likely chapter headings from a raw line."""
    cleaned = " ".join(line.split()).strip(" -\t")
    if not cleaned or len(cleaned) < 5 or len(cleaned) > 140:
        return None

    # Remove trailing TOC leader dots and page numbers.
    cleaned = re.sub(r"\.{2,}\s*\d+\s*$", "", cleaned).strip()
    cleaned = re.sub(r"\s+\d+\s*$", "", cleaned).strip()

    lower_cleaned = cleaned.lower()
    if lower_cleaned in {"contents", "table of contents"}:
        return None
    if lower_cleaned.startswith(("appendix", "references", "index", "bibliography")):
        return None

    explicit_chapter = re.compile(
        r"^(chapter|ch\.)\s*[0-9]+(?:\s*[:.\-]\s*|\s+).+",
        re.IGNORECASE,
    )
    numbered_chapter = re.compile(
        r"^[0-9]{1,2}\s*[\.\):-]\s*[A-Za-z].+"
    )
    spaced_numbered_chapter = re.compile(
        r"^[0-9]{1,2}\s+[A-Za-z][A-Za-z0-9 ,:&'()/\-]+$"
    )
    if explicit_chapter.match(cleaned):
        return cleaned
    if numbered_chapter.match(cleaned):
        return cleaned
    if spaced_numbered_chapter.match(cleaned) and not re.match(r"^[0-9]{1,2}\.[0-9]", cleaned):
        return cleaned
    return None


def _chapter_cache_key(file_paths: list[str]) -> tuple[str, ...]:
    """Builds a stable cache key from file paths and mtimes."""
    key_parts = []
    for file_path in sorted(file_paths):
        path = Path(file_path)
        try:
            mtime = int(path.stat().st_mtime)
        except OSError:
            mtime = 0
        key_parts.append(f"{path}:{mtime}")
    return tuple(key_parts)


def _chapter_cache_get(cache_key: tuple[str, ...]):
    cached = CHAPTER_ENTRY_CACHE.get(cache_key)
    if cached is not None:
        CHAPTER_ENTRY_CACHE.move_to_end(cache_key)
    return cached


def _chapter_cache_set(cache_key: tuple[str, ...], value):
    CHAPTER_ENTRY_CACHE[cache_key] = value
    CHAPTER_ENTRY_CACHE.move_to_end(cache_key)
    while len(CHAPTER_ENTRY_CACHE) > CHAPTER_ENTRY_CACHE_MAXSIZE:
        CHAPTER_ENTRY_CACHE.popitem(last=False)


def _chapter_disk_cache_path(cache_key: tuple[str, ...]) -> Path:
    digest = hashlib.sha256("||".join(cache_key).encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"chapter_entries_{digest}.json"


def _chapter_disk_cache_get(cache_key: tuple[str, ...]):
    cache_path = _chapter_disk_cache_path(cache_key)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if int(payload.get("version", -1)) != CHAPTER_DISK_CACHE_VERSION:
        return None

    entries = payload.get("entries")
    evidence_payload = payload.get("evidence_docs", [])
    if not isinstance(entries, list):
        return None

    evidence_docs = []
    for item in evidence_payload:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        try:
            page = int(item.get("page", -1))
        except (TypeError, ValueError):
            page = -1
        try:
            logical_page = int(item["logical_page"]) if item.get("logical_page") not in (None, "") else None
        except (TypeError, ValueError):
            logical_page = None
        try:
            logical_page_offset = (
                int(item["logical_page_offset"])
                if item.get("logical_page_offset") not in (None, "")
                else None
            )
        except (TypeError, ValueError):
            logical_page_offset = None
        evidence_docs.append(
            Document(
                page_content=str(item.get("page_content", "")),
                metadata={
                    "source": source,
                    "page": page,
                    "logical_page": logical_page,
                    "logical_page_offset": logical_page_offset,
                },
            )
        )
    return entries, evidence_docs


def _chapter_disk_cache_set(cache_key: tuple[str, ...], entries: list[dict], evidence_docs: list[Any]):
    cache_path = _chapter_disk_cache_path(cache_key)
    serialized_docs = []
    for doc in evidence_docs or []:
        metadata = getattr(doc, "metadata", {}) or {}
        serialized_docs.append(
            {
                "source": str(metadata.get("source", "")),
                "page": metadata.get("page", -1),
                "logical_page": metadata.get("logical_page"),
                "logical_page_offset": metadata.get("logical_page_offset"),
                "page_content": str(getattr(doc, "page_content", "")),
            }
        )
    payload = {
        "version": CHAPTER_DISK_CACHE_VERSION,
        "entries": entries,
        "evidence_docs": serialized_docs,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(cache_path)


def _doc_cache_key(file_path: str) -> str:
    path = Path(file_path)
    try:
        mtime = int(path.stat().st_mtime)
    except OSError:
        mtime = 0
    return f"{path}:{mtime}"


def _is_heading_continuation(line: str) -> bool:
    """Detects wrapped continuation lines in a table-of-contents entry."""
    cleaned = " ".join(line.split()).strip(" -\t")
    if not cleaned or len(cleaned) > 70:
        return False
    if re.match(r"^(chapter|ch\.)\s*[0-9ivxlcdm]+", cleaned, re.IGNORECASE):
        return False
    if re.match(r"^[0-9]{1,2}\s*[\.\):-]", cleaned):
        return False
    if re.match(r"^[0-9]{1,2}\s+[A-Za-z]", cleaned):
        return False
    if re.match(r"^[ivxlcdm]{1,6}\s*[\.\):-]", cleaned, re.IGNORECASE):
        return False
    if cleaned.lower() in {"contents", "table of contents"}:
        return False
    if not re.match(r"^\(?[A-Z][A-Za-z0-9 ,:&'()/\-\)]+$", cleaned):
        return False

    letters = [ch for ch in cleaned if ch.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    # Continuation lines for wrapped chapter headings are typically all-caps/near all-caps.
    return uppercase_ratio >= 0.75


def _extract_chapter_entry(line: str):
    """Extracts a chapter entry and optional TOC page number from a merged line."""
    cleaned = " ".join(line.split()).strip(" -\t")
    if not cleaned:
        return None

    # Pattern with trailing book page number, e.g. "31.DEPLOYMENT 118"
    with_page = re.match(r"^([0-9]{1,2})\s*[\.\):-]?\s*(.+?)\s+([0-9]{1,3})$", cleaned)
    if with_page:
        chapter_number = int(with_page.group(1))
        chapter_title = with_page.group(2).strip(" -\t")
        toc_page = int(with_page.group(3))
        return {
            "chapter_number": chapter_number,
            "title": chapter_title,
            "heading": f"{chapter_number}. {chapter_title}",
            "toc_page": toc_page,
        }

    heading = _extract_chapter_heading(cleaned)
    if not heading:
        return None

    number_and_title = re.match(r"^([0-9]{1,2})\s*[\.\):-]?\s*(.+)$", heading)
    if not number_and_title:
        return None

    chapter_number = int(number_and_title.group(1))
    chapter_title = number_and_title.group(2).strip(" -\t")
    return {
        "chapter_number": chapter_number,
        "title": chapter_title,
        "heading": f"{chapter_number}. {chapter_title}",
        "toc_page": None,
    }


def extract_chapter_entries(file_paths: list[str]):
    """
    Extracts chapter entries from table-of-contents and heading lines.
    Returns (entries, evidence_docs).
    """
    if not file_paths:
        return [], []

    cache_key = _chapter_cache_key(file_paths)
    cached = _chapter_cache_get(cache_key)
    if cached is not None:
        return cached
    disk_cached = _chapter_disk_cache_get(cache_key)
    if disk_cached is not None:
        _chapter_cache_set(cache_key, disk_cached)
        return disk_cached

    entries = []
    evidence_docs = []
    seen = set()

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue

        if path.suffix.lower() == ".pdf" and PDF_SUPPORT:
            loader = PyMuPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        try:
            pages = loader.load()
        except Exception:
            continue

        if not pages:
            continue

        # TOC-heavy scan region first.
        max_scan_pages = min(len(pages), 120)
        candidate_indices = set()
        for idx, page_doc in enumerate(pages[:max_scan_pages]):
            text_lower = page_doc.page_content.lower()
            if "table of contents" in text_lower or re.search(r"\bcontents\b", text_lower):
                # TOCs often span many continuation pages without repeating "contents".
                candidate_indices.update(range(idx, min(idx + 28, len(pages))))

        if not candidate_indices:
            candidate_indices = set(range(min(len(pages), 60)))

        found_for_file = 0
        for idx in sorted(candidate_indices):
            page_doc = pages[idx]
            raw_lines = page_doc.page_content.splitlines()
            line_idx = 0
            while line_idx < len(raw_lines):
                initial = " ".join(raw_lines[line_idx].split()).strip(" -\t")
                if not initial:
                    line_idx += 1
                    continue
                if not _extract_chapter_heading(initial):
                    line_idx += 1
                    continue

                # Merge wrapped TOC headings split across lines.
                look_ahead = line_idx + 1
                merged_heading = initial
                while look_ahead < len(raw_lines):
                    next_line = " ".join(raw_lines[look_ahead].split()).strip(" -\t")
                    if not next_line:
                        look_ahead += 1
                        continue
                    if re.match(r"^[0-9]{1,3}$", next_line):
                        merged_heading = f"{merged_heading} {next_line}".strip()
                        look_ahead += 1
                        continue
                    if _extract_chapter_heading(next_line):
                        break
                    if _is_heading_continuation(next_line):
                        merged_heading = f"{merged_heading} {next_line}".strip()
                        look_ahead += 1
                        continue
                    break

                entry = _extract_chapter_entry(merged_heading)
                if not entry:
                    line_idx = look_ahead
                    continue

                heading_key = f"{path}:{entry['chapter_number']}"
                if heading_key in seen:
                    line_idx = look_ahead
                    continue
                seen.add(heading_key)
                entry["source"] = str(path)
                entry["toc_doc_page"] = page_doc.metadata.get("page", -1)
                entries.append(entry)
                evidence_docs.append(page_doc)
                found_for_file += 1
                line_idx = look_ahead

        # Fallback pass: explicit "Chapter X ..." lines from early pages.
        if found_for_file < 2:
            for page_doc in pages[:min(len(pages), 80)]:
                raw_lines = page_doc.page_content.splitlines()
                line_idx = 0
                while line_idx < len(raw_lines):
                    heading = _extract_chapter_heading(raw_lines[line_idx])
                    if not heading or not re.match(r"^(chapter|ch\.)", heading, re.IGNORECASE):
                        line_idx += 1
                        continue
                    fallback = _extract_chapter_entry(heading)
                    if not fallback:
                        line_idx += 1
                        continue
                    heading_key = f"{path}:{fallback['chapter_number']}"
                    if heading_key in seen:
                        line_idx += 1
                        continue
                    seen.add(heading_key)
                    fallback["source"] = str(path)
                    fallback["toc_doc_page"] = page_doc.metadata.get("page", -1)
                    entries.append(fallback)
                    evidence_docs.append(page_doc)
                    line_idx += 1

    entries.sort(key=lambda e: (e["source"], e["chapter_number"]))

    # Drop ambiguous TOC pages that repeat too often (OCR artifacts like "... 101").
    page_counts = {}
    for entry in entries:
        toc_page = entry.get("toc_page")
        if toc_page is None:
            continue
        page_counts[toc_page] = page_counts.get(toc_page, 0) + 1
    for entry in entries:
        toc_page = entry.get("toc_page")
        if toc_page is not None and page_counts.get(toc_page, 0) > 2:
            entry["toc_page"] = None

    result = (entries, evidence_docs)
    _chapter_cache_set(cache_key, result)
    try:
        _chapter_disk_cache_set(cache_key, entries, evidence_docs)
    except Exception:
        # Disk cache should never block chapter extraction correctness.
        pass
    return result


def extract_chapter_headings(file_paths: list[str]):
    """Backward-compatible wrapper returning only heading strings."""
    entries, evidence_docs = extract_chapter_entries(file_paths)
    return [entry["heading"] for entry in entries], evidence_docs


def warm_chapter_metadata_cache(file_paths: list[str]) -> tuple[int, float]:
    """Precomputes chapter metadata so regex scans are not paid in the query hot path."""
    if not file_paths:
        return 0, 0.0
    start = time.perf_counter()
    entries, _ = extract_chapter_entries(file_paths)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return len(entries), elapsed_ms


def _normalize_for_match(text: str) -> str:
    normalized = re.sub(r"[^\w]+", " ", str(text or "").casefold(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _to_safe_console_text(text: str) -> str:
    """Converts Unicode-rich text to a console-safe ASCII representation."""
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


@lru_cache(maxsize=8)
def _load_document_pages_cached(cache_key: str, source_file: str):
    """Loads source pages and caches them by source path + mtime."""
    _ = cache_key
    path = Path(source_file)
    if path.suffix.lower() == ".pdf" and PDF_SUPPORT:
        loader = PyMuPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    try:
        pages = loader.load()
    except Exception:
        pages = []
    return pages


def _load_document_pages(source_file: str):
    return _load_document_pages_cached(_doc_cache_key(source_file), source_file)


def _match_requested_chapter(query: str, entries: list[dict]):
    """Matches user query to a specific chapter entry."""
    if not entries:
        return None

    number = _extract_chapter_number_from_query(query)
    if number is None:
        chapter_number = re.search(r"\b([1-9]|[1-2][0-9]|3[0-9])(?:st|nd|rd|th)?\b", query.lower())
        number = int(chapter_number.group(1)) if chapter_number else None
    if number is not None:
        for entry in entries:
            if entry["chapter_number"] == number:
                return entry

    tokens = tokenize_for_matching(query, min_len=3)
    stopwords = {
        "the", "this", "that", "book", "chapter", "chaper", "chapters",
        "explain", "summary", "summarize", "about", "please", "can", "you",
    }
    query_tokens = [t for t in tokens if t not in stopwords]
    if not query_tokens:
        return None

    best_entry = None
    best_score = 0
    for entry in entries:
        title_tokens = set(tokenize_for_matching(entry["title"], min_len=1))
        score = sum(1 for token in query_tokens if token in title_tokens)
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry if best_score > 0 else None


def _get_chapter_context_docs(target_entry: dict, entries: list[dict]):
    """Builds focused context docs for a target chapter from its source document."""
    source_file = target_entry["source"]
    pages = CHUNK_REGISTRY.get_source_documents(source_file)
    if not pages:
        # Fallback for legacy/unregistered sources.
        pages = _load_document_pages(source_file)
    if not pages:
        return []

    source_entries = sorted(
        [e for e in entries if e["source"] == source_file],
        key=lambda e: e["chapter_number"]
    )
    toc_cutoff = max((e.get("toc_doc_page", -1) for e in source_entries), default=-1) + 1
    chapter_num = target_entry["chapter_number"]
    chapter_title_norm = _normalize_for_match(target_entry["title"])
    title_probe = " ".join(chapter_title_norm.split()[:4])

    start_idx = None
    for idx in range(max(0, toc_cutoff), len(pages)):
        norm_text = _normalize_for_match(pages[idx].page_content)
        if str(chapter_num) in norm_text and title_probe and title_probe in norm_text:
            start_idx = idx
            break

    if start_idx is None:
        for idx in range(max(0, toc_cutoff), len(pages)):
            norm_text = _normalize_for_match(pages[idx].page_content)
            if title_probe and title_probe in norm_text:
                start_idx = idx
                break

    if start_idx is None:
        return []

    next_entry = None
    for entry in source_entries:
        if entry["chapter_number"] > chapter_num:
            next_entry = entry
            break

    end_idx = min(len(pages) - 1, start_idx + 8)
    if next_entry:
        next_title_norm = _normalize_for_match(next_entry["title"])
        next_probe = " ".join(next_title_norm.split()[:4])
        for idx in range(start_idx + 1, min(len(pages), start_idx + 30)):
            norm_text = _normalize_for_match(pages[idx].page_content)
            if str(next_entry["chapter_number"]) in norm_text and next_probe and next_probe in norm_text:
                end_idx = max(start_idx, idx - 1)
                break

    return pages[start_idx:end_idx + 1]


def _build_extractive_chapter_answer(target_entry: dict, chapter_context: list, query: str) -> str:
    """Builds a deterministic, cleaned explanation from chapter text only."""
    raw_lines = []
    for doc in chapter_context:
        for raw_line in doc.page_content.splitlines():
            cleaned = _to_safe_console_text(raw_line).strip()
            if not cleaned:
                continue
            # Drop obvious page/header/footer noise.
            if re.fullmatch(r"\d{1,3}", cleaned):
                continue
            if re.search(r"principles of building ai agents", cleaned, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"[A-Z]{2,}\s+[A-Z]{2,}", cleaned):
                # Author/footer-like all-caps labels (e.g. "SAM BHAGWAT")
                continue
            cleaned = re.sub(r"https?://\S+", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                raw_lines.append(cleaned)

    if not raw_lines:
        header = f"Chapter {target_entry['chapter_number']}: {target_entry['title']}"
        if target_entry.get("toc_page") is not None:
            header += f" (Book page {target_entry['toc_page']})"
        return f"{header}\n\nI could not extract enough chapter text to explain this chapter."

    # Merge wrapped lines into sentence-like chunks.
    chunks = []
    for line in raw_lines:
        # Repair OCR artifacts.
        line = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", line)
        line = re.sub(r"\b([A-Za-z])\s+([a-z]{3,})\b", r"\1\2", line)
        line = re.sub(r"^(?:[A-Za-z]\s+)?\d{1,2}\s+[A-Z][A-Z\s\-\(\):]{3,}\s*", "", line)
        line = " ".join(line.split()).strip(" -:;,.")
        if not line:
            continue
        if chunks and len(chunks[-1]) < 220 and not re.search(r"[.!?]$", chunks[-1]):
            chunks[-1] = f"{chunks[-1]} {line}".strip()
        else:
            chunks.append(line)

    candidates = []
    seen = set()
    for chunk in chunks:
        if len(chunk) < 55:
            continue
        chunk = re.sub(r"\s+", " ", chunk).strip()
        key = chunk.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(chunk)

    if not candidates:
        header = f"Chapter {target_entry['chapter_number']}: {target_entry['title']}"
        if target_entry.get("toc_page") is not None:
            header += f" (Book page {target_entry['toc_page']})"
        return f"{header}\n\nI could not extract enough chapter text to explain this chapter."

    tokens = tokenize_for_matching(query, min_len=3)
    title_tokens = tokenize_for_matching(target_entry["title"], min_len=3)
    stopwords = {"chapter", "chaper", "book", "explain", "summary", "summarize", "about", "this", "that", "the", "need"}
    query_tokens = [t for t in tokens if t not in stopwords]
    score_tokens = set(query_tokens + title_tokens)

    scored = []
    for idx, candidate in enumerate(candidates):
        lower = candidate.lower()
        score = sum(1 for token in score_tokens if token in lower)
        # Prefer lines that read like claims/definitions.
        if any(marker in lower for marker in (" is ", " are ", " can ", " should ", " advantage", " downside", " limitation", " works")):
            score += 1
        scored.append((score, idx))

    ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
    chosen_indices = sorted(idx for _, idx in ranked[:5])
    selected = []
    for idx in chosen_indices:
        text = candidates[idx]
        if len(text) > 320:
            text = text[:320].rsplit(" ", 1)[0] + "..."
        selected.append(text)

    lines = [f"Chapter {target_entry['chapter_number']}: {target_entry['title']}"]
    if target_entry.get("toc_page") is not None:
        lines.append("")
        lines.append(f"Book page (from contents): {target_entry['toc_page']}")
    lines.append("")
    lines.append("Extracted explanation from chapter text:")
    for text in selected:
        lines.append(f"- {text}")
    return "\n".join(lines)


def _render_chapter_lookup(target: dict, chapter_sources: list):
    """Renders a lookup-style answer for a single matched chapter."""
    lookup_lines = [f"Best chapter match: **{target['heading']}**", ""]
    if target.get("toc_page") is not None:
        lookup_lines.append(f"Book page (from contents): **{target['toc_page']}**")
    console.print(Panel(Markdown("\n".join(lookup_lines)), title="Answer", border_style="blue"))
    source_docs = [
        d for d in chapter_sources
        if d.metadata.get("source") == target["source"]
        and d.metadata.get("page", -1) == target.get("toc_doc_page", -999)
    ]
    console.print(Panel(format_sources(source_docs or chapter_sources), title="Sources", border_style="yellow"))

def _expand_context_legacy(final_docs):
    """
    Legacy expansion path that re-loads source files and re-splits chunks.
    Kept for backward compatibility with older indexed chunks and integrity tests.
    """
    if not final_docs:
        return []

    source_files = {doc.metadata.get("source") for doc in final_docs}
    all_pages_in_context = {
        (doc.metadata.get("source"), doc.metadata.get("page")) for doc in final_docs
    }

    full_doc_chunks_map = {}
    for source_file in source_files:
        if not source_file:
            continue
        source_str = str(source_file).strip()
        source_path = Path(source_str)
        is_url = source_str.lower().startswith(("http://", "https://"))
        if not is_url and not source_path.exists():
            # Test doubles and stale metadata may reference non-local files.
            continue
        try:
            if source_path.suffix.lower() == ".pdf" and PDF_SUPPORT:
                loader = PyMuPDFLoader(source_str)
            else:
                loader = TextLoader(source_str, encoding="utf-8")
            full_doc = loader.load()
        except Exception:
            # Legacy expansion should never break answer generation.
            continue

        if full_doc:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            full_doc_chunks_map[source_file] = text_splitter.split_documents(full_doc)

    expanded_chunks = {doc.page_content: doc for doc in final_docs}
    for source, chunks in full_doc_chunks_map.items():
        for i, chunk in enumerate(chunks):
            if (source, chunk.metadata.get("page")) in all_pages_in_context:
                for offset in (-1, 0, 1):
                    neighbor_idx = i + offset
                    if 0 <= neighbor_idx < len(chunks):
                        neighbor_chunk = chunks[neighbor_idx]
                        if neighbor_chunk.page_content not in expanded_chunks:
                            expanded_chunks[neighbor_chunk.page_content] = neighbor_chunk

    return sorted(
        expanded_chunks.values(),
        key=lambda d: (d.metadata.get("source"), d.metadata.get("page", -1), d.metadata.get("chunk_index", -1)),
    )


def expand_context(final_docs):
    """
    Expands retrieved context using pre-indexed chunk neighbors from the chunk registry.
    Falls back to legacy behavior only for chunks missing registry metadata.
    """
    if not final_docs:
        return []

    max_expanded_chunks = max(1, len(final_docs) * 2)
    needs_legacy_fallback = False
    seed_topic_pages = {
        int(page)
        for page in (_get_display_page_number(doc) for doc in final_docs)
        if page is not None
    }
    seed_min_page = min(seed_topic_pages) if seed_topic_pages else None
    seed_max_page = max(seed_topic_pages) if seed_topic_pages else None
    seed_sources = {
        str(doc.metadata.get("source"))
        for doc in final_docs
        if doc.metadata.get("source")
    }

    expanded_docs = []
    seen_keys = set()

    def _doc_key(doc):
        metadata = getattr(doc, "metadata", {}) or {}
        preview = " ".join(str(getattr(doc, "page_content", "")).split())[:120]
        return (
            str(metadata.get("source", "")),
            metadata.get("page", -1),
            metadata.get("chunk_index", -1),
            preview,
        )

    def _is_topic_bounded(candidate_doc) -> bool:
        metadata = getattr(candidate_doc, "metadata", {}) or {}
        source = str(metadata.get("source", "")).strip()
        if seed_sources and source and source not in seed_sources:
            return False
        if not seed_topic_pages:
            return False
        candidate_page = _get_display_page_number(candidate_doc)
        if candidate_page is None:
            return False
        candidate_page = int(candidate_page)
        if seed_min_page is not None and candidate_page < seed_min_page:
            return False
        if seed_max_page is not None and candidate_page > (seed_max_page + 1):
            return False
        return any(abs(candidate_page - seed_page) <= 2 for seed_page in seed_topic_pages)

    def _add_doc(doc, *, force: bool = False):
        if len(expanded_docs) >= max_expanded_chunks:
            return
        key = _doc_key(doc)
        if key in seen_keys:
            return
        if not force and not _is_topic_bounded(doc):
            return
        seen_keys.add(key)
        expanded_docs.append(doc)

    # Keep original retrieval ordering (proxy for relevance/similarity rank).
    for seed in final_docs:
        _add_doc(seed, force=True)

    seed_records: list[tuple[Any, str, Any, Any]] = []
    for doc in final_docs:
        seed_logical_page = doc.metadata.get("logical_page")
        seed_logical_offset = doc.metadata.get("logical_page_offset")
        chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
        if not chunk_id:
            resolved_chunk_id = CHUNK_REGISTRY.resolve_chunk_id(
                source=doc.metadata.get("source", ""),
                page=doc.metadata.get("page", -1),
                content=getattr(doc, "page_content", ""),
            )
            if resolved_chunk_id:
                doc.metadata["chunk_id"] = resolved_chunk_id
                chunk_id = resolved_chunk_id
        seed_records.append((doc, chunk_id, seed_logical_page, seed_logical_offset))

    seed_chunk_ids = [chunk_id for _, chunk_id, _, _ in seed_records if chunk_id]
    seed_rows_by_chunk = CHUNK_REGISTRY.get_chunks_batch(seed_chunk_ids) if seed_chunk_ids else {}
    neighbor_ids_by_seed: dict[str, list[str]] = {}
    unique_neighbor_ids: list[str] = []
    seen_neighbor_ids = set()

    for chunk_id in seed_chunk_ids:
        row = seed_rows_by_chunk.get(chunk_id)
        if not row:
            continue
        neighbor_ids = []
        if row.get("prev_chunk_id"):
            neighbor_ids.append(str(row["prev_chunk_id"]))
        neighbor_ids.append(str(row["chunk_id"]))
        if row.get("next_chunk_id"):
            neighbor_ids.append(str(row["next_chunk_id"]))
        neighbor_ids_by_seed[chunk_id] = neighbor_ids
        for neighbor_id in neighbor_ids:
            if neighbor_id not in seen_neighbor_ids:
                seen_neighbor_ids.add(neighbor_id)
                unique_neighbor_ids.append(neighbor_id)

    if unique_neighbor_ids:
        # Prime cache so per-seed document conversion does not trigger N+1 DB round-trips.
        CHUNK_REGISTRY.get_chunks_batch(unique_neighbor_ids)

    for doc, chunk_id, seed_logical_page, seed_logical_offset in seed_records:
        if not chunk_id:
            if seed_logical_page not in (None, ""):
                doc.metadata["logical_page"] = seed_logical_page
            if seed_logical_offset not in (None, ""):
                doc.metadata["logical_page_offset"] = seed_logical_offset
            needs_legacy_fallback = True
            continue

        registry_row = seed_rows_by_chunk.get(chunk_id)
        if registry_row:
            if registry_row.get("logical_page") not in (None, ""):
                doc.metadata["logical_page"] = registry_row.get("logical_page")
            elif seed_logical_page not in (None, ""):
                doc.metadata["logical_page"] = seed_logical_page
            if registry_row.get("logical_page_offset") not in (None, ""):
                doc.metadata["logical_page_offset"] = registry_row.get("logical_page_offset")
            elif seed_logical_offset not in (None, ""):
                doc.metadata["logical_page_offset"] = seed_logical_offset
            doc.metadata["chunk_index"] = registry_row.get("chunk_index", doc.metadata.get("chunk_index"))
            doc.metadata["source_mtime"] = registry_row.get("source_mtime", doc.metadata.get("source_mtime"))
            doc.metadata["prev_chunk_id"] = registry_row.get("prev_chunk_id", doc.metadata.get("prev_chunk_id"))
            doc.metadata["next_chunk_id"] = registry_row.get("next_chunk_id", doc.metadata.get("next_chunk_id"))
        else:
            if seed_logical_page not in (None, ""):
                doc.metadata["logical_page"] = seed_logical_page
            if seed_logical_offset not in (None, ""):
                doc.metadata["logical_page_offset"] = seed_logical_offset
            needs_legacy_fallback = True
            continue

        neighbor_ids = neighbor_ids_by_seed.get(chunk_id, [])
        if not neighbor_ids:
            needs_legacy_fallback = True
            continue

        for neighbor_doc in CHUNK_REGISTRY.get_documents_batch(neighbor_ids):
            if neighbor_doc.metadata.get("logical_page_offset") in (None, "") and seed_logical_offset not in (None, ""):
                neighbor_doc.metadata["logical_page_offset"] = seed_logical_offset
            _add_doc(neighbor_doc)
            if len(expanded_docs) >= max_expanded_chunks:
                break
        if len(expanded_docs) >= max_expanded_chunks:
            break

    if needs_legacy_fallback and len(expanded_docs) < max_expanded_chunks:
        for legacy_doc in _expand_context_legacy(final_docs):
            _add_doc(legacy_doc)
            if len(expanded_docs) >= max_expanded_chunks:
                break

    return expanded_docs


def handle_user_query(query, retriever, doc_chain, uploader, memory_manager=None, session_id=None, grounded: bool = True):
    """Processes a user's query through the RAG pipeline and prints the result."""
    raw_query = query
    effective_query = query
    followup_resolved = False
    if memory_manager and session_id:
        effective_query = memory_manager.resolve_followup_query(session_id, query)
        followup_resolved = effective_query.strip() != query.strip()
    memory_recall_intent = detect_memory_recall_query(raw_query) or detect_memory_recall_query(effective_query)

    ack_query = is_acknowledgment_query(query) or is_acknowledgment_query(effective_query)
    if ack_query:
        ack_answer = random.choice(ACK_RESPONSES)
        console.print(Panel(Markdown(ack_answer), title="Answer", border_style="blue"))
        console.print("[Perf] Retrieval: 0.00s | Generation: 0.00s | Total: 0.00s", markup=False)
        if memory_manager and session_id:
            memory_manager.register_user_query(
                session_id=session_id,
                raw_query=query,
                resolved_query=effective_query if effective_query != query else None,
                intent="acknowledgment",
                topic=None,
            )
            memory_manager.register_assistant_answer(
                session_id=session_id,
                answer=ack_answer,
                intent="acknowledgment",
                topic=None,
                metadata={"sources": []},
            )
            memory_manager.update_state(
                session_id,
                {"last_intent": "acknowledgment", "response_mode": "acknowledgment"},
            )
        return

    if (
        memory_manager
        and session_id
        and _is_pronoun_reference_query(raw_query)
        and not followup_resolved
        and not memory_recall_intent
    ):
        state = memory_manager.get_state(session_id)
        active_topic = str(state.get("active_topic") or "").strip()
        last_topic = str(state.get("last_topic") or "").strip()
        if not active_topic and not last_topic:
            clarify_msg = "Could you clarify what you're referring to?"
            console.print(Panel(Markdown(clarify_msg), title="Answer", border_style="blue"))
            console.print("[Perf] Retrieval: 0.00s | Generation: 0.00s | Total: 0.00s", markup=False)
            memory_manager.register_user_query(
                session_id=session_id,
                raw_query=raw_query,
                resolved_query=None,
                intent="clarification_needed",
                topic=None,
            )
            memory_manager.register_assistant_answer(
                session_id=session_id,
                answer=clarify_msg,
                intent="clarification_needed",
                topic=None,
                metadata={"sources": []},
            )
            memory_manager.update_state(
                session_id,
                {"last_intent": "clarification_needed", "response_mode": "clarification"},
            )
            return

    chapter_intent = classify_chapter_intent(effective_query)
    followup_memory_mode = bool(memory_manager and session_id and followup_resolved)
    tracked_intent = "memory_recall" if memory_recall_intent else ("followup_memory" if followup_memory_mode else chapter_intent)
    topic_guess = memory_manager.derive_topic(effective_query) if memory_manager and session_id else ""
    if memory_manager and session_id:
        memory_manager.register_user_query(
            session_id=session_id,
            raw_query=query,
            resolved_query=effective_query,
            intent=tracked_intent,
            topic=topic_guess,
        )

    retrieval_elapsed_s = 0.0
    generation_elapsed_s = 0.0

    def _finalize_answer(
        answer_text: str,
        source_docs: list,
        intent_label: str,
        topic: str = "",
        state_updates: dict[str, Any] | None = None,
        render_answer: bool = True,
    ):
        safe_answer = _to_safe_console_text(str(answer_text))
        if render_answer:
            console.print(Panel(Markdown(safe_answer), title="Answer", border_style="blue"))
        console.print(Panel(format_sources(source_docs), title="Sources", border_style="yellow"))
        total_elapsed_s = retrieval_elapsed_s + generation_elapsed_s
        console.print(
            f"[Perf] Retrieval: {retrieval_elapsed_s:.2f}s | Generation: {generation_elapsed_s:.2f}s | Total: {total_elapsed_s:.2f}s",
            markup=False,
        )
        logger.info(
            "qa_turn_completed",
            intent=intent_label,
            retrieval_s=round(retrieval_elapsed_s, 4),
            generation_s=round(generation_elapsed_s, 4),
            total_s=round(total_elapsed_s, 4),
            sources=len(source_docs or []),
        )
        if not (memory_manager and session_id):
            return

        metadata = {"sources": _collect_source_refs(source_docs)}
        if effective_query != query:
            metadata["resolved_query"] = effective_query
        final_topic = topic or topic_guess
        memory_manager.register_assistant_answer(
            session_id=session_id,
            answer=safe_answer,
            intent=intent_label,
            topic=final_topic or None,
            metadata=metadata,
        )
        updates = {
            "last_intent": intent_label,
            "response_mode": "memory_recall" if intent_label == "memory_recall" else (
                "deterministic_chapter"
                if intent_label in {"chapter_count", "chapter_list", "chapter_lookup", "chapter_lookup_failed", "chapter_other"}
                else "rag_llm"
            ),
        }
        non_anchor_intents = {
            "chapter_count",
            "chapter_list",
            "chapter_other",
            "chapter_lookup_failed",
            "chapter_explain_failed",
            "memory_recall",
            "acknowledgment",
        }
        should_track_topic = (
            bool(final_topic)
            and intent_label not in non_anchor_intents
            and not is_acknowledgment_query(effective_query)
            and memory_manager.has_semantic_topic_signal(effective_query)
        )
        if should_track_topic:
            updates["last_topic"] = final_topic
            updates["active_topic"] = final_topic
        if metadata["sources"] and (
            should_track_topic
            or (state_updates and "active_chapter" in state_updates)
            or intent_label == "general_qa"
        ):
            updates["active_source_refs"] = metadata["sources"]
        if state_updates:
            updates.update(state_updates)
        memory_manager.update_state(session_id, updates)

    def _coerce_stream_chunk(chunk: Any) -> str:
        if chunk is None:
            return ""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            for key in ("answer", "output_text", "text", "content"):
                value = chunk.get(key)
                if value is not None:
                    return str(value)
        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif hasattr(item, "text"):
                    parts.append(str(getattr(item, "text")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(chunk)

    def _clear_loading_line(label: str):
        clear_len = max(16, len(label) + 4)
        console.file.write("\r" + (" " * clear_len) + "\r")
        console.file.flush()

    def _invoke_with_ascii_spinner(payload: dict[str, Any], label: str = "Loading") -> str:
        with console.status(f"[bold cyan]{label}...[/bold cyan]", spinner="dots"):
            return str(doc_chain.invoke(payload))

    def _invoke_chain_with_stream(payload: dict[str, Any]) -> tuple[str, bool]:
        stream_fn = getattr(doc_chain, "stream", None)
        if not callable(stream_fn):
            return _invoke_with_ascii_spinner(payload), False

        stream_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        stream_done = threading.Event()

        def _stream_worker():
            try:
                for chunk in stream_fn(payload):
                    stream_queue.put(("chunk", chunk))
            except Exception as exc:
                stream_queue.put(("error", exc))
            finally:
                stream_done.set()

        worker = threading.Thread(target=_stream_worker, daemon=True)
        worker.start()

        pieces: list[str] = []
        emitted = False
        spinner = "|/-\\"
        spinner_idx = 0
        loading_label = "Loading"
        last_event_at = time.perf_counter()
        idle_timeout_s = float(STREAM_IDLE_TIMEOUT_S or 0.0)
        stream_error = None
        try:
            while True:
                try:
                    event, chunk = stream_queue.get(timeout=STREAM_POLL_INTERVAL_S)
                    last_event_at = time.perf_counter()
                except queue.Empty:
                    if stream_done.is_set():
                        break
                    if idle_timeout_s > 0.0 and (time.perf_counter() - last_event_at) > idle_timeout_s:
                        raise TimeoutError("Streaming response stalled.")
                    if not emitted:
                        console.file.write(f"\r{loading_label} {spinner[spinner_idx % len(spinner)]}")
                        console.file.flush()
                        spinner_idx += 1
                    continue

                if event == "error":
                    stream_error = chunk
                    raise chunk
                if event != "chunk":
                    continue

                text = _coerce_stream_chunk(chunk)
                if not text:
                    continue
                safe_text = _to_safe_console_text(text)
                if not safe_text:
                    continue
                if not emitted:
                    _clear_loading_line(loading_label)
                pieces.append(safe_text)
                console.print(safe_text, end="", markup=False, highlight=False, soft_wrap=True)
                emitted = True
            if emitted:
                console.print()
                return "".join(pieces), True
        except Exception as exc:
            stream_error = stream_error or exc
            _clear_loading_line(loading_label)
        finally:
            worker.join(timeout=1.0)

        if stream_error:
            console.print(f"[yellow]Streaming unavailable ({stream_error}); falling back to non-stream invocation.[/yellow]")
            logger.warning("streaming_fallback", error=str(stream_error))
        return _invoke_with_ascii_spinner(payload), False

    if True:
        retrieved_docs = None
        answer_intent_label = "general_qa"
        answer_topic = topic_guess
        answer_state_updates = None
        skip_context_expansion = False
        generation_query = effective_query
        retrieval_query_override = None

        if memory_recall_intent:
            history = memory_manager.get_conversation_history(session_id) if (memory_manager and session_id) else []
            recall_answer = answer_from_conversation_history(raw_query, history)
            _finalize_answer(
                recall_answer,
                [],
                intent_label="memory_recall",
                topic="conversation recall",
            )
            return

        if followup_memory_mode:
            followup_docs = _recover_followup_docs_from_memory(memory_manager, session_id, uploader)
            if followup_docs:
                retrieved_docs = followup_docs
                skip_context_expansion = True
                if memory_manager and session_id:
                    state = memory_manager.get_state(session_id)
                    answer_topic = (
                        topic_guess
                        or str(state.get("active_topic") or "").strip()
                        or str(state.get("last_topic") or "").strip()
                    )
            else:
                console.print("[yellow]Could not recover follow-up context from memory. Falling back to standard retrieval.[/yellow]")

        # Chapter intent path is handled before retrieval to avoid LLM hallucinations.
        if chapter_intent != "none" and not followup_memory_mode:
            chapter_entries, chapter_sources = extract_chapter_entries(uploader.get_all_paths())
            if chapter_entries:
                requested_chapter_number = _extract_chapter_number_from_query(effective_query)

                if chapter_intent == "count":
                    answer_lines = [
                        f"Total chapters found: **{len(chapter_entries)}**",
                        "",
                        f"Range: Chapter {chapter_entries[0]['chapter_number']} to Chapter {chapter_entries[-1]['chapter_number']}",
                    ]
                    _finalize_answer(
                        "\n".join(answer_lines),
                        chapter_sources,
                        intent_label="chapter_count",
                        topic="chapter count",
                    )
                    return

                if chapter_intent == "list":
                    lines = ["Exact chapter headings found in the document context:\n"]
                    for entry in chapter_entries:
                        if entry.get("toc_page") is not None:
                            lines.append(f"- {entry['heading']} (Book page {entry['toc_page']})")
                        else:
                            lines.append(f"- {entry['heading']}")
                    _finalize_answer(
                        "\n".join(lines),
                        chapter_sources,
                        intent_label="chapter_list",
                        topic="chapter list",
                    )
                    return

                if chapter_intent == "number_lookup" and requested_chapter_number is not None:
                    target = next(
                        (entry for entry in chapter_entries if entry["chapter_number"] == requested_chapter_number),
                        None,
                    )
                    if target:
                        chapter_context = _get_chapter_context_docs(target, chapter_entries)
                        if chapter_context:
                            # Keep chapter grounding concise for CPU-bound local inference.
                            retrieved_docs = chapter_context[:2]
                            skip_context_expansion = True
                        else:
                            retrieval_query_override = (
                                f"chapter {target['chapter_number']} {target['title']} overview summary"
                            )
                        generation_query = (
                            f"Explain what chapter {target['chapter_number']} ({target['title']}) is about in simple terms."
                        )
                        answer_intent_label = "chapter_explain"
                        answer_topic = f"chapter {target['chapter_number']} {target['title']}"
                        answer_state_updates = {
                            "active_chapter": target["chapter_number"],
                            "active_chapter_title": target["title"],
                        }
                    else:
                        _finalize_answer(
                            f"I could not find chapter **{requested_chapter_number}** in the detected table of contents.",
                            chapter_sources,
                            intent_label="chapter_lookup_failed",
                            topic="chapter lookup",
                        )
                        return

                if chapter_intent == "lookup":
                    target = _match_requested_chapter(effective_query, chapter_entries)
                    if target:
                        chapter_context = _get_chapter_context_docs(target, chapter_entries)
                        if chapter_context:
                            # Keep chapter grounding concise for CPU-bound local inference.
                            retrieved_docs = chapter_context[:2]
                            skip_context_expansion = True
                        else:
                            retrieval_query_override = (
                                f"chapter {target['chapter_number']} {target['title']} overview summary"
                            )
                        generation_query = (
                            f"Explain what chapter {target['chapter_number']} ({target['title']}) covers in simple terms."
                        )
                        answer_intent_label = "chapter_explain"
                        answer_topic = f"chapter {target['chapter_number']} {target['title']}"
                        answer_state_updates = {
                            "active_chapter": target["chapter_number"],
                            "active_chapter_title": target["title"],
                        }
                        # Continue to the shared RAG generation stage.
                    else:
                        _finalize_answer(
                            "Could not match a specific chapter for lookup. Try adding a clearer topic keyword.",
                            chapter_sources,
                            intent_label="chapter_lookup_failed",
                            topic="chapter lookup",
                        )
                        return

                if chapter_intent == "explain":
                    target = _match_requested_chapter(effective_query, chapter_entries)
                    if target:
                        chapter_context = _get_chapter_context_docs(target, chapter_entries)
                        if chapter_context:
                            # Pass only the first 2 pages of the chapter to prevent CPU LLM freeze.
                            retrieved_docs = chapter_context[:2]
                            answer_intent_label = "chapter_explain"
                            answer_topic = f"chapter {target['chapter_number']} {target['title']}"
                            answer_state_updates = {
                                "active_chapter": target["chapter_number"],
                                "active_chapter_title": target["title"],
                            }
                            skip_context_expansion = True
                            generation_query = effective_query
                        else:
                            console.print("[yellow]Could not isolate chapter pages reliably. Falling back to general QA path.[/yellow]")
                    else:
                        _finalize_answer(
                            "Could not match a specific chapter in your query. Ask with chapter number/title.",
                            chapter_sources,
                            intent_label="chapter_explain_failed",
                            topic="chapter explain",
                        )
                        return

                if chapter_intent == "chapter_other":
                    _finalize_answer(
                        "I detected a chapter-related question but could not classify it cleanly. "
                        "Try one of: chapter count, chapter list, chapter lookup, or explain chapter <number>.",
                        chapter_sources,
                        intent_label="chapter_other",
                        topic="chapters",
                    )
                    return
            else:
                console.print("[yellow]No explicit chapter headings found. Falling back to general QA path.[/yellow]")

        if retrieved_docs is None:
            retrieval_query = retrieval_query_override or effective_query
            if memory_manager and session_id:
                retrieval_query = memory_manager.enhance_retrieval_query(
                    session_id,
                    retrieval_query,
                    query_is_resolved=followup_resolved,
                    strict=followup_resolved,
                )

            retrieval_start = time.perf_counter()
            retrieved_docs = retriever.invoke(retrieval_query)
            retrieval_elapsed_s += time.perf_counter() - retrieval_start
            retrieved_docs = _filter_toc_like_docs(effective_query, retrieved_docs)

        if not retrieved_docs:
            console.print(Panel("[yellow]No relevant context found.[/yellow]", title="Warning"))
            if memory_manager and session_id:
                updates = {"last_intent": "no_context"}
                if topic_guess and memory_manager.has_semantic_topic_signal(effective_query):
                    updates["last_topic"] = topic_guess
                    updates["active_topic"] = topic_guess
                memory_manager.update_state(session_id, updates)
            return

        if skip_context_expansion:
            final_context = retrieved_docs
        else:
            final_context = expand_context(retrieved_docs)

        if doc_chain:
            simplify_request = bool(re.search(r"\b(simplify|simple|in simple terms?)\b", generation_query.lower()))
            if grounded:
                model_input = _build_grounded_question(generation_query)
                if answer_intent_label == "general_qa" and simplify_request:
                    model_input = (
                        f"{model_input}\n\n"
                        "Instructions:\n"
                        "- Rewrite in very simple everyday language.\n"
                        "- Keep it between 70 and 110 words.\n"
                        "- Use one short example.\n"
                        "- Avoid technical terms unless absolutely necessary.\n"
                        "- Do not use terms like: context window, semantic recall, topK, memory processors.\n"
                    )
            else:
                model_input = generation_query
                if memory_manager and session_id:
                    model_input = memory_manager.compose_model_input(session_id, generation_query)
            generation_start = time.perf_counter()
            answer, streamed = _invoke_chain_with_stream({"context": final_context, "input": model_input})
            generation_elapsed_s += time.perf_counter() - generation_start
            answer = _sanitize_answer_with_context(str(answer), final_context)
            if (
                grounded
                and answer_intent_label == "general_qa"
                and _needs_fuller_answer(generation_query)
                and _answer_is_too_brief(answer)
            ):
                refinement_input = (
                    f"{_build_grounded_question(generation_query)}\n\n"
                    "Refinement:\n"
                    "- Expand the answer depth while staying grounded.\n"
                    "- Write 2-3 compact paragraphs.\n"
                    "- Include one practical example from context."
                )
                refinement_start = time.perf_counter()
                answer = doc_chain.invoke({"context": final_context, "input": refinement_input})
                generation_elapsed_s += time.perf_counter() - refinement_start
                answer = _sanitize_answer_with_context(str(answer), final_context)
                streamed = False
            _finalize_answer(
                str(answer),
                final_context,
                intent_label=answer_intent_label,
                topic=answer_topic,
                state_updates=answer_state_updates,
                render_answer=not streamed,
            )
        else:
            console.print(Panel("[yellow]LLM not available. Cannot generate an answer.[/yellow]", title="Warning"))
            if memory_manager and session_id:
                updates = {"last_intent": "llm_unavailable"}
                if answer_state_updates:
                    updates.update(answer_state_updates)
                if answer_topic and memory_manager.has_semantic_topic_signal(effective_query):
                    updates["last_topic"] = answer_topic
                    updates["active_topic"] = answer_topic
                memory_manager.update_state(session_id, updates)


# --- Main Application Flow ---

def _resolve_upload_path(raw_input: str) -> tuple[Path | None, str | None]:
    """Normalizes and validates user-provided upload path."""
    cleaned = str(raw_input or "").strip().strip('"').strip("'")
    if not cleaned:
        return None, "Error: Empty path provided."
    try:
        resolved = Path(cleaned).expanduser().resolve(strict=True)
    except FileNotFoundError:
        return None, f"Error: File not found at '{cleaned}'"
    except OSError as exc:
        return None, f"Error: Invalid path '{cleaned}' ({exc})"

    if os.name == "nt":
        is_reserved_fn = getattr(os.path, "isreserved", None)
        if callable(is_reserved_fn) and is_reserved_fn(str(resolved)):
            return None, f"Error: Reserved path is not allowed: '{resolved}'"
    if not resolved.is_file():
        return None, f"Error: Path is not a regular file: '{resolved}'"
    return resolved, None


def handle_document_upload(uploader):
    """CLI flow for uploading a new document."""
    file_path_str = Prompt.ask("Enter the full path to your document")
    file_path, error_message = _resolve_upload_path(file_path_str)
    if file_path is None:
        console.print(f"[bold red]{error_message}[/bold red]")
        return
    title = Prompt.ask("Enter a custom title (optional)", default=file_path.name)
    upload_ok = uploader.upload_document(str(file_path), title)
    if not upload_ok:
        return
    with console.status("[bold cyan]Precomputing chapter metadata cache...[/bold cyan]", spinner="dots"):
        chapter_count, elapsed_ms = warm_chapter_metadata_cache(uploader.get_all_paths())
    console.print(f"[dim][perf] chapter metadata cache warmup: {elapsed_ms:.1f} ms (entries={chapter_count})[/dim]")

def handle_qa_session(uploader, memory_manager):
    """Initiates the RAG pipeline and enters the Q&A loop."""
    from .query_processor import QueryProcessor

    console.print("\n[bold]Select Retrieval Engine:[/bold]")
    console.print("1. Standard Hybrid (BM25 + Vector + Rerank) [High Accuracy, Slower]")
    console.print("2. Experimental SPLADE (Sparse Neural Search) [Fast, High Accuracy]")
    engine_choice = Prompt.ask("Choose an engine", choices=["1", "2"], default="1")
    retrieval_mode = "hybrid" if engine_choice == "1" else "splade"

    retriever, doc_chain = build_rag_pipeline(uploader, retrieval_mode=retrieval_mode)
    if not retriever:
        console.print("[bold red]Failed to initialize RAG pipeline. Please ensure documents are uploaded.[/bold red]")
        return

    with console.status("[bold cyan]Loading chapter metadata cache...[/bold cyan]", spinner="dots"):
        chapter_count, elapsed_ms = warm_chapter_metadata_cache(uploader.get_all_paths())
    console.print(f"[dim][perf] chapter metadata cache load: {elapsed_ms:.1f} ms (entries={chapter_count})[/dim]")

    session_id = memory_manager.start_session(uploader.get_all_paths())
    query_processor = QueryProcessor(
        retriever=retriever,
        doc_chain=doc_chain,
        uploader=uploader,
        memory_manager=memory_manager,
        session_id=session_id,
        grounded=True,
        query_handler=handle_user_query,
    )
    console.print("\n[bold green]Q&A Session Started.[/bold green] [italic]Type 'back' to return to menu.[/italic]")
    console.print(f"[green]Memory session: {session_id}[/green]")
    console.print(f"[green]Retrieval engine: {retrieval_mode}[/green]")
    while True:
        query = Prompt.ask("[bold cyan]Ask a question (or type 'back' to go back to the menu)[/bold cyan]")
        if query.lower() == 'back':
            break
        if query.strip():
            query_processor.process(query)

def main():
    """Main application loop."""
    display_welcome_banner()
    uploader = DocumentUploader()
    memory_manager = HierarchicalMemoryManager()

    try:
        while True:
            try:
                console.print("\n[bold]Main Menu:[/bold]")
                console.print("[green]1. Upload Document(s)[/green]")
                console.print("[cyan]2. List Uploaded Documents[/cyan]")
                console.print("[blue]3. Start Q&A Session[/blue]")
                console.print("[red]4. Exit[/red]")

                choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"])

                if choice == "1":
                    handle_document_upload(uploader)
                elif choice == "2":
                    uploader.list_documents()
                elif choice == "3":
                    handle_qa_session(uploader, memory_manager)
                elif choice == "4":
                    break
            except KeyboardInterrupt:
                break
    finally:
        memory_manager.close()
        if hasattr(CHUNK_REGISTRY, "close"):
            CHUNK_REGISTRY.close()

    console.print("\n[bold magenta]Goodbye! Hope you had a productive session.[/bold magenta]")
    sys.exit(0)

if __name__ == "__main__":
    main()
