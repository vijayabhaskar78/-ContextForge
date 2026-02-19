# ContextForge System Documentation

## 1) What This System Is

ContextForge is a CLI-based Retrieval-Augmented Generation (RAG) document Q&A system.

It lets a user:
- upload local documents (PDF/text/code files),
- index them into retrieval systems,
- ask questions in an interactive session,
- get grounded answers with source pages,
- keep conversational continuity using a hierarchical memory manager.

The current runnable entrypoint is:
- `python final/app.py`

---

## 2) What We Built

The system now includes:
- hybrid retrieval (BM25 + vector retrieval),
- cross-encoder reranking,
- chapter-aware deterministic routing (count/list/lookup/explain),
- hierarchical conversational memory with SQLite persistence,
- follow-up query resolution (pronouns and low-signal follow-ups),
- acknowledgment handling (no retrieval for `ok`, `thanks`, etc.),
- grounded answer prompting,
- post-generation acronym-expansion sanitizer to reduce unsupported expansions,
- source-page reporting for traceability,
- regression tests for chapter intents and memory behavior.

---

## 3) High-Level Architecture

Main modules:
- `final/app.py`: CLI, orchestration, query routing, chapter logic, generation flow.
- `final/rag_pipeline.py`: chunking, retriever construction, reranking, LLM chain creation.
- `final/memory_manager.py`: SQLite-backed hierarchical memory and follow-up resolution.
- `final/document_manager.py`: upload/list metadata and indexing status management.
- `final/config.py`: model names, toggles, paths, hardware detection.

Persistent stores:
- `workspace_documents/uploaded_docs.json`: uploaded document metadata.
- `workspace_documents/doc_*`: copied source documents.
- `vector_store_db/`: vector DB persistence.
- `runtime_cache/bm25_retriever.pkl`: BM25 cache.
- `runtime_cache/conversation_memory.sqlite`: session memory DB.

---

## 4) Tech Stack and Models

Core stack:
- Python + Rich CLI
- LangChain
- Chroma vector DB
- HuggingFace embeddings and reranker
- Ollama local LLM or Groq API (toggle-based)

Current default model config (`final/config.py`):
- local LLM: `granite3.3:2b`
- embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Generation options (`final/rag_pipeline.py`):
- `temperature=0.4`
- `top_p=0.95`
- `num_predict=1200`
- `repeat_penalty=1.15`

---

## 5) End-to-End Workflows

## 5.1 Startup Workflow

1. `final/app.py` boots.
2. `final/config.py` detects GPU/CPU and prints capability.
3. Main menu starts:
- upload documents,
- list documents,
- start Q&A session.

## 5.2 Document Upload Workflow

Handled by `DocumentUploader` in `final/document_manager.py`:

1. User provides file path and optional title.
2. File is copied into `workspace_documents/` with generated `doc_<timestamp>_<stem>` ID.
3. Metadata is stored in `uploaded_docs.json` with `is_indexed=False`.

## 5.3 RAG Pipeline Build Workflow

Handled by `build_rag_pipeline()` in `final/rag_pipeline.py`:

1. Initialize embeddings model.
2. If unindexed docs exist:
- load and split docs (`RecursiveCharacterTextSplitter` by default),
- insert/add chunks into Chroma,
- mark docs as indexed.
3. Load vector retriever from Chroma (`k=15`).
4. Build/load BM25 retriever cache (`k=15`).
5. Combine with `EnsembleRetriever` (weights `[0.5, 0.5]`).
6. Rerank with cross-encoder (`top_n=5`) via `ContextualCompressionRetriever`.
7. Initialize LLM + document chain prompt.
8. Return `(retriever, doc_chain)`.

## 5.4 Query Handling Workflow

Core orchestrator: `handle_user_query()` in `final/app.py`.

Execution order:

1. Resolve follow-up query from memory state:
- `effective_query = memory_manager.resolve_followup_query(session_id, query)`

2. Handle acknowledgments early:
- detects `ok`, `thanks`, `got it`, etc.
- returns short acknowledgment response,
- does not run retrieval/generation.

3. If pronoun-only follow-up cannot be resolved and no active/last topic:
- returns clarification prompt:
  `Could you clarify what you're referring to?`

4. Classify chapter intent:
- `none`, `count`, `list`, `number_lookup`, `lookup`, `explain`, `chapter_other`.

5. If chapter intent detected:
- run deterministic chapter pipeline (no LLM hallucination path).

6. Else run general RAG pipeline:
- retrieval query enhancement from memory manager,
- retrieve hybrid+rereanked docs,
- expand context with neighboring chunks,
- generate grounded answer with `doc_chain`,
- sanitize unsupported acronym expansions,
- print answer + sources,
- update memory/session state.

---

## 6) Chapter Workflow (Deterministic Path)

Implemented in `final/app.py`:
- chapter extraction functions parse TOC-like pages and explicit headings,
- handles noisy OCR and wrapped headings,
- caches parsed chapter entries by file mtime.

Supported chapter intents:
- `count`: returns exact chapter count and range.
- `list`: returns detected chapter headings (and TOC book pages when available).
- `number_lookup`: maps chapter number to title.
- `lookup`: maps topic/title terms to best chapter.
- `explain`: pulls chapter page range and builds deterministic extractive explanation.

This path is intentionally pre-retrieval and mostly non-generative to avoid chapter hallucinations.

---

## 7) Conversational Memory System

Implemented in `final/memory_manager.py`.

Memory layers:
- short-term: recent turns,
- episodic: token-overlap-based retrieval of prior turns,
- semantic/profile: extracted stable user facts,
- session state: active topic/chapter and intent tracking.

SQLite schema:
- `sessions`: state JSON, rolling summary, doc signature.
- `turns`: user/assistant turns with intent/topic/metadata.
- `profile_facts`: key-value user facts with confidence.

Key session state fields commonly used:
- `active_topic`
- `last_topic`
- `active_chapter`
- `active_chapter_title`
- `last_intent`
- `response_mode`

---

## 8) Follow-Up Resolution and Query Integrity

Follow-up resolution function:
- `resolve_followup_query(user_query, session_state)`

What it does:
- detects ambiguous follow-ups (`this`, `that`, `explain more`, `tell me more`),
- rewrites query using `active_topic`/`last_topic` or active chapter anchor,
- preserves explicit queries and acknowledgment turns unchanged.

Examples:
- `Explain this in simpler terms` + `active_topic=hierarchical memory`
  -> `Explain hierarchical memory in simpler terms`

Retrieval strictness:
- when query is already resolved, retrieval enhancement returns resolved query directly,
- no extra hint expansion in strict resolved mode.

Topic integrity protections:
- acknowledgments do not update active/last topic,
- low-signal and pronoun-only turns are guarded from topic corruption.

---

## 9) Grounded Generation Strategy

General answer generation is done through the document chain in `final/rag_pipeline.py` and question framing in `final/app.py`.

Grounding design:
- answers should stay inside provided context,
- fallback sentence when context truly lacks answer:
  `The answer is not available in the provided document.`

Post-generation sanitizer (`final/app.py`):
- removes acronym expansions not explicitly found in retrieved context.
- keeps acronym itself (for example, converts `RAG (unsupported expansion)` to `RAG`).

This reduces unsupported expansions while preserving readable answers.

---

## 10) Source Attribution

Every final answer prints:
- an answer panel,
- a sources panel listing filenames and page numbers.

For memory persistence, compact source refs are stored with assistant-turn metadata.

---

## 11) Testing and Validation

Current tests live in:
- `final/tests/test_memory_manager.py`
- `final/tests/test_chapter_intents.py`

Coverage highlights:
- follow-up resolution,
- acknowledgment handling,
- topic continuity after acknowledgment,
- strict resolved retrieval behavior,
- prompt uses resolved query (not raw ambiguous query),
- acronym sanitizer behavior,
- clarification behavior when pronoun has no anchor,
- exactly 100 chapter intent routing cases.

---

## 12) Operational Notes

Run:
- `python final/app.py`

Typical first run behavior:
- uploads are indexed into Chroma,
- BM25 cache is built,
- later sessions reuse persisted indexes/caches.

Important dependency notes:
- PDF support depends on `PyMuPDF` (`fitz`).
- local generation depends on Ollama and installed model.
- API path depends on Groq key and package if `USE_API_LLM=True`.

---

## 13) Current Workflow Summary (Compact)

User question path:

1. raw query
2. follow-up resolution
3. acknowledgment/clarification guard
4. chapter intent routing (deterministic branch) or general branch
5. retrieval query enhancement
6. hybrid retrieval + reranking
7. context expansion
8. grounded generation
9. answer sanitation
10. source display + memory/state update

This is the current "what we built" system from ingestion to grounded conversational QA.
