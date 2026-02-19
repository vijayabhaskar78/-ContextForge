# ContextForge Fixes Changelog

This document summarizes the major issues identified during debugging, why they happened, and exactly how they were fixed.

## 1) Chapter follow-ups repeated extractive text instead of conversational answers

### Problem
- Follow-up prompts such as `explain this` were returning repeated extracted chapter text.
- Behavior looked like a document viewer, not a conversational assistant.

### Root Cause
- Chapter explain path was using deterministic extractive output directly.

### Resolution
- Switched chapter answer generation to use the same LLM generation path with chapter context as grounding.
- Kept deterministic chapter extraction/routing, but changed only final answer generation behavior.

### Result
- Follow-ups now produce new contextual explanations instead of repeated extract blocks.

---

## 2) Chapter queries used a separate generation path (architectural split)

### Problem
- Normal queries used full RAG pipeline.
- Chapter explain had a separate inline generation path.

### Root Cause
- Chapter explain invoked generation in its own branch instead of flowing through shared generation stage.

### Resolution
- Refactored `handle_user_query` so chapter branches provide `retrieved_docs`, then continue into the shared generation block.
- Removed chapter-specific generation prompt bypass.

### Result
- Unified answer-generation architecture for normal and chapter flows.

---

## 3) Misclassification: `What is chapter 19?` treated as explain

### Problem
- Primary query `What is chapter 19?` was misclassified as explain instead of fast number lookup.

### Root Cause
- `"what is chapter"` was included in `is_chapter_explanation_query()` markers.

### Resolution
- Removed `"what is chapter"` from explanation markers.

### Result
- `What is chapter N?` now routes through `number_lookup` intent as intended.

---

## 4) CPU overload on chapter explain due to large raw chapter context

### Problem
- Explain flow could pass up to ~9 pages to local LLM, causing heavy latency/freezing.

### Root Cause
- Full chapter context was forwarded directly.

### Resolution
- Capped chapter explain grounding context to first 2 pages:
  - `retrieved_docs = chapter_context[:2]`

### Result
- Lower prompt load and faster local inference while preserving chapter grounding.

---

## 5) Metadata-only chapter answer on primary lookup

### Problem
- `What is chapter 19?` returned only metadata (`Best chapter match...`) rather than actual explanation.

### Root Cause
- Successful `number_lookup`/`lookup` branches finalized immediately with metadata response.

### Resolution
- On successful chapter lookup:
  - resolve chapter docs,
  - prepare explanation-oriented query,
  - continue into shared LLM generation path.
- Metadata-only response retained only for lookup failures.

### Result
- Primary chapter queries now return full generated explanations.

---

## 6) Follow-up routing re-ran chapter resolver (wrong conversational behavior)

### Problem
- After chapter question, follow-ups like `Explain this` re-ran chapter resolver.

### Root Cause
- Chapter routing executed even when follow-up referent had already been resolved from memory.

### Resolution
- Introduced follow-up-first mode:
  - if query was resolved as follow-up (`followup_resolved`), recover context from memory first,
  - skip chapter resolver in this case.
- Added explicit follow-up log phase:
  - `Phase 1: Resolving follow-up referent from memory...`

### Result
- Follow-ups reuse memory context and do not re-run chapter resolution.

---

## 7) Retrieval scope leakage in `expand_context`

### Problem
- Context expansion leaked into unrelated chapters (example: pages `25-27` expanded into `76-87`).

### Root Cause
- Neighbor expansion lacked strict topical/page boundary checks and had no strong size cap.

### Resolution
- Added topic-bounded expansion constraints in `expand_context`:
  - seed topic pages from initial retrieved docs,
  - include candidate only if within `+/-2` pages of any seed page,
  - enforce same-source guard,
  - hard cap total expanded docs to `len(retrieved_docs) * 2`,
  - keep seed/retrieval order priority.

### Result
- Expansion is bounded, topic-pure, and less likely to pollute LLM context.

---

## 8) Follow-up source recovery drift (logical vs physical page mismatch)

### Problem
- Follow-up reuse could recover wrong pages because refs stored display pages only.

### Root Cause
- `sources` metadata stored only user-facing page numbers; rehydration guessed physical pages.

### Resolution
- Extended source refs to store:
  - `doc_page` (physical page index),
  - `source_path` (full path hint),
  - display `page` retained for UX.
- Rehydration now prefers exact `doc_page` match first, then fallback strategies.
- Rehydration now also reapplies `logical_page`/`logical_page_offset` from stored refs onto recovered docs.

### Result
- Follow-up memory context now reuses the exact original pages more reliably.
- Displayed source pages remain stable across follow-ups (no jump from logical pages to unrelated physical page ranges).

---

## 9) Response depth inconsistency for definitional questions

### Problem
- Initial definitional prompts (for example, `What is hierarchical memory?`) could be too short compared to follow-up explanations.

### Root Cause
- Prompt guidance for fuller style was mainly keyed to explicit follow-up verbs (`explain`, `simplify`, etc.), and model outputs were sometimes terse.

### Resolution
- Expanded style-trigger detection in grounded question builder to include definitional intents:
  - `what is`, `what are`, `who is`, `define`
- Added one bounded refinement pass in `general_qa` only when the first grounded answer is still too brief.
- Refinement remains grounded and uses the same context/docs.

### Result
- Definitional first-turn answers are now more consistently detailed.
- Follow-up answers retain continuity and depth without changing source scope.

---

## 10) Prior-topic hint contamination on explicit new questions

### Problem
- Explicit new-topic queries (for example, `What is RAG?`) could be polluted by previous topic hints (for example, `hierarchical memory`), forcing retrieval to the wrong page region.

### Root Cause
- `enhance_retrieval_query()` appended `active_topic` hints even when the user had already asked an explicit standalone question.

### Resolution
- Added explicit-topic guard in retrieval enhancement:
  - if the resolved query matches explicit topic intent and is not ambiguous follow-up text, return it unchanged (no old-topic suffixing).

### Result
- New explicit questions now retrieve their own topical pages rather than inheriting prior-topic page clusters.

---

## Tests and Verification

The following test suites were repeatedly executed after each fix set:

- `final.tests.test_memory_manager`
- `final.tests.test_chapter_intents`
- `final.tests.test_logical_page_numbering`
- `final.tests.test_performance_optimizations`

Additional regression tests were added for:

- chapter intent routing (`What is chapter 19?` -> `number_lookup`),
- primary chapter query generating explanation (not metadata-only),
- follow-up not re-running chapter resolver,
- chapter explain 2-page context cap,
- source ref recovery preferring exact physical page (`doc_page`),
- expand_context bounded output and `+/-2` page purity.

---

## Files Updated (high-level)

- `final/app.py`
- `final/memory_manager.py`
- `final/tests/test_memory_manager.py`
- `final/tests/test_chapter_intents.py`
- `final/tests/test_performance_optimizations.py`
