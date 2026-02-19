import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from langchain_core.documents import Document


from contextforge import app
from contextforge.document_manager import ChunkRegistry
from contextforge.memory_manager import (
    HierarchicalMemoryManager,
    resolve_followup_query,
    is_acknowledgment_query,
)


class TestHierarchicalMemoryManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "memory.sqlite"
        self.mm = HierarchicalMemoryManager(self.db_path)
        self.session_id = self.mm.start_session(["book.pdf"])

    def tearDown(self):
        self.mm.close()
        self.tmp.cleanup()

    def test_state_roundtrip(self):
        self.mm.update_state(self.session_id, {"active_chapter": 20, "active_chapter_title": "ALTERNATIVES TO RAG"})
        state = self.mm.get_state(self.session_id)
        self.assertEqual(state.get("active_chapter"), 20)
        self.assertEqual(state.get("active_chapter_title"), "ALTERNATIVES TO RAG")

    def test_followup_resolution_uses_active_chapter(self):
        self.mm.update_state(self.session_id, {"active_chapter": 20, "active_chapter_title": "ALTERNATIVES TO RAG"})
        resolved = self.mm.resolve_followup_query(self.session_id, "explain this chapter")
        self.assertIn("chapter 20", resolved.lower())

    def test_pronoun_followup_resolution_uses_active_topic(self):
        self.mm.update_state(self.session_id, {"active_topic": "hierarchical memory", "last_topic": "hierarchical memory"})
        resolved = self.mm.resolve_followup_query(self.session_id, "Explain this in simpler terms.")
        self.assertEqual(resolved.lower(), "explain hierarchical memory in simpler terms.")

    def test_topic_continuity_uses_last_topic_when_active_missing(self):
        self.mm.update_state(self.session_id, {"last_topic": "hierarchical memory"})
        resolved = self.mm.resolve_followup_query(self.session_id, "tell me more")
        self.assertIn("hierarchical memory", resolved.lower())

    def test_no_false_positive_for_explicit_topic_query(self):
        self.mm.update_state(self.session_id, {"active_topic": "hierarchical memory", "last_topic": "hierarchical memory"})
        query = "Explain memory processors in simpler terms."
        resolved = self.mm.resolve_followup_query(self.session_id, query)
        self.assertEqual(resolved, query)

    def test_enhanced_retrieval_does_not_append_old_topic_for_explicit_new_question(self):
        self.mm.update_state(self.session_id, {"active_topic": "hierarchical memory", "last_topic": "hierarchical memory"})
        enhanced = self.mm.enhance_retrieval_query(self.session_id, "What is RAG?")
        self.assertEqual(enhanced.lower(), "what is rag?")

    def test_enhanced_retrieval_query_uses_topic_hint(self):
        self.mm.update_state(
            self.session_id,
            {"active_chapter": 31, "active_chapter_title": "DEPLOYMENT", "active_topic": "deployment", "last_topic": "deployment"},
        )
        enhanced = self.mm.enhance_retrieval_query(self.session_id, "what are the challenges?")
        self.assertIn("chapter 31", enhanced.lower())
        self.assertIn("deployment", enhanced.lower())

    def test_detect_memory_recall_query_patterns(self):
        self.assertTrue(app.detect_memory_recall_query("What did I ask before this?"))
        self.assertTrue(app.detect_memory_recall_query("What was my previous question"))
        self.assertTrue(app.detect_memory_recall_query("What did I ask before hierarchical memory?"))
        self.assertFalse(app.detect_memory_recall_query("What is hierarchical memory?"))

    def test_strict_retrieval_scope_for_resolved_followup(self):
        self.mm.update_state(
            self.session_id,
            {
                "active_topic": "hierarchical memory",
                "last_topic": "hierarchical memory",
                "active_chapter": 20,
                "active_chapter_title": "ALTERNATIVES TO RAG",
            },
        )
        resolved = self.mm.resolve_followup_query(self.session_id, "Explain this in simpler terms")
        enhanced = self.mm.enhance_retrieval_query(
            self.session_id,
            resolved,
            query_is_resolved=True,
            strict=True,
        )
        self.assertIn("hierarchical memory", enhanced.lower())
        self.assertNotIn("chapter 20", enhanced.lower())

    def test_build_memory_context_contains_recent_dialogue(self):
        self.mm.register_user_query(self.session_id, "What is chapter 20?", intent="number_lookup", topic="chapter 20")
        self.mm.register_assistant_answer(self.session_id, "Chapter 20 is ALTERNATIVES TO RAG.")
        context = self.mm.build_memory_context(self.session_id, "explain it")
        self.assertIn("Recent dialogue", context)
        self.assertIn("Chapter 20", context)

    def test_profile_fact_extraction(self):
        self.mm.register_user_query(self.session_id, "my name is alex", intent="none", topic="name")
        facts = self.mm.get_profile_facts(self.session_id)
        keys = {fact["fact_key"] for fact in facts}
        self.assertIn("name", keys)

    def test_function_resolve_followup_query(self):
        state = {"active_topic": "hierarchical memory", "last_topic": "hierarchical memory"}
        resolved = resolve_followup_query("Explain this in simpler terms", state)
        self.assertEqual(resolved.lower(), "explain hierarchical memory in simpler terms")

    def test_acknowledgment_classifier(self):
        self.assertTrue(is_acknowledgment_query("ok"))
        self.assertTrue(is_acknowledgment_query("thank you"))
        self.assertTrue(is_acknowledgment_query("got it."))
        self.assertFalse(is_acknowledgment_query("what is hierarchical memory"))

    def test_acknowledgment_does_not_change_topic(self):
        self.mm.update_state(self.session_id, {"active_topic": "hierarchical memory", "last_topic": "hierarchical memory"})
        retriever = _FakeRetriever()
        uploader = _FakeUploader()
        with redirect_stdout(io.StringIO()):
            app.handle_user_query("ok", retriever, None, uploader, self.mm, self.session_id)
        state = self.mm.get_state(self.session_id)
        self.assertEqual(state.get("active_topic"), "hierarchical memory")
        self.assertEqual(state.get("last_topic"), "hierarchical memory")
        self.assertEqual(state.get("last_intent"), "acknowledgment")

    def test_acknowledgment_does_not_trigger_retrieval(self):
        retriever = _FakeRetriever()
        uploader = _FakeUploader()
        with redirect_stdout(io.StringIO()):
            app.handle_user_query("thanks", retriever, None, uploader, self.mm, self.session_id)
        self.assertEqual(retriever.calls, 0)

    def test_followup_after_acknowledgment_keeps_topic_continuity(self):
        self.mm.update_state(self.session_id, {"active_topic": "hierarchical memory", "last_topic": "hierarchical memory"})
        retriever = _FakeRetriever()
        uploader = _FakeUploader()
        with redirect_stdout(io.StringIO()):
            app.handle_user_query("ok", retriever, None, uploader, self.mm, self.session_id)
            app.handle_user_query("explain this", retriever, None, uploader, self.mm, self.session_id)
        self.assertEqual(retriever.calls, 1)
        self.assertIn("hierarchical memory", retriever.last_query.lower())
        self.assertNotIn("referring to", retriever.last_query.lower())

    def test_topic_derivation_ignores_low_signal_queries(self):
        self.assertEqual(self.mm.derive_topic("ok"), "")
        self.assertEqual(self.mm.derive_topic("this"), "")
        self.assertEqual(self.mm.derive_topic("more details"), "")

    def test_memory_recall_before_this_returns_previous_user_question(self):
        docs = [
            Document(
                page_content="RAG combines retrieval and generation for grounded responses.",
                metadata={"source": "book.pdf", "page": 10},
            )
        ]
        retriever = _FakeRetrieverWithDocs(docs)
        doc_chain = _FakeDocChain(
            [
                (
                    "RAG combines retrieval and generation to ground answers with external context. "
                    "It helps assistants answer accurately using retrieved evidence from documents."
                )
            ]
        )
        uploader = _FakeUploader(paths=["book.pdf"])
        with redirect_stdout(io.StringIO()):
            app.handle_user_query("What is RAG?", retriever, doc_chain, uploader, self.mm, self.session_id, grounded=True)
            retriever_calls_before_recall = retriever.calls
            chain_calls_before_recall = doc_chain.calls
            app.handle_user_query(
                "What did I ask before this?",
                retriever,
                doc_chain,
                uploader,
                self.mm,
                self.session_id,
                grounded=True,
            )

        self.assertEqual(retriever.calls, retriever_calls_before_recall)
        self.assertEqual(doc_chain.calls, chain_calls_before_recall)
        last_turn = self.mm.get_recent_turns(self.session_id, limit=1)[0]
        self.assertEqual(last_turn["text"], "You asked: What is RAG?")

    def test_memory_recall_before_named_query_returns_correct_question(self):
        docs = [
            Document(
                page_content="MCP and hierarchical memory are both discussed.",
                metadata={"source": "book.pdf", "page": 20},
            )
        ]
        retriever = _FakeRetrieverWithDocs(docs)
        doc_chain = _FakeDocChain(
            [
                "MCP helps tools connect to models in a structured way for grounded operations.",
                "Hierarchical memory combines recent context with relevant long-term memory.",
            ]
        )
        uploader = _FakeUploader(paths=["book.pdf"])
        with redirect_stdout(io.StringIO()):
            app.handle_user_query("What is MCP?", retriever, doc_chain, uploader, self.mm, self.session_id, grounded=True)
            app.handle_user_query("What is hierarchical memory?", retriever, doc_chain, uploader, self.mm, self.session_id, grounded=True)
            retriever_calls_before_recall = retriever.calls
            chain_calls_before_recall = doc_chain.calls
            app.handle_user_query(
                "What did I ask before hierarchical memory?",
                retriever,
                doc_chain,
                uploader,
                self.mm,
                self.session_id,
                grounded=True,
            )

        self.assertEqual(retriever.calls, retriever_calls_before_recall)
        self.assertEqual(doc_chain.calls, chain_calls_before_recall)
        last_turn = self.mm.get_recent_turns(self.session_id, limit=1)[0]
        self.assertEqual(last_turn["text"], "You asked: What is MCP?")

    def test_memory_recall_before_phrase_after_followup_returns_initial_question(self):
        docs = [
            Document(
                page_content="Hierarchical memory and context continuity details.",
                metadata={"source": "book.pdf", "page": 26, "logical_page": 26},
            )
        ]
        retriever = _FakeRetrieverWithDocs(docs)
        doc_chain = _FakeDocChain(
            [
                "Hierarchical memory combines recent context and long-term memory retrieval for continuity.",
                "It means using recent chat plus key older facts to answer clearly.",
            ]
        )
        uploader = _FakeUploader(paths=["book.pdf"])
        with patch.object(app, "_recover_followup_docs_from_memory", return_value=docs):
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "What is hierarchical memory?",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )
                app.handle_user_query(
                    "Explain this simply",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )
                retriever_calls_before_recall = retriever.calls
                chain_calls_before_recall = doc_chain.calls
                app.handle_user_query(
                    "What did I ask before explain this simply?",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )

        self.assertEqual(retriever.calls, retriever_calls_before_recall)
        self.assertEqual(doc_chain.calls, chain_calls_before_recall)
        last_turn = self.mm.get_recent_turns(self.session_id, limit=1)[0]
        self.assertEqual(last_turn["text"], "You asked: What is hierarchical memory?")

    def test_memory_recall_new_session_without_history(self):
        retriever = _FakeRetriever()
        doc_chain = _FakeDocChain(["unused"])
        uploader = _FakeUploader(paths=["book.pdf"])
        with redirect_stdout(io.StringIO()):
            app.handle_user_query(
                "What did I ask before this?",
                retriever,
                doc_chain,
                uploader,
                self.mm,
                self.session_id,
                grounded=True,
            )

        self.assertEqual(retriever.calls, 0)
        self.assertEqual(doc_chain.calls, 0)
        last_turn = self.mm.get_recent_turns(self.session_id, limit=1)[0]
        self.assertEqual(last_turn["text"], "I do not have previous conversation in this session.")

    def test_grounded_pipeline_does_not_post_validate_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "context.txt"
            doc_path.write_text(
                "Hierarchical memory stores relevant and persistent long-term characteristics of users.",
                encoding="utf-8",
            )
            docs = [
                Document(
                    page_content="Hierarchical memory stores relevant and persistent long-term characteristics of users.",
                    metadata={"source": str(doc_path), "page": 0},
                )
            ]
            retriever = _FakeRetrieverWithDocs(docs)
            doc_chain = _FakeDocChain(["class HierarchicalMemory:\n    pass"])
            uploader = _FakeUploader()
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "What is hierarchical memory?",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )

        self.assertEqual(doc_chain.calls, 1, "Grounded mode should not perform hard post-validation retries.")
        last_turn = self.mm.get_recent_turns(self.session_id, limit=1)[0]
        self.assertEqual(last_turn["role"], "assistant")
        self.assertIn("class hierarchicalmemory", last_turn["text"].lower())

    def test_grounded_pipeline_refines_brief_definition_answer_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "context.txt"
            doc_path.write_text(
                "Hierarchical memory combines recent context with relevant long-term memory retrieval.",
                encoding="utf-8",
            )
            docs = [
                Document(
                    page_content="Hierarchical memory combines recent context with relevant long-term memory retrieval.",
                    metadata={"source": str(doc_path), "page": 0},
                )
            ]
            retriever = _FakeRetrieverWithDocs(docs)
            doc_chain = _FakeDocChain(
                [
                    "It combines short-term and long-term memory.",
                    (
                        "Hierarchical memory combines recent conversation state with selected long-term memory "
                        "so responses remain coherent over time. For example, an assistant can remember a user's "
                        "earlier project constraints while answering a new implementation question."
                    ),
                ]
            )
            uploader = _FakeUploader()
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "What is hierarchical memory?",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )

        self.assertEqual(doc_chain.calls, 2)
        self.assertIn("refinement", doc_chain.inputs[1].lower())
        last_turn = self.mm.get_recent_turns(self.session_id, limit=1)[0]
        self.assertIn("for example", last_turn["text"].lower())

    def test_grounded_pipeline_refines_simplify_query_to_plain_short_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "context.txt"
            doc_path.write_text(
                "Hierarchical memory combines recent context with relevant long-term memory retrieval.",
                encoding="utf-8",
            )
            docs = [
                Document(
                    page_content="Hierarchical memory combines recent context with relevant long-term memory retrieval.",
                    metadata={"source": str(doc_path), "page": 0},
                )
            ]
            retriever = _FakeRetrieverWithDocs(docs)
            doc_chain = _FakeDocChain(
                [
                    "It means the assistant remembers recent chat plus important older facts. For example, it remembers your project goal while answering your next question.",
                ]
            )
            uploader = _FakeUploader()
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "Simplify hierarchical memory",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )

        self.assertEqual(doc_chain.calls, 1)
        self.assertIn("rewrite in very simple everyday language", doc_chain.inputs[0].lower())
        last_turn = self.mm.get_recent_turns(self.session_id, limit=1)[0]
        self.assertIn("for example", last_turn["text"].lower())

    def test_grounded_pipeline_streams_when_chain_supports_stream(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "context.txt"
            doc_path.write_text(
                "Hierarchical memory combines recent context with relevant long-term memory retrieval.",
                encoding="utf-8",
            )
            docs = [
                Document(
                    page_content="Hierarchical memory combines recent context with relevant long-term memory retrieval.",
                    metadata={"source": str(doc_path), "page": 0},
                )
            ]
            retriever = _FakeRetrieverWithDocs(docs)
            doc_chain = _FakeStreamingDocChain(
                [
                    "It means the assistant remembers recent chat plus important older facts.",
                    " For example, it remembers your project goal while answering your next question.",
                ]
            )
            uploader = _FakeUploader()
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "Simplify hierarchical memory",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )

        self.assertEqual(doc_chain.stream_calls, 1)
        self.assertEqual(doc_chain.invoke_calls, 0)
        self.assertIn("rewrite in very simple everyday language", doc_chain.inputs[0].lower())
        last_turn = self.mm.get_recent_turns(self.session_id, limit=1)[0]
        self.assertIn("for example", last_turn["text"].lower())

    def test_generation_prompt_uses_resolved_question(self):
        self.mm.update_state(self.session_id, {"active_topic": "hierarchical memory", "last_topic": "hierarchical memory"})
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "context.txt"
            doc_path.write_text(
                "Hierarchical memory stores relevant and persistent long-term characteristics of users.",
                encoding="utf-8",
            )
            docs = [
                Document(
                    page_content="Hierarchical memory stores relevant and persistent long-term characteristics of users.",
                    metadata={"source": str(doc_path), "page": 0},
                )
            ]
            retriever = _FakeRetrieverWithDocs(docs)
            doc_chain = _FakeDocChain([
                "Hierarchical memory stores relevant and persistent long-term characteristics of users.",
            ])
            uploader = _FakeUploader()
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "explain this",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )

        self.assertGreaterEqual(doc_chain.calls, 1)
        prompt_input = doc_chain.inputs[0].lower()
        self.assertIn("question:\nexplain hierarchical memory", prompt_input)
        self.assertNotIn("question:\nexplain this", prompt_input)
        self.assertNotIn("\ndetailed answer:\n", prompt_input)
        self.assertNotIn("\nanswer:\n", prompt_input)
        self.assertNotIn("context:\n{context}", prompt_input)

    def test_chapter_explain_uses_llm_synthesis_not_extractive_repeat(self):
        retriever = _FakeRetriever()
        uploader = _FakeUploader(paths=["book.pdf"])
        doc_chain = _FakeDocChain(
            ["In simple terms, this chapter explains how chunking and retrieval power a RAG workflow."]
        )
        chapter_entries = [
            {
                "chapter_number": 19,
                "title": "RAG PIPELINES",
                "heading": "19. RAG PIPELINES",
                "source": "book.pdf",
                "toc_doc_page": 2,
                "toc_page": 73,
            }
        ]
        chapter_sources = [
            Document(
                page_content="19. RAG PIPELINES 73",
                metadata={"source": "book.pdf", "page": 2},
            )
        ]
        chapter_context = [
            Document(
                page_content=(
                    "Chapter 19 explains building a RAG pipeline with chunking, indexing, and querying "
                    "retrieved chunks for grounded responses."
                ),
                metadata={"source": "book.pdf", "page": 73},
            )
        ]

        with patch.object(app, "extract_chapter_entries", return_value=(chapter_entries, chapter_sources)), patch.object(
            app, "_get_chapter_context_docs", return_value=chapter_context
        ), patch.object(
            app,
            "_build_extractive_chapter_answer",
            side_effect=AssertionError("extractive answer path must not run when LLM is available"),
        ):
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "Explain chapter 19 simply",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )

        self.assertEqual(retriever.calls, 0)
        self.assertEqual(doc_chain.calls, 1)
        chapter_prompt = doc_chain.inputs[0].lower()
        self.assertIn("question:\nexplain chapter 19 simply", chapter_prompt)
        self.assertNotIn("resolved chapter:", chapter_prompt)
        last_turn = self.mm.get_recent_turns(self.session_id, limit=1)[0]
        self.assertNotIn("extracted explanation from chapter text", last_turn["text"].lower())
        self.assertIn("rag workflow", last_turn["text"].lower())
        state = self.mm.get_state(self.session_id)
        self.assertEqual(state.get("last_intent"), "chapter_explain")
        self.assertEqual(state.get("response_mode"), "rag_llm")

    def test_primary_chapter_number_query_generates_explanation(self):
        retriever = _FakeRetriever()
        uploader = _FakeUploader(paths=["book.pdf"])
        doc_chain = _FakeDocChain(["Chapter 19 introduces the RAG pipeline setup and core workflow."])
        chapter_entries = [
            {
                "chapter_number": 19,
                "title": "RAG PIPELINES",
                "heading": "19. RAG PIPELINES",
                "source": "book.pdf",
                "toc_doc_page": 2,
                "toc_page": 73,
            }
        ]
        chapter_sources = [
            Document(
                page_content="19. RAG PIPELINES 73",
                metadata={"source": "book.pdf", "page": 2},
            )
        ]
        chapter_context = [
            Document(page_content="Intro page one.", metadata={"source": "book.pdf", "page": 73}),
            Document(page_content="Intro page two.", metadata={"source": "book.pdf", "page": 74}),
            Document(page_content="Deep details page three.", metadata={"source": "book.pdf", "page": 75}),
        ]

        with patch.object(app, "extract_chapter_entries", return_value=(chapter_entries, chapter_sources)), patch.object(
            app, "_get_chapter_context_docs", return_value=chapter_context
        ):
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "What is chapter 19?",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )

        self.assertEqual(retriever.calls, 0)
        self.assertEqual(doc_chain.calls, 1)
        self.assertEqual(doc_chain.context_sizes[0], 2)
        chapter_prompt = doc_chain.inputs[0].lower()
        self.assertIn("question:\nexplain what chapter 19 (rag pipelines) is about in simple terms.", chapter_prompt)
        last_turn = self.mm.get_recent_turns(self.session_id, limit=1)[0]
        self.assertNotIn("best chapter match", last_turn["text"].lower())
        self.assertIn("chapter 19", last_turn["text"].lower())
        state = self.mm.get_state(self.session_id)
        self.assertEqual(state.get("last_intent"), "chapter_explain")
        self.assertEqual(state.get("active_chapter"), 19)
        self.assertTrue(state.get("active_source_refs"))

    def test_chapter_followup_pronoun_uses_resolved_chapter_for_llm_synthesis(self):
        self.mm.update_state(self.session_id, {"active_chapter": 19, "active_chapter_title": "RAG PIPELINES"})
        retriever = _FakeRetriever()
        uploader = _FakeUploader(paths=["book.pdf"])
        doc_chain = _FakeDocChain(["This chapter teaches a practical retrieval pattern."])
        chapter_entries = [
            {
                "chapter_number": 19,
                "title": "RAG PIPELINES",
                "heading": "19. RAG PIPELINES",
                "source": "book.pdf",
                "toc_doc_page": 2,
                "toc_page": 73,
            }
        ]
        chapter_sources = [
            Document(
                page_content="19. RAG PIPELINES 73",
                metadata={"source": "book.pdf", "page": 2},
            )
        ]
        chapter_context = [
            Document(
                page_content="RAG pipelines split docs, index chunks, and retrieve relevant parts for answers.",
                metadata={"source": "book.pdf", "page": 73},
            )
        ]

        with patch.object(app, "_recover_followup_docs_from_memory", return_value=chapter_context), patch.object(
            app,
            "extract_chapter_entries",
            side_effect=AssertionError("chapter resolver must not run for resolved follow-ups"),
        ):
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "Explain this",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )

        self.assertEqual(retriever.calls, 0)
        self.assertGreaterEqual(doc_chain.calls, 1)
        self.assertEqual(doc_chain.context_sizes[0], 1)
        chapter_prompt = doc_chain.inputs[0].lower()
        self.assertIn("question:\nexplain chapter 19 rag pipelines", chapter_prompt)
        self.assertNotIn("resolved chapter:", chapter_prompt)

    def test_chapter_explain_caps_context_to_first_two_pages(self):
        retriever = _FakeRetriever()
        uploader = _FakeUploader(paths=["book.pdf"])
        doc_chain = _FakeDocChain(["Short chapter summary response."])
        chapter_entries = [
            {
                "chapter_number": 19,
                "title": "RAG PIPELINES",
                "heading": "19. RAG PIPELINES",
                "source": "book.pdf",
                "toc_doc_page": 2,
                "toc_page": 73,
            }
        ]
        chapter_sources = [
            Document(
                page_content="19. RAG PIPELINES 73",
                metadata={"source": "book.pdf", "page": 2},
            )
        ]
        chapter_context = [
            Document(page_content="Chapter intro page one.", metadata={"source": "book.pdf", "page": 73}),
            Document(page_content="Chapter intro page two.", metadata={"source": "book.pdf", "page": 74}),
            Document(page_content="Deep chapter details page three.", metadata={"source": "book.pdf", "page": 75}),
        ]

        with patch.object(app, "extract_chapter_entries", return_value=(chapter_entries, chapter_sources)), patch.object(
            app, "_get_chapter_context_docs", return_value=chapter_context
        ):
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "Explain chapter 19",
                    retriever,
                    doc_chain,
                    uploader,
                    self.mm,
                    self.session_id,
                    grounded=True,
                )

        self.assertEqual(doc_chain.calls, 1)
        self.assertEqual(doc_chain.context_sizes[0], 2)

    def test_sanitize_answer_strips_unknown_acronym_expansions(self):
        docs = [
            Document(
                page_content="RAG is used to retrieve relevant chunks.",
                metadata={},
            )
        ]
        answer = "RAG (Relevance Assessment Graph) is useful."
        sanitized = app._sanitize_answer_with_context(answer, docs)
        self.assertIn("RAG", sanitized)
        self.assertNotIn("Relevance Assessment Graph", sanitized)

    def test_sanitize_answer_preserves_paragraph_breaks(self):
        docs = [Document(page_content="Hierarchical memory combines short-term and long-term memory.", metadata={})]
        answer = "Paragraph one has details.\n\nParagraph two has more details."
        sanitized = app._sanitize_answer_with_context(answer, docs)
        self.assertIn("\n\n", sanitized)
        self.assertNotIn("Paragraph one has details. Paragraph two", sanitized)

    def test_pronoun_without_topic_requests_clarification(self):
        retriever = _FakeRetriever()
        uploader = _FakeUploader()
        with redirect_stdout(io.StringIO()):
            app.handle_user_query("explain this", retriever, None, uploader, self.mm, self.session_id, grounded=True)
        self.assertEqual(retriever.calls, 0)
        state = self.mm.get_state(self.session_id)
        self.assertEqual(state.get("last_intent"), "clarification_needed")

    def test_toc_like_chunks_are_filtered_for_general_queries(self):
        toc_doc = Document(
            page_content=(
                "5. MODEL ROUTING AND STRUCTURED OUTPUT 18 "
                "6. TOOL CALLING 20 "
                "7. AGENT MEMORY 25"
            ),
            metadata={"source": "book.pdf", "page": 1},
        )
        content_doc = Document(
            page_content="Hierarchical memory combines recent messages with long-term memory retrieval.",
            metadata={"source": "book.pdf", "page": 40},
        )
        filtered = app._filter_toc_like_docs("What is hierarchical memory?", [toc_doc, content_doc])
        self.assertEqual(len(filtered), 1)
        self.assertIn("Hierarchical memory", filtered[0].page_content)

    def test_toc_like_chunks_are_retained_for_chapter_queries(self):
        toc_doc = Document(
            page_content=(
                "1. INTRODUCTION 1 "
                "2. AGENT MEMORY 25 "
                "3. DEPLOYMENT 41"
            ),
            metadata={"source": "book.pdf", "page": 1},
        )
        content_doc = Document(
            page_content="Agent memory helps preserve context over time.",
            metadata={"source": "book.pdf", "page": 25},
        )
        filtered = app._filter_toc_like_docs("show table of contents", [toc_doc, content_doc])
        self.assertEqual(len(filtered), 2)

    def test_format_sources_prefers_logical_page(self):
        docs = [
            Document(
                page_content="Hierarchical memory details.",
                metadata={"source": "book.pdf", "page": 40, "logical_page": 26},
            )
        ]
        formatted = app.format_sources(docs)
        self.assertIn("Pages: 26", formatted)
        self.assertNotIn("Pages: 41", formatted)

    def test_collect_source_refs_falls_back_to_physical_page(self):
        docs = [
            Document(
                page_content="Fallback page mapping.",
                metadata={"source": "book.pdf", "page": 40},
            )
        ]
        refs = app._collect_source_refs(docs)
        self.assertEqual(refs[0]["page"], 41)
        self.assertEqual(refs[0]["doc_page"], 40)

    def test_display_page_uses_logical_offset_when_logical_missing(self):
        doc = Document(
            page_content="Offset-based logical page mapping.",
            metadata={"source": "book.pdf", "page": 40, "logical_page_offset": 15},
        )
        self.assertEqual(app._get_display_page_number(doc), 26)

    def test_resolve_source_refs_prefers_doc_page_to_avoid_logical_page_drift(self):
        source_path = str(Path("book.pdf").resolve())
        pages = [
            Document(page_content=f"Physical page {idx + 1}", metadata={"source": source_path, "page": idx})
            for idx in range(60)
        ]
        source_refs = [{"source": "book.pdf", "source_path": source_path, "page": 25, "doc_page": 40}]

        with patch.object(app, "_load_document_pages", return_value=pages):
            recovered = app._resolve_source_refs_to_docs(source_refs, [source_path])

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].metadata.get("page"), 40)
        self.assertEqual(recovered[0].metadata.get("logical_page"), 25)
        self.assertEqual(app._get_display_page_number(recovered[0]), 25)

    def test_register_chunks_rejects_out_of_range_logical_page(self):
        with tempfile.TemporaryDirectory() as td:
            registry = ChunkRegistry(Path(td) / "chunk_registry.sqlite")
            try:
                docs = [
                    Document(
                        page_content="Bad logical page value",
                        metadata={"source": "book.pdf", "page": 0, "logical_page": 2024},
                    )
                ]
                registry.register_chunks(docs)
                row = registry.get_chunk(docs[0].metadata["chunk_id"])
                self.assertIsNotNone(row)
                self.assertIsNone(row["logical_page"])
                self.assertIsNone(docs[0].metadata["logical_page"])
            finally:
                registry.close()


class _FakeRetriever:
    def __init__(self):
        self.calls = 0
        self.last_query = ""

    def invoke(self, query):
        self.calls += 1
        self.last_query = str(query)
        return []


class _FakeRetrieverWithDocs:
    def __init__(self, docs):
        self.docs = docs
        self.calls = 0
        self.last_query = ""

    def invoke(self, query):
        self.calls += 1
        self.last_query = str(query)
        return self.docs


class _FakeDocChain:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.inputs = []
        self.context_sizes = []

    def invoke(self, payload):
        self.calls += 1
        self.inputs.append(payload.get("input", ""))
        context_docs = payload.get("context") or []
        self.context_sizes.append(len(context_docs))
        idx = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[idx]


class _FakeStreamingDocChain:
    def __init__(self, stream_chunks):
        self.stream_chunks = list(stream_chunks)
        self.stream_calls = 0
        self.invoke_calls = 0
        self.inputs = []
        self.context_sizes = []

    def stream(self, payload):
        self.stream_calls += 1
        self.inputs.append(payload.get("input", ""))
        context_docs = payload.get("context") or []
        self.context_sizes.append(len(context_docs))
        for chunk in self.stream_chunks:
            yield chunk

    def invoke(self, payload):
        self.invoke_calls += 1
        self.inputs.append(payload.get("input", ""))
        context_docs = payload.get("context") or []
        self.context_sizes.append(len(context_docs))
        return "".join(self.stream_chunks)


class _FakeUploader:
    def __init__(self, paths=None):
        self.paths = list(paths or [])

    def get_all_paths(self):
        return list(self.paths)


if __name__ == "__main__":
    unittest.main()
