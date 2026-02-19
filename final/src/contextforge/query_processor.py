"""Query processing orchestration layer for CLI sessions."""
from __future__ import annotations

from typing import Any, Callable


class QueryProcessor:
    """
    Encapsulates per-session query execution state.
    CLI code can remain focused on IO while this class owns query orchestration wiring.
    """

    def __init__(
        self,
        *,
        retriever: Any,
        doc_chain: Any,
        uploader: Any,
        memory_manager: Any = None,
        session_id: str | None = None,
        grounded: bool = True,
        query_handler: Callable[..., Any] | None = None,
    ):
        self.retriever = retriever
        self.doc_chain = doc_chain
        self.uploader = uploader
        self.memory_manager = memory_manager
        self.session_id = session_id
        self.grounded = bool(grounded)
        self.query_handler = query_handler

    def process(self, query: str):
        if not callable(self.query_handler):
            raise RuntimeError("QueryProcessor requires a callable query_handler.")
        return self.query_handler(
            query,
            self.retriever,
            self.doc_chain,
            self.uploader,
            self.memory_manager,
            self.session_id,
            self.grounded,
        )

