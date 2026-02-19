"""
Shared tokenization helpers for multilingual text matching.
"""
from __future__ import annotations

import re

_UNICODE_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize_for_matching(text: str, *, min_len: int = 1, limit: int | None = None) -> list[str]:
    """
    Tokenizes text with Unicode-aware word boundaries.
    Keeps letters/numbers from non-Latin scripts and normalizes via casefold().
    """
    safe_min_len = max(1, int(min_len))
    max_tokens = int(limit) if limit is not None else None

    out: list[str] = []
    for raw in _UNICODE_WORD_RE.findall(str(text or "").casefold()):
        token = raw.strip("_")
        if not token:
            continue
        if len(token) < safe_min_len:
            continue
        if not any(ch.isalnum() for ch in token):
            continue
        out.append(token)
        if max_tokens is not None and len(out) >= max_tokens:
            break
    return out
