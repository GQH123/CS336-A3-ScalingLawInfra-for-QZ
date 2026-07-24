from __future__ import annotations

import hashlib
import html
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply conservative normalization before hashing/tokenization."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0000", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def should_keep_text(
    text: str,
    *,
    seen_hashes: set[str],
    min_chars: int = 200,
    max_chars: int | None = None,
) -> bool:
    cleaned = clean_text(text)
    if len(cleaned) < min_chars:
        return False
    if max_chars is not None and len(cleaned) > max_chars:
        return False

    digest = document_hash(cleaned)
    if digest in seen_hashes:
        return False
    seen_hashes.add(digest)
    return True
