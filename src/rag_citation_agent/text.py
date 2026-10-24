"""Tokenisation helpers shared by retrieval and grounding checks."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being
    do does did how what when where which who whom whose why can could should would will shall may
    might must it its this that these those there their they them he she his her you your we our
    i me my not no yes than then so such about into over under per via also any all each both
    more most other some only own same too very just""".split()
)

_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_SUFFIXES = ("ations", "ation", "ingly", "edly", "ing", "ies", "ed", "es", "ly", "s")


def stem(token: str) -> str:
    if token.isdigit():
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokens(text: str) -> list[str]:
    """All lowercase word tokens (numbers keep their decimal/thousand separators)."""
    return [m.replace(",", "") for m in _WORD.findall(text.lower())]


def content_tokens(text: str) -> list[str]:
    """Stemmed, stopword-free tokens: what retrieval and support checks compare."""
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


def numbers(text: str) -> set[str]:
    return {t for t in tokens(text) if t[0].isdigit()}


def normalize(text: str) -> str:
    """Case/whitespace/punctuation-insensitive form used for quote matching."""
    return " ".join(tokens(text))


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(" ".join(text.split()))]
    return [p for p in parts if p]
