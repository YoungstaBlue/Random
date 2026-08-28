"""Tokenization, stop-word removal, and lemmatization.

Input:  raw legal text/documents.
Output: cleaned text strings ready for vectorization.

Uses spaCy for lemmatization when available; otherwise falls back to NLTK, and
finally to a dependency-free regex tokenizer + a built-in legal stop-word list
so the stage always runs.
"""
from __future__ import annotations

import re
from typing import Iterable, List

from ..schemas import CleanedDocument, RawDocument

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

# Compact English + legal-boilerplate stop-word set (fallback path).
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "as", "by", "at", "from", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "he", "she",
    "they", "them", "his", "her", "their", "we", "you", "i", "not", "no", "so",
    "such", "than", "into", "upon", "shall", "hereby", "herein", "whereas",
    "aforesaid", "thereof", "thereto", "pursuant", "said", "any", "all", "may",
}

# Very small lemma map for the fallback path (suffix rules cover the rest).
_LEMMA_OVERRIDES = {
    "held": "hold", "brought": "bring", "sought": "seek", "found": "find",
    "filed": "file", "ruled": "rule", "granted": "grant", "denied": "deny",
}


class Preprocessor:
    """Cleans raw documents into lemmatized, stop-word-free token strings."""

    def __init__(self, use_spacy: bool = True, use_nltk: bool = True) -> None:
        self._nlp = None
        self._nltk_lemmatizer = None
        self._nltk_stops = None

        if use_spacy:
            try:  # pragma: no cover - depends on optional install
                import spacy

                try:
                    self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                except OSError:
                    self._nlp = spacy.blank("en")
            except Exception:
                self._nlp = None

        if self._nlp is None and use_nltk:
            try:  # pragma: no cover - depends on optional install
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer

                self._nltk_lemmatizer = WordNetLemmatizer()
                self._nltk_stops = set(stopwords.words("english"))
            except Exception:
                self._nltk_lemmatizer = None

    # -- public API ---------------------------------------------------------
    def clean_text(self, text: str) -> List[str]:
        if self._nlp is not None:  # pragma: no cover - optional path
            return self._clean_spacy(text)
        if self._nltk_lemmatizer is not None:  # pragma: no cover - optional path
            return self._clean_nltk(text)
        return self._clean_regex(text)

    def process(self, doc: RawDocument) -> CleanedDocument:
        tokens = self.clean_text(doc.text)
        return CleanedDocument(
            doc_id=doc.doc_id,
            text=" ".join(tokens),
            tokens=tokens,
            metadata=doc.metadata,
        )

    def process_many(self, docs: Iterable[RawDocument]) -> List[CleanedDocument]:
        return [self.process(d) for d in docs]

    # -- backends -----------------------------------------------------------
    def _clean_spacy(self, text: str) -> List[str]:  # pragma: no cover
        out: List[str] = []
        for tok in self._nlp(text):
            if tok.is_space or tok.is_punct or tok.like_num:
                continue
            lemma = (tok.lemma_ or tok.text).lower().strip()
            if len(lemma) < 2 or lemma in _STOP_WORDS or tok.is_stop:
                continue
            out.append(lemma)
        return out

    def _clean_nltk(self, text: str) -> List[str]:  # pragma: no cover
        stops = self._nltk_stops or _STOP_WORDS
        out: List[str] = []
        for raw in _TOKEN_RE.findall(text.lower()):
            if raw in stops or len(raw) < 2:
                continue
            out.append(self._nltk_lemmatizer.lemmatize(raw))
        return out

    def _clean_regex(self, text: str) -> List[str]:
        out: List[str] = []
        for raw in _TOKEN_RE.findall(text.lower()):
            if raw in _STOP_WORDS or len(raw) < 2:
                continue
            out.append(_naive_lemma(raw))
        return out


def _naive_lemma(word: str) -> str:
    """Dependency-free lemmatizer: override table + conservative suffix rules."""
    if word in _LEMMA_OVERRIDES:
        return _LEMMA_OVERRIDES[word]
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""),
                         ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)] + repl
    return word
