"""Semantic search engine — Block A embedded inside Block B, Module 3.

This is where the vector-matrix engine becomes a legal capability: it wires
ingestion -> sparse vectorization -> LSH index -> accelerated cosine ranking
into a single "query by legal meaning" interface. It queries case law, statutes,
and precedent files by doctrine/context rather than pure keyword matching.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from ..config import Settings, get_settings
from ..ingestion.preprocess import Preprocessor
from ..index.lsh_index import LSHIndex
from ..query.similarity import CosineRanker
from ..schemas import CleanedDocument, QueryResult, RawDocument, SearchHit
from ..vectorize.sparse_vectorizer import SparseVectorizer, VectorizedCorpus

log = logging.getLogger("base44.analysis.semantic_search")


class SemanticSearchEngine:
    """Doctrine-aware retrieval over a legal corpus using the Block A pipeline."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.preprocessor = Preprocessor()
        self.vectorizer = SparseVectorizer(self.settings)
        self.ranker = CosineRanker()
        self.index: Optional[LSHIndex] = None
        self.corpus: Optional[VectorizedCorpus] = None
        self._snippets: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    def build(self, docs: Sequence[RawDocument]) -> "SemanticSearchEngine":
        if not docs:
            log.info("semantic index build skipped: empty corpus")
            self.corpus = None
            self.index = None
            return self
        cleaned: List[CleanedDocument] = self.preprocessor.process_many(docs)
        self._snippets = {d.doc_id: _snippet(raw.text) for d, raw in zip(cleaned, docs)}
        self.corpus = self.vectorizer.fit_transform(cleaned)
        self.index = LSHIndex(nbits=self.settings.lsh_nbits).build(self.corpus.matrix)
        log.info(
            "semantic index ready: %d docs, %d features, density=%.4f, lsh=%s",
            self.corpus.n_docs, self.corpus.n_features, self.corpus.density,
            self.index.backend,
        )
        return self

    def query(self, text: str, top_k: Optional[int] = None) -> QueryResult:
        if self.corpus is None or self.index is None:
            raise RuntimeError("SemanticSearchEngine.build() must run before query()")
        top_k = top_k or self.settings.query_top_k

        cleaned_tokens = self.preprocessor.clean_text(text)
        q_vec = self.vectorizer.transform_query(" ".join(cleaned_tokens))

        candidate_rows = self.index.candidates(q_vec)
        bucket = self.index.bucket_of(q_vec)
        ranked = self.ranker.rank(q_vec, self.corpus.matrix, candidate_rows, top_k)

        hits = [
            SearchHit(
                doc_id=self.corpus.doc_ids[row],
                score=round(score, 6),
                bucket=bucket if bucket >= 0 else None,
                snippet=self._snippets.get(self.corpus.doc_ids[row]),
            )
            for row, score in ranked
        ]
        return QueryResult(
            query=text,
            hits=hits,
            candidates_scanned=len(candidate_rows),
            total_docs=self.corpus.n_docs,
            backend={
                "index": self.index.backend,
                "similarity": self.ranker.backend,
                "device": self.ranker.device,
            },
        )


def _snippet(text: str, length: int = 160) -> str:
    flat = " ".join(text.split())
    return flat[:length] + ("…" if len(flat) > length else "")
