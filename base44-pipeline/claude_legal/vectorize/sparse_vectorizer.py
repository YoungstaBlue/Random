"""Convert cleaned text into numerical vectors using TF-IDF.

STRICT CONSTRAINT (Block A #2): vectors are stored as ``scipy.sparse`` CSR
matrices for memory efficiency across millions of documents. There is no dense
code path — ``.toarray()`` is never called on the corpus.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..config import Settings, get_settings
from ..schemas import CleanedDocument


@dataclass
class VectorizedCorpus:
    """Runtime artifact (kept out of the JSON schemas on purpose)."""
    doc_ids: List[str]
    matrix: sp.csr_matrix           # shape (n_docs, n_features), L2-normalized, sparse
    vectorizer: TfidfVectorizer

    @property
    def n_docs(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_features(self) -> int:
        return self.matrix.shape[1]

    @property
    def density(self) -> float:
        if self.matrix.shape[0] == 0 or self.matrix.shape[1] == 0:
            return 0.0
        return self.matrix.nnz / float(self.matrix.shape[0] * self.matrix.shape[1])


class SparseVectorizer:
    """TF-IDF vectorizer producing L2-normalized sparse CSR matrices.

    L2 normalization means a later dot product between two rows equals their
    cosine similarity — which the query layer exploits for fast ranking.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._vectorizer = TfidfVectorizer(
            max_features=self.settings.tfidf_max_features,
            ngram_range=(1, self.settings.tfidf_ngram_max),
            dtype=np.float32,
            norm=None,          # we normalize explicitly (keeps result sparse)
            sublinear_tf=True,
        )
        self._fitted = False

    def fit_transform(self, docs: Sequence[CleanedDocument]) -> VectorizedCorpus:
        texts = [d.text for d in docs]
        matrix = self._vectorizer.fit_transform(texts)      # sparse CSR
        matrix = normalize(matrix, norm="l2", axis=1, copy=False)
        self._fitted = True
        assert sp.issparse(matrix), "vectorization must remain sparse"
        return VectorizedCorpus(
            doc_ids=[d.doc_id for d in docs],
            matrix=matrix.tocsr().astype(np.float32),
            vectorizer=self._vectorizer,
        )

    def transform_query(self, text: str) -> sp.csr_matrix:
        """Vectorize a single query into a 1xN sparse row using the fitted vocab."""
        if not self._fitted:
            raise RuntimeError("SparseVectorizer must fit a corpus before querying")
        vec = self._vectorizer.transform([text])
        return normalize(vec, norm="l2", axis=1, copy=False).tocsr().astype(np.float32)
