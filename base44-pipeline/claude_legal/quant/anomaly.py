"""Vector-matrix anomaly detection for e-discovery.

Operates on the REAL sparse TF-IDF corpus produced by Block A (not simulated
embeddings). Because rows are L2-normalized, a sparse dot product yields cosine
similarity directly; documents unexpectedly close to a target (near-duplicates,
coordinated language) surface as anomalies for review.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..schemas import AnomalyMatch
from ..vectorize.sparse_vectorizer import VectorizedCorpus


class VectorAnomalyEngine:
    def __init__(self, corpus: VectorizedCorpus) -> None:
        self.corpus = corpus

    def identify_anomalies(
        self, target_idx: int = 0, threshold: float = 0.85,
        top_k: Optional[int] = None,
    ) -> List[AnomalyMatch]:
        if self.corpus is None or self.corpus.n_docs == 0:
            return []
        target = self.corpus.matrix[target_idx]                 # sparse row
        sims = np.asarray((self.corpus.matrix @ target.T).todense()).ravel()

        matches: List[AnomalyMatch] = []
        for idx, score in enumerate(sims):
            if idx == target_idx:
                continue
            if score > threshold:
                matches.append(AnomalyMatch(doc_id=self.corpus.doc_ids[idx],
                                            score=round(float(score), 6)))
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k] if top_k else matches
