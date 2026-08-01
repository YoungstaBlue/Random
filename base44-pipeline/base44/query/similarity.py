"""Cosine similarity between a query vector and the LSH-bucket candidates.

Because both corpus rows and the query are L2-normalized upstream, cosine
similarity reduces to a dot product. That dot product is dispatched to the
fastest available backend (Block A #4):

    * PyTorch  — SIMD on CPU, or GPU when ``cuda`` is available
    * CuPy     — GPU
    * NumPy    — SIMD-optimized BLAS fallback

Only the small candidate shortlist from the LSH index is scored, never the full
corpus, so the O(N^2) brute-force comparison is avoided end-to-end.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

log = logging.getLogger("base44.query")


def _detect_backend() -> str:
    try:  # pragma: no cover - optional dependency
        import torch  # noqa: F401
        return "torch"
    except Exception:
        pass
    try:  # pragma: no cover - optional dependency
        import cupy  # noqa: F401
        return "cupy"
    except Exception:
        pass
    return "numpy"


class CosineRanker:
    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or _detect_backend()
        self.device = "cpu"

    def rank(
        self,
        query_vec: sp.csr_matrix,
        corpus: sp.csr_matrix,
        candidate_rows: List[int],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Return ``[(row_index, score), ...]`` sorted by descending cosine sim."""
        if not candidate_rows:
            return []
        sub = corpus[candidate_rows]                      # sparse gather, stays sparse
        if self.backend == "torch":
            scores = self._dot_torch(query_vec, sub)
        elif self.backend == "cupy":  # pragma: no cover - optional dependency
            scores = self._dot_cupy(query_vec, sub)
        else:
            scores = self._dot_numpy(query_vec, sub)

        order = np.argsort(-scores)[:top_k]
        return [(candidate_rows[i], float(scores[i])) for i in order]

    # -- backends -----------------------------------------------------------
    def _dot_numpy(self, q: sp.csr_matrix, sub: sp.csr_matrix) -> np.ndarray:
        # Sparse-times-sparse dot product; result densified only for the shortlist.
        return np.asarray((sub @ q.T).todense()).ravel()

    def _dot_torch(self, q: sp.csr_matrix, sub: sp.csr_matrix) -> np.ndarray:  # pragma: no cover
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        q_t = torch.sparse_csr_tensor(
            torch.tensor(q.indptr, dtype=torch.int64),
            torch.tensor(q.indices, dtype=torch.int64),
            torch.tensor(q.data, dtype=torch.float32),
            size=q.shape,
        ).to(self.device).to_dense()
        s_t = torch.sparse_csr_tensor(
            torch.tensor(sub.indptr, dtype=torch.int64),
            torch.tensor(sub.indices, dtype=torch.int64),
            torch.tensor(sub.data, dtype=torch.float32),
            size=sub.shape,
        ).to(self.device).to_dense()
        scores = (s_t @ q_t.T).squeeze(-1)
        return scores.detach().cpu().numpy()

    def _dot_cupy(self, q: sp.csr_matrix, sub: sp.csr_matrix) -> np.ndarray:  # pragma: no cover
        import cupy as cp
        import cupyx.scipy.sparse as csp

        q_g = csp.csr_matrix(q)
        s_g = csp.csr_matrix(sub)
        scores = (s_g @ q_g.T).todense().ravel()
        return cp.asnumpy(scores)
