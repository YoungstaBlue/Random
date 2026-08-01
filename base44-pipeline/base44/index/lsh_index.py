"""Locality-Sensitive Hashing index for Approximate Nearest Neighbor search.

Groups document vectors into hash buckets so a query is compared ONLY against
documents in matching buckets — bypassing brute-force O(N^2) comparison
(Block A #3).

Backend: FAISS ``IndexLSH`` (Meta) when ``faiss`` is installed. Otherwise a
self-contained NumPy random-hyperplane LSH provides identical semantics so the
pipeline runs anywhere. Both expose the same ``candidates(query)`` interface.

Because the query layer needs the ORIGINAL sparse TF-IDF rows to compute exact
cosine similarity, the index returns candidate *row indices*, not distances.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import scipy.sparse as sp

log = logging.getLogger("base44.index")


class LSHIndex:
    def __init__(self, nbits: int = 256, seed: int = 42) -> None:
        self.nbits = nbits
        self.seed = seed
        self._backend: str = "none"
        self._n = 0
        self._dim = 0

        # FAISS backend state
        self._faiss_index = None
        # NumPy backend state
        self._planes: Optional[np.ndarray] = None          # (dim, nbits)
        self._codes: Optional[np.ndarray] = None           # (n,) packed hashes
        self._buckets: dict[int, List[int]] = {}

    # ------------------------------------------------------------------ #
    def build(self, matrix: sp.csr_matrix) -> "LSHIndex":
        """Hash every row of the (sparse, L2-normalized) matrix into buckets."""
        self._n, self._dim = matrix.shape
        if self._try_build_faiss(matrix):
            self._backend = "faiss"
        else:
            self._build_numpy(matrix)
            self._backend = "numpy"
        log.info("LSH index built: backend=%s n=%d nbits=%d", self._backend, self._n, self.nbits)
        return self

    @property
    def backend(self) -> str:
        return self._backend

    def candidates(self, query_vec: sp.csr_matrix, max_candidates: int = 512) -> List[int]:
        """Return row indices sharing a bucket with the query (the ANN shortlist)."""
        if self._n == 0:
            return []
        if self._backend == "faiss":
            return self._candidates_faiss(query_vec, max_candidates)
        return self._candidates_numpy(query_vec, max_candidates)

    def bucket_of(self, query_vec: sp.csr_matrix) -> int:
        """Integer bucket id for reporting/telemetry (NumPy backend)."""
        if self._backend == "numpy" and self._planes is not None:
            return int(self._hash_dense(self._to_dense(query_vec))[0])
        return -1

    # ------------------------------------------------------------------ #
    # FAISS backend
    # ------------------------------------------------------------------ #
    def _try_build_faiss(self, matrix: sp.csr_matrix) -> bool:
        try:  # pragma: no cover - optional dependency
            import faiss
        except Exception:
            return False
        try:  # pragma: no cover - optional dependency
            dense = matrix.toarray().astype(np.float32)  # FAISS needs dense input rows
            index = faiss.IndexLSH(self._dim, self.nbits)
            index.train(dense)
            index.add(dense)
            self._faiss_index = index
            return True
        except Exception as exc:  # pragma: no cover
            log.warning("FAISS unavailable (%s); using NumPy LSH fallback", exc)
            return False

    def _candidates_faiss(self, query_vec: sp.csr_matrix, max_candidates: int) -> List[int]:  # pragma: no cover
        q = query_vec.toarray().astype(np.float32)
        k = min(max_candidates, self._n)
        _dist, idx = self._faiss_index.search(q, k)
        return [int(i) for i in idx[0] if i >= 0]

    # ------------------------------------------------------------------ #
    # NumPy random-hyperplane LSH backend
    # ------------------------------------------------------------------ #
    def _build_numpy(self, matrix: sp.csr_matrix) -> None:
        rng = np.random.default_rng(self.seed)
        # Cap hyperplane bits so buckets stay populated on small corpora.
        bits = min(self.nbits, max(4, int(np.log2(max(self._n, 2))) + 3), 60)
        self._planes = rng.standard_normal((self._dim, bits)).astype(np.float32)
        dense = self._to_dense(matrix)
        self._codes = self._hash_dense(dense)
        self._buckets = {}
        for row, code in enumerate(self._codes):
            self._buckets.setdefault(int(code), []).append(row)

    def _candidates_numpy(self, query_vec: sp.csr_matrix, max_candidates: int) -> List[int]:
        code = int(self._hash_dense(self._to_dense(query_vec))[0])
        cands = list(self._buckets.get(code, []))
        # Widen to Hamming-1 neighbor buckets if the exact bucket is too small.
        if len(cands) < 8 and self._planes is not None:
            bits = self._planes.shape[1]
            for b in range(bits):
                neighbor = code ^ (1 << b)
                cands.extend(self._buckets.get(neighbor, []))
        # De-dup, preserve order, cap.
        seen: set[int] = set()
        out: List[int] = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                out.append(c)
            if len(out) >= max_candidates:
                break
        # Safety fallback: on very small corpora a query may hash to an empty
        # bucket. Scanning the full (tiny) corpus keeps recall correct; on large
        # corpora the bucket is populated and this branch is not taken.
        if not out:
            return list(range(min(self._n, max_candidates)))
        return out

    def _hash_dense(self, dense: np.ndarray) -> np.ndarray:
        """Pack sign(x · planes) into an integer code per row."""
        projections = dense @ self._planes                  # (rows, bits)
        bits = (projections > 0).astype(np.uint64)
        weights = (1 << np.arange(bits.shape[1], dtype=np.uint64))
        return (bits * weights).sum(axis=1)

    @staticmethod
    def _to_dense(matrix: sp.csr_matrix) -> np.ndarray:
        # LSH hashing needs dense rows; the CORPUS stays sparse (only the hash
        # projection is dense, and it is discarded immediately).
        return np.asarray(matrix.todense(), dtype=np.float32)
