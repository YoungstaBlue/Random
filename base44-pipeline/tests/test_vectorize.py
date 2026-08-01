import numpy as np
import scipy.sparse as sp

from base44.ingestion.preprocess import Preprocessor
from base44.vectorize.sparse_vectorizer import SparseVectorizer


def _clean(corpus):
    pre = Preprocessor(use_spacy=False, use_nltk=False)
    return pre.process_many(corpus)


def test_matrix_is_sparse_and_normalized(corpus):
    vec = SparseVectorizer()
    result = vec.fit_transform(_clean(corpus))
    # STRICT: must be a scipy sparse CSR matrix, never dense.
    assert sp.issparse(result.matrix)
    assert result.matrix.format == "csr"
    assert result.density < 1.0
    # L2-normalized rows -> row norms are ~1 (or 0 for empty rows).
    norms = np.sqrt(result.matrix.multiply(result.matrix).sum(axis=1)).A.ravel()
    for n in norms:
        assert abs(n - 1.0) < 1e-5 or n == 0.0


def test_query_vector_shares_vocabulary(corpus):
    vec = SparseVectorizer()
    result = vec.fit_transform(_clean(corpus))
    q = vec.transform_query("wrongful termination retaliation")
    assert sp.issparse(q)
    assert q.shape[1] == result.n_features
