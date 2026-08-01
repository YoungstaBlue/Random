from base44.index.lsh_index import LSHIndex
from base44.ingestion.preprocess import Preprocessor
from base44.query.similarity import CosineRanker
from base44.vectorize.sparse_vectorizer import SparseVectorizer


def _corpus_matrix(corpus):
    pre = Preprocessor(use_spacy=False, use_nltk=False)
    vec = SparseVectorizer()
    return vec, vec.fit_transform(pre.process_many(corpus))


def test_lsh_returns_candidate_subset(corpus):
    vec, result = _corpus_matrix(corpus)
    index = LSHIndex(nbits=64).build(result.matrix)
    q = vec.transform_query("wrongful termination retaliation unsafe conditions")
    cands = index.candidates(q)
    assert isinstance(cands, list)
    assert all(0 <= c < result.n_docs for c in cands)


def test_cosine_ranker_orders_by_similarity(corpus):
    vec, result = _corpus_matrix(corpus)
    index = LSHIndex(nbits=64).build(result.matrix)
    ranker = CosineRanker(backend="numpy")
    q = vec.transform_query("wrongful termination retaliation unsafe working conditions")
    cands = index.candidates(q) or list(range(result.n_docs))
    ranked = ranker.rank(q, result.matrix, cands, top_k=4)
    assert ranked, "expected at least one hit"
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)             # descending order
    # d1 is the wrongful-termination/retaliation doc -> should top the ranking.
    assert result.doc_ids[ranked[0][0]] == "d1"
