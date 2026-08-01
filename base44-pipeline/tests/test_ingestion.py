from base44.ingestion.preprocess import Preprocessor, _naive_lemma
from base44.schemas import RawDocument


def test_stopwords_and_lemmatization():
    pre = Preprocessor(use_spacy=False, use_nltk=False)
    tokens = pre.clean_text("The plaintiffs filed motions and the courts ruled.")
    assert "the" not in tokens and "and" not in tokens        # stop-words removed
    assert "plaintiff" in tokens                              # 'plaintiffs' -> lemma
    assert "file" in tokens                                   # 'filed' -> override lemma


def test_process_returns_cleaned_document():
    pre = Preprocessor(use_spacy=False, use_nltk=False)
    doc = RawDocument(doc_id="x", text="Witnesses testified about the evidence.")
    cleaned = pre.process(doc)
    assert cleaned.doc_id == "x"
    assert cleaned.text and cleaned.tokens
    assert cleaned.text == " ".join(cleaned.tokens)


def test_naive_lemma_rules():
    assert _naive_lemma("companies") == "company"
    assert _naive_lemma("held") == "hold"
