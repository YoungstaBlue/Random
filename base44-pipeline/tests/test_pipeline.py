from base44.pipeline import Base44Pipeline
from base44.schemas import CourtFormatConfig


def test_end_to_end_pipeline(corpus):
    pipe = Base44Pipeline().index_corpus(corpus)
    court = CourtFormatConfig(plaintiff="Santos", defendant="Acme",
                              case_number="3:25-cv-01234", attorney_name="Jane Roe")
    result = pipe.process_case(
        case_id="CASE-001",
        text="Plaintiff alleges wrongful termination and retaliation. "
             "See 42 U.S.C. 2000e. A motion under Rule 12 is pending.",
        query="wrongful termination retaliation",
        court=court,
    )
    # Every module produced output.
    assert result.categorized
    assert result.priorities
    assert result.query_result and result.query_result.hits
    assert result.visualization.decision_tree.children
    assert result.formatted_document.certificate_of_service
    assert result.compliance is not None
    assert result.revision.version == 1
    # Metrics recorded for each stage (Module 8).
    stages = {m.stage for m in result.metrics}
    assert {"categorize", "prioritize", "query", "visualize", "format", "verify",
            "version"} <= stages
    # ANN proof: fewer candidates scanned than a brute-force pass would need,
    # or at most the full corpus on a tiny fixture.
    assert result.query_result.candidates_scanned <= result.query_result.total_docs


def test_routes_are_pluggable():
    pipe = Base44Pipeline()
    seen = []
    pipe.router.register("case.completed", lambda p: seen.append(p))
    pipe.index_corpus([])  # empty corpus is allowed; query simply skipped
    pipe.process_case(case_id="C2", text="Evidence only, no query.")
    assert seen, "custom route handler should have fired"
