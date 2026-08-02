"""End-to-end demo of the base44 pipeline — runs fully offline (no credentials).

    python examples/run_pipeline.py

Flow: load sample legal corpus -> build Block A semantic index -> process a case
payload (categorize, prioritize, semantic query, visualize, format, verify,
version) -> print each stage so the whole data flow is observable.
"""
from __future__ import annotations

import json
from pathlib import Path

from base44.ingestion.loaders import load_documents
from base44.pipeline import Base44Pipeline
from base44.schemas import CaseFinancials, CourtFormatConfig

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "sample_case"


def main() -> None:
    print("=" * 72)
    print("base44 — Unified Legal Analysis & Vector-Matrix Pipeline (demo)")
    print("=" * 72)

    pipe = Base44Pipeline()

    # 1) Block A: ingest + sparse-vectorize + LSH-index the corpus.
    docs = load_documents(CORPUS_DIR)
    pipe.index_corpus(docs)
    corpus = pipe.search.corpus
    print(f"\n[1] Indexed {len(docs)} docs | "
          f"{corpus.n_features} sparse features | density={corpus.density:.4f} | "
          f"LSH backend={pipe.search.index.backend} | "
          f"similarity backend={pipe.search.ranker.backend}")

    # 2) Run a full case through the pipeline.
    case_text = (CORPUS_DIR / "complaint.txt").read_text(encoding="utf-8")
    court = CourtFormatConfig(
        court_name="IN THE UNITED STATES DISTRICT COURT",
        district="District of Puerto Rico",
        case_number="3:25-cv-01234",
        plaintiff="Maria Santos",
        defendant="Acme Logistics Inc.",
        document_title="Plaintiff's Response to Motion to Dismiss",
        attorney_name="Jane Roe",
        attorney_bar="PR-98765",
    )
    financials = CaseFinancials(
        damages_claimed=1_500_000.0, litigation_cost=200_000.0,
        evidence_strength=0.8, witness_credibility=0.6,
        economic_impact=1.2, prob_win_baseline=0.65,
    )
    result = pipe.process_case(
        case_id="CASE-001",
        text=case_text,
        query="retaliation for reporting unsafe working conditions",
        court=court,
        financials=financials,
    )

    # 3) Show each stage's output.
    print("\n[2] Workload categorization (Module 6):")
    for bucket, items in result.categorized.model_dump().items():
        print(f"    {bucket:22s}: {len(items)} item(s)")

    print("\n[3] Top priorities (Module 6):")
    for t in result.priorities[:5]:
        print(f"    {t.task_id} [{t.kind.value}] score={t.priority_score}  {t.summary[:60]}")

    print("\n[4] Semantic search hits (Module 3 / Block A):")
    qr = result.query_result
    print(f"    scanned {qr.candidates_scanned}/{qr.total_docs} docs via {qr.backend}")
    for h in qr.hits:
        print(f"    {h.score:.4f}  {h.doc_id}  — {h.snippet}")

    print("\n[5] Visualization payloads (Module 4):")
    viz = result.visualization
    print(f"    decision tree branches : {len(viz.decision_tree.children)}")
    print(f"    relational graph       : {len(viz.relational_graph.nodes)} nodes, "
          f"{len(viz.relational_graph.edges)} edges")
    print(f"    mind map clusters      : {len(viz.mind_map.children)}")

    print("\n[6] Court-rules formatted caption (Module 5):")
    print("    " + result.formatted_document.caption.replace("\n", "\n    "))

    print("\n[7] Compliance report (Module 7):")
    print(f"    passed={result.compliance.passed} "
          f"issues={result.compliance.issues} "
          f"citations={len(result.compliance.citations_checked)}")

    print("\n[8] Version control (Module 5):")
    r = result.revision
    print(f"    v{r.version} by {r.author} @ {r.timestamp.isoformat()} "
          f"hash={r.content_hash[:12]}…")

    print("\n[9] Quantitative strategy tier:")
    q = result.quant
    print(f"    static expected value : ${q.static_expected_value:,.2f}")
    print(f"    Monte Carlo p5/p50/p95: ${q.monte_carlo.p5:,.0f} / "
          f"${q.monte_carlo.p50:,.0f} / ${q.monte_carlo.p95:,.0f}")
    print(f"    Bayesian win prob     : {q.bayesian_win_probability:.2%}")
    print(f"    gradient ∇f (ev/wit/ec): {q.gradient_leverage.d_evidence:,.0f} / "
          f"{q.gradient_leverage.d_witness:,.0f} / {q.gradient_leverage.d_economic:,.0f}")
    if q.nash_equilibrium:
        print(f"    Nash equilibrium      : Plaintiff→{q.nash_equilibrium.plaintiff_strategy} | "
              f"Defendant→{q.nash_equilibrium.defendant_strategy}")
    print(f"    recommendation        : {q.settlement_recommendation}")

    print("\n[10] Performance metrics (Module 8):")
    for m in result.metrics:
        print(f"    {m.stage:12s} {m.duration_ms:8.3f} ms")

    print("\nFull result JSON (truncated) ↓")
    print(json.dumps(result.model_dump(mode="json"), default=str)[:600] + " …")


if __name__ == "__main__":
    main()
