"""Base44Pipeline — the single orchestrator that wires every module together.

This is the ONLY place modules are connected (Block A requirement #3). Adding a
new module means registering it here or on the router; existing module code is
never edited. Data flows:

    ingest -> categorize -> prioritize -> (Block A) semantic search
           -> visualize -> format -> verify -> version -> metrics

External-integration edges (cloud sync, webhooks, LLM) activate only when
configured; otherwise they no-op so the full flow always completes.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from .analysis.semantic_search import SemanticSearchEngine
from .config import Settings, get_settings
from .formatting.court_rules import CourtRulesFormatter
from .formatting.versioning import VersionControl
from .ingestion.security import SemanticFirewall
from .quant.engine import QuantitativeStrategyEngine
from .routing.router import PayloadRouter
from .routing.webhooks import WebhookNotifier
from .schemas import (
    CaseFinancials,
    CourtFormatConfig,
    PipelineResult,
    RawDocument,
    VisualizationBundle,
)
from .visualization.decision_tree import build_decision_tree
from .visualization.graphs import build_relational_graph
from .visualization.mind_map import build_mind_map
from .workload.categorize import WorkloadCategorizer
from .workload.metrics import MetricsRecorder
from .workload.priority import assign_priorities
from .workload.verify import ComplianceChecker

log = logging.getLogger("base44.pipeline")


class Base44Pipeline:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

        # Module 2 — routing surface
        self.notifier = WebhookNotifier(self.settings)
        self.router = PayloadRouter(notifier=self.notifier)

        # Module 1 (ingestion guard) — adversarial semantic firewall
        self.firewall = SemanticFirewall(quarantine=True)

        # Module 6/7 — workload + verification
        self.categorizer = WorkloadCategorizer()
        self.compliance = ComplianceChecker()

        # Module 3 — semantic search (Block A engine)
        self.search = SemanticSearchEngine(self.settings)

        # Module 5 — formatting + versioning
        self.formatter = CourtRulesFormatter()
        self.versioning = VersionControl()

        self._corpus_ready = False
        self._register_default_routes()

    # ------------------------------------------------------------------ #
    def index_corpus(self, docs: Sequence[RawDocument]) -> "Base44Pipeline":
        """Build the Block A semantic index over a document corpus.

        Documents pass through the adversarial firewall first; poisoned docs are
        quarantined before any text reaches the vectorizer or LLM.
        """
        safe_docs = self.firewall.scan_documents(list(docs))
        self.search.build(safe_docs)
        self._corpus_ready = self.search.corpus is not None
        return self

    def process_case(
        self,
        case_id: str,
        text: str,
        query: Optional[str] = None,
        court: Optional[CourtFormatConfig] = None,
        financials: Optional[CaseFinancials] = None,
        author: str = "base44",
    ) -> PipelineResult:
        """Run the full pipeline over one case payload and return a PipelineResult."""
        metrics = MetricsRecorder()

        with metrics.stage("categorize"):
            categorized = self.categorizer.categorize(text)
        self.router.dispatch("case.categorized", categorized)

        with metrics.stage("prioritize"):
            priorities = assign_priorities(categorized)

        query_result = None
        if query and self._corpus_ready:
            with metrics.stage("query") as _:
                query_result = self.search.query(query)
            # attach retrieval telemetry for the adaptive-optimization hook
            metrics.metrics[-1].detail.update(
                candidates_scanned=query_result.candidates_scanned,
                total_docs=query_result.total_docs,
            )
            self.router.dispatch("case.searched", query_result)

        with metrics.stage("visualize"):
            visualization = VisualizationBundle(
                decision_tree=build_decision_tree(case_id, categorized),
                relational_graph=build_relational_graph(case_id, categorized),
                mind_map=build_mind_map(case_id, categorized),
            )
        self.router.dispatch("case.visualized", visualization)

        formatted = None
        if court is not None:
            with metrics.stage("format"):
                formatted = self.formatter.format(court, body=text)

        with metrics.stage("verify"):
            compliance = self.compliance.check(text, cfg=court, formatted=formatted)

        with metrics.stage("version"):
            revision = self.versioning.commit(
                doc_id=case_id, content=text, author=author,
                summary=f"Processed case {case_id}",
            )

        quant = None
        if financials is not None:
            with metrics.stage("quant"):
                quant = QuantitativeStrategyEngine(financials).analyze(
                    corpus=self.search.corpus if self._corpus_ready else None,
                )
            self.router.dispatch("case.quantified", quant)

        result = PipelineResult(
            case_id=case_id,
            categorized=categorized,
            priorities=priorities,
            query_result=query_result,
            visualization=visualization,
            formatted_document=formatted,
            compliance=compliance,
            revision=revision,
            quant=quant,
            metrics=metrics.metrics,
        )
        adjustments = metrics.suggest_adjustments()
        if adjustments:
            log.info("adaptive suggestions: %s", adjustments)
        self.router.dispatch("case.completed", result)
        return result

    # ------------------------------------------------------------------ #
    def _register_default_routes(self) -> None:
        # Handlers are intentionally thin; they exist so external services can
        # subscribe to lifecycle events without touching process_case().
        self.router.register("case.categorized", lambda p: log.debug("categorized"))
        self.router.register("case.searched", lambda p: log.debug("searched"))
        self.router.register("case.visualized", lambda p: log.debug("visualized"))
        self.router.register("case.quantified", lambda p: log.debug("quantified"))
        self.router.register("case.completed", lambda p: log.debug("completed"))
