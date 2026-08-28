"""Pipeline Fine-Tuning & Performance Logging (Module 8).

Latency & accuracy audit: times each pipeline stage and records structured
metrics. ``suggest_adjustments`` provides the adaptive-optimization hook that
proposes routing/config tweaks from observed outcomes.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List

from ..schemas import StageMetric

log = logging.getLogger("claude_legal.metrics")


class MetricsRecorder:
    def __init__(self) -> None:
        self.metrics: List[StageMetric] = []

    @contextmanager
    def stage(self, name: str, **detail: Any):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = (time.perf_counter() - start) * 1000.0
            self.metrics.append(StageMetric(stage=name, duration_ms=round(duration, 3),
                                            detail=detail))
            log.info("stage=%s duration_ms=%.3f %s", name, duration, detail)

    def total_ms(self) -> float:
        return round(sum(m.duration_ms for m in self.metrics), 3)

    def suggest_adjustments(self) -> Dict[str, str]:
        """Adaptive-optimization hook (Module 8): heuristics over recorded stages."""
        suggestions: Dict[str, str] = {}
        by_stage = {m.stage: m for m in self.metrics}

        vec = by_stage.get("vectorize")
        if vec and vec.duration_ms > 1000:
            suggestions["tfidf_max_features"] = (
                "Vectorization slow; lower CLAUDE_LEGAL_TFIDF_MAX_FEATURES or reduce ngram range."
            )
        query = by_stage.get("query")
        if query and query.detail.get("candidates_scanned", 0) > query.detail.get("total_docs", 1) * 0.5:
            suggestions["lsh_nbits"] = (
                "LSH buckets too coarse (scanning >50% of corpus); raise CLAUDE_LEGAL_LSH_NBITS."
            )
        return suggestions
