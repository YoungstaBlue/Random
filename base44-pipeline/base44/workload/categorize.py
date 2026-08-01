"""Workload Categorization (Module 6).

Parses incoming case payloads into the four legal buckets: raw evidence,
statutory authority, precedent case law, and procedural rules. Uses lightweight
lexical signals; the semantic engine (Block A) can refine these downstream.
"""
from __future__ import annotations

import re
from typing import List

from ..schemas import CategorizedPayload

# Signals for each bucket (checked in priority order per line).
_STATUTE_RE = re.compile(r"\b\d+\s+U\.S\.C\.|\b§\s?\d+|\bStat\.\b|\bstatute\b", re.I)
_PRECEDENT_RE = re.compile(r"\bv\.\s+[A-Z]|\b\d+\s+[A-Z][A-Za-z.]+\s+\d+\b|\bprecedent\b", re.I)
_PROCEDURE_RE = re.compile(r"\bFed\.\s?R\.|\bRule\s+\d+|\bdeadline\b|\bfile\b|\bmotion\b|"
                           r"\bserve\b|\bhearing\b", re.I)


class WorkloadCategorizer:
    def categorize(self, text: str) -> CategorizedPayload:
        payload = CategorizedPayload()
        for line in self._segments(text):
            if _STATUTE_RE.search(line):
                payload.statutory_authority.append(line)
            elif _PRECEDENT_RE.search(line):
                payload.precedent.append(line)
            elif _PROCEDURE_RE.search(line):
                payload.procedural_rules.append(line)
            else:
                payload.evidence.append(line)
        return payload

    @staticmethod
    def _segments(text: str) -> List[str]:
        # Split on newlines and sentence boundaries; keep non-trivial segments.
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [p.strip() for p in parts if len(p.strip()) > 3]
