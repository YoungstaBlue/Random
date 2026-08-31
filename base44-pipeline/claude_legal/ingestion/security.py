"""Adversarial Semantic Firewall — prompt-injection defense for ingested docs.

Legal documents are attacker-controlled input: a hostile party can embed
prompt-injection ("poison pill") text hoping a downstream LLM obeys it. This
firewall scans raw documents at ingestion and quarantines matches before any
text reaches the vectorizer or the LLM analyst (Module 3 constraint).

Detection is signature + light-heuristic based (no ML dependency). It is a
guardrail, not a guarantee; keep the LLM's own system-prompt constraints too.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Sequence

from ..schemas import RawDocument

log = logging.getLogger("claude_legal.ingestion.security")


class SecurityException(Exception):
    """Raised when a document trips the semantic firewall."""


# Known injection signatures (case-insensitive substring match).
_SIGNATURES = [
    "ignore all previous",
    "ignore previous instructions",
    "disregard the above",
    "override protocol",
    "system prompt",
    "you are now",
    "developer mode",
    "reveal your instructions",
    "act as though",
]

# Heuristic patterns that co-occur with injection attempts.
_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?)", re.I),
    re.compile(r"(disregard|forget)\s+(everything|all|the\s+above)", re.I),
    re.compile(r"award\s+zero\s+damages", re.I),
]


@dataclass
class FirewallHit:
    doc_id: str
    signature: str
    excerpt: str


class SemanticFirewall:
    def __init__(self, quarantine: bool = True) -> None:
        # quarantine=True -> drop offending docs and continue;
        # quarantine=False -> raise SecurityException (halt).
        self.quarantine = quarantine

    def scan_text(self, text: str) -> List[str]:
        """Return the list of triggered signatures for a single text (empty = clean)."""
        lowered = text.lower()
        hits = [sig for sig in _SIGNATURES if sig in lowered]
        for pat in _PATTERNS:
            m = pat.search(text)
            if m:
                hits.append(m.group(0).strip().lower())
        return sorted(set(hits))

    def scan_documents(self, docs: Sequence[RawDocument]) -> List[RawDocument]:
        """Scan a corpus. Clean docs pass through; poisoned docs are quarantined
        (or raise, when ``quarantine=False``)."""
        clean: List[RawDocument] = []
        for doc in docs:
            hits = self.scan_text(doc.text)
            if not hits:
                clean.append(doc)
                continue
            msg = f"POISON PILL DETECTED in '{doc.doc_id}': {hits}"
            if not self.quarantine:
                raise SecurityException(msg)
            log.warning("%s — quarantined", msg)
        return clean
