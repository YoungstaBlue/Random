"""Citation & jurisdiction verification (Module 3 / Module 7 pre-check).

Extracts legal citations and classifies jurisdiction (federal / state / US
territory such as Puerto Rico). Uses the ``eyecite`` library when installed for
robust extraction; otherwise a regex extractor covers the common reporter
patterns so verification always runs.

NOTE: "verified" here means *well-formed and jurisdiction-classified* — purely
offline pattern-matching. When ``CLAUDE_LEGAL_COURTLISTENER_ENABLED=true``,
``verify_text`` also resolves each citation against the real CourtListener
citation-lookup API (see ``analysis/citator.py``) and populates the
``resolved``/``case_name``/``courtlistener_url`` fields on each ``Citation``.
That confirms a citation names a real, on-file opinion; it is still not an
authoritative good-law/negative-treatment check — see the caveat in
``citator.py``. The citator is disabled by default (offline-safe), so this
module never makes a network call unless explicitly configured to.
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..config import Settings, get_settings
from ..schemas import Citation
from .citator import CourtListenerCitator

# e.g. "347 U.S. 483", "410 F.3d 1200", "5 P.R. Offic. Trans. 1"
_CITE_RE = re.compile(
    r"\b(\d{1,4})\s+"
    r"([A-Z][A-Za-z.]*(?:\s?[0-9][a-z]{0,2})?(?:\s[A-Z][A-Za-z.]+){0,3})\.?\s+"
    r"(\d{1,5})\b"
)

_FEDERAL_REPORTERS = {"U.S.", "S. Ct.", "L. Ed.", "F.", "F.2d", "F.3d", "F. Supp.",
                      "F. Supp. 2d", "F. Supp. 3d", "Fed. Cl."}
_TERRITORY_HINTS = {
    "P.R.": "US territory — Puerto Rico",
    "Offic. Trans.": "US territory — Puerto Rico",
    "D.P.R.": "US territory — Puerto Rico (District of Puerto Rico)",
    "V.I.": "US territory — U.S. Virgin Islands",
    "Guam": "US territory — Guam",
}


class CitationVerifier:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._citator = CourtListenerCitator(self.settings)

    def extract(self, text: str) -> List[Citation]:
        out: List[Citation] = []
        for m in _CITE_RE.finditer(text):
            volume, reporter, page = m.group(1), m.group(2).strip(), m.group(3)
            normalized = f"{volume} {reporter} {page}"
            out.append(
                Citation(
                    raw=m.group(0).strip(),
                    normalized=normalized,
                    jurisdiction=self._classify(reporter),
                    verified=True,     # well-formed & classified
                    notes=None,
                )
            )
        return out

    def verify_text(self, text: str) -> List[Citation]:
        """Prefer eyecite when available, else regex; then resolve against
        CourtListener if the citator is enabled."""
        citations = self._extract_text(text)
        if self._citator.enabled:
            citations = self._citator.resolve(citations)
        return citations

    def _extract_text(self, text: str) -> List[Citation]:
        try:  # pragma: no cover - optional dependency
            from eyecite import get_citations

            cites = get_citations(text)
            if cites:
                out: List[Citation] = []
                for c in cites:
                    raw = getattr(c, "matched_text", lambda: str(c))()
                    reporter = getattr(getattr(c, "groups", {}), "get", lambda *_: None)("reporter")
                    out.append(
                        Citation(
                            raw=raw,
                            normalized=getattr(c, "corrected_citation", lambda: raw)(),
                            jurisdiction=self._classify(reporter or ""),
                            verified=True,
                        )
                    )
                return out
        except Exception:
            pass
        return self.extract(text)

    @staticmethod
    def _classify(reporter: str) -> str:
        for hint, label in _TERRITORY_HINTS.items():
            if hint in reporter:
                return label
        if any(reporter.startswith(fr) or reporter == fr for fr in _FEDERAL_REPORTERS):
            return "federal"
        return "state"
