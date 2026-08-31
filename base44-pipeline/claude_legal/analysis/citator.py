"""Real citation resolution against CourtListener (Module 3/7 — the citator edge).

``analysis/citations.py`` extracts and jurisdiction-classifies citations entirely
offline; that only proves a citation is well-formed, not that it names a real,
on-file opinion. This module closes that specific gap by calling CourtListener's
public citation-lookup API (Free Law Project, https://www.courtlistener.com),
which parses citation text and, when it finds a match, returns the opinion
cluster it resolved to.

Endpoint: ``POST {base_url}/api/rest/v4/citation-lookup/`` with body
``{"text": <citation text>}``. Response is a list of per-citation match objects;
each carries a ``clusters`` list when CourtListener found a match. Field names
follow CourtListener's documented v4 citation-lookup response shape as of this
writing; this module could not be live-verified against the real API in the
sandboxed environment it was built in (outbound network to courtlistener.com is
blocked by that environment's proxy policy), so every field is read
defensively with ``.get()`` and a shape mismatch degrades to "not resolved"
rather than raising. Smoke-test against the live API before relying on this in
production.

IMPORTANT CAVEAT — this is an existence check, not a good-law check. A
resolved citation means CourtListener has an opinion on file matching the
citation text. It does NOT mean the case is still good law: CourtListener's
free API does not provide automated negative-treatment / overruled detection
(that is what paid citators like Westlaw KeyCite or Lexis Shepard's do). The
``citation_count`` field, where available, is a raw citing-opinion count, not a
treatment signal — a heavily-cited case can still have been overruled.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..config import Settings, get_settings
from ..schemas import Citation

log = logging.getLogger("claude_legal.analysis.citator")

_LOOKUP_PATH = "/api/rest/v4/citation-lookup/"
_TIMEOUT_S = 15.0


class CourtListenerCitator:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.citator_enabled

    def resolve(self, citations: List[Citation]) -> List[Citation]:
        """Enrich ``citations`` in place with a real CourtListener lookup.

        No-ops (returns the list unchanged) when the citator is disabled, so
        callers never need to branch on configuration.
        """
        if not self.enabled or not citations:
            return citations
        for citation in citations:
            self._resolve_one(citation)
        return citations

    def _resolve_one(self, citation: Citation) -> None:
        try:
            matches = self._lookup(citation.normalized or citation.raw)
        except Exception as exc:  # pragma: no cover - network
            log.error("CourtListener lookup failed for '%s': %s", citation.raw, exc)
            citation.notes = _append_note(citation.notes, f"citator error: {exc}")
            return

        match = matches[0] if matches else {}
        clusters = match.get("clusters") or []
        if not clusters:
            citation.resolved = False
            status = match.get("status")
            if status and status != 200:
                citation.notes = _append_note(
                    citation.notes, f"CourtListener: no match (status {status})"
                )
            else:
                citation.notes = _append_note(citation.notes, "CourtListener: no match")
            return

        cluster = clusters[0]
        citation.resolved = True
        citation.cluster_id = cluster.get("id")
        citation.case_name = cluster.get("case_name") or cluster.get("caseName")
        citation.citation_count = cluster.get("citation_count") or cluster.get("citationCount")
        abs_url = cluster.get("absolute_url") or cluster.get("absoluteUrl")
        if abs_url:
            base = self.settings.courtlistener_base_url.rstrip("/")
            citation.courtlistener_url = abs_url if abs_url.startswith("http") else f"{base}{abs_url}"

    def _lookup(self, text: str) -> List[Dict[str, Any]]:
        url = self.settings.courtlistener_base_url.rstrip("/") + _LOOKUP_PATH
        body = {"text": text}
        headers = {"Content-Type": "application/json"}
        if self.settings.courtlistener_api_token:
            headers["Authorization"] = f"Token {self.settings.courtlistener_api_token}"

        try:  # pragma: no cover - network
            import httpx

            resp = httpx.post(url, json=body, headers=headers, timeout=_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
        except ImportError:
            data = self._post_urllib(url, body, headers)

        return data if isinstance(data, list) else []

    @staticmethod
    def _post_urllib(url: str, body: dict, headers: dict) -> Any:  # pragma: no cover - network
        import urllib.request

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _append_note(existing: Optional[str], addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition
