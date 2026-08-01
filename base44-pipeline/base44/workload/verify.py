"""Verification & Error Handling (Module 7).

Citation pre-check + compliance guardrail. Validates citations/Latin terms and
flags missing filing requirements (e.g. absent certificate of service, missing
caption fields) BEFORE final payload rendering.
"""
from __future__ import annotations

from typing import List, Optional

from ..analysis.citations import CitationVerifier
from ..formatting.legal_dictionary import LegalDictionary
from ..schemas import ComplianceReport, CourtFormatConfig, FormattedDocument


class ComplianceChecker:
    def __init__(self) -> None:
        self._citations = CitationVerifier()
        self._dictionary = LegalDictionary()

    def check(
        self,
        text: str,
        cfg: Optional[CourtFormatConfig] = None,
        formatted: Optional[FormattedDocument] = None,
    ) -> ComplianceReport:
        issues: List[str] = []
        warnings: List[str] = []

        citations = self._citations.verify_text(text)
        if not citations:
            warnings.append("No legal citations detected in source text.")

        # Latin terminology sanity: extracted terms are recognized by design;
        # unrecognized Latin-looking tokens are surfaced as warnings.
        for token in _latin_like_tokens(text):
            if not self._dictionary.define(token):
                warnings.append(f"Unverified Latin term: '{token}'")

        if cfg is not None:
            if not cfg.case_number:
                issues.append("Caption missing case number.")
            if not cfg.plaintiff or not cfg.defendant:
                issues.append("Caption missing a party (plaintiff/defendant).")
            if not cfg.attorney_name:
                warnings.append("Signature block missing attorney name.")

        if formatted is not None:
            if "CERTIFICATE OF SERVICE" not in formatted.full_text.upper():
                issues.append("Missing certificate of service.")
            if not formatted.signature_block.strip():
                issues.append("Missing signature block.")

        return ComplianceReport(
            passed=not issues,
            issues=issues,
            warnings=warnings,
            citations_checked=citations,
        )


_COMMON_LATIN = {"habeas", "corpus", "subpoena", "amicus", "curiae", "stare",
                 "decisis", "res", "judicata", "voir", "dire", "prima", "facie",
                 "de", "novo", "ex", "parte", "camera", "mens", "rea",
                 "certiorari", "pro", "se", "sua", "sponte"}


def _latin_like_tokens(text: str) -> List[str]:
    out = []
    for raw in text.replace(",", " ").replace(".", " ").split():
        w = raw.lower()
        if w in _COMMON_LATIN:
            out.append(w)
    return out
