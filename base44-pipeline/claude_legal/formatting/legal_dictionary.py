"""Legal Dictionary Engine (Module 5).

Extracts legal Latin terminology from text and defines it. Ships with a compact
built-in glossary; a larger reference (e.g. a Black's Law Dictionary data file)
can be supplied via ``extra_terms`` without code changes.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

_BUILTIN: Dict[str, str] = {
    "habeas corpus": "A writ requiring a person under arrest to be brought before "
                     "a judge, testing the legality of the detention.",
    "subpoena": "A writ ordering a person to attend and testify or produce evidence.",
    "amicus curiae": "'Friend of the court'; a non-party who offers information "
                     "bearing on the case.",
    "stare decisis": "The doctrine that courts follow precedent in deciding cases.",
    "res judicata": "A matter already judged; bars re-litigation of the same claim.",
    "voir dire": "Preliminary examination of prospective jurors or witnesses.",
    "prima facie": "On its face; sufficient to establish a fact unless rebutted.",
    "de novo": "Anew; a standard of review deciding the matter without deference.",
    "ex parte": "By or for one party, without notice to the other.",
    "in camera": "In private (in a judge's chambers), out of public view.",
    "mens rea": "The mental element (guilty mind) of a crime.",
    "certiorari": "A writ by which a higher court reviews a lower court's decision.",
    "pro se": "Representing oneself without a lawyer.",
    "sua sponte": "Of its own accord; action taken by a court without a party's request.",
}


class LegalDictionary:
    def __init__(self, extra_terms: Optional[Dict[str, str]] = None) -> None:
        self.terms = dict(_BUILTIN)
        if extra_terms:
            self.terms.update({k.lower(): v for k, v in extra_terms.items()})
        # Longest-first so multi-word terms match before single words.
        self._patterns = sorted(self.terms, key=len, reverse=True)

    def define(self, term: str) -> Optional[str]:
        return self.terms.get(term.lower())

    def extract(self, text: str) -> Dict[str, str]:
        """Return {term: definition} for every glossary term present in text."""
        found: Dict[str, str] = {}
        lowered = text.lower()
        for term in self._patterns:
            if re.search(r"\b" + re.escape(term) + r"\b", lowered):
                found[term] = self.terms[term]
        return found

    def annotate(self, text: str) -> List[str]:
        return [f"{t}: {d}" for t, d in self.extract(text).items()]
