from base44.analysis.citations import CitationVerifier
from base44.formatting.court_rules import CourtRulesFormatter
from base44.formatting.legal_dictionary import LegalDictionary
from base44.schemas import CourtFormatConfig
from base44.workload.verify import ComplianceChecker


def test_court_formatter_includes_required_sections():
    cfg = CourtFormatConfig(plaintiff="A", defendant="B", case_number="1:23-cv-1",
                            attorney_name="Jane Roe")
    doc = CourtRulesFormatter().format(cfg, body="Argument text.")
    assert "CERTIFICATE OF SERVICE" in doc.full_text
    assert doc.signature_block
    assert "Case No. 1:23-cv-1" in doc.caption


def test_compliance_flags_missing_fields():
    cfg = CourtFormatConfig()  # no case number / parties
    report = ComplianceChecker().check("Some text with no citations.", cfg=cfg)
    assert report.passed is False
    assert any("case number" in i.lower() for i in report.issues)


def test_citation_classifies_puerto_rico():
    cites = CitationVerifier().extract("See 5 P.R. Offic. Trans. 1 and 411 U.S. 792.")
    jurisdictions = {c.jurisdiction for c in cites}
    assert any("Puerto Rico" in (j or "") for j in jurisdictions)
    assert "federal" in jurisdictions


def test_legal_dictionary_extracts_latin():
    found = LegalDictionary().extract("The court issued a writ of habeas corpus sua sponte.")
    assert "habeas corpus" in found
    assert "sua sponte" in found
