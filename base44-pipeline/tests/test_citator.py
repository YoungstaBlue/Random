from claude_legal.analysis.citations import CitationVerifier
from claude_legal.analysis.citator import CourtListenerCitator
from claude_legal.config import Settings
from claude_legal.schemas import Citation


def _settings(**overrides):
    return Settings(_env_file=None, **overrides)


def test_citator_disabled_by_default_is_offline_safe():
    citator = CourtListenerCitator(_settings())
    assert citator.enabled is False

    citation = Citation(raw="347 U.S. 483")
    out = citator.resolve([citation])
    assert out[0].resolved is None  # untouched — no network call made


def test_citator_resolve_no_ops_when_disabled_via_citation_verifier():
    verifier = CitationVerifier(_settings())
    cites = verifier.verify_text("See 347 U.S. 483 and 410 F.3d 1200.")
    assert cites
    assert all(c.resolved is None for c in cites)


def test_citator_matches_populate_citation(monkeypatch):
    citator = CourtListenerCitator(_settings(courtlistener_enabled=True))

    def fake_lookup(self, text):
        return [{
            "citation": text,
            "status": 200,
            "clusters": [{
                "id": 108713,
                "case_name": "Brown v. Board of Education",
                "absolute_url": "/opinion/108713/brown-v-board-of-education/",
                "citation_count": 4321,
            }],
        }]

    monkeypatch.setattr(CourtListenerCitator, "_lookup", fake_lookup)

    citation = Citation(raw="347 U.S. 483", normalized="347 U.S. 483")
    out = citator.resolve([citation])[0]

    assert out.resolved is True
    assert out.cluster_id == 108713
    assert out.case_name == "Brown v. Board of Education"
    assert out.citation_count == 4321
    assert out.courtlistener_url == (
        "https://www.courtlistener.com/opinion/108713/brown-v-board-of-education/"
    )


def test_citator_no_match_marks_unresolved(monkeypatch):
    citator = CourtListenerCitator(_settings(courtlistener_enabled=True))
    monkeypatch.setattr(
        CourtListenerCitator, "_lookup",
        lambda self, text: [{"citation": text, "status": 404, "clusters": []}],
    )

    citation = Citation(raw="999 Made.Up 1")
    out = citator.resolve([citation])[0]

    assert out.resolved is False
    assert out.notes and "no match" in out.notes.lower()


def test_citator_network_error_is_caught_and_noted(monkeypatch):
    citator = CourtListenerCitator(_settings(courtlistener_enabled=True))

    def boom(self, text):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(CourtListenerCitator, "_lookup", boom)

    citation = Citation(raw="347 U.S. 483")
    out = citator.resolve([citation])[0]

    assert out.resolved is None
    assert out.notes and "citator error" in out.notes.lower()
