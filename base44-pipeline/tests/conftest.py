import pytest

from base44.schemas import RawDocument


@pytest.fixture
def corpus():
    return [
        RawDocument(doc_id="d1", text="The plaintiff alleges wrongful termination "
                    "and retaliation for reporting unsafe working conditions."),
        RawDocument(doc_id="d2", text="McDonnell Douglas burden shifting framework "
                    "governs Title VII discrimination claims and pretext analysis."),
        RawDocument(doc_id="d3", text="A motion to dismiss under Rule 12 tests the "
                    "sufficiency of the complaint and requires a certificate of service."),
        RawDocument(doc_id="d4", text="The defendant filed an answer denying the "
                    "allegations and asserting a reduction in force defense."),
    ]
