from datetime import date, timedelta

from claude_legal.workload.categorize import WorkloadCategorizer
from claude_legal.workload.priority import assign_priorities


def test_categorize_into_four_buckets():
    text = (
        "Plaintiff was terminated after filing a complaint. "
        "See 42 U.S.C. 2000e for the governing statute. "
        "The court applied McDonnell Douglas v. Green as precedent. "
        "A motion to dismiss under Rule 12 has a filing deadline."
    )
    cat = WorkloadCategorizer().categorize(text)
    assert cat.statutory_authority
    assert cat.precedent
    assert cat.procedural_rules


def test_priority_prefers_near_deadlines():
    from claude_legal.schemas import CategorizedPayload

    cat = CategorizedPayload(procedural_rules=["file response", "schedule hearing"])
    today = date(2025, 1, 1)
    tasks = assign_priorities(cat, today=today,
                              deadlines={"T0001": today + timedelta(days=1)})
    assert tasks[0].priority_score >= tasks[-1].priority_score
