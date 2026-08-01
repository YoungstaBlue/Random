"""Priority Assignment (Module 6).

Ranks processing tasks by procedural deadline proximity and bucket weight, so
deadline-driven filings surface first.
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from ..schemas import CategorizedPayload, PayloadKind, PriorityTask

# Base weight per bucket — procedural items carry deadlines and rank highest.
_KIND_WEIGHT = {
    PayloadKind.PROCEDURAL_RULE: 1.0,
    PayloadKind.STATUTORY_AUTHORITY: 0.7,
    PayloadKind.PRECEDENT: 0.5,
    PayloadKind.EVIDENCE: 0.3,
}


def assign_priorities(
    payload: CategorizedPayload,
    today: Optional[date] = None,
    deadlines: Optional[dict[str, date]] = None,
) -> List[PriorityTask]:
    today = today or date.today()
    deadlines = deadlines or {}
    tasks: List[PriorityTask] = []
    counter = 0

    buckets = [
        (PayloadKind.PROCEDURAL_RULE, payload.procedural_rules),
        (PayloadKind.STATUTORY_AUTHORITY, payload.statutory_authority),
        (PayloadKind.PRECEDENT, payload.precedent),
        (PayloadKind.EVIDENCE, payload.evidence),
    ]
    for kind, items in buckets:
        for item in items:
            counter += 1
            task_id = f"T{counter:04d}"
            deadline = deadlines.get(task_id) or deadlines.get(item)
            urgency = _urgency(today, deadline)
            score = round(_KIND_WEIGHT[kind] + urgency, 4)
            tasks.append(
                PriorityTask(task_id=task_id, kind=kind, summary=item[:120],
                             deadline=deadline, priority_score=score)
            )
    tasks.sort(key=lambda t: t.priority_score, reverse=True)
    return tasks


def _urgency(today: date, deadline: Optional[date]) -> float:
    if deadline is None:
        return 0.0
    days = (deadline - today).days
    if days <= 0:
        return 2.0            # overdue / due today — maximum urgency
    return round(1.0 / (1.0 + days), 4)
