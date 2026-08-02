"""Expected-value baseline for a case (actuarial math).

    EV = P(win) * damages - P(loss) * litigation_cost
"""
from __future__ import annotations

from ..schemas import CaseFinancials


class ExpectedValueEngine:
    def __init__(self, financials: CaseFinancials) -> None:
        self.financials = financials

    def calculate_static_ev(self) -> float:
        p_win = self.financials.prob_win_baseline
        value = self.financials.damages_claimed
        cost = self.financials.litigation_cost
        return (p_win * value) - ((1.0 - p_win) * cost)
