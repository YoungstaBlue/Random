"""Game-theoretic settlement model.

Builds a 2x2 payoff tensor over Plaintiff strategies {Settle, Trial} and Defendant
strategies {Pay, Fight} and finds a pure-strategy Nash equilibrium (mutual best
responses). No pure NE => caller should fall back to a mixed strategy.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..schemas import NashEquilibrium
from .expected_value import ExpectedValueEngine

_PLAINTIFF = ("Settle", "Trial")
_DEFENDANT = ("Pay", "Fight")


class StrategicPayoffMatrix:
    def __init__(self, ev_engine: ExpectedValueEngine, settlement_ratio: float = 0.90) -> None:
        self.ev_engine = ev_engine
        self.settlement_ratio = settlement_ratio

    def build_payoff_tensor(self) -> np.ndarray:
        ev_trial = self.ev_engine.calculate_static_ev()
        cost = self.ev_engine.financials.litigation_cost
        settlement = ev_trial * self.settlement_ratio

        # Shape (plaintiff, defendant, player): player 0 = plaintiff, 1 = defendant.
        payoffs = np.zeros((2, 2, 2))
        payoffs[0, 0] = [settlement, -settlement]              # Settle / Pay
        payoffs[0, 1] = [-(cost * 0.1), 0.0]                   # Settle / Fight (bluff called)
        payoffs[1, 0] = [ev_trial, -ev_trial]                  # Trial / Pay (defendant folds)
        payoffs[1, 1] = [ev_trial - cost, -ev_trial - cost]    # Trial / Fight (mutual cost)
        return payoffs

    def find_pure_nash_equilibrium(self, payoffs: np.ndarray) -> Optional[NashEquilibrium]:
        for i in range(2):
            for j in range(2):
                p1 = payoffs[i, j, 0]
                p2 = payoffs[i, j, 1]
                p1_best = all(p1 >= payoffs[r, j, 0] for r in range(2))
                p2_best = all(p2 >= payoffs[i, c, 1] for c in range(2))
                if p1_best and p2_best:
                    return NashEquilibrium(
                        plaintiff_strategy=_PLAINTIFF[i],
                        defendant_strategy=_DEFENDANT[j],
                        plaintiff_payoff=float(p1),
                        defendant_payoff=float(p2),
                        is_pure=True,
                    )
        return None
