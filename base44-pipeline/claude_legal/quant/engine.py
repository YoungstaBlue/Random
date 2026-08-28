"""Quantitative Strategy Engine — orchestrates the litigation-analytics modules.

Combines expected value, Bayesian updating, calculus leverage, Monte Carlo, and
game theory (plus optional corpus anomaly detection) into a single
QuantitativeStrategyReport with a settlement recommendation.
"""
from __future__ import annotations

from typing import Optional

from ..schemas import CaseFinancials, QuantitativeStrategyReport
from ..vectorize.sparse_vectorizer import VectorizedCorpus
from .anomaly import VectorAnomalyEngine
from .bayesian import BayesianInferenceNetwork
from .calculus import MultivariableCalculusEngine
from .expected_value import ExpectedValueEngine
from .game_theory import StrategicPayoffMatrix


class QuantitativeStrategyEngine:
    def __init__(self, financials: CaseFinancials) -> None:
        self.financials = financials
        self.ev_engine = ExpectedValueEngine(financials)
        self.calc_engine = MultivariableCalculusEngine(financials)
        self.game = StrategicPayoffMatrix(self.ev_engine)
        self.bayes = BayesianInferenceNetwork(financials.prob_win_baseline)

    def analyze(
        self,
        corpus: Optional[VectorizedCorpus] = None,
        mc_iterations: int = 10_000,
        mc_seed: Optional[int] = None,
        new_evidence_likelihoods: tuple[float, float] = (0.85, 0.20),
    ) -> QuantitativeStrategyReport:
        static_ev = self.ev_engine.calculate_static_ev()
        gradient = self.calc_engine.calculate_gradient_leverage()
        monte_carlo = self.calc_engine.run_monte_carlo_simulation(
            iterations=mc_iterations, seed=mc_seed
        )
        payoffs = self.game.build_payoff_tensor()
        nash = self.game.find_pure_nash_equilibrium(payoffs)
        posterior = self.bayes.update_posterior(*new_evidence_likelihoods)

        anomalies = []
        if corpus is not None and corpus.n_docs > 1:
            anomalies = VectorAnomalyEngine(corpus).identify_anomalies(top_k=10)

        return QuantitativeStrategyReport(
            static_expected_value=round(static_ev, 2),
            bayesian_win_probability=round(posterior, 6),
            gradient_leverage=gradient,
            monte_carlo=monte_carlo,
            nash_equilibrium=nash,
            anomalies=anomalies,
            settlement_recommendation=self._recommend(static_ev, posterior, nash),
        )

    @staticmethod
    def _recommend(ev: float, posterior: float, nash) -> str:
        stance = nash.plaintiff_strategy if nash else "Mixed strategy"
        if posterior >= 0.66 and ev > 0:
            lean = "Strong position; anchor negotiations near full expected value."
        elif posterior >= 0.5:
            lean = "Favorable but contested; a discounted settlement de-risks variance."
        else:
            lean = "Weak position; prioritize an early settlement to cap downside."
        return f"Game-theoretic stance: {stance}. {lean}"
