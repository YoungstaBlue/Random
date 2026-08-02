"""Multivariable calculus + stochastic modeling of the damages function.

The damages model D = f(evidence, witness, economic) is non-linear. Its gradient
∇f identifies the highest-leverage inputs (where a marginal improvement yields the
largest damages shift), and a Monte Carlo simulation characterizes the award's
distribution under variance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import approx_fprime

from ..schemas import CaseFinancials, GradientLeverage, MonteCarloSummary


class MultivariableCalculusEngine:
    def __init__(self, financials: CaseFinancials) -> None:
        self.financials = financials
        self.x0 = np.array([
            financials.evidence_strength,
            financials.witness_credibility,
            financials.economic_impact,
        ], dtype=float)

    @staticmethod
    def damages_function(x: np.ndarray) -> float:
        """Non-linear damages model (exponential in evidence, log in economics)."""
        evidence, witness, economic = float(x[0]), float(x[1]), float(x[2])
        return (500_000 * (max(evidence, 0.0) ** 1.5)
                + 250_000 * witness
                + 100_000 * np.log1p(max(economic, 0.0)))

    def calculate_gradient_leverage(self) -> GradientLeverage:
        epsilon = np.sqrt(np.finfo(float).eps)
        grad = approx_fprime(self.x0, self.damages_function, epsilon)
        return GradientLeverage(
            d_evidence=float(grad[0]),
            d_witness=float(grad[1]),
            d_economic=float(grad[2]),
        )

    def run_monte_carlo_simulation(
        self, iterations: int = 10_000, volatility: float = 0.2,
        seed: Optional[int] = None,
    ) -> MonteCarloSummary:
        rng = np.random.default_rng(seed)
        mean_damages = self.damages_function(self.x0)
        scale = abs(mean_damages) * volatility
        samples = rng.normal(loc=mean_damages, scale=scale, size=iterations)
        return MonteCarloSummary(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            p5=float(np.percentile(samples, 5)),
            p50=float(np.percentile(samples, 50)),
            p95=float(np.percentile(samples, 95)),
            iterations=iterations,
        )
