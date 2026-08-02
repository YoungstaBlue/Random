"""Quantitative Strategy tier — litigation analytics for base44.

Turns case financials + the Block A vector corpus into a decision-support report:
expected value, Bayesian win-probability updating, multivariable-calculus leverage
points, Monte Carlo damage distribution, game-theoretic settlement equilibrium,
and cosine-similarity anomaly detection over the e-discovery corpus.
"""
from .anomaly import VectorAnomalyEngine
from .bayesian import BayesianInferenceNetwork
from .calculus import MultivariableCalculusEngine
from .engine import QuantitativeStrategyEngine
from .expected_value import ExpectedValueEngine
from .game_theory import StrategicPayoffMatrix

__all__ = [
    "QuantitativeStrategyEngine",
    "ExpectedValueEngine",
    "MultivariableCalculusEngine",
    "StrategicPayoffMatrix",
    "BayesianInferenceNetwork",
    "VectorAnomalyEngine",
]
