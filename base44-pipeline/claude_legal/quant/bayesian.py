"""Bayesian updating of win probability as new evidence arrives.

    P(H|E) = P(E|H)P(H) / [P(E|H)P(H) + P(E|~H)P(~H)]
"""
from __future__ import annotations


class BayesianInferenceNetwork:
    def __init__(self, initial_prior: float) -> None:
        self.prior = float(initial_prior)

    def update_posterior(self, likelihood_given_true: float,
                         likelihood_given_false: float) -> float:
        numerator = likelihood_given_true * self.prior
        denominator = numerator + likelihood_given_false * (1.0 - self.prior)
        if denominator == 0:
            return self.prior
        self.prior = numerator / denominator
        return self.prior
