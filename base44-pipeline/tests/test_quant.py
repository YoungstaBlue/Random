import math

from base44.quant.engine import QuantitativeStrategyEngine
from base44.quant.expected_value import ExpectedValueEngine
from base44.schemas import CaseFinancials


def _fin(**kw):
    base = dict(damages_claimed=1_500_000.0, litigation_cost=200_000.0,
               evidence_strength=0.8, witness_credibility=0.6,
               economic_impact=1.2, prob_win_baseline=0.65)
    base.update(kw)
    return CaseFinancials(**base)


def test_expected_value_formula():
    fin = _fin()
    ev = ExpectedValueEngine(fin).calculate_static_ev()
    expected = 0.65 * 1_500_000 - 0.35 * 200_000
    assert math.isclose(ev, expected)


def test_gradient_has_three_finite_components():
    report = QuantitativeStrategyEngine(_fin()).analyze(mc_iterations=1000, mc_seed=1)
    g = report.gradient_leverage
    assert all(math.isfinite(v) for v in (g.d_evidence, g.d_witness, g.d_economic))
    # Evidence enters with exponent 1.5 -> its marginal leverage should be largest.
    assert g.d_evidence > g.d_witness


def test_monte_carlo_is_reproducible_and_ordered():
    eng = QuantitativeStrategyEngine(_fin())
    r1 = eng.analyze(mc_iterations=5000, mc_seed=42).monte_carlo
    r2 = QuantitativeStrategyEngine(_fin()).analyze(mc_iterations=5000, mc_seed=42).monte_carlo
    assert r1.mean == r2.mean                 # same seed -> deterministic
    assert r1.p5 <= r1.p50 <= r1.p95          # percentiles ordered


def test_bayesian_update_raises_win_probability_with_strong_evidence():
    report = QuantitativeStrategyEngine(_fin(prob_win_baseline=0.65)).analyze(
        mc_iterations=500, mc_seed=1, new_evidence_likelihoods=(0.85, 0.20))
    assert report.bayesian_win_probability > 0.65


def test_nash_equilibrium_is_valid():
    report = QuantitativeStrategyEngine(_fin()).analyze(mc_iterations=500, mc_seed=1)
    ne = report.nash_equilibrium
    if ne is not None:
        assert ne.plaintiff_strategy in {"Settle", "Trial"}
        assert ne.defendant_strategy in {"Pay", "Fight"}
    assert report.settlement_recommendation
