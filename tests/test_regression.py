"""
Regression tests for PDIG-D-26-00342 major revision.

Pins the deterministic headline anchors and asserts seeding determinism.
Run with:  pytest tests/test_regression.py -q
"""
import os
import sys

import numpy as np
import pytest

# Make analysis/ importable regardless of CWD.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(REPO, "analysis")
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)

from barrier_visualization import BarrierRemovalModel  # noqa: E402

SEED = 42

# ---- Frozen deterministic anchors (MUST NOT CHANGE) -------------------------
BASELINE_EXACT = 1.8009337500000003e-05
MAX_SINGLE_BARRIER_GAIN_PCT = 0.005402801250000001  # Legal Knowledge Gap
THREE_WAY_SHARE_PCT = 87.60303281625

PAIRWISE_EXPECTED = {
    "data_integration x data_accuracy": 0.436,
    "data_integration x institutional": 4.514,
    "data_accuracy x institutional": 7.026,
}
INDIVIDUAL_EXPECTED = {
    "data_integration": 0.022,
    "data_accuracy": 0.035,
    "institutional": 0.362,
}


@pytest.fixture
def model():
    return BarrierRemovalModel(seed=SEED)


def test_baseline_success(model):
    assert model.calculate_baseline_success() == pytest.approx(BASELINE_EXACT, rel=0, abs=1e-18)


def test_baseline_pct_headline(model):
    assert round(model.calculate_baseline_success() * 100, 4) == 0.0018


def test_max_single_barrier_gain(model):
    ind = model.individual_barrier_effects()
    row = ind.loc[ind["marginal_effect_pct"].idxmax()]
    assert row["barrier_name"] == "Legal Knowledge Gap"
    assert row["marginal_effect_pct"] == pytest.approx(MAX_SINGLE_BARRIER_GAIN_PCT, abs=1e-12)
    assert round(row["marginal_effect_pct"], 4) == 0.0054


def test_three_way_interaction_share(model):
    inter = model.interaction_effects()
    tw = inter[inter["type"] == "three-way"]["effect_pct"].values[0]
    assert tw == pytest.approx(THREE_WAY_SHARE_PCT, abs=1e-9)
    assert round(tw, 1) == 87.6


def test_pairwise_and_individual_interactions(model):
    inter = model.interaction_effects()
    for _, r in inter.iterrows():
        if r["type"] == "pairwise":
            assert r["effect_pct"] == pytest.approx(PAIRWISE_EXPECTED[r["term"]], abs=1e-2), r["term"]
        elif r["type"] == "individual":
            assert r["effect_pct"] == pytest.approx(INDIVIDUAL_EXPECTED[r["term"]], abs=1e-2), r["term"]


def test_shapley_determinism():
    s1 = BarrierRemovalModel(seed=SEED).shapley_values()
    s2 = BarrierRemovalModel(seed=SEED).shapley_values()
    assert (s1["shapley_value"].values == s2["shapley_value"].values).all()
    assert s1.iloc[0]["barrier_name"] == "Legal Knowledge Gap"


def test_shapley_sums_to_full_effect(model):
    baseline = model.calculate_baseline_success()
    full = model.calculate_success_removing_barriers(list(model.barriers.keys())) - baseline
    s = model.shapley_values()
    assert s["shapley_value"].sum() == pytest.approx(full, rel=1e-9)


def test_stepwise_random_determinism():
    a = BarrierRemovalModel(seed=SEED).stepwise_removal_strategies()["random"]
    b = BarrierRemovalModel(seed=SEED).stepwise_removal_strategies()["random"]
    assert np.array_equal(a["success_probability"].values, b["success_probability"].values)


def test_sensitivity_determinism_smoke():
    import importlib
    sa = importlib.import_module("sensitivity_analysis")
    model = sa.BarrierModel()
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)
    s1 = sa.SobolSensitivityAnalysis(model, n_samples=500, rng=rng1).calculate_indices()
    s2 = sa.SobolSensitivityAnalysis(model, n_samples=500, rng=rng2).calculate_indices()
    assert np.array_equal(s1["S1"].values, s2["S1"].values)
    assert np.array_equal(s1["ST"].values, s2["ST"].values)


def test_copula_independence_crosscheck():
    import importlib
    cr = importlib.import_module("copula_robustness")
    d0 = cr.analytic_series(cr.P)
    assert round(d0["base"] * 100, 4) == 0.0018
    assert round(d0["three_way_share"] * 100, 1) == 87.6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
