"""
Cross-check the custom Sobol sensitivity implementation against SALib.

The manuscript pipeline uses a self-contained Monte Carlo A/B (Saltelli/Jansen)
Sobol estimator in analysis/sensitivity_analysis.py rather than depending on
SALib. This test validates that the custom estimator agrees with SALib's
reference implementation on the same multiplicative barrier model, so the
"Sobol indices" claim is externally corroborated rather than self-referential.

SALib is an OPTIONAL, test-only dependency. If it (or numpy) is unavailable the
whole module is skipped, so `make test` still passes in a minimal environment.
"""
import os
import sys

import numpy as np
import pytest

SALib = pytest.importorskip("SALib", reason="SALib not installed; crosscheck skipped")
from SALib.sample import saltelli as salib_saltelli  # noqa: E402
from SALib.analyze import sobol as salib_sobol  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(REPO, "analysis")
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)

import sensitivity_analysis as sa  # noqa: E402

SEED = 42


def _bounds(model, bounds_factor=0.3):
    base = model.barrier_probs
    lower = np.maximum(0.05, base * (1 - bounds_factor))
    upper = np.minimum(0.95, base * (1 + bounds_factor))
    return lower, upper


def test_custom_sobol_matches_salib_ranking():
    """
    The custom estimator and SALib should agree on which barriers carry the
    most first-order variance (rank agreement), and total-order indices should
    dominate first-order ones (interaction-heavy model) in both.
    """
    model = sa.BarrierModel()
    n = model.n_barriers
    lower, upper = _bounds(model)

    # ---- Custom estimator (our pipeline) -----------------------------------
    rng = np.random.default_rng(SEED)
    custom = sa.SobolSensitivityAnalysis(model, n_samples=8192, rng=rng)
    custom_df = custom.calculate_indices(bounds_factor=0.3)
    s1_custom = custom_df["S1"].to_numpy()
    st_custom = custom_df["ST"].to_numpy()

    # ---- SALib reference ---------------------------------------------------
    problem = {
        "num_vars": n,
        "names": [f"b{i}" for i in range(n)],
        "bounds": list(zip(lower.tolist(), upper.tolist())),
    }
    # Deterministic SALib sample.
    try:
        X = salib_saltelli.sample(problem, 1024, calc_second_order=False, seed=SEED)
    except TypeError:  # older SALib without seed kwarg
        np.random.seed(SEED)
        X = salib_saltelli.sample(problem, 1024, calc_second_order=False)
    Y = np.array([model.calculate_success(x) for x in X])
    Si = salib_sobol.analyze(problem, Y, calc_second_order=False, seed=SEED)
    s1_salib = np.array(Si["S1"])
    st_salib = np.array(Si["ST"])

    # Both methods: total-order should on aggregate exceed first-order
    # (strong interaction structure of the multiplicative model).
    assert st_custom.sum() > s1_custom.sum()
    assert st_salib.sum() > s1_salib.sum()

    # Rank agreement on the top-3 most influential barriers (first-order).
    top_custom = set(np.argsort(s1_custom)[-3:])
    top_salib = set(np.argsort(s1_salib)[-3:])
    assert len(top_custom & top_salib) >= 2, (
        f"Top-3 S1 ranking disagreement: custom={top_custom} salib={top_salib}"
    )
