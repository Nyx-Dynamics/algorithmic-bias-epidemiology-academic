"""
Major-4 robustness: relax multiplicative independence.

(A) Gaussian-copula correlated barriers (R2 critical path): equicorrelation rho
    over 11 Bernoulli pass-events; also a block (within-layer) variant.
(B) Repeated-attempt topology (R1): p_i^(m) = 1-(1-p_i)^m, m=2,3 (analytic).

Reports, per topology: baseline all-pass P, max single-BARRIER gain, max single-LAYER
gain, and the three-way interaction SHARE using the manuscript's own decomposition
(baseline-relative factorial contrast). Seed = 42, common random numbers across rho.
"""
import numpy as np
from scipy.stats import norm

SEED = 42
N = 100_000

# 11 barriers: (name, layer, pass_prob)  -- from R2 handoff / barrier_definitions
BARRIERS = [
    ("Rapid Data Transmission", "L1", 0.30),
    ("Multi-System Integration", "L1", 0.55),
    ("Permanent Storage", "L1", 0.45),
    ("Error Detection Difficulty", "L2", 0.35),
    ("Correction Process Barriers", "L2", 0.35),
    ("Incomplete Correction Propagation", "L2", 0.40),
    ("Awareness Gap", "L3", 0.30),
    ("Record Access Barriers", "L3", 0.55),
    ("Legal Knowledge Gap", "L3", 0.25),
    ("Legal Resource Barriers", "L3", 0.40),
    ("Systemic Bias in Algorithms", "L3", 0.30),
]
P = np.array([b[2] for b in BARRIERS])
LAYERS = ["L1", "L2", "L3"]
LAYER_OF = np.array([b[1] for b in BARRIERS])
LAYER_IDX = {L: np.where(LAYER_OF == L)[0] for L in LAYERS}


def success_probs_from_passed(passed):
    """Given boolean pass matrix (N x 11), return f(S) for all 8 layer-removal sets."""
    f = {}
    for r in range(8):
        removed = [LAYERS[k] for k in range(3) if (r >> k) & 1]
        eff = passed.copy()
        for L in removed:
            eff[:, LAYER_IDX[L]] = True  # removed layer -> always pass
        f[frozenset(removed)] = eff.all(axis=1).mean()
    return f


def decompose(f):
    """Manuscript decomposition: baseline-relative individual/pairwise/three-way."""
    base = f[frozenset()]
    ind = {L: f[frozenset([L])] - base for L in LAYERS}
    pair_int = {}
    for a, b in [("L1", "L2"), ("L1", "L3"), ("L2", "L3")]:
        joint = f[frozenset([a, b])] - base
        pair_int[(a, b)] = joint - (ind[a] + ind[b])
    all_effect = f[frozenset(LAYERS)] - base  # = 1 - base
    three_way = all_effect - (sum(ind.values()) + sum(pair_int.values()))
    share = three_way / all_effect if all_effect > 0 else float("nan")
    return base, ind, pair_int, three_way, all_effect, share


def max_single_barrier_gain(passed, base):
    """Max gain from making exactly one barrier always-pass (Prop-1 linked)."""
    best = 0.0
    for i in range(11):
        eff = passed.copy()
        eff[:, i] = True
        best = max(best, eff.all(axis=1).mean() - base)
    return best


def run_copula(rho, E, block=False):
    """E: N x 11 standard normals (common random numbers). Returns metrics dict."""
    if block:
        Sigma = np.zeros((11, 11))
        for L in LAYERS:
            idx = LAYER_IDX[L]
            for i in idx:
                for j in idx:
                    Sigma[i, j] = 1.0 if i == j else rho
        np.fill_diagonal(Sigma, 1.0)
    else:
        Sigma = (1 - rho) * np.eye(11) + rho * np.ones((11, 11))
    Ltri = np.linalg.cholesky(Sigma)
    Z = E @ Ltri.T
    U = norm.cdf(Z)
    passed = U < P  # marginal pass prob = P exactly
    f = success_probs_from_passed(passed)
    base, ind, pair, tw, alleff, share = decompose(f)
    msb = max_single_barrier_gain(passed, base)
    msl = max(ind.values())
    return dict(rho=rho, base=base, three_way_share=share, three_way_abs=tw,
               max_single_barrier_gain=msb, max_single_layer_gain=msl,
               total_gain=alleff)


def analytic_series(pvec):
    """Independent serial baseline + decomposition (matches manuscript method)."""
    def succ(removed_layers):
        pp = pvec.copy()
        for L in removed_layers:
            pp[LAYER_IDX[L]] = 1.0
        return np.prod(pp)
    f = {}
    for r in range(8):
        removed = [LAYERS[k] for k in range(3) if (r >> k) & 1]
        f[frozenset(removed)] = succ(removed)
    base, ind, pair, tw, alleff, share = decompose(f)
    # max single barrier
    msb = 0.0
    for i in range(11):
        pp = pvec.copy(); pp[i] = 1.0
        msb = max(msb, np.prod(pp) - base)
    return dict(base=base, three_way_share=share, max_single_barrier_gain=msb,
                max_single_layer_gain=max(ind.values()), total_gain=alleff)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    E = rng.standard_normal((N, 11))  # common random numbers across all rho

    print("=" * 92)
    print("(A) GAUSSIAN COPULA (equicorrelation rho) — n=%d, seed=%d" % (N, SEED))
    print("=" * 92)
    print(f"{'rho':>5} {'baseline P':>13} {'3way share':>11} {'maxSingleBar':>13} "
          f"{'maxSingleLayer':>15} {'totalGain':>10}")
    rhos = [round(0.05 * k, 2) for k in range(11)]  # 0.0 .. 0.5
    for rho in rhos:
        m = run_copula(rho, E)
        print(f"{rho:>5.2f} {m['base']*100:>12.4f}% {m['three_way_share']*100:>10.1f}% "
              f"{m['max_single_barrier_gain']*100:>12.4f}% {m['max_single_layer_gain']*100:>14.2f}% "
              f"{m['total_gain']*100:>9.2f}%")

    print("\n(A') BLOCK COPULA (within-layer rho, cross-layer 0)")
    print(f"{'rho':>5} {'baseline P':>13} {'3way share':>11} {'maxSingleBar':>13}")
    for rho in rhos:
        m = run_copula(rho, E, block=True)
        print(f"{rho:>5.2f} {m['base']*100:>12.4f}% {m['three_way_share']*100:>10.1f}% "
              f"{m['max_single_barrier_gain']*100:>12.4f}%")

    print("\n" + "=" * 92)
    print("(B) REPEATED-ATTEMPT topology  p_i^(m) = 1-(1-p_i)^m  (analytic, independent)")
    print("=" * 92)
    print(f"{'m':>3} {'baseline P':>13} {'3way share':>11} {'maxSingleBar':>13} {'maxSingleLayer':>15}")
    for m_att in [1, 2, 3]:
        pv = 1 - (1 - P) ** m_att
        d = analytic_series(pv)
        print(f"{m_att:>3} {d['base']*100:>12.4f}% {d['three_way_share']*100:>10.1f}% "
              f"{d['max_single_barrier_gain']*100:>12.4f}% {d['max_single_layer_gain']*100:>14.2f}%")

    # independence cross-check vs manuscript
    d0 = analytic_series(P)
    print("\nIndependence cross-check (should match 0.0018%% / 87.6%%): "
          f"baseline={d0['base']*100:.4f}%, three_way_share={d0['three_way_share']*100:.1f}%")
