"""
COPULA / ALTERNATIVE-TOPOLOGY ROBUSTNESS (Major 4)
Algorithmic Bias Epidemiology Framework

Relaxes the multiplicative-independence assumption of the baseline serial model:

  (A)  Gaussian-copula equicorrelation over the 11 Bernoulli pass-events, sweeping
       the correlation rho. Marginal pass probabilities are preserved exactly.
  (A') Block (within-layer) copula variant: correlation only within a layer.
  (B)  Repeated-attempt topology  p_i^(m) = 1 - (1 - p_i)^m  (analytic, m = 1,2,3).

For every topology / rho we report the manuscript-style baseline-relative factorial
decomposition: baseline all-pass probability, max single-BARRIER gain, max single-LAYER
gain, three-way interaction SHARE, and total achievable gain.

Common random numbers (one standard-normal draw reused across all rho) make the sweep
smooth and reproducible. Deterministic via numpy.random.default_rng(seed).

Emits:
  <outdir>/copula_sweep.csv                    (equicorrelation + block + repeated-attempt)
  <figdir>/FigX_copula_robustness.<fmt>        (baseline P and 3-way share vs rho)

Adapted from analysis/alternative_topology.py (same logic, CLI-wrapped).

Author: AC Demidont, DO
Nyx Dynamics LLC
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

DEFAULT_SEED = 42
DEFAULT_N = 100_000

# 11 barriers: (name, layer, pass_prob) -- from barrier_definitions.csv
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
    msb = 0.0
    for i in range(11):
        pp = pvec.copy(); pp[i] = 1.0
        msb = max(msb, np.prod(pp) - base)
    return dict(base=base, three_way_share=share, max_single_barrier_gain=msb,
                max_single_layer_gain=max(ind.values()), total_gain=alleff)


def frange(lo, hi, step):
    """Inclusive float range (robust to fp accumulation)."""
    n = int(round((hi - lo) / step))
    return [round(lo + step * k, 6) for k in range(n + 1)]


def parse_args():
    p = argparse.ArgumentParser(
        description="Major-4 copula / alternative-topology robustness sweep (seeded)."
    )
    p.add_argument('--seed', type=int, default=DEFAULT_SEED,
                   help=f'Random seed for common random numbers (default {DEFAULT_SEED}).')
    p.add_argument('--rho-min', type=float, default=0.0, help='Minimum equicorrelation rho.')
    p.add_argument('--rho-max', type=float, default=0.5, help='Maximum equicorrelation rho.')
    p.add_argument('--rho-step', type=float, default=0.05, help='rho sweep step.')
    p.add_argument('--n', type=int, default=DEFAULT_N, help='Monte Carlo sample size.')
    p.add_argument('--outdir', type=str, default='.', help='Directory for copula_sweep.csv.')
    p.add_argument('--figdir', type=str, default='.', help='Directory for the figure.')
    p.add_argument('--fig-format', type=str, default='png',
                   help='Figure format extension (default png; tif/tiff/eps ok).')
    p.add_argument('--dpi', type=int, default=300, help='Figure resolution (default 300).')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)
    ext = args.fig_format.lower()
    if ext == 'tif':
        ext = 'tiff'

    rng = np.random.default_rng(args.seed)
    E = rng.standard_normal((args.n, 11))  # common random numbers across all rho
    rhos = frange(args.rho_min, args.rho_max, args.rho_step)

    rows = []

    print("=" * 92)
    print("(A) GAUSSIAN COPULA (equicorrelation rho) — n=%d, seed=%d" % (args.n, args.seed))
    print("=" * 92)
    print(f"{'rho':>5} {'baseline P':>13} {'3way share':>11} {'maxSingleBar':>13} "
          f"{'maxSingleLayer':>15} {'totalGain':>10}")
    for rho in rhos:
        m = run_copula(rho, E)
        rows.append(dict(topology='copula_equicorrelation', param=rho, **m))
        print(f"{rho:>5.2f} {m['base']*100:>12.4f}% {m['three_way_share']*100:>10.1f}% "
              f"{m['max_single_barrier_gain']*100:>12.4f}% {m['max_single_layer_gain']*100:>14.2f}% "
              f"{m['total_gain']*100:>9.2f}%")

    print("\n(A') BLOCK COPULA (within-layer rho, cross-layer 0)")
    print(f"{'rho':>5} {'baseline P':>13} {'3way share':>11} {'maxSingleBar':>13}")
    for rho in rhos:
        m = run_copula(rho, E, block=True)
        rows.append(dict(topology='copula_block', param=rho, **m))
        print(f"{rho:>5.2f} {m['base']*100:>12.4f}% {m['three_way_share']*100:>10.1f}% "
              f"{m['max_single_barrier_gain']*100:>12.4f}%")

    print("\n" + "=" * 92)
    print("(B) REPEATED-ATTEMPT topology  p_i^(m) = 1-(1-p_i)^m  (analytic, independent)")
    print("=" * 92)
    print(f"{'m':>3} {'baseline P':>13} {'3way share':>11} {'maxSingleBar':>13} {'maxSingleLayer':>15}")
    for m_att in [1, 2, 3]:
        pv = 1 - (1 - P) ** m_att
        d = analytic_series(pv)
        rows.append(dict(topology='repeated_attempt', param=m_att,
                         rho=np.nan, base=d['base'],
                         three_way_share=d['three_way_share'],
                         three_way_abs=np.nan,
                         max_single_barrier_gain=d['max_single_barrier_gain'],
                         max_single_layer_gain=d['max_single_layer_gain'],
                         total_gain=d['total_gain']))
        print(f"{m_att:>3} {d['base']*100:>12.4f}% {d['three_way_share']*100:>10.1f}% "
              f"{d['max_single_barrier_gain']*100:>12.4f}% {d['max_single_layer_gain']*100:>14.2f}%")

    # independence cross-check vs manuscript
    d0 = analytic_series(P)
    print("\nIndependence cross-check (should match 0.0018%% / 87.6%%): "
          f"baseline={d0['base']*100:.4f}%, three_way_share={d0['three_way_share']*100:.1f}%")

    # ---- Write CSV ----------------------------------------------------------
    df = pd.DataFrame(rows, columns=[
        'topology', 'param', 'rho', 'base', 'three_way_share', 'three_way_abs',
        'max_single_barrier_gain', 'max_single_layer_gain', 'total_gain'])
    csv_path = os.path.join(args.outdir, 'copula_sweep.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n-> {csv_path}")

    # ---- Figure -------------------------------------------------------------
    eq = df[df.topology == 'copula_equicorrelation']
    bl = df[df.topology == 'copula_block']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(eq['param'], eq['base'] * 100, 'o-', color='#2c7fb8',
             linewidth=2, markersize=7, label='Equicorrelation')
    ax1.plot(bl['param'], bl['base'] * 100, 's--', color='#31a354',
             linewidth=2, markersize=6, label='Block (within-layer)')
    ax1.axhline(d0['base'] * 100, color='#e34a33', linestyle=':',
                label='Independent baseline (0.0018%)')
    ax1.set_xlabel(r'Copula correlation $\rho$', fontsize=13)
    ax1.set_ylabel('Baseline all-pass probability (%)', fontsize=13)
    ax1.set_title('A. Baseline recourse probability vs correlation',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(eq['param'], eq['three_way_share'] * 100, 'o-', color='#2c7fb8',
             linewidth=2, markersize=7, label='Equicorrelation')
    ax2.plot(bl['param'], bl['three_way_share'] * 100, 's--', color='#31a354',
             linewidth=2, markersize=6, label='Block (within-layer)')
    ax2.axhline(d0['three_way_share'] * 100, color='#e34a33', linestyle=':',
                label='Independent (87.6%)')
    ax2.set_xlabel(r'Copula correlation $\rho$', fontsize=13)
    ax2.set_ylabel('Three-way interaction share (%)', fontsize=13)
    ax2.set_title('B. Three-way interaction share vs correlation',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.figdir, f'FigX_copula_robustness.{ext}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"-> {fig_path}")


if __name__ == "__main__":
    main()
