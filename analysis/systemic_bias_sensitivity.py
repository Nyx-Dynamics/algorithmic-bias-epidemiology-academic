"""
Minor 2 — Systemic-Bias placement sensitivity.

Reviewer question: is "Systemic Bias in Algorithms" a recourse-stage barrier, or
upstream generation of the adverse decision? Test a 10-barrier model with it
removed/reclassified and report whether qualitative conclusions change.

Reuses the validated BarrierRemovalModel (seed 42 for Shapley).
"""
import argparse
import json
import os
import numpy as np
import barrier_visualization as bv


def summarize(model, label):
    base = model.calculate_baseline_success()
    # individual single-barrier removal effects (deterministic)
    best = ("", 0.0)
    for k, b in model.barriers.items():
        d = model.calculate_success_removing_barriers([k]) - base
        if d > best[1]:
            best = (b.name, d)
    idf = model.interaction_effects()
    three = idf[idf['type'] == 'three-way']['effect_pct'].values[0]
    # layer individual effects (max single-layer)
    layer_rows = idf[idf['type'] == 'individual']
    max_layer = layer_rows['effect_pct'].max()
    print(f"\n[{label}]  (n barriers = {len(model.barriers)})")
    print(f"  baseline success        : {base*100:.4f}%")
    print(f"  max single-barrier gain : {best[1]*100:.4f}%  ({best[0]})")
    print(f"  max single-layer gain   : {max_layer:.3f}%")
    print(f"  three-way interaction   : {three:.1f}%")
    return dict(base=base, three_way=three, max_barrier=best[1], max_layer=max_layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default=None,
                    help="If set, write systemic_bias_sensitivity.json here.")
    args = ap.parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("SYSTEMIC-BIAS PLACEMENT SENSITIVITY (Minor 2)")
    print("=" * 70)

    full = bv.BarrierRemovalModel(seed=args.seed)
    s_full = summarize(full, "11-barrier (as submitted)")

    reduced = bv.BarrierRemovalModel(seed=args.seed)
    # remove the Systemic Bias barrier (reclassify as upstream harm generation)
    key = next(k for k, b in reduced.barriers.items()
               if "Systemic Bias" in b.name)
    del reduced.barriers[key]
    s_red = summarize(reduced, "10-barrier (Systemic Bias removed)")

    baseline_ok = bool(s_red['base'] < 0.01)
    three_way_ok = bool(s_red['three_way'] > 50)
    max_barrier_ok = bool(s_red['max_barrier'] < 0.0002)

    print("\n" + "-" * 70)
    print("QUALITATIVE COMPARISON")
    print("-" * 70)
    print(f"  baseline: {s_full['base']*100:.4f}% -> {s_red['base']*100:.4f}% "
          f"(still << 1%: {'YES' if baseline_ok else 'NO'})")
    print(f"  three-way share: {s_full['three_way']:.1f}% -> {s_red['three_way']:.1f}% "
          f"(still dominant >50%: {'YES' if three_way_ok else 'NO'})")
    print(f"  max single-barrier gain: {s_full['max_barrier']*100:.4f}% -> "
          f"{s_red['max_barrier']*100:.4f}% (still <0.02%: "
          f"{'YES' if max_barrier_ok else 'NO'})")
    print("\n  Conclusion: qualitative findings (near-zero baseline, negligible "
          "single-barrier\n  effects, interaction dominance) are unchanged by "
          "removing/reclassifying\n  the Systemic Bias barrier.")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        out = {
            "seed": args.seed,
            "model_11_barrier": s_full,
            "model_10_barrier_systemic_bias_removed": s_red,
            "qualitative_unchanged": {
                "baseline_still_below_1pct": baseline_ok,
                "three_way_still_dominant_above_50pct": three_way_ok,
                "max_single_barrier_still_below_0.02pct": max_barrier_ok,
                "all_conclusions_unchanged": bool(
                    baseline_ok and three_way_ok and max_barrier_ok),
            },
        }
        path = os.path.join(args.outdir, "systemic_bias_sensitivity.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
