"""
Baseline invariant verification for PDIG-D-26-00342.

Recomputes the frozen headline invariants directly from the seeded model,
asserts them, and (optionally) emits a machine-readable verification report.

Usage:
    python analysis/verify_baseline.py [--seed 42] [--json PATH] [--check-outputs DIR ...]
Exits non-zero if any invariant fails.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))

from barrier_visualization import BarrierRemovalModel  # noqa: E402

# Frozen anchors.
BASELINE_EXACT = 1.8009337500000003e-05
MAX_GAIN_PCT = 0.005402801250000001
THREE_WAY_SHARE_PCT = 87.60303281625


def compute(seed):
    m = BarrierRemovalModel(seed=seed)
    base = m.calculate_baseline_success()
    ind = m.individual_barrier_effects()
    row = ind.loc[ind["marginal_effect_pct"].idxmax()]
    inter = m.interaction_effects()
    tw = float(inter[inter["type"] == "three-way"]["effect_pct"].values[0])
    return {
        "baseline_success": float(base),
        "baseline_pct": float(base * 100),
        "max_single_barrier_gain_pct": float(row["marginal_effect_pct"]),
        "max_single_barrier_name": str(row["barrier_name"]),
        "three_way_interaction_share_pct": tw,
    }


def check(vals):
    results = []
    def add(name, ok, got, expected):
        results.append({"check": name, "pass": bool(ok), "got": got, "expected": expected})
    add("baseline_success_exact",
        abs(vals["baseline_success"] - BASELINE_EXACT) < 1e-18,
        vals["baseline_success"], BASELINE_EXACT)
    add("baseline_pct_0.0018",
        round(vals["baseline_pct"], 4) == 0.0018, round(vals["baseline_pct"], 4), 0.0018)
    add("max_single_barrier_gain_0.0054",
        round(vals["max_single_barrier_gain_pct"], 4) == 0.0054,
        round(vals["max_single_barrier_gain_pct"], 4), 0.0054)
    add("max_gain_barrier_is_legal_knowledge",
        vals["max_single_barrier_name"] == "Legal Knowledge Gap",
        vals["max_single_barrier_name"], "Legal Knowledge Gap")
    add("three_way_share_87.6",
        round(vals["three_way_interaction_share_pct"], 1) == 87.6,
        round(vals["three_way_interaction_share_pct"], 1), 87.6)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--check-outputs", nargs="*", default=[],
                    help="Paths that must exist for verification to pass.")
    args = ap.parse_args()

    vals = compute(args.seed)
    results = check(vals)

    missing = [p for p in args.check_outputs if not os.path.exists(p)]
    for p in args.check_outputs:
        results.append({"check": f"output_exists:{p}", "pass": os.path.exists(p),
                        "got": os.path.exists(p), "expected": True})

    all_pass = all(r["pass"] for r in results)
    report = {"seed": args.seed, "all_pass": all_pass, "values": vals, "checks": results}

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)

    print("BASELINE INVARIANT VERIFICATION (seed=%d)" % args.seed)
    for r in results:
        print("  [%s] %s  got=%s expected=%s" %
              ("PASS" if r["pass"] else "FAIL", r["check"], r["got"], r["expected"]))
    if args.json:
        print("  -> %s" % args.json)
    if missing:
        print("  MISSING OUTPUTS: %s" % ", ".join(missing))
    print("RESULT:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
