"""
Run-manifest and software-versions writer for PDIG-D-26-00342 reproducibility.

Captures git SHA, timestamp, Python + package versions, seed, input-file
checksums, executed targets, generated output paths, and headline results into
outputs/reproducibility/run_manifest.json, plus a software_versions.txt.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))


def sha256(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "UNKNOWN"


def pkg_versions():
    vers = {}
    for name in ["numpy", "scipy", "pandas", "matplotlib"]:
        try:
            mod = __import__(name)
            vers[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            vers[name] = "not-installed"
    return vers


def headline(seed):
    from barrier_visualization import BarrierRemovalModel
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
        "three_way_interaction_share_pct": tw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=os.path.join(REPO, "outputs", "reproducibility", "run_manifest.json"))
    ap.add_argument("--versions-out", type=str,
                    default=os.path.join(REPO, "outputs", "reproducibility", "software_versions.txt"))
    ap.add_argument("--targets", nargs="*", default=[], help="Executed Make targets.")
    ap.add_argument("--inputs", nargs="*",
                    default=[os.path.join(REPO, "analysis", "barrier_definitions.csv")],
                    help="Input files to checksum.")
    ap.add_argument("--outputs", nargs="*", default=[], help="Generated output paths to record.")
    args = ap.parse_args()

    manifest = {
        "git_commit_sha": git_sha(),
        "run_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": pkg_versions(),
        "random_seed": args.seed,
        "input_checksums": {os.path.relpath(p, REPO): sha256(p) for p in args.inputs},
        "executed_targets": args.targets,
        "generated_outputs": [os.path.relpath(p, REPO) for p in args.outputs],
        "headline_results": headline(args.seed),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.versions_out)), exist_ok=True)
    with open(args.versions_out, "w") as fh:
        fh.write("Python %s\n" % sys.version.replace("\n", " "))
        fh.write("platform %s\n" % platform.platform())
        for k, v in manifest["package_versions"].items():
            fh.write("%s %s\n" % (k, v))

    print("-> %s" % args.out)
    print("-> %s" % args.versions_out)


if __name__ == "__main__":
    main()
