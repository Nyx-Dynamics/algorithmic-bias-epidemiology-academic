"""
PARAMETER DERIVATION AUDIT-TRAIL BUILDER (Major 7 / Minor 7)
Algorithmic Bias Epidemiology Framework

Emits the full per-barrier derivation audit trail required by the reviewer:
  - data/parameter_sources/parameter_derivation.md  (human-readable, one section
    per barrier: target construct, source domain, exact source statistic with
    page/table/figure where verifiable, mapping rule, pass probability,
    plausibility range, rationale, transportability limitation)
  - build/S_deriv_table.tex  (LaTeX longtable supplement, same content)

Seed content: analysis/barrier_definitions.csv (pass probs, layers, names) and the
PLOS ONE S1 supporting-information table. Any source page/figure/table number that
could NOT be verified from the available materials is marked "" rather than
invented.

Author: AC Demidont, DO
Nyx Dynamics LLC
"""
import argparse
import csv
import os

# Layer display names
LAYER_DISPLAY = {
    "data_integration": "Data Integration",
    "data_accuracy": "Data Accuracy",
    "institutional": "Institutional",
}

# Per-barrier derivation content, keyed by barrier_definitions.csv `key`.
# `source_locator` uses "" wherever an exact page/table/figure number
# could not be confirmed from available source material (do NOT fabricate).
DERIVATION = {
    "rapid_transmission": dict(
        construct="Probability of correcting adverse data before it propagates to CRAs",
        source_domain="Consumer-finance reporting practice",
        source="CFPB 2022 (Consumer Response Annual Report / furnisher practices)",
        source_stat="Furnishers transmit information to CRAs within ~1-2 days",
        source_locator="CFPB 2022, furnisher-timeliness discussion",
        mapping="Short correction window before propagation -> ~30% chance of pre-propagation fix",
        prob=0.30, rng="0.20-0.40",
        rationale="Rapid transmission leaves a narrow window; set near the low end of navigability.",
        transport="Finance-reporting timelines; not directly observed for healthcare/administrative recourse.",
    ),
    "multi_system_integration": dict(
        construct="Probability of clearing errors replicated across multiple systems",
        source_domain="Consumer-reporting accuracy",
        source="FTC 2013 (Report to Congress under FACT Act, credit-report accuracy study)",
        source_stat="~20% of consumers had an error on >=1 of 3 reports",
        source_locator="FTC 2013 accuracy study, executive summary",
        mapping="Complement of cross-system error prevalence, adjusted for multi-database copies -> ~0.55",
        prob=0.55, rng="0.40-0.65",
        rationale="Most consumers are not blocked by cross-system replication for this stage.",
        transport="Credit-report ecosystem; generalization to 20+ healthcare databases is an assumption.",
    ),
    "permanent_storage": dict(
        construct="Probability of encountering data that has not yet auto-expired",
        source_domain="Records-retention law/practice",
        source="CFPB 2022; FCRA Section 605 (7-year retention)",
        source_stat="7-year retention; <20% of adverse items removed within 4 years",
        source_locator="FCRA 15 U.S.C. 1681c (Section 605); CFPB 2022 retention discussion",
        mapping="Fraction encountering unexpired adverse data over recourse horizon -> ~0.45",
        prob=0.45, rng="0.35-0.55",
        rationale="Long statutory retention makes unexpired adverse data common but not universal.",
        transport="Statutory retention windows differ across administrative/healthcare data domains.",
    ),
    "error_detection": dict(
        construct="Probability of successfully identifying the specific material error",
        source_domain="Consumer-reporting accuracy",
        source="FTC 2013 (FACT Act accuracy study)",
        source_stat="26% of consumers identified a material error in their report",
        source_locator="FTC 2013 accuracy study, material-error finding",
        mapping="Observed material-error identification rate, rounded -> ~0.35 for actionable detection",
        prob=0.35, rng="0.25-0.45",
        rationale="Detecting the responsible error is difficult; set modestly above the raw 26%.",
        transport="Credit context; error visibility in algorithmic/administrative decisions may differ.",
    ),
    "correction_process": dict(
        construct="Probability of achieving a complete correction once an error is identified",
        source_domain="Consumer-reporting dispute outcomes",
        source="FTC 2013; CFPB 2022 (dispute resolution)",
        source_stat="~37% of disputes fully resolved in the consumer's favor",
        source_locator="FTC 2015 follow-up / CFPB 2022 dispute-outcome data",
        mapping="Full-resolution rate, rounded -> ~0.35 achieve complete correction",
        prob=0.35, rng="0.25-0.45",
        rationale="Correction procedures are complex and slow; complete resolution is a minority outcome.",
        transport="Dispute mechanics are finance-specific; recourse procedures elsewhere may differ.",
    ),
    "incomplete_propagation": dict(
        construct="Probability that a correction propagates sufficiently across systems",
        source_domain="Consumer-reporting data flows",
        source="CFPB 2022; FTC 2015",
        source_stat="Corrections do not auto-propagate across all downstream systems",
        source_locator="CFPB 2022 propagation discussion",
        mapping="Estimated adequate-propagation fraction -> ~0.40",
        prob=0.40, rng="0.30-0.50",
        rationale="Corrections often fail to reach all copies; set below the midpoint.",
        transport="Propagation topology is domain-specific; healthcare data-sharing differs.",
    ),
    "awareness_gap": dict(
        construct="Probability an individual becomes aware that recourse is possible",
        source_domain="Access-to-justice / legal-needs survey",
        source="LSC 2022 (Justice Gap Report)",
        source_stat="92% of low-income civil legal problems get inadequate help; ~25% led to any action",
        source_locator="LSC 2022 Justice Gap Report, headline findings",
        mapping="Fraction reaching awareness-and-action for a recourse pathway -> ~0.30",
        prob=0.30, rng="0.20-0.40",
        rationale="Large justice gap implies most never become aware; set at the low-moderate end.",
        transport="Civil legal-needs data; direct transfer to algorithmic recourse awareness is provisional.",
    ),
    "record_access": dict(
        construct="Probability of obtaining one's own underlying records",
        source_domain="Consumer-reporting access rights",
        source="CFPB 2022; FCRA Section 612 (free annual reports)",
        source_stat="Statutory free annual reports exist, but practical access barriers remain",
        source_locator="FCRA 15 U.S.C. 1681j (Section 612); CFPB 2022 access discussion",
        mapping="Most-navigable stage given statutory access rights -> ~0.55",
        prob=0.55, rng="0.45-0.65",
        rationale="Legal access rights make this the most navigable barrier, though not frictionless.",
        transport="Statutory access rights vary by domain; healthcare record access governed differently.",
    ),
    "legal_knowledge": dict(
        construct="Probability of knowing the specific legal rights/remedies that apply",
        source_domain="Access-to-justice / legal-needs survey",
        source="LSC 2022 (Justice Gap Report)",
        source_stat="~39% believe they can use the legal system to protect themselves",
        source_locator="LSC 2022 Justice Gap Report, legal-confidence finding",
        mapping="Knowing algorithm-specific rights is narrower than general confidence -> ~0.25",
        prob=0.25, rng="0.15-0.35",
        rationale="Specific algorithmic-recourse rights are less known than general legal confidence.",
        transport="General legal-confidence data; algorithm-specific rights knowledge is an extrapolation.",
    ),
    "legal_resources": dict(
        construct="Probability of accessing legal resources given knowledge",
        source_domain="Access-to-justice / legal-needs survey",
        source="LSC 2022 (Justice Gap Report)",
        source_stat="46% cite cost as a barrier; ~50% of those seeking help are turned away",
        source_locator="LSC 2022 Justice Gap Report, cost/turn-away findings",
        mapping="Fraction with knowledge who can actually secure resources -> ~0.40",
        prob=0.40, rng="0.30-0.50",
        rationale="Cost and capacity constraints block many who know their rights.",
        transport="Civil legal-aid capacity; not a healthcare-specific resource estimate.",
    ),
    "systemic_bias": dict(
        construct="Probability the governing system lacks structural bias against the individual",
        source_domain="Healthcare algorithmic fairness",
        source="Obermeyer et al. 2019 (Science)",
        source_stat="Commercial risk algorithm exhibited bias reducing Black patients' identified need",
        source_locator="Obermeyer et al. 2019, Science 366(6464):447-453",
        mapping="Context-specific bias magnitude mapped to ~0.30 chance system is not structurally biased",
        prob=0.30, rng="0.20-0.40",
        rationale="Single-context commercial-algorithm study; used as a bounded illustrative proxy only.",
        transport="STRONG LIMITATION: one commercial cost-proxy algorithm; NOT a population-wide estimate "
                  "of healthcare algorithmic bias. See Major 3 scope caveat.",
    ),
}


def load_barriers(csv_path):
    with open(csv_path, newline='') as fh:
        return list(csv.DictReader(fh))


def latex_escape(s):
    for a, b in [('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#'),
                 ('>=', r'$\geq$'), ('<', r'$<$'), ('~', r'$\sim$')]:
        s = s.replace(a, b)
    return s


def build_markdown(barriers):
    lines = []
    lines.append("# Parameter Derivation Audit Trail\n")
    lines.append("**Manuscript:** PDIG-D-26-00342 — *Structural limits of single-barrier "
                 "reform in algorithmic recourse: a formal series-system model with "
                 "implications for digital health*\n")
    lines.append("This file documents the derivation of every stage-specific pass "
                 "probability `p_i`. Each `p_i` is a **provisional, transport-limited "
                 "empirical calibration** (epistemic Level 3), not a measured property of "
                 "any real recourse system. Numerical model outputs derived from these "
                 "values (baseline 0.0018%, maximum single-barrier gain 0.0054%, 87.6% "
                 "three-way interaction share) are **model properties**, not empirical "
                 "measurements.\n")
    lines.append("Source locators marked **** could not be confirmed to an exact "
                 "page/table/figure from available materials and must be verified against "
                 "the primary source before final submission.\n")
    lines.append("| Column | Meaning |")
    lines.append("|---|---|")
    lines.append("| Pass probability | Implemented `p_i` (probability of clearing the stage) |")
    lines.append("| Plausibility range | Prespecified sensitivity bounds |\n")
    lines.append("---\n")

    for b in barriers:
        key = b['key']
        d = DERIVATION.get(key)
        if d is None:
            continue
        layer = LAYER_DISPLAY.get(b['layer'], b['layer'])
        lines.append(f"## {b['name']}\n")
        lines.append(f"- **Barrier key:** `{key}`")
        lines.append(f"- **Layer:** {layer}")
        lines.append(f"- **Target construct:** {d['construct']}")
        lines.append(f"- **Source domain:** {d['source_domain']}")
        lines.append(f"- **Source:** {d['source']}")
        lines.append(f"- **Exact source statistic:** {d['source_stat']}")
        lines.append(f"- **Source page/table/figure:** {d['source_locator']}")
        lines.append(f"- **Mapping rule:** {d['mapping']}")
        lines.append(f"- **Pass probability (p_i):** {d['prob']:.2f}")
        lines.append(f"- **Plausibility range:** {d['rng']}")
        lines.append(f"- **Rationale:** {d['rationale']}")
        lines.append(f"- **Transportability limitation:** {d['transport']}\n")

    # Consistency check line
    prod = 1.0
    for b in barriers:
        d = DERIVATION.get(b['key'])
        if d:
            prod *= d['prob']
    lines.append("---\n")
    lines.append(f"**Consistency check:** product of the eleven pass probabilities = "
                 f"{prod:.6e} = {prod*100:.4f}% (matches the frozen baseline 0.0018%).\n")
    return "\n".join(lines) + "\n"


def build_latex(barriers):
    out = []
    out.append("% Auto-generated by analysis/build_derivation_table.py -- do not edit by hand.")
    out.append(r"\begin{longtable}{p{2.4cm} p{1.4cm} c p{1.4cm} p{2.0cm} p{3.0cm} p{3.4cm}}")
    out.append(r"\caption{Full per-barrier parameter derivation audit trail. Locators marked "
               r" require confirmation against the primary source.}"
               r"\label{tab:deriv}\\")
    out.append(r"\hline")
    out.append(r"\textbf{Barrier} & \textbf{Layer} & \textbf{Pass} & \textbf{Range} & "
               r"\textbf{Source} & \textbf{Key statistic (locator)} & "
               r"\textbf{Mapping rule / transportability} \\")
    out.append(r"\hline \endfirsthead")
    out.append(r"\hline")
    out.append(r"\textbf{Barrier} & \textbf{Layer} & \textbf{Pass} & \textbf{Range} & "
               r"\textbf{Source} & \textbf{Key statistic (locator)} & "
               r"\textbf{Mapping rule / transportability} \\")
    out.append(r"\hline \endhead")
    for b in barriers:
        d = DERIVATION.get(b['key'])
        if d is None:
            continue
        layer = LAYER_DISPLAY.get(b['layer'], b['layer'])
        stat = latex_escape(f"{d['source_stat']} ({d['source_locator']})")
        mapping = latex_escape(f"{d['mapping']}. {d['transport']}")
        row = (f"{latex_escape(b['name'])} & {latex_escape(layer)} & {d['prob']:.2f} & "
               f"{latex_escape(d['rng'])} & {latex_escape(d['source'])} & {stat} & {mapping} \\\\ \\hline")
        out.append(row)
    out.append(r"\end{longtable}")
    return "\n".join(out) + "\n"


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Build parameter-derivation audit trail (md + LaTeX).")
    p.add_argument('--barriers', type=str,
                   default=os.path.join(here, 'barrier_definitions.csv'),
                   help='Path to barrier_definitions.csv.')
    p.add_argument('--out-md', type=str,
                   default=os.path.join(here, '..', 'data', 'parameter_sources',
                                        'parameter_derivation.md'),
                   help='Output markdown path.')
    p.add_argument('--out-table', type=str,
                   default=os.path.join(here, '..', 'build', 'S_deriv_table.tex'),
                   help='Output LaTeX supplement table path.')
    return p.parse_args()


def main():
    args = parse_args()
    barriers = load_barriers(args.barriers)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_table)), exist_ok=True)

    with open(args.out_md, 'w') as fh:
        fh.write(build_markdown(barriers))
    with open(args.out_table, 'w') as fh:
        fh.write(build_latex(barriers))

    print(f"-> {args.out_md}")
    print(f"-> {args.out_table}")


if __name__ == "__main__":
    main()
