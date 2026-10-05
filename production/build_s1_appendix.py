"""Generate S1_Appendix.tex from the R1 supplement source.

PLOS requires that a bundled Supporting Information file label its components
alphabetically and that they be cited as "Table A in S1 Appendix". This script applies
that relabelling mechanically so the mapping is auditable and cannot drift from the
manuscript by hand-editing.

It also repairs three defects carried by the source:

  1. A caption still referred to "[verify]" markers that no longer exist in the table
     body -- a dangling reference, and the mirror of the defect repaired in
     analysis/build_derivation_table.py. Replaced with that file's wording so the
     supplement and the repository state the same caveat.

  2. LaTeX float counters produced double labels ("Figure 1. Fig A. ...", and a section
     heading reading "Table B" above a caption reading "Table 3"). Automatic caption
     prefixes are suppressed and the alphabetical label is placed inside the caption.

  3. Reviewer-round artifacts -- "(Major 4)", "(corrected)" -- would otherwise publish.

Usage:  python3 production/build_s1_appendix.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "manuscript" / "supplement_v5.tex"
OUT = REPO / "production" / "S1_Appendix.tex"

# Section headings and the one in-text cross-reference.
RELABEL = [
    ("S1 Note. Scope of claims", "Text A. Scope of claims"),
    ("S-Deriv Table. Parameter derivation audit trail (all 11 barriers)",
     "Table A. Parameter derivation audit trail (all 11 barriers)"),
    ("S-Deriv Table. Parameter derivation audit trail",
     "Table A. Parameter derivation audit trail"),
    ("S2 Table. Individual barrier removal effects",
     "Table B. Individual barrier removal effects"),
    ("S3 Table. Layer interaction decomposition",
     "Table C. Layer interaction decomposition"),
    ("S4 Table. Shapley value attribution", "Table D. Shapley value attribution"),
    ("S5 Table. Sobol sensitivity indices", "Table E. Sobol sensitivity indices"),
    ("S6 Table. Robustness under bounded perturbation",
     "Table F. Robustness under bounded perturbation"),
    ("S7 Table. Signal-to-noise and output variability",
     "Table G. Signal-to-noise and output variability"),
    ("S-Copula Table. Robustness to alternative topologies",
     "Table H. Robustness to alternative topologies"),
    ("S-Copula Table. Alternative-topology robustness",
     "Table H. Robustness to alternative topologies"),
    ("S1 Fig. ", "Fig A. "),
    ("S2 Fig. ", "Fig B. "),
    ("S3 Fig. ", "Fig C. "),
    ("(Table~S-Deriv)", "(Table A)"),
]

# Table captions, in document order, receive these labels.
TABLE_LABELS = list("ABCDEFGH")

OLD_CAVEAT = ("Source page/table references marked [verify] are to be confirmed at "
              "finalization.")
NEW_CAVEAT = ("Source locators identify the specific discussion or finding within each "
              "source rather than an exact page, table, or figure number; they were not "
              "confirmed to page-level precision against the primary sources.")


def build() -> str:
    text = SRC.read_text()

    for old, new in RELABEL:
        text = text.replace(old, new)

    assert OLD_CAVEAT in text, "caveat sentence not found; source may have changed"
    text = text.replace(OLD_CAVEAT, NEW_CAVEAT)

    # Reviewer-round artifacts must not reach print.
    text = re.sub(r"\s*\((?:Major|Minor) \d+\)", "", text)
    text = text.replace(" (corrected)", "")

    # Place the alphabetical label inside each table caption, in document order.
    labels = iter(TABLE_LABELS)

    def label_caption(m: re.Match) -> str:
        return m.group(1) + f"Table {next(labels)}. " + m.group(2)

    # Tables use two caption commands in this source: \captionof{table}{...} for the
    # landscape longtable and \caption{...} inside table environments. Figure captions
    # share the \caption form, so they are skipped -- they already carry "Fig A." from
    # the relabel map above.
    text, n = re.subn(
        r"(\\(?:captionof\{table\}|caption)\{\{\\bf\s*)(?!Fig )(\S)",
        label_caption, text)
    if n != len(TABLE_LABELS):
        raise SystemExit(f"expected {len(TABLE_LABELS)} table captions, found {n}")

    # Suppress LaTeX's automatic "Table N."/"Figure N." prefixes so the alphabetical
    # label is the only one a reader sees.
    text = text.replace(
        r"\usepackage{array}",
        "\\usepackage{caption}\n\\captionsetup{labelformat=empty}\n\\usepackage{array}", 1)

    note = (r"\noindent\textit{Components of this appendix are labelled alphabetically "
            r"and should be cited as, for example, ``Table A in S1 Appendix'' and "
            r"``Fig A in S1 Appendix''.}" "\n\n" r"\tableofcontents")
    text = text.replace(r"\tableofcontents", note, 1)

    text = text.replace(
        "% PDIG-D-26-00342 v5 -- revised supplement (Major Revision)",
        "% PDIG-D-26-00342R1 -- S1 Appendix (production)\n"
        "% GENERATED by production/build_s1_appendix.py from manuscript/supplement_v5.tex.\n"
        "% Do not hand-edit: change the source and regenerate.")
    return text


def main() -> int:
    OUT.write_text(build())
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            OUT.name], cwd=OUT.parent, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stdout[-2000:])
            return 1
    print(f"built {OUT.with_suffix('.pdf').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
