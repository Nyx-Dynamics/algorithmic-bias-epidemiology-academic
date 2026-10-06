"""Relabel the prepared R1 supporting information into a PLOS S1 Appendix.

Source: the R1 supplement authored for the revision,
~/Downloads/PDIG-D-26-00342/v6_upload/3_Supporting_Information_S1_File.docx, which the
upload checklist lists as the Supporting Information file and which was never uploaded
to Editorial Manager -- EM still holds the February PDF.

PLOS requires that a bundled Supporting Information file label its components
alphabetically and that they be cited as "Table A in S1 Appendix". This script applies
that relabelling to the .docx directly, editing the label runs in word/document.xml and
leaving every table, number, figure and style untouched. Nothing is retyped or
reformatted, so the content remains exactly what was prepared for the revision.

Usage:  python3 production/build_s1_appendix_docx.py
"""
from __future__ import annotations

import html
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = Path.home() / "Downloads" / "PDIG-D-26-00342" / "v6_upload" / "3_Supporting_Information_S1_File.docx"
OUT = REPO / "production" / "S1_Appendix.docx"

TABLE_LETTERS = "ABCDEFG"   # Supplementary Tables 1-7
FIG_LETTERS = "ABC"         # Supplementary Figures 1-3

NOTE = ("Components of this appendix are labelled alphabetically and are cited as, for "
        "example, “Table A in S1 Appendix” and “Fig A in S1 Appendix”.")


def relabel(xml: str) -> tuple[str, list[str]]:
    log: list[str] = []

    def table_sub(m: re.Match) -> str:
        n = int(m.group(1))
        return f"Table {TABLE_LETTERS[n - 1]}"

    def fig_sub(m: re.Match) -> str:
        n = int(m.group(1))
        return f"Fig {FIG_LETTERS[n - 1]}"

    xml, n_t = re.subn(r"Supplementary Table ([1-7])", table_sub, xml)
    log.append(f"table labels and cross-references rewritten: {n_t}")
    xml, n_f = re.subn(r"Supplementary Figure ([1-3])", fig_sub, xml)
    log.append(f"figure labels and cross-references rewritten: {n_f}")

    # Caption labels used "Table A:"; PLOS style is "Table A."
    xml, n_c = re.subn(r"(Table [A-G]|Fig [A-C]):", r"\1.", xml)
    log.append(f"label separators normalised to a period: {n_c}")

    # Document title becomes the file label.
    xml, n_ttl = re.subn(r"(<w:t[^>]*>)Supplementary Information(</w:t>)",
                         r"\1S1 Appendix\2", xml)
    log.append(f"title set to 'S1 Appendix': {n_ttl}")
    if n_ttl != 1:
        raise SystemExit("expected exactly one title run")
    return xml, log


def insert_note(xml: str) -> str:
    """Add the citation-guidance line immediately after the title paragraph."""
    m = re.search(r"<w:p\b[^>]*>(?:(?!</w:p>).)*?S1 Appendix.*?</w:p>", xml, re.S)
    if not m:
        raise SystemExit("title paragraph not found")
    para = (f'<w:p><w:pPr><w:pStyle w:val="BodyText"/></w:pPr><w:r><w:rPr><w:i/>'
            f'</w:rPr><w:t xml:space="preserve">{html.escape(NOTE)}</w:t></w:r></w:p>')
    return xml[:m.end()] + para + xml[m.end():]


def main() -> int:
    if not SRC.exists():
        print(f"source not found: {SRC}")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, OUT)

    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
        parts = {n: z.read(n) for n in names}

    xml = parts["word/document.xml"].decode("utf8")
    xml, log = relabel(xml)
    xml = insert_note(xml)
    parts["word/document.xml"] = xml.encode("utf8")

    # Rewrite the package, preserving every other part byte-for-byte.
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, parts[n])

    for line in log:
        print(f"  {line}")
    print(f"  wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
