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

# The figures are uploaded separately, so the appendix contains tables only and the note
# must not invite a "Fig A in S1 Appendix" citation that would point at nothing.
NOTE = ("Tables in this appendix are labelled alphabetically and are cited as, for "
        "example, “Table A in S1 Appendix”.")


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

    # Table G covers three things -- the equicorrelation copula sweep, the within-layer
    # block variant, and the repeated-attempt rows (m = 2, 3). Its original title named
    # only the first, which left the manuscript's citation of the repeated-attempt
    # results pointing at a table that did not appear to contain them. Retitled so the
    # table announces its own contents.
    old_g = "Table G. Robustness to alternative topologies (seed 42, n = 100,000)."
    new_g = ("Table G. Robustness to alternative topologies: correlated barriers and "
             "repeated attempts (seed 42, n = 100,000).")
    xml, n_g = re.subn(re.escape(old_g), new_g, xml)
    log.append(f"Table G retitled: {n_g}")
    if n_g != 1:
        raise SystemExit("expected exactly one Table G title")

    # Document title becomes the file label.
    xml, n_ttl = re.subn(r"(<w:t[^>]*>)Supplementary Information(</w:t>)",
                         r"\1S1 Appendix\2", xml)
    log.append(f"title set to 'S1 Appendix': {n_ttl}")
    if n_ttl != 1:
        raise SystemExit("expected exactly one title run")
    return xml, log


FIG_LABEL = re.compile(r"^(Fig [A-C])\.\s*(.+)$")

# The third supporting figure is byte-identical to main Figure 6
# (sha256 adf7157c099c9fd583a2da597a9d15b59ad8e0e30c65f064178c791a2467a2ae, confirmed
# against the file Editorial Manager already holds). Publishing it as supporting
# information would print the same figure twice. It is dropped, and nothing is lost: the
# main Fig 6 caption already carries the cross-reference to Table G in S1 Appendix.
DUPLICATE_OF_MAIN = {3: "byte-identical to main Fig 6"}


def extract_figure_captions(xml: str) -> list[tuple[str, str]]:
    """Pull the figure captions out before they are removed.

    The figures move to Editorial Manager as separate Supporting Information items, so
    their captions belong in the manuscript's Supporting information section, not in the
    appendix. PLOS numbers separate items S1 Fig, S2 Fig, S3 Fig -- letters are only for
    components inside a bundled file.
    """
    text = html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml)))
    out = []
    for line in (l.strip() for l in text.split("\n")):
        m = FIG_LABEL.match(line)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def strip_figures(xml: str) -> tuple[str, int, int]:
    """Remove figure images and their captions from the appendix body."""
    # Drop any paragraph that contains a drawing.
    paras = re.findall(r"<w:p\b(?:(?!<w:p\b).)*?</w:p>", xml, re.S)
    n_img = 0
    for para in paras:
        if "<w:drawing>" in para:
            xml = xml.replace(para, "", 1)
            n_img += 1
    # Drop the caption paragraphs.
    n_cap = 0
    for para in re.findall(r"<w:p\b(?:(?!<w:p\b).)*?</w:p>", xml, re.S):
        text = html.unescape(re.sub(r"<[^>]+>", "", para)).strip()
        if FIG_LABEL.match(text):
            xml = xml.replace(para, "", 1)
            n_cap += 1
    return xml, n_img, n_cap


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

    captions = extract_figure_captions(xml)
    xml, n_img, n_cap = strip_figures(xml)
    log.append(f"figure images removed: {n_img}; figure captions removed: {n_cap}")
    if (n_img, n_cap) != (3, 3):
        raise SystemExit(f"expected 3 images and 3 captions, removed {n_img} and {n_cap}")

    xml = insert_note(xml)
    parts["word/document.xml"] = xml.encode("utf8")

    # Drop the image parts and their relationships so no dangling reference remains.
    rels = parts["word/_rels/document.xml.rels"].decode("utf8")
    rels = re.sub(r"<Relationship\b[^>]*media/[^>]*/>", "", rels)
    parts["word/_rels/document.xml.rels"] = rels.encode("utf8")
    names = [n for n in names if not n.startswith("word/media/")]
    log.append("image parts and relationships dropped")

    # Rewrite the package, preserving every other part byte-for-byte.
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, parts[n])

    # The figure captions now belong in the manuscript, beside the separate SI figures.
    cap_path = OUT.parent / "SI_FIGURE_CAPTIONS.md"
    n_keep = len(captions) - len(DUPLICATE_OF_MAIN)
    files = ", ".join(f"`S{i}_Fig.tif`" for i in range(1, n_keep + 1))
    lines = ["# Supporting information figure captions",
             "",
             f"{n_keep} figures are uploaded to Editorial Manager as separate Supporting",
             f"Information items ({files}). PLOS numbers separate items S1/S2/S3, not A/B/C",
             "-- letters apply only to components inside a bundled file. These captions belong",
             "in the manuscript's Supporting information section, one paragraph each, after",
             "the S1 Appendix entry.",
             ""]
    kept = 0
    for i, (_, title) in enumerate(captions, start=1):
        if i in DUPLICATE_OF_MAIN:
            lines += [f"> Supporting figure {i} omitted: {DUPLICATE_OF_MAIN[i]}. "
                      f"Do not upload it.", ""]
            continue
        kept += 1
        # These captions live in the manuscript now, outside the appendix, so a bare
        # "Table G" no longer identifies its location.
        title = re.sub(r"\bTable G\b(?! in S1 Appendix)", "Table G in S1 Appendix", title)
        lines += [f"**S{kept} Fig.** {title}", ""]
    cap_path.write_text("\n".join(lines))
    log.append(f"wrote {cap_path.name} with {len(captions)} caption(s)")

    for line in log:
        print(f"  {line}")
    print(f"  wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
