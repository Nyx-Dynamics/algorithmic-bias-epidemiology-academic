"""Emit the manuscript's Supporting information caption from S1_Appendix.docx.

Generated rather than written by hand so the caption cannot drift from the appendix it
describes. Drift is exactly what went wrong here: an earlier caption was written against a
differently-ordered supplement and mislabelled almost every component.

Titles are reproduced verbatim from the appendix, including capitalisation, so proper
nouns (Shapley, Sobol) are preserved and no wording is introduced that the appendix does
not itself use.

Usage:  python3 production/build_si_caption.py
"""
from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

APPENDIX = Path(__file__).resolve().parents[1] / "production" / "S1_Appendix.docx"


def items() -> list[tuple[str, str]]:
    xml = zipfile.ZipFile(APPENDIX).read("word/document.xml").decode("utf8")
    text = html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml)))
    found: list[tuple[str, str]] = []
    for line in (s.strip() for s in text.split("\n")):
        m = re.match(r"^((?:Table [A-G]|Fig [A-C]))\.\s*(.+)$", line)
        if not m:
            continue
        label, title = m.group(1), m.group(2).strip()
        # Figure labels run on into descriptive prose; keep the title sentence only.
        title = title.split(". ")[0].rstrip(".")
        if not any(lbl == label for lbl, _ in found):
            found.append((label, title))
    return found


def caption() -> str:
    found = items()
    body = "; ".join(f"{lbl}, {ttl}" for lbl, ttl in found)
    # The heading follows the contents: it has been wrong in both directions already,
    # once promising figures the appendix did not hold and once omitting figures it did.
    has_figs = any(lbl.startswith("Fig") for lbl, _ in found)
    kind = "tables and figures" if has_figs else "tables"
    return f"S1 Appendix. Supporting {kind}. {body}."


if __name__ == "__main__":
    print(caption())
