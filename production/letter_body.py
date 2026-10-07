"""Emit the Author Comments as plain text for Editorial Manager's Letter Body box.

The source is written in Markdown for review; EM's Letter Body field is plain text and
renders ** and ` literally. This strips the markup and the editorial header, leaving only
the letter itself.

Usage:  python3 production/letter_body.py
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent / "AUTHOR_COMMENTS.md"


def plain(md: str) -> str:
    body = md.split("---", 1)[1].strip()        # drop the editorial header
    body = re.sub(r"\*\*(.+?)\*\*", r"\1", body, flags=re.S)   # bold
    body = re.sub(r"(?<!\w)`([^`]+)`(?!\w)", r"\1", body)      # inline code
    body = re.sub(r"^> ?", "    ", body, flags=re.M)           # blockquote -> indent
    return body


if __name__ == "__main__":
    print(plain(SRC.read_text()))
