"""Check the compiled paper's pagination, references, fonts and build diagnostics."""
import argparse
from pathlib import Path
import re

import pymupdf


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(path):
    with pymupdf.open(path) as document:
        pages = [page.get_text() for page in document]
        require(len(pages) == 13, "expected 12 body pages and one references page")
        require(pages[-1].lstrip().startswith("REFERENCES"), "references must start on page 13")
        require("ACKNOWLEDGMENT OF AI-GENERATED CONTENT" in pages[-1],
                "AI disclosure must appear with the references on page 13")
        body = "\n".join(pages[:-1])
        require(not re.search(r"^(?:APPENDIX|REFERENCES)\b", body, re.M), "unexpected body heading")
        require(re.findall(r"^TABLE ([IVX]+)$", body, re.M) ==
                ["I", "II", "III", "IV", "V", "VI"], "wrong number or order of tables")
        numbers = [int(i) for i in re.findall(r"^\[(\d+)\]", pages[-1], re.M)]
        require(numbers == list(range(1, 31)), "expected 30 consecutive bibliography entries")
        for page in document:
            require(page.rect.width == 612 and page.rect.height == 792, "expected US letter pages")
            for font in page.get_fonts():
                require(document.extract_font(font[0])[-1], f"font not embedded: {font[3]}")
    log = path.with_suffix(".log").read_text(encoding="utf8", errors="replace")
    blg = path.with_suffix(".blg").read_text(encoding="utf8", errors="replace")
    require(not re.search(r"warning:|underfull|overfull|undefined references|^!", log, re.I | re.M),
            "LaTeX diagnostics present; inspect the final .log")
    require(not re.search(r"^Warning--|^I found|error messages", blg, re.M),
            "BibTeX diagnostics present; inspect the .blg")
    print("12 body pages, references and AI disclosure on page 13, 6 tables, 30 references, embedded fonts, clean build logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="compiled PDF with its .log and .blg beside it")
    check(parser.parse_args().pdf)
