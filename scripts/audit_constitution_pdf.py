
from pathlib import Path
import hashlib
import json
import re

import fitz


PDF_FILE = Path(
    "data/raw/constitution/constitution_official_2024.pdf"
)

OUT_DIR = Path("data/processed/constitution")
REPORT_DIR = Path("outputs")

PAGES_FILE = OUT_DIR / "constitution_official_pages.jsonl"
REPORT_FILE = REPORT_DIR / "constitution_pdf_audit.txt"
SUMMARY_FILE = REPORT_DIR / "constitution_pdf_audit.json"


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main():
    if not PDF_FILE.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_FILE}\n"
            "Check the filename and location."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    page_stats = []
    previews = []
    candidate_headings = []
    total_characters = 0
    empty_pages = 0

    # These are only exploratory patterns. They do not
    # determine final Article boundaries.
    heading_pattern = re.compile(
        r"(?im)^\s*(?:ARTICLE\s+)?"
        r"(?:1|21|395)\s*[.\-–—]\s+\S.*$"
    )

    with fitz.open(PDF_FILE) as pdf, open(
        PAGES_FILE, "w", encoding="utf-8"
    ) as writer:

        if pdf.needs_pass:
            raise ValueError("The PDF is password protected.")

        page_count = len(pdf)

        for page_index in range(page_count):
            page = pdf[page_index]

            # Preserve the extracted text without deleting
            # footnotes, page numbers, or amendment markers.
            text = page.get_text("text")

            record = {
                "document_type": "constitution_source_page",
                "source_file": PDF_FILE.name,
                "pdf_page": page_index + 1,
                "text": text
            }

            writer.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            char_count = len(text)
            total_characters += char_count

            if not text.strip():
                empty_pages += 1

            page_stats.append({
                "pdf_page": page_index + 1,
                "characters": char_count
            })

            # Keep a preview of the opening pages.
            if page_index < 3:
                previews.append(
                    f"PDF PAGE {page_index + 1}\n"
                    + text[:2500]
                )

            # Locate the preface without assuming its page.
            if re.search(r"(?im)^\s*PREFACE\s*$", text):
                previews.append(
                    f"PREFACE CANDIDATE — PDF PAGE {page_index + 1}\n"
                    + text[:3500]
                )

            # Locate possible Article headings.
            for match in heading_pattern.finditer(text):
                candidate_headings.append({
                    "pdf_page": page_index + 1,
                    "line": match.group(0).strip()
                })

            # Locate possible Schedule headings.
            if re.search(
                r"\bFIRST\s+SCHEDULE\b",
                text,
                flags=re.IGNORECASE
            ):
                candidate_headings.append({
                    "pdf_page": page_index + 1,
                    "line": "FIRST SCHEDULE candidate"
                })

    summary = {
        "source_file": PDF_FILE.name,
        "source_sha256": sha256_file(PDF_FILE),
        "pdf_pages": page_count,
        "total_characters": total_characters,
        "empty_text_pages": empty_pages,
        "candidate_headings": candidate_headings,
        "page_statistics": page_stats
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    report = [
        "CONSTITUTION PDF AUDIT",
        "=" * 70,
        f"Source: {PDF_FILE.name}",
        f"PDF pages: {page_count}",
        f"Total characters: {total_characters}",
        f"Empty text pages: {empty_pages}",
        "",
        "CANDIDATE ARTICLE / SCHEDULE HEADINGS",
        "-" * 70,
        json.dumps(
            candidate_headings,
            indent=2,
            ensure_ascii=False
        ),
        "",
        "OPENING PAGES AND PREFACE",
        "-" * 70,
        "\n\n".join(previews)
    ]

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("=" * 70)
    print("CONSTITUTION PDF AUDIT")
    print("=" * 70)
    print("PDF pages:", page_count)
    print("Total characters:", total_characters)
    print("Empty text pages:", empty_pages)
    print("Candidate headings:", len(candidate_headings))
    print("\nSaved:", PAGES_FILE)
    print("Saved:", REPORT_FILE)
    print("Saved:", SUMMARY_FILE)


if __name__ == "__main__":
    main()