
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

OUTPUT_FILE = OUT_DIR / "constitution_page_split_test.jsonl"
REPORT_FILE = REPORT_DIR / "constitution_page_split_test.txt"

PAGES_TO_TEST = [33, 34, 42, 43]


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def get_lines(page):
    """Extract individual text lines with their coordinates."""
    lines = []

    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            text = "".join(
                span["text"] for span in line["spans"]
            ).strip()

            if text:
                lines.append({
                    "text": text,
                    "bbox": [round(v, 2) for v in line["bbox"]]
                })

    # The inspected pages use a single-column layout.
    lines.sort(key=lambda item: (
        round(item["bbox"][1], 1),
        item["bbox"][0]
    ))

    return lines


def is_separator(text):
    """Recognize the printed footnote rule."""
    compact = re.sub(r"\s+", "", text)
    return bool(re.fullmatch(r"_{12,}", compact))


def is_header(text, bbox):
    """Recognize only the known repeated page headers."""
    if bbox[3] > 210:
        return False

    return bool(
        re.fullmatch(
            r"THE\s+CONSTITUTION\s+OF\s+INDIA",
            text,
            flags=re.IGNORECASE
        )
        or re.fullmatch(
            r"\(Part\s+[IVXLCDM]+\..*\)",
            text,
            flags=re.IGNORECASE
        )
    )


def is_page_number(text, bbox, page):
    """Identify isolated page numbers in known margin areas."""
    if not re.fullmatch(r"\d{1,4}", text):
        return False

    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) / 2

    top_margin = y1 < 210 and x0 < 180
    bottom_center = (
        y0 > page.rect.height * 0.75
        and abs(center_x - page.rect.width / 2) < 30
    )

    return top_margin or bottom_center


def split_page(page):
    lines = get_lines(page)

    separators = [
        line for line in lines
        if is_separator(line["text"])
        and line["bbox"][1] > page.rect.height * 0.38
    ]

    flags = []

    if not separators:
        separator_y = None
        flags.append("footnote_separator_not_found")
    else:
        separator_y = separators[0]["bbox"][1]

        if len(separators) > 1:
            flags.append("multiple_separator_candidates")

    header_lines = []
    page_number_lines = []
    main_lines = []
    footnote_lines = []

    for line in lines:
        text = line["text"]
        bbox = line["bbox"]

        if is_separator(text):
            continue

        if is_page_number(text, bbox, page):
            page_number_lines.append(line)
            continue

        if is_header(text, bbox):
            header_lines.append(line)
            continue

        if separator_y is not None and bbox[1] >= separator_y:
            footnote_lines.append(line)
        else:
            # If no separator exists, preserve the text
            # and flag the page for manual review.
            main_lines.append(line)

    def join_text(items):
        return "\n".join(item["text"] for item in items)

    return {
        "raw_text": page.get_text("text"),
        "main_text": join_text(main_lines),
        "footnotes_text": join_text(footnote_lines),
        "header_text": join_text(header_lines),
        "page_number_text": join_text(page_number_lines),
        "separator_y": separator_y,
        "main_lines": main_lines,
        "footnote_lines": footnote_lines,
        "quality_flags": flags
    }


def main():
    if not PDF_FILE.exists():
        raise FileNotFoundError(PDF_FILE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    source_hash = sha256_file(PDF_FILE)
    reports = []

    with fitz.open(PDF_FILE) as pdf, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as writer:

        for page_number in PAGES_TO_TEST:
            page = pdf[page_number - 1]
            result = split_page(page)

            record = {
                "document_type": "constitution_page_split_test",
                "source_file": PDF_FILE.name,
                "source_sha256": source_hash,
                "source_edition": "2024-05-01",
                "pdf_page": page_number,
                **result
            }

            writer.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            reports.append(
                "=" * 80 + "\n"
                f"PDF PAGE {page_number}\n"
                f"Separator y: {result['separator_y']}\n"
                f"Main characters: {len(result['main_text'])}\n"
                f"Footnote characters: {len(result['footnotes_text'])}\n"
                f"Flags: {result['quality_flags']}\n\n"
                "--- MAIN TEXT ---\n"
                f"{result['main_text']}\n\n"
                "--- FOOTNOTES ---\n"
                f"{result['footnotes_text']}\n"
            )

            print(
                f"Page {page_number}: "
                f"main={len(result['main_text'])} chars, "
                f"footnotes={len(result['footnotes_text'])} chars, "
                f"flags={result['quality_flags']}"
            )

    REPORT_FILE.write_text(
        "\n\n".join(reports),
        encoding="utf-8"
    )

    print("\nSaved:", OUTPUT_FILE)
    print("Saved:", REPORT_FILE)


if __name__ == "__main__":
    main()