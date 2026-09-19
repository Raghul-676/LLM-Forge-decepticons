from pathlib import Path
from collections import Counter
import hashlib
import json
import re

import fitz
import pandas as pd


PDF_FILE = Path(
    "data/raw/constitution/constitution_official_2024.pdf"
)

OUT_DIR = Path("data/processed/constitution")
REPORT_DIR = Path("outputs")

OUTPUT_FILE = OUT_DIR / "constitution_pages_structured.jsonl"
SUMMARY_FILE = REPORT_DIR / "constitution_page_split_summary.json"
REVIEW_FILE = REPORT_DIR / "constitution_page_split_review.csv"


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def get_lines(page):
    lines = []

    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            text = "".join(
                span["text"]
                for span in line["spans"]
            ).strip()

            if text:
                lines.append({
                    "text": text,
                    "bbox": [
                        round(v, 2)
                        for v in line["bbox"]
                    ]
                })

    lines.sort(
        key=lambda item: (
            round(item["bbox"][1], 1),
            item["bbox"][0]
        )
    )

    return lines


def is_separator(text):
    compact = re.sub(r"\s+", "", text)

    return bool(
        re.fullmatch(r"_{12,}", compact)
    )


def is_header(text, bbox):
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
    if not re.fullmatch(r"\d{1,4}", text):
        return False

    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) / 2

    top_margin = (
        y1 < 210
        and x0 < 180
    )

    bottom_center = (
        y0 > page.rect.height * 0.75
        and abs(
            center_x - page.rect.width / 2
        ) < 30
    )

    return top_margin or bottom_center


def split_page(page):
    lines = get_lines(page)

    separators = [
        line
        for line in lines
        if is_separator(line["text"])
        and line["bbox"][1]
        > page.rect.height * 0.38
    ]

    flags = []

    if not separators:
        separator_y = None
    else:
        separator_y = separators[0]["bbox"][1]

        if len(separators) > 1:
            flags.append(
                "multiple_separator_candidates"
            )

    header_lines = []
    page_number_lines = []
    main_lines = []
    footnote_lines = []

    for line in lines:
        text = line["text"]
        bbox = line["bbox"]

        if is_separator(text):
            continue

        if is_page_number(
            text,
            bbox,
            page
        ):
            page_number_lines.append(line)
            continue

        if is_header(text, bbox):
            header_lines.append(line)
            continue

        if (
            separator_y is not None
            and bbox[1] >= separator_y
        ):
            footnote_lines.append(line)
        else:
            main_lines.append(line)

    def join_text(items):
        return "\n".join(
            item["text"]
            for item in items
        )

    return {
        "main_text": join_text(main_lines),
        "footnotes_text":
            join_text(footnote_lines),
        "header_text":
            join_text(header_lines),
        "page_number_text":
            join_text(page_number_lines),
        "separator_y": separator_y,
        "quality_flags": flags
    }


def main():
    if not PDF_FILE.exists():
        raise FileNotFoundError(PDF_FILE)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    source_hash = sha256_file(PDF_FILE)

    flag_counts = Counter()
    page_type_counts = Counter()
    review_rows = []

    total_main_chars = 0
    total_footnote_chars = 0

    with fitz.open(PDF_FILE) as pdf, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as writer:

        for page_index in range(len(pdf)):
            page_number = page_index + 1
            page = pdf[page_index]

            result = split_page(page)

            main_text = result["main_text"]
            footnotes_text = result[
                "footnotes_text"
            ]

            total_main_chars += len(main_text)
            total_footnote_chars += len(
                footnotes_text
            )

            flags = result[
                "quality_flags"
            ]

            flag_counts.update(flags)

            if footnotes_text:
                page_type = "main_plus_footnotes"
            elif main_text:
                page_type = "main_only"
            else:
                page_type = "empty_after_split"

            page_type_counts[
                page_type
            ] += 1

            if (
                not main_text.strip()
                or flags
            ):
                review_rows.append({
                    "pdf_page": page_number,
                    "main_characters":
                        len(main_text),
                    "footnote_characters":
                        len(footnotes_text),
                    "flags":
                        "; ".join(flags),
                    "main_preview":
                        main_text[:300]
                })

            record = {
                "document_type":
                    "constitution_source_page",
                "source_file":
                    PDF_FILE.name,
                "source_sha256":
                    source_hash,
                "source_edition":
                    "2024-05-01",
                "pdf_page":
                    page_number,
                "main_text":
                    main_text,
                "footnotes_text":
                    footnotes_text,
                "header_text":
                    result["header_text"],
                "page_number_text":
                    result["page_number_text"],
                "separator_y":
                    result["separator_y"],
                "quality_flags":
                    flags
            }

            writer.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            if (
                page_number % 50 == 0
                or page_number == len(pdf)
            ):
                print(
                    f"[{page_number}/{len(pdf)}]"
                )

    summary = {
        "source_file":
            PDF_FILE.name,
        "source_edition":
            "2024-05-01",
        "source_sha256":
            source_hash,
        "pages_processed":
            sum(page_type_counts.values()),
        "page_type_counts":
            dict(page_type_counts),
        "total_main_characters":
            total_main_chars,
        "total_footnote_characters":
            total_footnote_chars,
        "flag_counts":
            dict(flag_counts),
        "review_pages":
            len(review_rows)
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    pd.DataFrame(
        review_rows
    ).to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print(
        "CONSTITUTION FULL PAGE SPLIT SUMMARY"
    )
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved:", OUTPUT_FILE)
    print("Saved:", SUMMARY_FILE)
    print("Saved:", REVIEW_FILE)


if __name__ == "__main__":
    main()