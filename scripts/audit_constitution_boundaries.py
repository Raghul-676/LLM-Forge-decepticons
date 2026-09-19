
from pathlib import Path
from collections import Counter
import json
import re


SOURCE_FILE = Path(
    "data/processed/constitution/constitution_official_pages.jsonl"
)

STRUCTURED_FILE = Path(
    "data/processed/constitution/constitution_pages_structured.jsonl"
)

REPORT_FILE = Path(
    "outputs/constitution_boundary_audit.txt"
)

SUMMARY_FILE = Path(
    "outputs/constitution_boundary_audit.json"
)


def load_pages(path):
    pages = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            number = record["pdf_page"]

            if number in pages:
                raise ValueError(
                    f"Duplicate page {number} in {path}"
                )

            pages[number] = record

    return pages


def text_tokens(text):
    """
    Ignore only standalone printed separator rules.
    Token comparison detects missing/extra text but
    does not prove that reading order is correct.
    """
    text = re.sub(
        r"(?m)^[ \t]*_{12,}[ \t]*$",
        "",
        text
    )

    return Counter(re.findall(r"\S+", text))


def find_candidates(pages, label, pattern):
    matches = []

    for number, record in sorted(pages.items()):
        text = record.get("main_text", "")

        for match in re.finditer(
            pattern, text, flags=re.IGNORECASE | re.MULTILINE
        ):
            matches.append({
                "pdf_page": number,
                "preview": text[
                    max(0, match.start() - 80):
                    min(len(text), match.end() + 450)
                ]
            })

    return {
        "label": label,
        "matches": matches
    }


def main():
    source = load_pages(SOURCE_FILE)
    structured = load_pages(STRUCTURED_FILE)

    if set(source) != set(structured):
        raise ValueError(
            "Source and structured page numbers do not match."
        )

    mismatches = []
    no_separator = []

    for number in sorted(source):
        original = source[number]["text"]
        record = structured[number]

        reconstructed = "\n".join([
            record.get("header_text", ""),
            record.get("page_number_text", ""),
            record.get("main_text", ""),
            record.get("footnotes_text", "")
        ])

        original_tokens = text_tokens(original)
        reconstructed_tokens = text_tokens(reconstructed)

        missing = original_tokens - reconstructed_tokens
        extra = reconstructed_tokens - original_tokens

        if missing or extra:
            mismatches.append({
                "pdf_page": number,
                "missing_token_count": sum(missing.values()),
                "extra_token_count": sum(extra.values()),
                "missing_examples": missing.most_common(10),
                "extra_examples": extra.most_common(10)
            })

        if record.get("separator_y") is None:
            no_separator.append(number)

    patterns = [
        (
            "Article 1",
            r"^[ \t]*(?:\d{1,3}\[)?1\.[ \t]+"
            r"Name and territory of the Union\b"
        ),
        (
            "Article 21",
            r"^[ \t]*(?:\d{1,3}\[)?21\.[ \t]+"
            r"Protection of life and personal liberty\b"
        ),
        (
            "Article 395",
            r"^[ \t]*(?:\d{1,3}\[)?395\.[ \t]+Repeals\b"
        ),
        (
            "First Schedule heading",
            r"^[ \t]*(?:THE[ \t]+)?FIRST[ \t]+SCHEDULE[ \t]*$"
        )
    ]

    boundaries = [
        find_candidates(structured, label, pattern)
        for label, pattern in patterns
    ]

    # Inspect a few pages without separators, plus known pages.
    sample_pages = set(
        [1, 2, 3, 4, 33, 34, 42, 43, max(structured)]
    )

    sample_pages.update(
        n for n in no_separator if n >= 33
    )

    sample_pages = sorted(sample_pages)[:14]

    summary = {
        "source_pages": len(source),
        "structured_pages": len(structured),
        "token_mismatch_pages": len(mismatches),
        "pages_without_separator": len(no_separator),
        "no_separator_page_numbers": no_separator,
        "boundary_candidates": boundaries,
        "mismatches": mismatches
    }

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    report = [
        "CONSTITUTION BOUNDARY AND INTEGRITY AUDIT",
        "=" * 70,
        f"Pages: {len(structured)}",
        f"Token mismatch pages: {len(mismatches)}",
        f"Pages without separator: {len(no_separator)}",
        "",
        "BOUNDARY CANDIDATES",
        "=" * 70,
        json.dumps(boundaries, indent=2, ensure_ascii=False),
        "",
        "TOKEN MISMATCHES (FIRST 20)",
        "=" * 70,
        json.dumps(mismatches[:20], indent=2, ensure_ascii=False),
        "",
        "SAMPLE PAGE PREVIEWS",
        "=" * 70
    ]

    for number in sample_pages:
        record = structured[number]

        report.extend([
            f"\n--- PDF PAGE {number} ---",
            f"Separator: {record.get('separator_y')}",
            f"Main characters: {len(record.get('main_text', ''))}",
            "MAIN TEXT:",
            record.get("main_text", "")[:700],
            "FOOTNOTES:",
            record.get("footnotes_text", "")[:250]
        ])

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("=" * 70)
    print("CONSTITUTION BOUNDARY AUDIT")
    print("=" * 70)
    print("Pages checked:", len(structured))
    print("Token mismatch pages:", len(mismatches))
    print("Pages without separator:", len(no_separator))

    print("\nBoundary candidates:")
    for item in boundaries:
        pages = sorted(set(
            m["pdf_page"] for m in item["matches"]
        ))
        print(f"  {item['label']}: {pages}")

    print("\nSaved:", REPORT_FILE)
    print("Saved:", SUMMARY_FILE)


if __name__ == "__main__":
    main()