from pathlib import Path
import json
import re


INPUT_FILE = Path(
    "data/processed/constitution/constitution_pages_structured.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/constitution/constitution_articles_pilot.jsonl"
)

REPORT_FILE = Path(
    "outputs/constitution_articles_pilot.txt"
)

PAGES = [33, 34]

# Expected headings in this controlled two-page pilot.
EXPECTED = [
    ("1", "Name and territory of the Union"),
    ("2", "Admission or establishment of new States"),
    ("2A", "Sikkim to be associated with the Union"),
    ("3", "Formation of new States and alteration of areas"),
    ("4", "Laws made under articles 2 and 3"),
]


def load_pages():
    selected = {}

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record["pdf_page"] in PAGES:
                selected[record["pdf_page"]] = record

    if set(selected) != set(PAGES):
        raise ValueError("One or more pilot pages are missing.")

    return selected


def main():
    pages = load_pages()

    # Join pages in their original order.
    # Footnotes are not included in the Article body.
    page_texts = [
        pages[number]["main_text"]
        for number in PAGES
    ]

    combined = "\n".join(page_texts)

    # Locate only the five expected headings.
    # This prevents ordinary clause numbers from being
    # mistaken for new Articles in this pilot.
    matches = []

    for number, heading in EXPECTED:
        heading_pattern = re.escape(heading).replace(
            r"\ ", r"\s+"
        )

        pattern = (
            r"(?m)^[ \t]*"
            r"(?:\d{1,3}\[)?"
            + re.escape(number)
            + r"\.[ \t]*"
            + r"\[?"
            + heading_pattern
        )

        found = list(re.finditer(pattern, combined))

        if len(found) != 1:
            raise ValueError(
                f"Expected exactly one heading for Article "
                f"{number}, found {len(found)}"
            )

        matches.append({
            "number": number,
            "heading": heading,
            "start": found[0].start()
        })

    matches.sort(key=lambda item: item["start"])

    actual_order = [m["number"] for m in matches]
    expected_order = [n for n, _ in EXPECTED]

    if actual_order != expected_order:
        raise ValueError(
            f"Unexpected Article order: {actual_order}"
        )

    # Map positions in the joined text back to PDF pages.
    page_ranges = []
    cursor = 0

    for number, text in zip(PAGES, page_texts):
        page_ranges.append((
            number,
            cursor,
            cursor + len(text)
        ))
        cursor += len(text) + 1

    def pages_for_span(start, end):
        return [
            number
            for number, p_start, p_end in page_ranges
            if start < p_end and end > p_start
        ]

    records = []
    report = []

    for index, match in enumerate(matches):
        start = match["start"]

        end = (
            matches[index + 1]["start"]
            if index + 1 < len(matches)
            else len(combined)
        )

        article_text = combined[start:end].strip()
        source_pages = pages_for_span(start, end)

        record = {
            "document_id": (
                f"constitution_2024_article_{match['number']}"
            ),
            "document_type": "constitution_article",
            "title": match["heading"],
            "article_number": match["number"],
            "jurisdiction": "India",
            "text": article_text,
            "metadata": {
                "source_file":
                    "constitution_official_2024.pdf",
                "source_edition": "2024-05-01",
                "source_pages": source_pages,
                "extraction_status": "pilot_unverified",
                "footnotes": [],
                "footnote_status":
                    "retained_in_source_pages",
                "legal_status": "unverified"
            }
        }

        records.append(record)

        report.extend([
            "=" * 70,
            f"ARTICLE {match['number']}",
            f"Pages: {source_pages}",
            f"Characters: {len(article_text)}",
            "",
            article_text,
            ""
        ])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(
                record, ensure_ascii=False
            ) + "\n")

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("Articles extracted:", len(records))

    for record in records:
        print(
            f"Article {record['article_number']}: "
            f"{len(record['text'])} chars, "
            f"pages {record['metadata']['source_pages']}"
        )

    print("\nSaved:", OUTPUT_FILE)
    print("Saved:", REPORT_FILE)


if __name__ == "__main__":
    main()