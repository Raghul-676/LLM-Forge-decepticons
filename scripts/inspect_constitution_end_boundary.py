from pathlib import Path
import json

SOURCE_FILE = Path(
    "data/processed/constitution/constitution_pages_structured.jsonl"
)

OUTPUT_FILE = Path(
    "outputs/constitution_end_boundary_inspection.txt"
)

PAGES_TO_INSPECT = list(range(281, 291)) + [382, 383, 384, 385]

def main():
    pages = {}

    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pages[record["pdf_page"]] = record

    report = []

    for page_number in PAGES_TO_INSPECT:
        record = pages.get(page_number)

        if not record:
            continue

        report.append("=" * 80)
        report.append(f"PDF PAGE {page_number}")
        report.append("=" * 80)

        report.append("\n--- MAIN TEXT ---\n")
        report.append(
            record.get("main_text", "")[:5000]
        )

        report.append("\n--- FOOTNOTES ---\n")
        report.append(
            record.get("footnotes_text", "")[:1500]
        )

        report.append("")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("Inspected pages:", PAGES_TO_INSPECT)
    print("Saved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()