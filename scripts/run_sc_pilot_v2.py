
from pathlib import Path
from collections import Counter
import json

import pandas as pd

from parse_sc_format_aware import parse_judgment


PDF_FOLDER = Path("data/raw/supreme_court/pdfs")

# Use the EXACT same 200 PDFs from the original pilot.
ORIGINAL_REPORT = Path(
    "outputs/sc_pilot_200_quality.csv"
)

OUTPUT_FILE = Path(
    "data/processed/supreme_court/sc_pilot_200_v2.jsonl"
)

REPORT_FILE = Path(
    "outputs/sc_pilot_200_v2_quality.csv"
)

SUMMARY_FILE = Path(
    "outputs/sc_pilot_200_v2_summary.json"
)


def main():

    if not ORIGINAL_REPORT.exists():
        raise FileNotFoundError(
            f"Original pilot report not found: {ORIGINAL_REPORT}"
        )

    original_df = pd.read_csv(ORIGINAL_REPORT)

    filenames = (
        original_df["source_file"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if len(filenames) != 200:
        raise ValueError(
            f"Expected 200 original filenames, found {len(filenames)}"
        )

    if len(set(filenames)) != len(filenames):
        raise ValueError(
            "Duplicate filenames found in original pilot report."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    report_rows = []

    status_counts = Counter()
    format_counts = Counter()
    text_status_counts = Counter()
    field_counts = Counter()
    flag_counts = Counter()

    saved_count = 0

    fields = [
        "petitioner",
        "respondent",
        "judgment_date",
        "bench",
        "citation_block",
        "headnote",
        "judgment_text"
    ]

    print("=" * 70)
    print("SUPREME COURT PILOT V2")
    print("Using the same 200 PDFs as the original pilot")
    print("=" * 70)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

        for index, filename in enumerate(filenames, start=1):

            pdf_path = PDF_FOLDER / filename

            print(f"[{index}/{len(filenames)}] {filename}")

            try:

                record = parse_judgment(pdf_path)

                meta = record["metadata"]

                # Save each record immediately.
                output.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )

                saved_count += 1

                status = meta.get(
                    "parse_status",
                    "unknown"
                )

                document_format = meta.get(
                    "document_format",
                    "unknown"
                )

                text_status = meta.get(
                    "text_status",
                    "unknown"
                )

                flags = meta.get(
                    "quality_flags",
                    []
                )

                status_counts[status] += 1
                format_counts[document_format] += 1
                text_status_counts[text_status] += 1
                flag_counts.update(flags)

                for field in fields:
                    if record.get(field):
                        field_counts[field] += 1

                report_rows.append({
                    "source_file": filename,
                    "status": status,
                    "document_format": document_format,
                    "text_status": text_status,
                    "title": record.get("title"),
                    "case_number": record.get("case_number"),
                    "date": record.get("judgment_date"),
                    "page_count": meta.get("page_count"),
                    "character_count": meta.get("character_count"),
                    "judgment_characters":
                        meta.get("judgment_character_count"),
                    "has_petitioner":
                        bool(record.get("petitioner")),
                    "has_respondent":
                        bool(record.get("respondent")),
                    "has_date":
                        bool(record.get("judgment_date")),
                    "has_bench":
                        bool(record.get("bench")),
                    "has_citation":
                        bool(record.get("citation_block")),
                    "has_headnote":
                        bool(record.get("headnote")),
                    "has_judgment":
                        bool(record.get("judgment_text")),
                    "quality_flags": "; ".join(flags),
                    "source_sha256":
                        meta.get("source_sha256")
                })

            except Exception as exc:

                error_message = str(exc)

                if error_message == "empty_file":
                    error_type = "empty_file"
                elif isinstance(exc, FileNotFoundError):
                    error_type = "file_not_found"
                else:
                    error_type = "extraction_error"

                status_counts[error_type] += 1

                report_rows.append({
                    "source_file": filename,
                    "status": error_type,
                    "document_format": None,
                    "text_status": "unusable",
                    "quality_flags": error_message
                })

                print("ERROR:", error_message)

    # Save quality report.
    report_df = pd.DataFrame(report_rows)

    report_df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "attempted": len(filenames),
        "saved": saved_count,
        "status_counts": dict(status_counts),
        "format_counts": dict(format_counts),
        "text_status_counts": dict(text_status_counts),
        "field_counts": dict(field_counts),
        "quality_flags": dict(flag_counts)
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("PILOT V2 SUMMARY")
    print("=" * 70)

    print("PDFs attempted:", len(filenames))
    print("Documents saved:", saved_count)

    print("\nStatus counts:")
    for key, count in status_counts.items():
        print(f"  {key}: {count}")

    print("\nDocument formats:")
    for key, count in format_counts.items():
        print(f"  {key}: {count}")

    print("\nText status:")
    for key, count in text_status_counts.items():
        print(f"  {key}: {count}")

    print("\nField counts:")
    for field, count in field_counts.items():
        print(f"  {field}: {count}/{saved_count}")

    print("\nQuality flags:")
    for flag, count in flag_counts.items():
        print(f"  {flag}: {count}")

    print("\nSaved:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)
    print("Summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()