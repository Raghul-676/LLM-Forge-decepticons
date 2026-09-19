from pathlib import Path
import pandas as pd
import fitz


PDF_FOLDER = Path("data/raw/supreme_court/pdfs")

REPORT_FILE = Path(
    "outputs/sc_pilot_200_quality.csv"
)

OUTPUT_FILE = Path(
    "outputs/sc_failure_inspection.txt"
)


# ---------------------------------------------------------
# Load pilot quality report
# ---------------------------------------------------------

df = pd.read_csv(REPORT_FILE)

unsupported = df[
    df["quality_flags"]
    .fillna("")
    .str.contains("unsupported_header_format", regex=False)
]

errors = df[
    df["status"] == "extraction_error"
]


print("Unsupported headers:", len(unsupported))
print("Extraction errors:", len(errors))


# ---------------------------------------------------------
# Inspect five unsupported PDFs
# ---------------------------------------------------------

report_parts = []

for _, row in unsupported.head(5).iterrows():

    filename = row["source_file"]
    pdf_path = PDF_FOLDER / filename

    print("Inspecting:", filename)

    try:

        with fitz.open(pdf_path) as document:

            page_count = len(document)

            # Only first two pages are needed to inspect
            # the document's header and structure.
            pages = []

            for page_index in range(min(2, page_count)):

                pages.append(
                    document[page_index].get_text("text")
                )

        preview = "\n".join(pages)

        report_parts.append(
            "=" * 80 + "\n"
            f"FILE: {filename}\n"
            f"PAGES: {page_count}\n"
            f"FILE SIZE: {pdf_path.stat().st_size} bytes\n"
            "=" * 80 + "\n\n"
            "----- FIRST TWO PAGES -----\n"
            f"{preview[:10000]}\n\n"
        )

    except Exception as exc:

        report_parts.append(
            "=" * 80 + "\n"
            f"FILE: {filename}\n"
            f"ERROR: {exc}\n\n"
        )


# ---------------------------------------------------------
# Show all extraction errors from the pilot
# ---------------------------------------------------------

report_parts.append(
    "\n" + "=" * 80 + "\n"
    "EXTRACTION ERRORS FROM PILOT\n"
    + "=" * 80 + "\n\n"
)

for _, row in errors.iterrows():

    report_parts.append(
        f"FILE: {row['source_file']}\n"
        f"ERROR: {row['quality_flags']}\n\n"
    )


# ---------------------------------------------------------
# Save inspection report
# ---------------------------------------------------------

OUTPUT_FILE.write_text(
    "\n".join(report_parts),
    encoding="utf-8"
)

print("\nSaved:", OUTPUT_FILE)
print("Inspection complete.")