
# from pathlib import Path
# from collections import Counter
# import json
# # import random

# import pandas as pd

# from parse_sc_judgments_test import parse_judgment


# PDF_FOLDER = Path("data/raw/supreme_court/pdfs")

# OUTPUT_FILE = Path(
#     "data/processed/supreme_court/sc_pilot_200.jsonl"
# )

# REPORT_FILE = Path(
#     "outputs/sc_pilot_200_quality.csv"
# )

# SUMMARY_FILE = Path(
#     "outputs/sc_pilot_200_summary.json"
# )

# SAMPLE_SIZE = 200
# RANDOM_SEED = 42


# def main():
#     all_pdfs = sorted(PDF_FOLDER.glob("*.pdf"))

#     if not all_pdfs:
#         raise FileNotFoundError(
#             f"No PDFs found in {PDF_FOLDER}"
#         )

#     # Deterministic random sample across the collection.
#     rng = random.Random(RANDOM_SEED)

#     sample_pdfs = sorted(
#         rng.sample(
#             all_pdfs,
#             min(SAMPLE_SIZE, len(all_pdfs))
#         )
#     )

#     OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
#     REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

#     report_rows = []
#     field_counts = Counter()
#     flag_counts = Counter()
#     status_counts = Counter()

#     print("=" * 70)
#     print("SUPREME COURT PILOT")
#     print("Available PDFs:", len(all_pdfs))
#     print("Selected PDFs:", len(sample_pdfs))
#     print("=" * 70)

#     with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

#         for index, pdf_path in enumerate(sample_pdfs, start=1):

#             print(
#                 f"[{index}/{len(sample_pdfs)}] "
#                 f"{pdf_path.name}"
#             )

#             try:
#                 record = parse_judgment(pdf_path)

#                 # Save immediately instead of keeping all
#                 # judgment texts in RAM.
#                 output.write(
#                     json.dumps(record, ensure_ascii=False)
#                     + "\n"
#                 )

#                 meta = record["metadata"]
#                 flags = meta["quality_flags"]

#                 status_counts[meta["parse_status"]] += 1
#                 flag_counts.update(flags)

#                 fields = [
#                     "petitioner",
#                     "respondent",
#                     "judgment_date",
#                     "bench",
#                     "citation_block",
#                     "headnote",
#                     "judgment_text"
#                 ]

#                 for field in fields:
#                     if record.get(field):
#                         field_counts[field] += 1

#                 report_rows.append({
#                     "source_file": pdf_path.name,
#                     "status": meta["parse_status"],
#                     "date": record["judgment_date"],
#                     "title": record["title"],
#                     "page_count": meta["page_count"],
#                     "character_count": meta["character_count"],
#                     "judgment_characters":
#                         meta["judgment_character_count"],
#                     "has_petitioner": bool(record["petitioner"]),
#                     "has_respondent": bool(record["respondent"]),
#                     "has_bench": bool(record["bench"]),
#                     "has_citation": bool(record["citation_block"]),
#                     "has_headnote": bool(record["headnote"]),
#                     "has_judgment": bool(record["judgment_text"]),
#                     "legacy_header_found":
#                         meta["legacy_header_found"],
#                     "quality_flags": "; ".join(flags),
#                     "source_sha256": meta["source_sha256"]
#                 })

#             except Exception as exc:

#                 status_counts["extraction_error"] += 1

#                 report_rows.append({
#                     "source_file": pdf_path.name,
#                     "status": "extraction_error",
#                     "quality_flags": str(exc)
#                 })

#                 print("ERROR:", exc)

#     # Save the quality report.
#     report_df = pd.DataFrame(report_rows)

#     report_df.to_csv(
#         REPORT_FILE,
#         index=False,
#         encoding="utf-8-sig"
#     )

#     saved_count = sum(
#         count
#         for status, count in status_counts.items()
#         if status != "extraction_error"
#     )

#     summary = {
#         "available_pdfs": len(all_pdfs),
#         "attempted": len(sample_pdfs),
#         "saved": saved_count,
#         "status_counts": dict(status_counts),
#         "field_counts": dict(field_counts),
#         "quality_flags": dict(flag_counts)
#     }

#     SUMMARY_FILE.write_text(
#         json.dumps(summary, indent=2),
#         encoding="utf-8"
#     )

#     print("\n" + "=" * 70)
#     print("PILOT SUMMARY")
#     print("=" * 70)

#     print("PDFs attempted:", len(sample_pdfs))
#     print("Documents saved:", saved_count)

#     print("\nStatus counts:")
#     for status, count in status_counts.items():
#         print(f"  {status}: {count}")

#     print("\nField counts:")
#     for field, count in field_counts.items():
#         print(f"  {field}: {count}/{saved_count}")

#     print("\nQuality flags:")
#     for flag, count in flag_counts.items():
#         print(f"  {flag}: {count}")

#     print("\nSaved:", OUTPUT_FILE)
#     print("Report:", REPORT_FILE)
#     print("Summary:", SUMMARY_FILE)


# if __name__ == "__main__":
#     main()



from pathlib import Path
from collections import Counter
import json

import pandas as pd

from parse_sc_judgments_test import parse_judgment


# ============================================================
# PATHS
# ============================================================

PDF_FOLDER = Path(
    "../data/raw/supreme_court/pdfs"
)

OUTPUT_FILE = Path(
    "data/processed/supreme_court/sc_all.jsonl"
)

REPORT_FILE = Path(
    "outputs/sc_all_quality.csv"
)

SUMMARY_FILE = Path(
    "outputs/sc_all_summary.json"
)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Find ALL PDFs
    # --------------------------------------------------------

    all_pdfs = sorted(
        PDF_FOLDER.glob("*.pdf")
    )

    if not all_pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_FOLDER}"
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    report_rows = []

    field_counts = Counter()
    flag_counts = Counter()
    status_counts = Counter()

    print("=" * 70)
    print("SUPREME COURT FULL DATASET PARSING")
    print("=" * 70)

    print(
        "PDF folder:",
        PDF_FOLDER
    )

    print(
        "Total PDFs found:",
        len(all_pdfs)
    )

    print(
        "Output:",
        OUTPUT_FILE
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Parse every PDF
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for index, pdf_path in enumerate(
            all_pdfs,
            start=1
        ):

            print(
                f"[{index}/{len(all_pdfs)}] "
                f"{pdf_path.name}"
            )

            try:

                # --------------------------------------------
                # Parse judgment
                # --------------------------------------------

                record = parse_judgment(
                    pdf_path
                )

                # --------------------------------------------
                # Save immediately
                #
                # We do not keep judgment bodies in RAM.
                # --------------------------------------------

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                # Flush periodically so that data already parsed
                # is physically written even during a long run.
                if index % 100 == 0:
                    output.flush()

                # --------------------------------------------
                # Metadata / quality information
                # --------------------------------------------

                meta = record.get(
                    "metadata",
                    {}
                )

                flags = meta.get(
                    "quality_flags",
                    []
                )

                parse_status = meta.get(
                    "parse_status",
                    "unknown"
                )

                status_counts[
                    parse_status
                ] += 1

                flag_counts.update(
                    flags
                )

                # --------------------------------------------
                # Count extracted fields
                # --------------------------------------------

                fields = [
                    "petitioner",
                    "respondent",
                    "judgment_date",
                    "bench",
                    "citation_block",
                    "headnote",
                    "judgment_text"
                ]

                for field in fields:

                    if record.get(field):
                        field_counts[
                            field
                        ] += 1

                # --------------------------------------------
                # Add quality-report row
                # --------------------------------------------

                report_rows.append({
                    "source_file":
                        pdf_path.name,

                    "status":
                        parse_status,

                    "date":
                        record.get(
                            "judgment_date"
                        ),

                    "title":
                        record.get(
                            "title"
                        ),

                    "page_count":
                        meta.get(
                            "page_count"
                        ),

                    "character_count":
                        meta.get(
                            "character_count"
                        ),

                    "judgment_characters":
                        meta.get(
                            "judgment_character_count"
                        ),

                    "has_petitioner":
                        bool(
                            record.get(
                                "petitioner"
                            )
                        ),

                    "has_respondent":
                        bool(
                            record.get(
                                "respondent"
                            )
                        ),

                    "has_bench":
                        bool(
                            record.get(
                                "bench"
                            )
                        ),

                    "has_citation":
                        bool(
                            record.get(
                                "citation_block"
                            )
                        ),

                    "has_headnote":
                        bool(
                            record.get(
                                "headnote"
                            )
                        ),

                    "has_judgment":
                        bool(
                            record.get(
                                "judgment_text"
                            )
                        ),

                    "legacy_header_found":
                        meta.get(
                            "legacy_header_found"
                        ),

                    "quality_flags":
                        "; ".join(flags),

                    "source_sha256":
                        meta.get(
                            "source_sha256"
                        )
                })

            # ------------------------------------------------
            # Do not stop entire run because of one bad PDF.
            # ------------------------------------------------

            except Exception as exc:

                status_counts[
                    "extraction_error"
                ] += 1

                report_rows.append({
                    "source_file":
                        pdf_path.name,

                    "status":
                        "extraction_error",

                    "date":
                        None,

                    "title":
                        None,

                    "page_count":
                        None,

                    "character_count":
                        None,

                    "judgment_characters":
                        None,

                    "has_petitioner":
                        False,

                    "has_respondent":
                        False,

                    "has_bench":
                        False,

                    "has_citation":
                        False,

                    "has_headnote":
                        False,

                    "has_judgment":
                        False,

                    "legacy_header_found":
                        None,

                    "quality_flags":
                        str(exc),

                    "source_sha256":
                        None
                })

                print(
                    "ERROR:",
                    type(exc).__name__,
                    "-",
                    exc
                )

            # ------------------------------------------------
            # Progress checkpoint
            # ------------------------------------------------

            if (
                index % 500 == 0
                or index == len(all_pdfs)
            ):

                print(
                    "\n"
                    + "-" * 70
                )

                print(
                    f"Progress: "
                    f"{index}/{len(all_pdfs)}"
                )

                print(
                    "Parsed/saved so far:",
                    index
                    - status_counts[
                        "extraction_error"
                    ]
                )

                print(
                    "Extraction errors:",
                    status_counts[
                        "extraction_error"
                    ]
                )

                print(
                    "-" * 70
                    + "\n"
                )

    # ========================================================
    # QUALITY REPORT
    # ========================================================

    report_df = pd.DataFrame(
        report_rows
    )

    report_df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    saved_count = (
        len(all_pdfs)
        - status_counts[
            "extraction_error"
        ]
    )

    summary = {
        "available_pdfs":
            len(all_pdfs),

        "attempted":
            len(all_pdfs),

        "saved":
            saved_count,

        "extraction_errors":
            status_counts[
                "extraction_error"
            ],

        "status_counts":
            dict(
                status_counts
            ),

        "field_counts":
            dict(
                field_counts
            ),

        "quality_flags":
            dict(
                flag_counts
            ),

        "output_file":
            str(
                OUTPUT_FILE
            ),

        "report_file":
            str(
                REPORT_FILE
            )
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # ========================================================
    # FINAL TERMINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FULL SUPREME COURT PARSING SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "PDFs attempted:",
        len(all_pdfs)
    )

    print(
        "Documents saved:",
        saved_count
    )

    print(
        "Extraction errors:",
        status_counts[
            "extraction_error"
        ]
    )

    print(
        "\nStatus counts:"
    )

    for status, count in (
        status_counts.items()
    ):

        print(
            f"  {status}: {count}"
        )

    print(
        "\nField counts:"
    )

    for field, count in (
        field_counts.items()
    ):

        print(
            f"  {field}: "
            f"{count}/{saved_count}"
        )

    print(
        "\nQuality flags:"
    )

    for flag, count in (
        flag_counts.items()
    ):

        print(
            f"  {flag}: {count}"
        )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "Report:"
    )

    print(
        REPORT_FILE
    )

    print(
        "Summary:"
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()