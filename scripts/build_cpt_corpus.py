from pathlib import Path
from collections import Counter
import hashlib
import json
import re


# ============================================================
# INPUT FILES
# ============================================================

CENTRAL_ACTS_FILE = Path(
    "../data/processed/central_acts/central_acts_sections.jsonl"
)

CONSTITUTION_FILE = Path(
    "../data/processed/constitution/constitution_pages_structured.jsonl"
)

SUPREME_COURT_FILE = Path(
    "data/processed/supreme_court/sc_all.jsonl"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_FILE = Path(
    "data/training/cpt/legal_cpt_corpus.jsonl"
)

SUMMARY_FILE = Path(
    "data/training/cpt/legal_cpt_corpus_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

# We already validated that the core Constitution text starts
# at PDF page 33 and Article 395 ends at PDF page 283.
#
# This intentionally excludes:
# - front matter
# - table of contents
# - schedules
# - appendices
# - footnotes
#
# You can add those later if wanted.
CONSTITUTION_START_PAGE = 33
CONSTITUTION_END_PAGE = 283


# Very tiny records are usually not useful for CPT.
MIN_TEXT_CHARS = 100


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    """
    Conservative legal-text cleaning.

    We preserve paragraph structure and punctuation while fixing
    obvious whitespace problems.
    """

    if not text:
        return ""

    text = str(text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace non-breaking spaces.
    text = text.replace("\u00a0", " ")

    # Remove spaces before newline.
    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text
    )

    # Collapse repeated spaces, but preserve newlines.
    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text
    )

    # Avoid huge runs of empty lines.
    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text
    )

    return text.strip()


def text_hash(text):
    """
    Used for exact duplicate detection.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_jsonl(path):
    """
    Stream JSONL one record at a time.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                yield json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {exc}"
                )


def safe_value(value):
    """
    Convert empty strings into None.
    """

    if value is None:
        return None

    if isinstance(value, str):

        value = value.strip()

        if not value:
            return None

    return value


# ============================================================
# CENTRAL ACTS
# ============================================================

def process_central_acts(
    output,
    seen_hashes,
    counters
):

    print("\n" + "=" * 70)
    print("PROCESSING CENTRAL ACTS")
    print("=" * 70)

    for record in load_jsonl(
        CENTRAL_ACTS_FILE
    ):

        counters[
            "central_act_input_records"
        ] += 1

        raw_text = clean_text(
            record.get("text")
        )

        if len(raw_text) < MIN_TEXT_CHARS:

            counters[
                "central_act_skipped_short"
            ] += 1

            continue

        metadata = record.get(
            "metadata",
            {}
        )

        title = safe_value(
            record.get("title")
        )

        section = safe_value(
            metadata.get("section")
        )

        heading = safe_value(
            metadata.get("heading")
        )

        # ----------------------------------------------------
        # Build a natural-text header.
        #
        # Useful document structure is included in the text,
        # while implementation metadata such as SHA256 and paths
        # stays outside the training text.
        # ----------------------------------------------------

        header_parts = []

        if title:
            header_parts.append(
                title
            )

        if section:
            header_parts.append(
                section
            )

        if (
            heading
            and heading not in raw_text[:300]
        ):
            header_parts.append(
                heading
            )

        if header_parts:

            training_text = (
                "\n".join(header_parts)
                + "\n\n"
                + raw_text
            )

        else:

            training_text = raw_text

        training_text = clean_text(
            training_text
        )

        content_hash = text_hash(
            training_text
        )

        if content_hash in seen_hashes:

            counters[
                "central_act_duplicates"
            ] += 1

            continue

        seen_hashes.add(
            content_hash
        )

        document_id = (
            record.get("document_id")
            or f"central_act_{content_hash[:20]}"
        )

        output_record = {
            "id": document_id,
            "source_type": "central_act",
            "text": training_text,
            "metadata": {
                "title":
                    title,

                "jurisdiction":
                    record.get(
                        "jurisdiction",
                        "India"
                    ),

                "act_id":
                    safe_value(
                        metadata.get(
                            "act_id"
                        )
                    ),

                "section":
                    section,

                "heading":
                    heading,

                "source_file":
                    safe_value(
                        metadata.get(
                            "source_file"
                        )
                    ),

                "source_version":
                    safe_value(
                        metadata.get(
                            "source_version"
                        )
                    ),

                "legal_status":
                    safe_value(
                        metadata.get(
                            "legal_status"
                        )
                    )
            }
        }

        output.write(
            json.dumps(
                output_record,
                ensure_ascii=False
            )
            + "\n"
        )

        counters[
            "central_act_saved"
        ] += 1


# ============================================================
# CONSTITUTION
# ============================================================

def process_constitution(
    output,
    seen_hashes,
    counters
):

    print("\n" + "=" * 70)
    print("PROCESSING CONSTITUTION")
    print("=" * 70)

    for record in load_jsonl(
        CONSTITUTION_FILE
    ):

        counters[
            "constitution_input_records"
        ] += 1

        pdf_page = record.get(
            "pdf_page"
        )

        # ----------------------------------------------------
        # Only include the validated core constitutional text.
        # ----------------------------------------------------

        if pdf_page is None:

            counters[
                "constitution_skipped_no_page"
            ] += 1

            continue

        if not (
            CONSTITUTION_START_PAGE
            <= pdf_page
            <= CONSTITUTION_END_PAGE
        ):

            counters[
                "constitution_skipped_outside_core"
            ] += 1

            continue

        raw_text = clean_text(
            record.get(
                "main_text"
            )
        )

        if len(raw_text) < MIN_TEXT_CHARS:

            counters[
                "constitution_skipped_short"
            ] += 1

            continue

        # Do not include footnotes_text here.
        #
        # Some constitutional footnotes describe substituted,
        # historical or uncommenced text. For CPT v1 we keep the
        # validated main constitutional text only.

        training_text = clean_text(
            "Constitution of India\n\n"
            + raw_text
        )

        content_hash = text_hash(
            training_text
        )

        if content_hash in seen_hashes:

            counters[
                "constitution_duplicates"
            ] += 1

            continue

        seen_hashes.add(
            content_hash
        )

        source_file = safe_value(
            record.get(
                "source_file"
            )
        )

        source_edition = safe_value(
            record.get(
                "source_edition"
            )
        )

        document_id = (
            f"constitution_page_{pdf_page}"
        )

        output_record = {
            "id": document_id,
            "source_type": "constitution",
            "text": training_text,
            "metadata": {
                "title":
                    "Constitution of India",

                "jurisdiction":
                    "India",

                "pdf_page":
                    pdf_page,

                "source_file":
                    source_file,

                "source_edition":
                    source_edition
            }
        }

        output.write(
            json.dumps(
                output_record,
                ensure_ascii=False
            )
            + "\n"
        )

        counters[
            "constitution_saved"
        ] += 1


# ============================================================
# SUPREME COURT
# ============================================================

def process_supreme_court(
    output,
    seen_hashes,
    counters
):

    print("\n" + "=" * 70)
    print("PROCESSING SUPREME COURT JUDGMENTS")
    print("=" * 70)

    for index, record in enumerate(
        load_jsonl(
            SUPREME_COURT_FILE
        ),
        start=1
    ):

        counters[
            "supreme_court_input_records"
        ] += 1

        metadata = record.get(
            "metadata",
            {}
        )

        # ----------------------------------------------------
        # If the parser explicitly marked text as unusable,
        # exclude it.
        #
        # This is deliberately tolerant because different parser
        # versions may not have a text_status field.
        # ----------------------------------------------------

        text_status = metadata.get(
            "text_status"
        )

        if (
            text_status is not None
            and text_status != "usable"
        ):

            counters[
                "supreme_court_skipped_non_usable"
            ] += 1

            continue

        # Prefer judgment_text.
        # Fall back to text only if needed.
        raw_text = (
            record.get(
                "judgment_text"
            )
            or record.get(
                "text"
            )
            or ""
        )

        raw_text = clean_text(
            raw_text
        )

        if len(raw_text) < MIN_TEXT_CHARS:

            counters[
                "supreme_court_skipped_short"
            ] += 1

            continue

        title = safe_value(
            record.get("title")
        )

        judgment_date = safe_value(
            record.get(
                "judgment_date"
            )
        )

        citation = safe_value(
            record.get(
                "citation_block"
            )
        )

        # ----------------------------------------------------
        # Build natural text header.
        # ----------------------------------------------------

        header_parts = [
            "Supreme Court of India"
        ]

        if title:
            header_parts.append(
                title
            )

        if judgment_date:
            header_parts.append(
                f"Judgment Date: {judgment_date}"
            )

        # Citation blocks can sometimes be very large or noisy.
        # We keep them in metadata rather than injecting them into
        # the CPT text.

        training_text = clean_text(
            "\n".join(
                header_parts
            )
            + "\n\n"
            + raw_text
        )

        content_hash = text_hash(
            training_text
        )

        if content_hash in seen_hashes:

            counters[
                "supreme_court_duplicates"
            ] += 1

            continue

        seen_hashes.add(
            content_hash
        )

        document_id = (
            record.get(
                "document_id"
            )
            or f"sc_{content_hash[:20]}"
        )

        output_record = {
            "id": document_id,
            "source_type": "supreme_court",
            "text": training_text,
            "metadata": {
                "title":
                    title,

                "jurisdiction":
                    record.get(
                        "jurisdiction",
                        "India"
                    ),

                "court":
                    record.get(
                        "court",
                        "Supreme Court of India"
                    ),

                "judgment_date":
                    judgment_date,

                "citation":
                    citation,

                "source_file":
                    safe_value(
                        metadata.get(
                            "source_file"
                        )
                    ),

                "parse_status":
                    safe_value(
                        metadata.get(
                            "parse_status"
                        )
                    ),

                "text_status":
                    safe_value(
                        metadata.get(
                            "text_status"
                        )
                    )
            }
        }

        output.write(
            json.dumps(
                output_record,
                ensure_ascii=False
            )
            + "\n"
        )

        counters[
            "supreme_court_saved"
        ] += 1

        if index % 1000 == 0:

            print(
                f"Processed Supreme Court "
                f"records: {index}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING UNIFIED CPT LEGAL CORPUS")
    print("=" * 70)

    print(
        "Central Acts:",
        CENTRAL_ACTS_FILE
    )

    print(
        "Constitution:",
        CONSTITUTION_FILE
    )

    print(
        "Supreme Court:",
        SUPREME_COURT_FILE
    )

    # --------------------------------------------------------
    # Validate files before starting.
    # --------------------------------------------------------

    input_files = [
        CENTRAL_ACTS_FILE,
        CONSTITUTION_FILE,
        SUPREME_COURT_FILE
    ]

    for path in input_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    counters = Counter()

    # Global exact duplicate detection across all datasets.
    seen_hashes = set()

    # --------------------------------------------------------
    # Write unified file
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        process_central_acts(
            output,
            seen_hashes,
            counters
        )

        process_constitution(
            output,
            seen_hashes,
            counters
        )

        process_supreme_court(
            output,
            seen_hashes,
            counters
        )

    # --------------------------------------------------------
    # Totals
    # --------------------------------------------------------

    total_saved = (
        counters[
            "central_act_saved"
        ]
        + counters[
            "constitution_saved"
        ]
        + counters[
            "supreme_court_saved"
        ]
    )

    total_duplicates = (
        counters[
            "central_act_duplicates"
        ]
        + counters[
            "constitution_duplicates"
        ]
        + counters[
            "supreme_court_duplicates"
        ]
    )

    summary = {
        "output_file":
            str(OUTPUT_FILE),

        "total_saved":
            total_saved,

        "source_counts": {
            "central_act":
                counters[
                    "central_act_saved"
                ],

            "constitution":
                counters[
                    "constitution_saved"
                ],

            "supreme_court":
                counters[
                    "supreme_court_saved"
                ]
        },

        "duplicates_removed":
            total_duplicates,

        "minimum_text_characters":
            MIN_TEXT_CHARS,

        "constitution_pages_included": {
            "start":
                CONSTITUTION_START_PAGE,

            "end":
                CONSTITUTION_END_PAGE
        },

        "details":
            dict(counters)
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CPT CORPUS BUILD COMPLETE")
    print("=" * 70)

    print(
        "Central Act records:",
        counters[
            "central_act_saved"
        ]
    )

    print(
        "Constitution records:",
        counters[
            "constitution_saved"
        ]
    )

    print(
        "Supreme Court records:",
        counters[
            "supreme_court_saved"
        ]
    )

    print(
        "\nTotal CPT documents:",
        total_saved
    )

    print(
        "Exact duplicates removed:",
        total_duplicates
    )

    print(
        "\nSaved corpus:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nSaved summary:"
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()