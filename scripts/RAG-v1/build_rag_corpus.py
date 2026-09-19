from pathlib import Path
from collections import Counter
import json
import re


# ============================================================
# INPUT FILES
# ============================================================

CENTRAL_ACTS_FILE = Path(
    "data/processed/central_acts/central_acts_sections.jsonl"
)

CONSTITUTION_FILE = Path(
    "data/processed/constitution/constitution_pages_structured.jsonl"
)

SUPREME_COURT_FILE = Path(
    "data/processed/supreme_court/sc_pilot_200_v2.jsonl"
)


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = Path("data/rag/v1")

OUTPUT_FILE = OUT_DIR / "legal_documents.jsonl"
SUMMARY_FILE = OUT_DIR / "legal_documents_summary.json"


# ============================================================
# CHUNK SETTINGS
# ============================================================

# BGE-small supports up to 512 tokens.
# ~220 words gives us a conservative margin.
MAX_WORDS = 220
OVERLAP_WORDS = 40


# Constitution body boundaries already validated earlier.
CONSTITUTION_START_PAGE = 33
CONSTITUTION_END_PAGE = 283


def clean_whitespace(text):
    """
    Conservative cleanup only.
    Preserve legal wording; normalize whitespace.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing spaces but preserve paragraph structure.
    lines = [
        re.sub(r"[ \t]+$", "", line)
        for line in text.splitlines()
    ]

    text = "\n".join(lines)

    # Collapse 3+ blank lines to maximum 2.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(text, max_words=MAX_WORDS, overlap_words=OVERLAP_WORDS):
    """
    Simple word-based chunker for Prototype v1.

    We intentionally avoid aggressive sentence rewriting.
    """
    text = clean_whitespace(text)

    if not text:
        return []

    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks = []

    start = 0

    while start < len(words):
        end = min(
            start + max_words,
            len(words)
        )

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap_words

    return chunks


def write_record(writer, record, stats):
    writer.write(
        json.dumps(
            record,
            ensure_ascii=False
        ) + "\n"
    )

    stats["documents"] += 1
    stats["source_types"][record["metadata"]["source_type"]] += 1


def process_central_acts(writer, stats):
    print("\nProcessing Central Acts...")

    source_records = 0

    with open(
        CENTRAL_ACTS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            if not line.strip():
                continue

            source = json.loads(line)
            source_records += 1

            text = clean_whitespace(
                source.get("text", "")
            )

            if not text:
                stats["skipped_empty"] += 1
                continue

            chunks = chunk_text(text)

            metadata = source.get(
                "metadata",
                {}
            )

            for chunk_index, chunk in enumerate(
                chunks,
                start=1
            ):

                record = {
                    "id": (
                        f"{source['document_id']}"
                        f"_chunk_{chunk_index}"
                    ),

                    "text": chunk,

                    "metadata": {
                        "source_type":
                            "central_act",

                        "document_id":
                            source.get("document_id"),

                        "title":
                            source.get("title"),

                        "jurisdiction":
                            source.get(
                                "jurisdiction",
                                "India"
                            ),

                        "section":
                            metadata.get("section"),

                        "heading":
                            metadata.get("heading"),

                        "hierarchy":
                            metadata.get("hierarchy"),

                        "act_id":
                            metadata.get("act_id"),

                        "enactment_date_raw":
                            metadata.get(
                                "enactment_date_raw"
                            ),

                        "source_file":
                            metadata.get(
                                "source_file"
                            ),

                        "source_path":
                            metadata.get(
                                "source_path"
                            ),

                        "source_version":
                            metadata.get(
                                "source_version"
                            ),

                        "legal_status":
                            metadata.get(
                                "legal_status"
                            ),

                        "chunk_index":
                            chunk_index,

                        "chunk_count":
                            len(chunks)
                    }
                }

                write_record(
                    writer,
                    record,
                    stats
                )

    stats["central_act_source_records"] = source_records


def process_constitution(writer, stats):
    print("Processing Constitution...")

    source_pages = 0

    with open(
        CONSTITUTION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            if not line.strip():
                continue

            source = json.loads(line)

            pdf_page = source.get(
                "pdf_page"
            )

            if not (
                CONSTITUTION_START_PAGE
                <= pdf_page
                <= CONSTITUTION_END_PAGE
            ):
                continue

            source_pages += 1

            text = clean_whitespace(
                source.get(
                    "main_text",
                    ""
                )
            )

            if not text:
                stats["skipped_empty"] += 1
                continue

            chunks = chunk_text(text)

            for chunk_index, chunk in enumerate(
                chunks,
                start=1
            ):

                record = {
                    "id": (
                        f"constitution_2024_"
                        f"page_{pdf_page}_"
                        f"chunk_{chunk_index}"
                    ),

                    "text": chunk,

                    "metadata": {
                        "source_type":
                            "constitution",

                        "title":
                            "Constitution of India",

                        "jurisdiction":
                            "India",

                        "source_file":
                            source.get(
                                "source_file"
                            ),

                        "source_edition":
                            source.get(
                                "source_edition"
                            ),

                        "pdf_page":
                            pdf_page,

                        "quality_flags":
                            source.get(
                                "quality_flags",
                                []
                            ),

                        "chunk_index":
                            chunk_index,

                        "chunk_count":
                            len(chunks)
                    }
                }

                write_record(
                    writer,
                    record,
                    stats
                )

    stats["constitution_source_pages"] = source_pages


def process_supreme_court(writer, stats):
    print("Processing Supreme Court judgments...")

    total_records = 0
    usable_records = 0
    skipped_non_usable = 0

    with open(
        SUPREME_COURT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            if not line.strip():
                continue

            source = json.loads(line)
            total_records += 1

            metadata = source.get(
                "metadata",
                {}
            )

            if (
                metadata.get("text_status")
                != "usable"
            ):
                skipped_non_usable += 1
                continue

            text = clean_whitespace(
                source.get(
                    "judgment_text",
                    ""
                )
            )

            if not text:
                stats["skipped_empty"] += 1
                continue

            usable_records += 1

            chunks = chunk_text(text)

            document_id = source.get(
                "document_id"
            )

            for chunk_index, chunk in enumerate(
                chunks,
                start=1
            ):

                record = {
                    "id": (
                        f"{document_id}"
                        f"_chunk_{chunk_index}"
                    ),

                    "text": chunk,

                    "metadata": {
                        "source_type":
                            "supreme_court",

                        "document_id":
                            document_id,

                        "title":
                            source.get("title"),

                        "jurisdiction":
                            source.get(
                                "jurisdiction",
                                "India"
                            ),

                        "court":
                            source.get("court"),

                        "judgment_date":
                            source.get(
                                "judgment_date"
                            ),

                        "petitioner":
                            source.get(
                                "petitioner"
                            ),

                        "respondent":
                            source.get(
                                "respondent"
                            ),

                        "bench":
                            source.get(
                                "bench"
                            ),

                        "citation":
                            source.get(
                                "citation_block"
                            ),

                        "language":
                            source.get(
                                "language"
                            ),

                        "source_file":
                            metadata.get(
                                "source_file"
                            ),

                        "document_format":
                            metadata.get(
                                "document_format"
                            ),

                        "parse_status":
                            metadata.get(
                                "parse_status"
                            ),

                        "quality_flags":
                            metadata.get(
                                "quality_flags",
                                []
                            ),

                        "text_status":
                            metadata.get(
                                "text_status"
                            ),

                        "chunk_index":
                            chunk_index,

                        "chunk_count":
                            len(chunks)
                    }
                }

                write_record(
                    writer,
                    record,
                    stats
                )

    stats["supreme_court_total_records"] = total_records
    stats["supreme_court_usable_records"] = usable_records
    stats[
        "supreme_court_skipped_non_usable"
    ] = skipped_non_usable


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stats = {
        "documents": 0,
        "source_types": Counter(),
        "skipped_empty": 0
    }

    print("=" * 70)
    print("RAG PROTOTYPE V1 - CORPUS BUILDER")
    print("=" * 70)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as writer:

        process_central_acts(
            writer,
            stats
        )

        process_constitution(
            writer,
            stats
        )

        process_supreme_court(
            writer,
            stats
        )

    summary = {
        "total_rag_documents":
            stats["documents"],

        "source_type_counts":
            dict(
                stats["source_types"]
            ),

        "central_act_source_records":
            stats.get(
                "central_act_source_records",
                0
            ),

        "constitution_source_pages":
            stats.get(
                "constitution_source_pages",
                0
            ),

        "supreme_court_total_records":
            stats.get(
                "supreme_court_total_records",
                0
            ),

        "supreme_court_usable_records":
            stats.get(
                "supreme_court_usable_records",
                0
            ),

        "supreme_court_skipped_non_usable":
            stats.get(
                "supreme_court_skipped_non_usable",
                0
            ),

        "skipped_empty":
            stats["skipped_empty"],

        "chunk_settings": {
            "max_words":
                MAX_WORDS,

            "overlap_words":
                OVERLAP_WORDS
        }
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("RAG CORPUS SUMMARY")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()