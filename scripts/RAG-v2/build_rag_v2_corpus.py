import json
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIG
# ============================================================

RAG_VERSION = "v2"

# A ~320-word legal chunk normally stays reasonably close to
# BGE-small's 512-token input limit.
CHUNK_WORDS = 320

# Small overlap helps preserve rules/reasoning split across chunks.
OVERLAP_WORDS = 50

# Skip truly tiny accidental fragments.
MIN_WORDS = 20


# ============================================================
# FIND PROJECT ROOT
# ============================================================

def find_project_root():

    script_path = Path(__file__).resolve()

    for parent in script_path.parents:

        if (parent / "scripts").is_dir():
            return parent

    raise RuntimeError(
        "Could not locate project root."
    )


PROJECT_ROOT = find_project_root()


# ============================================================
# FIND DATA ROOT
# ============================================================

def find_data_root():

    # Your recent cleaned CPT corpus was under scripts/data,
    # so prefer that location first.
    candidates = [
        PROJECT_ROOT / "scripts" / "data",
        PROJECT_ROOT / "data",
    ]

    relative_input = Path(
        "training/cpt/clean/legal_cpt_clean.jsonl"
    )

    for data_root in candidates:

        candidate_file = (
            data_root
            / relative_input
        )

        if candidate_file.exists():

            return data_root

    checked = "\n".join(
        str(
            root
            / relative_input
        )
        for root in candidates
    )

    raise FileNotFoundError(
        "Could not find legal_cpt_clean.jsonl.\n"
        "Checked:\n"
        + checked
    )


DATA_ROOT = find_data_root()


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    DATA_ROOT
    / "training"
    / "cpt"
    / "clean"
    / "legal_cpt_clean.jsonl"
)

OUTPUT_DIR = (
    DATA_ROOT
    / "rag"
    / "v2"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "legal_documents.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "legal_documents_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):

    if value is None:
        return None

    if isinstance(
        value,
        str
    ):

        value = value.strip()

        if not value:
            return None

    return value


def get_value(
    record,
    metadata,
    *keys
):

    for key in keys:

        value = clean_value(
            record.get(key)
        )

        if value is not None:
            return value

        value = clean_value(
            metadata.get(key)
        )

        if value is not None:
            return value

    return None


def normalize_text(text):

    if not text:
        return ""

    # For dense retrieval, preserving every original newline
    # is unnecessary. Normalized spacing also reduces file size.
    return " ".join(
        str(text).split()
    )


# ============================================================
# WORD CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_words=CHUNK_WORDS,
    overlap_words=OVERLAP_WORDS
):

    words = text.split()

    total_words = len(words)

    if total_words == 0:
        return []

    if total_words <= chunk_words:

        return [
            {
                "text": " ".join(words),
                "word_start": 0,
                "word_end": total_words,
                "word_count": total_words,
            }
        ]

    stride = (
        chunk_words
        - overlap_words
    )

    if stride <= 0:

        raise ValueError(
            "OVERLAP_WORDS must be smaller "
            "than CHUNK_WORDS."
        )

    chunks = []

    start = 0

    while start < total_words:

        end = min(
            start + chunk_words,
            total_words
        )

        chunk_words_list = (
            words[start:end]
        )

        # Do not write accidental tiny tails.
        # Because of overlap this should be uncommon.
        if (
            len(chunk_words_list)
            >= MIN_WORDS
        ):

            chunks.append({
                "text":
                    " ".join(
                        chunk_words_list
                    ),

                "word_start":
                    start,

                "word_end":
                    end,

                "word_count":
                    len(
                        chunk_words_list
                    ),
            })

        if end >= total_words:
            break

        start += stride

    return chunks


# ============================================================
# RETRIEVAL HEADER
# ============================================================

def build_retrieval_header(
    source_type,
    record,
    metadata
):

    lines = []

    # --------------------------------------------------------
    # CENTRAL ACT
    # --------------------------------------------------------

    if source_type == "central_act":

        title = get_value(
            record,
            metadata,
            "title"
        )

        section = get_value(
            record,
            metadata,
            "section"
        )

        heading = get_value(
            record,
            metadata,
            "heading"
        )

        if title:
            lines.append(
                f"Act: {title}"
            )

        if section:
            lines.append(
                f"Section: {section}"
            )

        if heading:
            lines.append(
                f"Heading: {heading}"
            )

    # --------------------------------------------------------
    # CONSTITUTION
    # --------------------------------------------------------

    elif source_type == "constitution":

        lines.append(
            "Constitution of India"
        )

        page = get_value(
            record,
            metadata,
            "pdf_page",
            "page"
        )

        if page is not None:

            lines.append(
                f"PDF page: {page}"
            )

    # --------------------------------------------------------
    # SUPREME COURT
    # --------------------------------------------------------

    elif source_type == "supreme_court":

        lines.append(
            "Supreme Court of India"
        )

        title = get_value(
            record,
            metadata,
            "title"
        )

        judgment_date = get_value(
            record,
            metadata,
            "judgment_date",
            "date"
        )

        citation = get_value(
            record,
            metadata,
            "citation",
            "citation_block"
        )

        if title:
            lines.append(
                f"Case: {title}"
            )

        if judgment_date:
            lines.append(
                f"Judgment date: "
                f"{judgment_date}"
            )

        if citation:

            citation_text = (
                " ".join(
                    str(
                        citation
                    ).split()
                )
            )

            # Avoid accidentally inserting an enormous
            # citation block into every chunk.
            citation_text = (
                citation_text[:500]
            )

            lines.append(
                f"Citation: "
                f"{citation_text}"
            )

    return "\n".join(
        lines
    )


# ============================================================
# COMPACT RAG METADATA
# ============================================================

def build_chunk_metadata(
    record,
    original_metadata,
    source_type,
    document_id,
    chunk_index,
    chunk_count,
    word_start,
    word_end,
    word_count
):

    metadata = {
        "document_id":
            document_id,

        "source_type":
            source_type,

        "chunk_index":
            chunk_index,

        "chunk_count":
            chunk_count,

        "word_start":
            word_start,

        "word_end":
            word_end,

        "word_count":
            word_count,

        "rag_version":
            RAG_VERSION,
    }

    # Only preserve useful provenance/search metadata.
    # Repeating the entire CPT metadata object across hundreds
    # of thousands of chunks would make the JSONL unnecessarily
    # large.

    useful_fields = [
        "title",
        "section",
        "heading",
        "hierarchy",

        "pdf_page",
        "page",

        "court",
        "judgment_date",

        "petitioner",
        "respondent",

        "citation",
        "citation_block",

        "act_id",
        "enactment_date_raw",

        "source_file",
        "source_path",
        "source_sha256",
        "source_url",
        "source_version",

        "legal_status",

        "parse_status",
        "document_format",
        "text_status",
    ]

    for field in useful_fields:

        value = get_value(
            record,
            original_metadata,
            field
        )

        if value is not None:

            metadata[field] = value

    return metadata


# ============================================================
# BUILD RAG CORPUS
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "FULL INDIAN LEGAL RAG V2 CORPUS BUILDER"
    )

    print(
        "=" * 78
    )

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "Data root:",
        DATA_ROOT
    )

    print(
        "\nInput:",
        INPUT_FILE
    )

    print(
        "Output:",
        OUTPUT_FILE
    )

    print(
        "\nChunk size:",
        CHUNK_WORDS,
        "words"
    )

    print(
        "Overlap:",
        OVERLAP_WORDS,
        "words"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    documents_read = 0

    documents_written = 0

    chunks_written = 0

    skipped_documents = 0

    total_input_words = 0

    total_chunk_words = 0

    docs_by_source = Counter()

    chunks_by_source = Counter()

    input_words_by_source = Counter()

    chunk_words_by_source = Counter()

    max_chunks_per_document = 0

    max_chunk_document_id = None

    chunk_distribution = defaultdict(
        int
    )

    # --------------------------------------------------------
    # STREAM INPUT
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as input_handle, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_handle:

        for line_number, line in enumerate(
            input_handle,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSON at "
                    f"input line {line_number}"
                ) from exc

            documents_read += 1

            original_metadata = (
                record.get(
                    "metadata"
                )
                or {}
            )

            source_type = (
                record.get(
                    "source_type"
                )
                or original_metadata.get(
                    "source_type"
                )
                or record.get(
                    "document_type"
                )
                or "unknown"
            )

            source_type = str(
                source_type
            )

            document_id = (
                record.get("id")
                or record.get(
                    "document_id"
                )
                or original_metadata.get(
                    "document_id"
                )
                or f"doc_{line_number}"
            )

            document_id = str(
                document_id
            )

            raw_text = record.get(
                "text",
                ""
            )

            text = normalize_text(
                raw_text
            )

            if not text:

                skipped_documents += 1
                continue

            word_count = len(
                text.split()
            )

            if word_count < MIN_WORDS:

                skipped_documents += 1
                continue

            documents_written += 1

            docs_by_source[
                source_type
            ] += 1

            input_words_by_source[
                source_type
            ] += word_count

            total_input_words += (
                word_count
            )

            chunks = chunk_text(
                text
            )

            if not chunks:

                skipped_documents += 1
                continue

            chunk_count = len(
                chunks
            )

            if (
                chunk_count
                > max_chunks_per_document
            ):

                max_chunks_per_document = (
                    chunk_count
                )

                max_chunk_document_id = (
                    document_id
                )

            # Coarse diagnostic bucket.
            if chunk_count == 1:

                chunk_distribution[
                    "1"
                ] += 1

            elif chunk_count <= 5:

                chunk_distribution[
                    "2-5"
                ] += 1

            elif chunk_count <= 20:

                chunk_distribution[
                    "6-20"
                ] += 1

            elif chunk_count <= 100:

                chunk_distribution[
                    "21-100"
                ] += 1

            else:

                chunk_distribution[
                    "101+"
                ] += 1

            retrieval_header = (
                build_retrieval_header(
                    source_type,
                    record,
                    original_metadata
                )
            )

            # ------------------------------------------------
            # WRITE CHUNKS
            # ------------------------------------------------

            for chunk_index, chunk in enumerate(
                chunks
            ):

                chunk_body = (
                    chunk[
                        "text"
                    ]
                )

                if retrieval_header:

                    retrieval_text = (
                        retrieval_header
                        + "\n\n"
                        + chunk_body
                    )

                else:

                    retrieval_text = (
                        chunk_body
                    )

                chunk_id = (
                    f"{document_id}"
                    f"__ragv2_"
                    f"{chunk_index:05d}"
                )

                metadata = (
                    build_chunk_metadata(
                        record=
                            record,

                        original_metadata=
                            original_metadata,

                        source_type=
                            source_type,

                        document_id=
                            document_id,

                        chunk_index=
                            chunk_index,

                        chunk_count=
                            chunk_count,

                        word_start=
                            chunk[
                                "word_start"
                            ],

                        word_end=
                            chunk[
                                "word_end"
                            ],

                        word_count=
                            chunk[
                                "word_count"
                            ],
                    )
                )

                rag_record = {
                    "id":
                        chunk_id,

                    "document_id":
                        document_id,

                    "source_type":
                        source_type,

                    "text":
                        retrieval_text,

                    "metadata":
                        metadata,
                }

                output_handle.write(
                    json.dumps(
                        rag_record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                chunks_written += 1

                chunks_by_source[
                    source_type
                ] += 1

                chunk_words_by_source[
                    source_type
                ] += chunk[
                    "word_count"
                ]

                total_chunk_words += (
                    chunk[
                        "word_count"
                    ]
                )

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (
                documents_read
                % 1000
                == 0
            ):

                print(
                    f"Processed "
                    f"{documents_read:,} docs"
                    f" | chunks="
                    f"{chunks_written:,}"
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    # BGE-small embedding dimension = 384.
    embedding_bytes = (
        chunks_written
        * 384
        * 4
    )

    embedding_gib = (
        embedding_bytes
        / (
            1024 ** 3
        )
    )

    summary = {
        "rag_version":
            RAG_VERSION,

        "input_file":
            str(
                INPUT_FILE
            ),

        "output_file":
            str(
                OUTPUT_FILE
            ),

        "chunking": {
            "chunk_words":
                CHUNK_WORDS,

            "overlap_words":
                OVERLAP_WORDS,

            "stride_words":
                CHUNK_WORDS
                - OVERLAP_WORDS,

            "minimum_words":
                MIN_WORDS,
        },

        "documents_read":
            documents_read,

        "documents_written":
            documents_written,

        "documents_skipped":
            skipped_documents,

        "chunks_written":
            chunks_written,

        "total_input_words":
            total_input_words,

        "total_chunk_words_with_overlap":
            total_chunk_words,

        "documents_by_source":
            dict(
                docs_by_source
            ),

        "chunks_by_source":
            dict(
                chunks_by_source
            ),

        "input_words_by_source":
            dict(
                input_words_by_source
            ),

        "chunk_words_by_source":
            dict(
                chunk_words_by_source
            ),

        "document_chunk_distribution":
            dict(
                chunk_distribution
            ),

        "largest_document_by_chunk_count": {
            "document_id":
                max_chunk_document_id,

            "chunk_count":
                max_chunks_per_document,
        },

        "estimated_float32_embedding_size_gib":
            round(
                embedding_gib,
                3
            ),
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as summary_handle:

        json.dump(
            summary,
            summary_handle,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # DISPLAY FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RAG V2 CORPUS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"\nDocuments read: "
        f"{documents_read:,}"
    )

    print(
        f"Documents written: "
        f"{documents_written:,}"
    )

    print(
        f"Documents skipped: "
        f"{skipped_documents:,}"
    )

    print(
        f"\nRAG chunks written: "
        f"{chunks_written:,}"
    )

    print(
        f"Input words: "
        f"{total_input_words:,}"
    )

    print(
        f"Chunk words including overlap: "
        f"{total_chunk_words:,}"
    )

    print(
        "\nSOURCE DISTRIBUTION"
    )

    print(
        "-" * 78
    )

    for source_type in sorted(
        chunks_by_source
    ):

        docs = docs_by_source[
            source_type
        ]

        chunks = chunks_by_source[
            source_type
        ]

        percentage = (
            chunks
            / chunks_written
            * 100
        )

        print(
            f"{source_type:<22}"
            f" docs="
            f"{docs:>8,}"
            f" | chunks="
            f"{chunks:>9,}"
            f" | "
            f"{percentage:>6.2f}%"
        )

    print(
        "\nDOCUMENT CHUNK DISTRIBUTION"
    )

    print(
        "-" * 78
    )

    for bucket in [
        "1",
        "2-5",
        "6-20",
        "21-100",
        "101+",
    ]:

        print(
            f"{bucket:<10}: "
            f"{chunk_distribution[bucket]:,}"
        )

    print(
        "\nLargest document:"
    )

    print(
        max_chunk_document_id
    )

    print(
        "Chunks:",
        max_chunks_per_document
    )

    print(
        "\nEstimated BGE embedding matrix "
        "size (float32):"
    )

    print(
        f"{embedding_gib:.3f} GiB"
    )

    print(
        "\nOUTPUT FILES"
    )

    print(
        "-" * 78
    )

    print(
        "RAG corpus:",
        OUTPUT_FILE
    )

    print(
        "Summary:",
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()