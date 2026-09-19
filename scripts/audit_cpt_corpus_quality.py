from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import re

import numpy as np
from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

DATASET_FILE = Path(
    "data/training/cpt/legal_cpt_corpus.jsonl"
)

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B-Base"

OUTPUT_DIR = Path(
    "data/training/cpt/quality"
)

SUMMARY_FILE = OUTPUT_DIR / "cpt_quality_summary.json"

DOCUMENT_STATS_FILE = OUTPUT_DIR / "cpt_document_stats.csv"

SOURCE_STATS_FILE = OUTPUT_DIR / "cpt_source_stats.csv"

LARGE_DOCUMENTS_FILE = OUTPUT_DIR / "cpt_large_documents.csv"

TOKEN_RATIO_OUTLIERS_FILE = (
    OUTPUT_DIR / "cpt_token_word_outliers.csv"
)

VERNACULAR_SC_FILE = (
    OUTPUT_DIR / "cpt_supreme_court_vernacular.csv"
)

DUPLICATE_CASES_FILE = (
    OUTPUT_DIR / "cpt_potential_duplicate_sc_cases.csv"
)


# ------------------------------------------------------------
# Quality thresholds
# ------------------------------------------------------------

LARGE_DOCUMENT_THRESHOLDS = [
    50_000,
    100_000,
    250_000
]

# >3 tokenizer tokens per whitespace word is worth inspecting.
TOKEN_WORD_OUTLIER_THRESHOLD = 3.0


# ============================================================
# VERNACULAR LANGUAGE TAGS
#
# These are based on common filename suffixes in the Supreme
# Court dataset. This is not full language detection.
# ============================================================

VERNACULAR_TAGS = {
    "HIN": "Hindi",
    "MAL": "Malayalam",
    "GUJ": "Gujarati",
    "TAM": "Tamil",
    "TEL": "Telugu",
    "KAN": "Kannada",
    "BEN": "Bengali",
    "MAR": "Marathi",
    "PUN": "Punjabi",
    "ORI": "Odia",
    "ODI": "Odia",
    "ASM": "Assamese",
    "URD": "Urdu"
}


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):

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
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                )


def count_words(text):

    return len(
        text.split()
    )


def safe_text(value):

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# SUPREME COURT LANGUAGE CLASSIFICATION
# ============================================================

def classify_sc_language(record):

    """
    Classify a Supreme Court record using filename/document-ID
    metadata.

    Returns:
        ("vernacular_tagged", "Malayalam")
        ("vernacular_tagged", "Unknown")
        ("english_or_untagged", None)

    This intentionally does NOT scan all 1+ billion characters
    for Unicode language detection.
    """

    metadata = record.get(
        "metadata",
        {}
    )

    document_id = safe_text(
        record.get("id")
    )

    source_file = safe_text(
        metadata.get("source_file")
    )

    combined = (
        document_id
        + " "
        + source_file
    ).upper()

    # Explicit folder/name marker.
    if "VERNACULAR" in combined:

        for tag, language in (
            VERNACULAR_TAGS.items()
        ):

            pattern = (
                rf"(?:_|-){tag}"
                rf"(?:_|-|\.|$)"
            )

            if re.search(
                pattern,
                combined
            ):

                return (
                    "vernacular_tagged",
                    language
                )

        return (
            "vernacular_tagged",
            "Unknown"
        )

    # Some files may have a language suffix even if
    # "vernacular" is absent.
    for tag, language in (
        VERNACULAR_TAGS.items()
    ):

        pattern = (
            rf"(?:_|-){tag}"
            rf"(?:_|-|\.|$)"
        )

        if re.search(
            pattern,
            combined
        ):

            return (
                "vernacular_tagged",
                language
            )

    return (
        "english_or_untagged",
        None
    )


# ============================================================
# SUPREME COURT CASE KEY
# ============================================================

def extract_sc_case_key(document_id):

    """
    Example:

    sc_36350-2010___supremecourt__...
            ↓
    36350-2010

    Used only as a POTENTIAL duplicate indicator.

    We deliberately require:
        number-year

    so values such as sc_-0___... are not grouped together.
    """

    if not document_id:

        return None

    document_id = str(
        document_id
    )

    if document_id.startswith("sc_"):

        document_id = (
            document_id[3:]
        )

    first_part = document_id.split(
        "___",
        1
    )[0]

    if re.fullmatch(
        r"\d{1,8}-\d{4}",
        first_part
    ):

        return first_part

    return None


# ============================================================
# CSV WRITER
# ============================================================

def write_csv(
    path,
    rows,
    fieldnames
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("CPT CORPUS QUALITY AUDIT")
    print("=" * 72)

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nDataset:",
        DATASET_FILE
    )

    print(
        "Tokenizer:",
        MODEL_NAME
    )

    # ========================================================
    # LOAD TOKENIZER
    # ========================================================

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True
        )
    )

    # We are measuring arbitrary-size documents rather than
    # passing them to the model. Setting a very large limit
    # prevents irrelevant max-length warnings.
    tokenizer.model_max_length = (
        10**12
    )

    print(
        "Tokenizer loaded."
    )

    # ========================================================
    # GLOBAL COUNTERS
    # ========================================================

    total_documents = 0
    total_words = 0
    total_tokens = 0
    total_characters = 0

    source_document_counts = Counter()
    source_word_counts = Counter()
    source_token_counts = Counter()
    source_character_counts = Counter()

    source_token_lists = defaultdict(
        list
    )

    # ========================================================
    # THRESHOLD COUNTERS
    # ========================================================

    large_document_counts = {
        threshold: 0
        for threshold
        in LARGE_DOCUMENT_THRESHOLDS
    }

    token_ratio_counts = {
        "ratio_ge_2": 0,
        "ratio_ge_3": 0,
        "ratio_ge_5": 0,
        "ratio_ge_10": 0
    }

    # ========================================================
    # SUPREME COURT COUNTERS
    # ========================================================

    sc_language_counts = Counter()

    sc_language_token_counts = Counter()

    sc_detected_languages = Counter()

    sc_case_groups = defaultdict(
        list
    )

    # ========================================================
    # DATA WE KEEP FOR REPORTS
    # ========================================================

    document_stats = []

    large_documents = []

    ratio_outliers = []

    vernacular_records = []

    all_token_counts = []
    all_word_counts = []
    all_token_word_ratios = []

    # ========================================================
    # PROCESS DATASET
    # ========================================================

    print(
        "\nAuditing corpus..."
    )

    for index, record in enumerate(
        load_jsonl(DATASET_FILE),
        start=1
    ):

        text = record.get(
            "text",
            ""
        )

        if not text:

            continue

        document_id = safe_text(
            record.get("id")
        )

        source_type = safe_text(
            record.get("source_type")
        )

        if not source_type:

            source_type = "unknown"

        metadata = record.get(
            "metadata",
            {}
        )

        title = safe_text(
            metadata.get("title")
        )

        source_file = safe_text(
            metadata.get("source_file")
        )

        # ----------------------------------------------------
        # Basic counts
        # ----------------------------------------------------

        character_count = len(
            text
        )

        word_count = count_words(
            text
        )

        # ----------------------------------------------------
        # Exact tokenizer token count
        # ----------------------------------------------------

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        token_count = len(
            token_ids
        )

        # Free large token list as soon as possible.
        del token_ids

        # ----------------------------------------------------
        # Token / word ratio
        # ----------------------------------------------------

        if word_count > 0:

            token_word_ratio = (
                token_count
                / word_count
            )

        else:

            token_word_ratio = 0.0

        # ----------------------------------------------------
        # Global totals
        # ----------------------------------------------------

        total_documents += 1

        total_characters += (
            character_count
        )

        total_words += (
            word_count
        )

        total_tokens += (
            token_count
        )

        # ----------------------------------------------------
        # Per-source totals
        # ----------------------------------------------------

        source_document_counts[
            source_type
        ] += 1

        source_word_counts[
            source_type
        ] += word_count

        source_token_counts[
            source_type
        ] += token_count

        source_character_counts[
            source_type
        ] += character_count

        source_token_lists[
            source_type
        ].append(
            token_count
        )

        # ----------------------------------------------------
        # Global distribution arrays
        # ----------------------------------------------------

        all_token_counts.append(
            token_count
        )

        all_word_counts.append(
            word_count
        )

        all_token_word_ratios.append(
            token_word_ratio
        )

        # ====================================================
        # LARGE DOCUMENT FLAGS
        # ====================================================

        exceeded_thresholds = []

        for threshold in (
            LARGE_DOCUMENT_THRESHOLDS
        ):

            if token_count >= threshold:

                large_document_counts[
                    threshold
                ] += 1

                exceeded_thresholds.append(
                    threshold
                )

        if exceeded_thresholds:

            large_documents.append({
                "id":
                    document_id,

                "source_type":
                    source_type,

                "title":
                    title,

                "source_file":
                    source_file,

                "words":
                    word_count,

                "tokens":
                    token_count,

                "tokens_per_word":
                    round(
                        token_word_ratio,
                        4
                    ),

                "largest_threshold_exceeded":
                    max(
                        exceeded_thresholds
                    )
            })

        # ====================================================
        # TOKEN / WORD RATIO FLAGS
        # ====================================================

        if token_word_ratio >= 2:

            token_ratio_counts[
                "ratio_ge_2"
            ] += 1

        if token_word_ratio >= 3:

            token_ratio_counts[
                "ratio_ge_3"
            ] += 1

        if token_word_ratio >= 5:

            token_ratio_counts[
                "ratio_ge_5"
            ] += 1

        if token_word_ratio >= 10:

            token_ratio_counts[
                "ratio_ge_10"
            ] += 1

        if (
            token_word_ratio
            >= TOKEN_WORD_OUTLIER_THRESHOLD
        ):

            ratio_outliers.append({
                "id":
                    document_id,

                "source_type":
                    source_type,

                "title":
                    title,

                "source_file":
                    source_file,

                "words":
                    word_count,

                "tokens":
                    token_count,

                "tokens_per_word":
                    round(
                        token_word_ratio,
                        4
                    )
            })

        # ====================================================
        # SUPREME COURT-SPECIFIC AUDIT
        # ====================================================

        language_class = ""
        detected_language = ""

        case_key = None

        if source_type == "supreme_court":

            (
                language_class,
                detected_language
            ) = classify_sc_language(
                record
            )

            sc_language_counts[
                language_class
            ] += 1

            sc_language_token_counts[
                language_class
            ] += token_count

            if detected_language:

                sc_detected_languages[
                    detected_language
                ] += 1

            if (
                language_class
                == "vernacular_tagged"
            ):

                vernacular_records.append({
                    "id":
                        document_id,

                    "language":
                        detected_language,

                    "title":
                        title,

                    "source_file":
                        source_file,

                    "words":
                        word_count,

                    "tokens":
                        token_count,

                    "tokens_per_word":
                        round(
                            token_word_ratio,
                            4
                        )
                })

            # ------------------------------------------------
            # Potential duplicate case grouping
            # ------------------------------------------------

            case_key = (
                extract_sc_case_key(
                    document_id
                )
            )

            if case_key:

                sc_case_groups[
                    case_key
                ].append({
                    "id":
                        document_id,

                    "title":
                        title,

                    "source_file":
                        source_file,

                    "tokens":
                        token_count,

                    "language_class":
                        language_class,

                    "language":
                        detected_language
                })

        # ====================================================
        # SAVE PER-DOCUMENT STATS
        # ====================================================

        document_stats.append({
            "id":
                document_id,

            "source_type":
                source_type,

            "title":
                title,

            "source_file":
                source_file,

            "characters":
                character_count,

            "words":
                word_count,

            "tokens":
                token_count,

            "tokens_per_word":
                round(
                    token_word_ratio,
                    4
                ),

            "sc_language_class":
                language_class,

            "sc_language":
                detected_language,

            "sc_case_key":
                case_key or ""
        })

        # ====================================================
        # PROGRESS
        # ====================================================

        if index % 1000 == 0:

            print(
                f"Processed "
                f"{index:,} records | "
                f"{total_tokens:,} tokens"
            )

    # ========================================================
    # DISTRIBUTION STATISTICS
    # ========================================================

    token_array = np.array(
        all_token_counts,
        dtype=np.int64
    )

    word_array = np.array(
        all_word_counts,
        dtype=np.int64
    )

    ratio_array = np.array(
        all_token_word_ratios,
        dtype=np.float64
    )

    token_percentiles = {
        "p50":
            float(
                np.percentile(
                    token_array,
                    50
                )
            ),

        "p75":
            float(
                np.percentile(
                    token_array,
                    75
                )
            ),

        "p90":
            float(
                np.percentile(
                    token_array,
                    90
                )
            ),

        "p95":
            float(
                np.percentile(
                    token_array,
                    95
                )
            ),

        "p99":
            float(
                np.percentile(
                    token_array,
                    99
                )
            ),

        "p99_5":
            float(
                np.percentile(
                    token_array,
                    99.5
                )
            )
    }

    ratio_percentiles = {
        "p50":
            float(
                np.percentile(
                    ratio_array,
                    50
                )
            ),

        "p90":
            float(
                np.percentile(
                    ratio_array,
                    90
                )
            ),

        "p95":
            float(
                np.percentile(
                    ratio_array,
                    95
                )
            ),

        "p99":
            float(
                np.percentile(
                    ratio_array,
                    99
                )
            )
    }

    # ========================================================
    # SOURCE STATISTICS
    # ========================================================

    source_stats = []

    for source_type in sorted(
        source_document_counts.keys()
    ):

        documents = (
            source_document_counts[
                source_type
            ]
        )

        words = (
            source_word_counts[
                source_type
            ]
        )

        tokens = (
            source_token_counts[
                source_type
            ]
        )

        characters = (
            source_character_counts[
                source_type
            ]
        )

        token_values = np.array(
            source_token_lists[
                source_type
            ],
            dtype=np.int64
        )

        source_stats.append({
            "source_type":
                source_type,

            "documents":
                documents,

            "documents_percent":
                round(
                    100
                    * documents
                    / total_documents,
                    3
                ),

            "characters":
                characters,

            "words":
                words,

            "tokens":
                tokens,

            "token_percent":
                round(
                    100
                    * tokens
                    / total_tokens,
                    3
                ),

            "average_tokens_per_document":
                round(
                    tokens
                    / documents,
                    2
                ),

            "median_tokens_per_document":
                round(
                    float(
                        np.median(
                            token_values
                        )
                    ),
                    2
                ),

            "p95_tokens_per_document":
                round(
                    float(
                        np.percentile(
                            token_values,
                            95
                        )
                    ),
                    2
                ),

            "max_tokens":
                int(
                    np.max(
                        token_values
                    )
                )
        })

    # ========================================================
    # POTENTIAL DUPLICATE CASE REPORT
    # ========================================================

    duplicate_case_rows = []

    duplicate_case_group_count = 0
    duplicate_case_document_count = 0

    for case_key, documents in sorted(
        sc_case_groups.items()
    ):

        if len(documents) <= 1:

            continue

        duplicate_case_group_count += 1

        duplicate_case_document_count += (
            len(documents)
        )

        token_values = [
            item["tokens"]
            for item in documents
        ]

        for item in documents:

            duplicate_case_rows.append({
                "case_key":
                    case_key,

                "group_size":
                    len(documents),

                "id":
                    item["id"],

                "title":
                    item["title"],

                "source_file":
                    item["source_file"],

                "tokens":
                    item["tokens"],

                "group_min_tokens":
                    min(
                        token_values
                    ),

                "group_max_tokens":
                    max(
                        token_values
                    ),

                "language_class":
                    item[
                        "language_class"
                    ],

                "language":
                    item[
                        "language"
                    ]
            })

    # ========================================================
    # SORT REPORTS
    # ========================================================

    large_documents.sort(
        key=lambda row: row["tokens"],
        reverse=True
    )

    ratio_outliers.sort(
        key=lambda row:
            row["tokens_per_word"],
        reverse=True
    )

    vernacular_records.sort(
        key=lambda row: row["tokens"],
        reverse=True
    )

    document_stats.sort(
        key=lambda row: row["tokens"],
        reverse=True
    )

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    write_csv(
        DOCUMENT_STATS_FILE,
        document_stats,
        [
            "id",
            "source_type",
            "title",
            "source_file",
            "characters",
            "words",
            "tokens",
            "tokens_per_word",
            "sc_language_class",
            "sc_language",
            "sc_case_key"
        ]
    )

    write_csv(
        SOURCE_STATS_FILE,
        source_stats,
        [
            "source_type",
            "documents",
            "documents_percent",
            "characters",
            "words",
            "tokens",
            "token_percent",
            "average_tokens_per_document",
            "median_tokens_per_document",
            "p95_tokens_per_document",
            "max_tokens"
        ]
    )

    write_csv(
        LARGE_DOCUMENTS_FILE,
        large_documents,
        [
            "id",
            "source_type",
            "title",
            "source_file",
            "words",
            "tokens",
            "tokens_per_word",
            "largest_threshold_exceeded"
        ]
    )

    write_csv(
        TOKEN_RATIO_OUTLIERS_FILE,
        ratio_outliers,
        [
            "id",
            "source_type",
            "title",
            "source_file",
            "words",
            "tokens",
            "tokens_per_word"
        ]
    )

    write_csv(
        VERNACULAR_SC_FILE,
        vernacular_records,
        [
            "id",
            "language",
            "title",
            "source_file",
            "words",
            "tokens",
            "tokens_per_word"
        ]
    )

    write_csv(
        DUPLICATE_CASES_FILE,
        duplicate_case_rows,
        [
            "case_key",
            "group_size",
            "id",
            "title",
            "source_file",
            "tokens",
            "group_min_tokens",
            "group_max_tokens",
            "language_class",
            "language"
        ]
    )

    # ========================================================
    # LANGUAGE TOKEN PERCENTAGES
    # ========================================================

    total_sc_tokens = sum(
        sc_language_token_counts.values()
    )

    sc_language_summary = {}

    for classification, count in (
        sc_language_counts.items()
    ):

        tokens = (
            sc_language_token_counts[
                classification
            ]
        )

        sc_language_summary[
            classification
        ] = {
            "documents":
                count,

            "tokens":
                tokens,

            "token_percent_of_sc":
                (
                    round(
                        100
                        * tokens
                        / total_sc_tokens,
                        3
                    )
                    if total_sc_tokens
                    else 0
                )
        }

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {
        "dataset":
            str(DATASET_FILE),

        "tokenizer":
            MODEL_NAME,

        "total_documents":
            total_documents,

        "total_characters":
            total_characters,

        "total_words":
            total_words,

        "total_tokens":
            total_tokens,

        "overall_tokens_per_word":
            (
                total_tokens
                / total_words
                if total_words
                else 0
            ),

        "document_token_percentiles":
            token_percentiles,

        "token_word_ratio_percentiles":
            ratio_percentiles,

        "large_document_counts": {
            str(threshold):
                large_document_counts[
                    threshold
                ]

            for threshold
            in LARGE_DOCUMENT_THRESHOLDS
        },

        "token_word_ratio_counts":
            token_ratio_counts,

        "token_word_outlier_threshold":
            TOKEN_WORD_OUTLIER_THRESHOLD,

        "source_statistics":
            source_stats,

        "supreme_court_language_classification":
            sc_language_summary,

        "supreme_court_detected_vernacular_languages":
            dict(
                sc_detected_languages
            ),

        "potential_duplicate_sc_cases": {
            "case_groups":
                duplicate_case_group_count,

            "documents_in_duplicate_groups":
                duplicate_case_document_count
        },

        "report_files": {
            "document_stats":
                str(
                    DOCUMENT_STATS_FILE
                ),

            "source_stats":
                str(
                    SOURCE_STATS_FILE
                ),

            "large_documents":
                str(
                    LARGE_DOCUMENTS_FILE
                ),

            "token_word_outliers":
                str(
                    TOKEN_RATIO_OUTLIERS_FILE
                ),

            "supreme_court_vernacular":
                str(
                    VERNACULAR_SC_FILE
                ),

            "potential_duplicate_sc_cases":
                str(
                    DUPLICATE_CASES_FILE
                )
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

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CPT CORPUS QUALITY AUDIT COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nDocuments: "
        f"{total_documents:,}"
    )

    print(
        f"Words: "
        f"{total_words:,}"
    )

    print(
        f"Tokens: "
        f"{total_tokens:,}"
    )

    print(
        f"Overall tokens / word: "
        f"{total_tokens / total_words:.3f}"
    )

    # --------------------------------------------------------
    # Source distribution
    # --------------------------------------------------------

    print(
        "\nTOKENS BY SOURCE"
    )

    print(
        "-" * 72
    )

    for row in source_stats:

        print(
            f"{row['source_type']:<20} "
            f"docs={row['documents']:>8,} | "
            f"tokens={row['tokens']:>12,} | "
            f"{row['token_percent']:>6.2f}%"
        )

    # --------------------------------------------------------
    # Large documents
    # --------------------------------------------------------

    print(
        "\nLARGE DOCUMENTS"
    )

    print(
        "-" * 72
    )

    for threshold in (
        LARGE_DOCUMENT_THRESHOLDS
    ):

        print(
            f">= {threshold:,} tokens: "
            f"{large_document_counts[threshold]:,}"
        )

    # --------------------------------------------------------
    # Token/word ratios
    # --------------------------------------------------------

    print(
        "\nTOKEN / WORD RATIO"
    )

    print(
        "-" * 72
    )

    print(
        f"Median: "
        f"{ratio_percentiles['p50']:.3f}"
    )

    print(
        f"P95: "
        f"{ratio_percentiles['p95']:.3f}"
    )

    print(
        f"P99: "
        f"{ratio_percentiles['p99']:.3f}"
    )

    print(
        f">= 2.0: "
        f"{token_ratio_counts['ratio_ge_2']:,}"
    )

    print(
        f">= 3.0: "
        f"{token_ratio_counts['ratio_ge_3']:,}"
    )

    print(
        f">= 5.0: "
        f"{token_ratio_counts['ratio_ge_5']:,}"
    )

    print(
        f">= 10.0: "
        f"{token_ratio_counts['ratio_ge_10']:,}"
    )

    # --------------------------------------------------------
    # Supreme Court language
    # --------------------------------------------------------

    print(
        "\nSUPREME COURT LANGUAGE TAGS"
    )

    print(
        "-" * 72
    )

    for classification, info in (
        sc_language_summary.items()
    ):

        print(
            f"{classification:<22} "
            f"docs={info['documents']:>7,} | "
            f"tokens={info['tokens']:>12,} | "
            f"{info['token_percent_of_sc']:>6.2f}% "
            f"of SC tokens"
        )

    if sc_detected_languages:

        print(
            "\nDetected vernacular tags:"
        )

        for language, count in sorted(
            sc_detected_languages.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            print(
                f"  {language}: "
                f"{count:,}"
            )

    # --------------------------------------------------------
    # Potential duplicate cases
    # --------------------------------------------------------

    print(
        "\nPOTENTIAL DUPLICATE SUPREME COURT CASES"
    )

    print(
        "-" * 72
    )

    print(
        "Duplicate case-key groups:",
        f"{duplicate_case_group_count:,}"
    )

    print(
        "Documents in those groups:",
        f"{duplicate_case_document_count:,}"
    )

    # --------------------------------------------------------
    # Largest 10
    # --------------------------------------------------------

    print(
        "\nTOP 10 LARGEST DOCUMENTS"
    )

    print(
        "-" * 72
    )

    for rank, row in enumerate(
        document_stats[:10],
        start=1
    ):

        print(
            f"{rank:>2}. "
            f"{row['tokens']:>10,} tokens | "
            f"{row['tokens_per_word']:>7.3f} t/w | "
            f"{row['source_type']} | "
            f"{row['id']}"
        )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    print(
        "\nREPORTS"
    )

    print(
        "-" * 72
    )

    print(
        "Summary:",
        SUMMARY_FILE
    )

    print(
        "Per-document stats:",
        DOCUMENT_STATS_FILE
    )

    print(
        "Source stats:",
        SOURCE_STATS_FILE
    )

    print(
        "Large documents:",
        LARGE_DOCUMENTS_FILE
    )

    print(
        "Token/word outliers:",
        TOKEN_RATIO_OUTLIERS_FILE
    )

    print(
        "Vernacular SC records:",
        VERNACULAR_SC_FILE
    )

    print(
        "Potential duplicate SC cases:",
        DUPLICATE_CASES_FILE
    )


if __name__ == "__main__":
    main()