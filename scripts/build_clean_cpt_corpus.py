from pathlib import Path
from collections import Counter, defaultdict
import csv
import json


# ============================================================
# PROJECT PATHS
# ============================================================

# Works whether you run from:
#   D:\Projects\LLM_training
# or:
#   D:\Projects\LLM_training\scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUT FILES
# ============================================================

CPT_FILE = (
    PROJECT_ROOT / "scripts"
    / "data"
    / "training"
    / "cpt"
    / "legal_cpt_corpus.jsonl"
)

DOCUMENT_STATS_FILE = (
    PROJECT_ROOT / "scripts"
    / "data"
    / "training"
    / "cpt"
    / "quality"
    / "cpt_document_stats.csv"
)

DUPLICATE_PAIRS_FILE = (
    PROJECT_ROOT / "scripts"
    / "data"
    / "training"
    / "cpt"
    / "quality"
    / "sc_near_duplicate_pairs.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT / "scripts"
    / "data"
    / "training"
    / "cpt"
    / "clean"
)

CLEAN_CORPUS_FILE = (
    OUTPUT_DIR
    / "legal_cpt_clean.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "legal_cpt_clean_summary.json"
)

EXCLUSIONS_FILE = (
    OUTPUT_DIR
    / "legal_cpt_exclusions.csv"
)

DUPLICATE_CLUSTERS_FILE = (
    OUTPUT_DIR
    / "legal_cpt_duplicate_clusters.csv"
)


# ============================================================
# CLEANING POLICY
# ============================================================

# Only Supreme Court documents are filtered using this ratio.
#
# Acts and Constitution are never rejected because of
# token/word ratio.
TOKEN_WORD_RATIO_THRESHOLD = 5.0


# Only these two duplicate classifications are automatically
# used for deduplication.
#
# IMPORTANT:
# "review" is deliberately NOT included.
DUPLICATE_CLASSES = {
    "exact_duplicate",
    "probable_near_duplicate",
}


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):

    if value is None:
        return ""

    return str(value).strip()


def safe_int(value):

    try:
        return int(float(value))

    except (TypeError, ValueError):
        return 0


def safe_float(value):

    try:
        return float(value)

    except (TypeError, ValueError):
        return 0.0


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
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {exc}"
                )


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
# UNION-FIND
#
# Used to convert duplicate PAIRS into duplicate CLUSTERS.
#
# Example:
#
# A ~= B
# B ~= C
#
# becomes:
#
# {A, B, C}
#
# We then keep only one canonical document.
# ============================================================

class UnionFind:

    def __init__(self):

        self.parent = {}
        self.rank = {}

    def add(self, item):

        if item not in self.parent:

            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):

        if self.parent[item] != item:

            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(self, a, b):

        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if (
            self.rank[root_a]
            < self.rank[root_b]
        ):

            root_a, root_b = (
                root_b,
                root_a
            )

        self.parent[root_b] = root_a

        if (
            self.rank[root_a]
            == self.rank[root_b]
        ):

            self.rank[root_a] += 1

    def groups(self):

        grouped = defaultdict(list)

        for item in self.parent:

            root = self.find(item)

            grouped[root].append(
                item
            )

        return list(
            grouped.values()
        )


# ============================================================
# LOAD DOCUMENT STATS
#
# These were generated by:
#
# audit_cpt_corpus_quality.py
#
# This means we do NOT need to tokenize 305M tokens again.
# ============================================================

def load_document_stats():

    print(
        "\nLoading document quality statistics..."
    )

    stats = {}

    with open(
        DOCUMENT_STATS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            document_id = safe_text(
                row.get("id")
            )

            if not document_id:
                continue

            if document_id in stats:

                raise ValueError(
                    f"Duplicate document ID in stats: "
                    f"{document_id}"
                )

            stats[document_id] = {
                "id":
                    document_id,

                "source_type":
                    safe_text(
                        row.get(
                            "source_type"
                        )
                    ),

                "title":
                    safe_text(
                        row.get(
                            "title"
                        )
                    ),

                "source_file":
                    safe_text(
                        row.get(
                            "source_file"
                        )
                    ),

                "characters":
                    safe_int(
                        row.get(
                            "characters"
                        )
                    ),

                "words":
                    safe_int(
                        row.get(
                            "words"
                        )
                    ),

                "tokens":
                    safe_int(
                        row.get(
                            "tokens"
                        )
                    ),

                "tokens_per_word":
                    safe_float(
                        row.get(
                            "tokens_per_word"
                        )
                    ),

                "sc_language_class":
                    safe_text(
                        row.get(
                            "sc_language_class"
                        )
                    ),

                "sc_language":
                    safe_text(
                        row.get(
                            "sc_language"
                        )
                    ),

                "sc_case_key":
                    safe_text(
                        row.get(
                            "sc_case_key"
                        )
                    )
            }

    print(
        "Document stats loaded:",
        f"{len(stats):,}"
    )

    return stats


# ============================================================
# INITIAL QUALITY FILTERS
# ============================================================

def determine_quality_exclusions(
    stats
):
    """
    Determine documents that should be removed BEFORE
    duplicate resolution.

    Current automatic filters:

    1. Supreme Court vernacular-tagged documents
    2. Supreme Court token/word ratio >= 5.0

    Central Acts and Constitution are not affected by these
    filters.
    """

    print(
        "\nApplying quality-filter rules..."
    )

    excluded = {}

    counts = Counter()
    token_counts = Counter()

    for document_id, info in (
        stats.items()
    ):

        source_type = info[
            "source_type"
        ]

        if source_type != "supreme_court":
            continue

        reasons = []

        # ----------------------------------------------------
        # Rule 1:
        # Remove vernacular-tagged SC documents from English
        # CPT v1.
        # ----------------------------------------------------

        if (
            info["sc_language_class"]
            == "vernacular_tagged"
        ):

            reasons.append(
                "sc_vernacular_tagged"
            )

        # ----------------------------------------------------
        # Rule 2:
        # Extreme tokenizer inefficiency / likely non-English
        # or extraction problem.
        # ----------------------------------------------------

        if (
            info["tokens_per_word"]
            >= TOKEN_WORD_RATIO_THRESHOLD
        ):

            reasons.append(
                "sc_token_word_ratio_ge_5"
            )

        if not reasons:
            continue

        # Primary reason is deterministic.
        if (
            "sc_vernacular_tagged"
            in reasons
        ):

            primary_reason = (
                "sc_vernacular_tagged"
            )

        else:

            primary_reason = (
                "sc_token_word_ratio_ge_5"
            )

        excluded[document_id] = {
            "primary_reason":
                primary_reason,

            "all_reasons":
                reasons,

            "canonical_id":
                "",

            "duplicate_cluster_id":
                ""
        }

        counts[
            primary_reason
        ] += 1

        token_counts[
            primary_reason
        ] += info["tokens"]

    print(
        "Initial quality exclusions:",
        f"{len(excluded):,}"
    )

    print(
        "  Vernacular:",
        f"{counts['sc_vernacular_tagged']:,}"
    )

    print(
        "  Token/word >= "
        f"{TOKEN_WORD_RATIO_THRESHOLD}:",
        f"{counts['sc_token_word_ratio_ge_5']:,}"
    )

    return (
        excluded,
        counts,
        token_counts
    )


# ============================================================
# BUILD DUPLICATE CLUSTERS
# ============================================================

def build_duplicate_clusters():

    print(
        "\nBuilding duplicate clusters..."
    )

    union_find = UnionFind()

    edge_counts = Counter()

    duplicate_edges = 0

    with open(
        DUPLICATE_PAIRS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            classification = safe_text(
                row.get(
                    "classification"
                )
            )

            if (
                classification
                not in DUPLICATE_CLASSES
            ):
                continue

            id_a = safe_text(
                row.get("id_a")
            )

            id_b = safe_text(
                row.get("id_b")
            )

            if not id_a or not id_b:
                continue

            union_find.union(
                id_a,
                id_b
            )

            duplicate_edges += 1

            edge_counts[
                classification
            ] += 1

    clusters = [
        sorted(group)
        for group in union_find.groups()
        if len(group) > 1
    ]

    clusters.sort(
        key=lambda group: (
            -len(group),
            group[0]
        )
    )

    print(
        "Duplicate edges used:",
        f"{duplicate_edges:,}"
    )

    print(
        "  Exact:",
        f"{edge_counts['exact_duplicate']:,}"
    )

    print(
        "  Probable near duplicate:",
        f"{edge_counts['probable_near_duplicate']:,}"
    )

    print(
        "Connected duplicate clusters:",
        f"{len(clusters):,}"
    )

    return clusters


# ============================================================
# LOAD SMALL AMOUNT OF METADATA FOR DUPLICATE CANDIDATES
#
# We do NOT load the entire 1+ GB corpus into RAM.
# ============================================================

def load_duplicate_candidate_metadata(
    clusters,
    quality_excluded
):

    candidate_ids = set()

    for cluster in clusters:

        for document_id in cluster:

            # No need to evaluate canonical quality for a record
            # already excluded by language/token-ratio rules.
            if document_id not in quality_excluded:

                candidate_ids.add(
                    document_id
                )

    print(
        "\nLoading metadata for duplicate candidates..."
    )

    print(
        "Eligible duplicate candidates:",
        f"{len(candidate_ids):,}"
    )

    metadata_map = {}

    if not candidate_ids:
        return metadata_map

    for record in load_jsonl(
        CPT_FILE
    ):

        document_id = safe_text(
            record.get("id")
        )

        if document_id not in candidate_ids:
            continue

        metadata = record.get(
            "metadata",
            {}
        )

        metadata_map[
            document_id
        ] = {
            "text_status":
                safe_text(
                    metadata.get(
                        "text_status"
                    )
                ),

            "parse_status":
                safe_text(
                    metadata.get(
                        "parse_status"
                    )
                ),

            "title":
                safe_text(
                    metadata.get(
                        "title"
                    )
                ),

            "source_file":
                safe_text(
                    metadata.get(
                        "source_file"
                    )
                ),

            "court":
                safe_text(
                    metadata.get(
                        "court"
                    )
                ),

            "judgment_date":
                safe_text(
                    metadata.get(
                        "judgment_date"
                    )
                ),

            "citation":
                safe_text(
                    metadata.get(
                        "citation"
                    )
                )
        }

        if (
            len(metadata_map)
            == len(candidate_ids)
        ):
            break

    missing = (
        candidate_ids
        - metadata_map.keys()
    )

    if missing:

        print(
            "WARNING: duplicate candidate metadata "
            f"missing for {len(missing):,} records."
        )

    print(
        "Candidate metadata loaded:",
        f"{len(metadata_map):,}"
    )

    return metadata_map


# ============================================================
# CANONICAL RECORD QUALITY
# ============================================================

def parser_quality_score(
    parse_status
):

    status = safe_text(
        parse_status
    ).lower()

    if not status:
        return 1

    bad_markers = [
        "error",
        "failed",
        "failure",
        "invalid",
    ]

    for marker in bad_markers:

        if marker in status:
            return 0

    if "review" in status:
        return 1

    return 2


def text_status_score(
    text_status
):

    status = safe_text(
        text_status
    ).lower()

    if status == "usable":
        return 2

    # Older parser records may have no text_status.
    if not status:
        return 1

    return 0


def metadata_completeness_score(
    metadata
):

    fields = [
        "title",
        "source_file",
        "court",
        "judgment_date",
        "citation",
    ]

    return sum(
        1
        for field in fields
        if metadata.get(field)
    )


def choose_canonical_document(
    candidate_ids,
    stats,
    metadata_map
):
    """
    Deterministic canonical-record selection.

    Priority:

    1. usable text_status
    2. better parser status
    3. richer metadata
    4. slightly longer token count
    5. lexical document ID tie-breaker

    Returns ONE document ID.
    """

    candidates = []

    for document_id in candidate_ids:

        stat = stats.get(
            document_id,
            {}
        )

        metadata = metadata_map.get(
            document_id,
            {}
        )

        candidate = {
            "id":
                document_id,

            "text_status_score":
                text_status_score(
                    metadata.get(
                        "text_status"
                    )
                ),

            "parser_quality_score":
                parser_quality_score(
                    metadata.get(
                        "parse_status"
                    )
                ),

            "metadata_score":
                metadata_completeness_score(
                    metadata
                ),

            "tokens":
                safe_int(
                    stat.get(
                        "tokens"
                    )
                )
        }

        candidates.append(
            candidate
        )

    # Sort best candidate FIRST.
    candidates.sort(
        key=lambda item: (
            -item["text_status_score"],
            -item["parser_quality_score"],
            -item["metadata_score"],
            -item["tokens"],
            item["id"]
        )
    )

    return candidates[0]["id"]


# ============================================================
# APPLY DUPLICATE CLEANING
# ============================================================

def apply_duplicate_cleaning(
    clusters,
    quality_excluded,
    stats,
    metadata_map
):

    print(
        "\nResolving duplicate clusters..."
    )

    duplicate_excluded = {}

    cluster_report_rows = []

    clusters_with_multiple_eligible = 0
    clusters_with_one_eligible = 0
    clusters_with_zero_eligible = 0

    removed_documents = 0
    removed_tokens = 0

    for cluster_number, cluster in enumerate(
        clusters,
        start=1
    ):

        cluster_id = (
            f"dup_cluster_{cluster_number:05d}"
        )

        eligible_ids = [
            document_id
            for document_id in cluster
            if document_id not in quality_excluded
        ]

        filtered_ids = [
            document_id
            for document_id in cluster
            if document_id in quality_excluded
        ]

        # ----------------------------------------------------
        # All documents were already rejected by quality rules.
        # ----------------------------------------------------

        if len(eligible_ids) == 0:

            clusters_with_zero_eligible += 1

            canonical_id = ""

        # ----------------------------------------------------
        # Only one good document remains.
        # No further deduplication necessary.
        # ----------------------------------------------------

        elif len(eligible_ids) == 1:

            clusters_with_one_eligible += 1

            canonical_id = (
                eligible_ids[0]
            )

        # ----------------------------------------------------
        # Multiple eligible duplicates remain.
        # Choose canonical, remove others.
        # ----------------------------------------------------

        else:

            clusters_with_multiple_eligible += 1

            canonical_id = (
                choose_canonical_document(
                    eligible_ids,
                    stats,
                    metadata_map
                )
            )

            for document_id in eligible_ids:

                if document_id == canonical_id:
                    continue

                duplicate_excluded[
                    document_id
                ] = {
                    "primary_reason":
                        "duplicate_exact_or_probable",

                    "all_reasons": [
                        "duplicate_exact_or_probable"
                    ],

                    "canonical_id":
                        canonical_id,

                    "duplicate_cluster_id":
                        cluster_id
                }

                removed_documents += 1

                removed_tokens += (
                    stats.get(
                        document_id,
                        {}
                    ).get(
                        "tokens",
                        0
                    )
                )

        # ----------------------------------------------------
        # Cluster audit report
        # ----------------------------------------------------

        total_cluster_tokens = sum(
            stats.get(
                document_id,
                {}
            ).get(
                "tokens",
                0
            )
            for document_id in cluster
        )

        eligible_tokens = sum(
            stats.get(
                document_id,
                {}
            ).get(
                "tokens",
                0
            )
            for document_id in eligible_ids
        )

        cluster_report_rows.append({
            "cluster_id":
                cluster_id,

            "cluster_size":
                len(cluster),

            "eligible_after_quality_filters":
                len(eligible_ids),

            "already_quality_filtered":
                len(filtered_ids),

            "canonical_id":
                canonical_id,

            "total_cluster_tokens":
                total_cluster_tokens,

            "eligible_tokens":
                eligible_tokens,

            "document_ids":
                " | ".join(cluster),

            "quality_filtered_ids":
                " | ".join(filtered_ids)
        })

    print(
        "Duplicate clusters with multiple "
        "eligible documents:",
        f"{clusters_with_multiple_eligible:,}"
    )

    print(
        "Duplicate clusters with one eligible document:",
        f"{clusters_with_one_eligible:,}"
    )

    print(
        "Duplicate clusters fully quality-filtered:",
        f"{clusters_with_zero_eligible:,}"
    )

    print(
        "Documents removed by deduplication:",
        f"{removed_documents:,}"
    )

    print(
        "Tokens removed by deduplication:",
        f"{removed_tokens:,}"
    )

    return (
        duplicate_excluded,
        cluster_report_rows
    )


# ============================================================
# MERGE EXCLUSION REASONS
# ============================================================

def merge_exclusions(
    quality_excluded,
    duplicate_excluded
):

    exclusions = {}

    for document_id, data in (
        quality_excluded.items()
    ):

        exclusions[
            document_id
        ] = dict(data)

    for document_id, data in (
        duplicate_excluded.items()
    ):

        # Normally there should be no overlap because duplicate
        # selection only considers records surviving quality
        # filtering.
        if document_id in exclusions:

            existing = exclusions[
                document_id
            ]

            existing[
                "all_reasons"
            ] = list(
                dict.fromkeys(
                    existing[
                        "all_reasons"
                    ]
                    + data[
                        "all_reasons"
                    ]
                )
            )

        else:

            exclusions[
                document_id
            ] = dict(data)

    return exclusions


# ============================================================
# WRITE FINAL CLEAN CORPUS
# ============================================================

def write_clean_corpus(
    stats,
    exclusions
):

    print(
        "\nWriting final cleaned CPT corpus..."
    )

    source_before_docs = Counter()
    source_before_tokens = Counter()

    source_after_docs = Counter()
    source_after_tokens = Counter()

    exclusion_reason_docs = Counter()
    exclusion_reason_tokens = Counter()

    total_input_documents = 0
    total_input_tokens = 0

    total_output_documents = 0
    total_output_tokens = 0

    missing_stats = []

    exclusion_rows = []

    with open(
        CLEAN_CORPUS_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for index, record in enumerate(
            load_jsonl(CPT_FILE),
            start=1
        ):

            document_id = safe_text(
                record.get("id")
            )

            source_type = safe_text(
                record.get(
                    "source_type"
                )
            )

            stat = stats.get(
                document_id
            )

            if stat is None:

                missing_stats.append(
                    document_id
                )

                raise ValueError(
                    "Missing quality statistics for "
                    f"document: {document_id}"
                )

            tokens = stat["tokens"]

            total_input_documents += 1
            total_input_tokens += tokens

            source_before_docs[
                source_type
            ] += 1

            source_before_tokens[
                source_type
            ] += tokens

            # ------------------------------------------------
            # Excluded
            # ------------------------------------------------

            if document_id in exclusions:

                exclusion = exclusions[
                    document_id
                ]

                primary_reason = (
                    exclusion[
                        "primary_reason"
                    ]
                )

                exclusion_reason_docs[
                    primary_reason
                ] += 1

                exclusion_reason_tokens[
                    primary_reason
                ] += tokens

                exclusion_rows.append({
                    "id":
                        document_id,

                    "source_type":
                        source_type,

                    "title":
                        stat["title"],

                    "source_file":
                        stat["source_file"],

                    "tokens":
                        tokens,

                    "words":
                        stat["words"],

                    "tokens_per_word":
                        stat[
                            "tokens_per_word"
                        ],

                    "sc_language_class":
                        stat[
                            "sc_language_class"
                        ],

                    "sc_language":
                        stat[
                            "sc_language"
                        ],

                    "primary_reason":
                        primary_reason,

                    "all_reasons":
                        "; ".join(
                            exclusion[
                                "all_reasons"
                            ]
                        ),

                    "duplicate_cluster_id":
                        exclusion.get(
                            "duplicate_cluster_id",
                            ""
                        ),

                    "canonical_id":
                        exclusion.get(
                            "canonical_id",
                            ""
                        )
                })

                continue

            # ------------------------------------------------
            # Keep exact original CPT record
            # ------------------------------------------------

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

            total_output_documents += 1
            total_output_tokens += tokens

            source_after_docs[
                source_type
            ] += 1

            source_after_tokens[
                source_type
            ] += tokens

            if index % 5000 == 0:

                print(
                    f"Processed {index:,} records | "
                    f"kept {total_output_documents:,}"
                )

    return {
        "total_input_documents":
            total_input_documents,

        "total_input_tokens":
            total_input_tokens,

        "total_output_documents":
            total_output_documents,

        "total_output_tokens":
            total_output_tokens,

        "source_before_docs":
            source_before_docs,

        "source_before_tokens":
            source_before_tokens,

        "source_after_docs":
            source_after_docs,

        "source_after_tokens":
            source_after_tokens,

        "exclusion_reason_docs":
            exclusion_reason_docs,

        "exclusion_reason_tokens":
            exclusion_reason_tokens,

        "exclusion_rows":
            exclusion_rows
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("FINAL CLEAN CPT CORPUS BUILDER")
    print("=" * 76)

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "\nInput CPT corpus:",
        CPT_FILE
    )

    print(
        "Document stats:",
        DOCUMENT_STATS_FILE
    )

    print(
        "Duplicate audit:",
        DUPLICATE_PAIRS_FILE
    )

    # ========================================================
    # VALIDATE INPUT FILES
    # ========================================================

    required_files = [
        CPT_FILE,
        DOCUMENT_STATS_FILE,
        DUPLICATE_PAIRS_FILE
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 1. LOAD QUALITY STATS
    # ========================================================

    stats = load_document_stats()

    # ========================================================
    # 2. QUALITY FILTERS
    # ========================================================

    (
        quality_excluded,
        quality_reason_counts,
        quality_reason_tokens
    ) = determine_quality_exclusions(
        stats
    )

    # ========================================================
    # 3. DUPLICATE CLUSTERS
    # ========================================================

    clusters = (
        build_duplicate_clusters()
    )

    # ========================================================
    # 4. LOAD METADATA NEEDED FOR CANONICAL SELECTION
    # ========================================================

    metadata_map = (
        load_duplicate_candidate_metadata(
            clusters,
            quality_excluded
        )
    )

    # ========================================================
    # 5. RESOLVE DUPLICATES
    # ========================================================

    (
        duplicate_excluded,
        cluster_report_rows
    ) = apply_duplicate_cleaning(
        clusters,
        quality_excluded,
        stats,
        metadata_map
    )

    # ========================================================
    # 6. COMBINE ALL EXCLUSIONS
    # ========================================================

    exclusions = merge_exclusions(
        quality_excluded,
        duplicate_excluded
    )

    print(
        "\nTotal documents scheduled "
        "for exclusion:",
        f"{len(exclusions):,}"
    )

    # ========================================================
    # 7. WRITE CLEAN CORPUS
    # ========================================================

    results = write_clean_corpus(
        stats,
        exclusions
    )

    # ========================================================
    # 8. SAVE EXCLUSION AUDIT
    # ========================================================

    exclusion_rows = (
        results[
            "exclusion_rows"
        ]
    )

    exclusion_rows.sort(
        key=lambda row: (
            row["primary_reason"],
            -row["tokens"],
            row["id"]
        )
    )

    write_csv(
        EXCLUSIONS_FILE,
        exclusion_rows,
        [
            "id",
            "source_type",
            "title",
            "source_file",
            "tokens",
            "words",
            "tokens_per_word",
            "sc_language_class",
            "sc_language",
            "primary_reason",
            "all_reasons",
            "duplicate_cluster_id",
            "canonical_id"
        ]
    )

    # ========================================================
    # 9. SAVE DUPLICATE CLUSTER REPORT
    # ========================================================

    write_csv(
        DUPLICATE_CLUSTERS_FILE,
        cluster_report_rows,
        [
            "cluster_id",
            "cluster_size",
            "eligible_after_quality_filters",
            "already_quality_filtered",
            "canonical_id",
            "total_cluster_tokens",
            "eligible_tokens",
            "document_ids",
            "quality_filtered_ids"
        ]
    )

    # ========================================================
    # 10. BUILD SOURCE SUMMARY
    # ========================================================

    source_summary = {}

    all_sources = sorted(
        set(
            results[
                "source_before_docs"
            ].keys()
        )
        |
        set(
            results[
                "source_after_docs"
            ].keys()
        )
    )

    for source_type in all_sources:

        before_docs = (
            results[
                "source_before_docs"
            ][source_type]
        )

        before_tokens = (
            results[
                "source_before_tokens"
            ][source_type]
        )

        after_docs = (
            results[
                "source_after_docs"
            ][source_type]
        )

        after_tokens = (
            results[
                "source_after_tokens"
            ][source_type]
        )

        source_summary[
            source_type
        ] = {
            "before_documents":
                before_docs,

            "before_tokens":
                before_tokens,

            "after_documents":
                after_docs,

            "after_tokens":
                after_tokens,

            "documents_removed":
                before_docs
                - after_docs,

            "tokens_removed":
                before_tokens
                - after_tokens,

            "final_token_percent":
                (
                    round(
                        100
                        * after_tokens
                        / results[
                            "total_output_tokens"
                        ],
                        4
                    )
                    if results[
                        "total_output_tokens"
                    ]
                    else 0
                )
        }

    # ========================================================
    # 11. EXCLUSION SUMMARY
    # ========================================================

    exclusion_summary = {}

    for reason in sorted(
        results[
            "exclusion_reason_docs"
        ].keys()
    ):

        exclusion_summary[
            reason
        ] = {
            "documents":
                results[
                    "exclusion_reason_docs"
                ][reason],

            "tokens":
                results[
                    "exclusion_reason_tokens"
                ][reason]
        }

    # ========================================================
    # 12. SUMMARY JSON
    # ========================================================

    removed_documents = (
        results[
            "total_input_documents"
        ]
        - results[
            "total_output_documents"
        ]
    )

    removed_tokens = (
        results[
            "total_input_tokens"
        ]
        - results[
            "total_output_tokens"
        ]
    )

    retained_percent = (
        100
        * results[
            "total_output_tokens"
        ]
        / results[
            "total_input_tokens"
        ]
        if results[
            "total_input_tokens"
        ]
        else 0
    )

    summary = {
        "input_file":
            str(CPT_FILE),

        "output_file":
            str(CLEAN_CORPUS_FILE),

        "cleaning_policy": {
            "keep_central_acts":
                True,

            "keep_constitution":
                True,

            "remove_sc_vernacular_tagged":
                True,

            "remove_sc_tokens_per_word_gte":
                TOKEN_WORD_RATIO_THRESHOLD,

            "deduplicate_using_classes":
                sorted(
                    DUPLICATE_CLASSES
                ),

            "keep_duplicate_review_pairs":
                True,

            "keep_likely_distinct_pairs":
                True,

            "remove_long_documents_based_only_on_length":
                False
        },

        "before": {
            "documents":
                results[
                    "total_input_documents"
                ],

            "tokens":
                results[
                    "total_input_tokens"
                ]
        },

        "after": {
            "documents":
                results[
                    "total_output_documents"
                ],

            "tokens":
                results[
                    "total_output_tokens"
                ],

            "tokens_retained_percent":
                retained_percent
        },

        "removed": {
            "documents":
                removed_documents,

            "tokens":
                removed_tokens
        },

        "exclusions_by_primary_reason":
            exclusion_summary,

        "source_summary":
            source_summary,

        "duplicate_summary": {
            "clusters":
                len(clusters),

            "documents_removed_as_duplicates":
                len(
                    duplicate_excluded
                )
        },

        "audit_files": {
            "exclusions":
                str(
                    EXCLUSIONS_FILE
                ),

            "duplicate_clusters":
                str(
                    DUPLICATE_CLUSTERS_FILE
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
    # FINAL TERMINAL REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "FINAL CLEAN CPT CORPUS COMPLETE"
    )

    print(
        "=" * 76
    )

    print(
        "\nBEFORE CLEANING"
    )

    print(
        "-" * 76
    )

    print(
        "Documents:",
        f"{results['total_input_documents']:,}"
    )

    print(
        "Tokens:",
        f"{results['total_input_tokens']:,}"
    )

    print(
        "\nEXCLUSIONS"
    )

    print(
        "-" * 76
    )

    for reason, data in (
        exclusion_summary.items()
    ):

        print(
            f"{reason:<35} "
            f"docs={data['documents']:>6,} | "
            f"tokens={data['tokens']:>12,}"
        )

    print(
        "\nTotal documents removed:",
        f"{removed_documents:,}"
    )

    print(
        "Total tokens removed:",
        f"{removed_tokens:,}"
    )

    print(
        "\nFINAL CLEAN CORPUS"
    )

    print(
        "-" * 76
    )

    print(
        "Documents:",
        f"{results['total_output_documents']:,}"
    )

    print(
        "Tokens:",
        f"{results['total_output_tokens']:,}"
    )

    print(
        "Tokens retained:",
        f"{retained_percent:.2f}%"
    )

    print(
        "\nFINAL SOURCE DISTRIBUTION"
    )

    print(
        "-" * 76
    )

    for source_type, data in (
        source_summary.items()
    ):

        print(
            f"{source_type:<20} "
            f"docs={data['after_documents']:>8,} | "
            f"tokens={data['after_tokens']:>12,} | "
            f"{data['final_token_percent']:>6.2f}%"
        )

    print(
        "\nOUTPUT FILES"
    )

    print(
        "-" * 76
    )

    print(
        "Clean corpus:",
        CLEAN_CORPUS_FILE
    )

    print(
        "Summary:",
        SUMMARY_FILE
    )

    print(
        "Exclusions audit:",
        EXCLUSIONS_FILE
    )

    print(
        "Duplicate clusters:",
        DUPLICATE_CLUSTERS_FILE
    )

    print(
        "\nOriginal corpus was NOT modified."
    )


if __name__ == "__main__":
    main()