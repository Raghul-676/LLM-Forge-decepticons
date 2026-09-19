from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import json
import re


# ============================================================
# CONFIG
# ============================================================

CPT_FILE = Path(
    "data/training/cpt/legal_cpt_corpus.jsonl"
)

QUALITY_DIR = Path(
    "data/training/cpt/quality"
)

PAIR_REPORT_FILE = (
    QUALITY_DIR / "sc_near_duplicate_pairs.csv"
)

GROUP_REPORT_FILE = (
    QUALITY_DIR / "sc_near_duplicate_groups.csv"
)

SUMMARY_FILE = (
    QUALITY_DIR / "sc_near_duplicate_summary.json"
)


# ------------------------------------------------------------
# Fingerprint configuration
# ------------------------------------------------------------

# Number of minimum hashed shingles kept for each document.
SIGNATURE_SIZE = 128

# Number of words in each shingle.
SHINGLE_SIZE = 5


# ------------------------------------------------------------
# Classification thresholds
# ------------------------------------------------------------

# Exact normalized text match.
EXACT_DUPLICATE = "exact_duplicate"

# Extremely strong evidence of same substantive text.
PROBABLE_DUPLICATE = "probable_near_duplicate"

# Similar enough that we should inspect it manually.
REVIEW = "review"

# Probably genuinely different documents.
DISTINCT = "likely_distinct"


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


def safe_text(value):

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# CASE KEY
# ============================================================

def extract_case_key(document_id):
    """
    Example:

    sc_439-2015___jonew__judis__43070
             ↓
    439-2015

    This is ONLY used to create candidate groups.
    Sharing a case key does NOT automatically mean duplicate.
    """

    document_id = safe_text(
        document_id
    )

    if document_id.startswith("sc_"):
        document_id = document_id[3:]

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
# TEXT NORMALIZATION
# ============================================================

def normalize_for_duplicate_detection(text):
    """
    More aggressive than CPT cleaning because this copy is used
    ONLY for duplicate comparison.

    It does not modify the actual CPT corpus.
    """

    text = safe_text(text)

    text = text.lower()

    # Normalize common Unicode punctuation.
    text = (
        text
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_title(title):

    title = normalize_for_duplicate_detection(
        title
    )

    # Remove punctuation for title comparison.
    title = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        title
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    return title.strip()


# ============================================================
# STABLE HASH
# ============================================================

def stable_hash64(text):
    """
    Deterministic 64-bit hash.
    """

    digest = hashlib.blake2b(
        text.encode(
            "utf-8",
            errors="ignore"
        ),
        digest_size=8
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False
    )


# ============================================================
# EXACT NORMALIZED HASH
# ============================================================

def normalized_sha256(normalized_text):

    return hashlib.sha256(
        normalized_text.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()


# ============================================================
# WORD TOKENIZATION
# ============================================================

WORD_PATTERN = re.compile(
    r"\w+",
    flags=re.UNICODE
)


def get_words(normalized_text):

    return WORD_PATTERN.findall(
        normalized_text
    )


# ============================================================
# BOTTOM-K SHINGLE SIGNATURE
# ============================================================

def build_signature(words):
    """
    Build a compact content fingerprint.

    We calculate hashes of 5-word shingles and retain the
    SIGNATURE_SIZE smallest hash values.

    This is similar to a bottom-k / MinHash-style sketch.

    Important advantage:
    small header/footer differences do not completely destroy
    similarity, unlike hashing the entire document.
    """

    if not words:
        return tuple()

    # For tiny documents, reduce shingle size naturally.
    if len(words) < SHINGLE_SIZE:

        return (
            stable_hash64(
                " ".join(words)
            ),
        )

    # Python heapq is a min-heap.
    # Store negative values to simulate a max-heap containing
    # the smallest SIGNATURE_SIZE hashes.
    heap = []

    seen_hashes = set()

    limit = (
        len(words)
        - SHINGLE_SIZE
        + 1
    )

    for i in range(limit):

        shingle = " ".join(
            words[
                i:
                i + SHINGLE_SIZE
            ]
        )

        hash_value = stable_hash64(
            shingle
        )

        # Repeated boilerplate should not dominate the signature.
        if hash_value in seen_hashes:
            continue

        seen_hashes.add(
            hash_value
        )

        if len(heap) < SIGNATURE_SIZE:

            heapq.heappush(
                heap,
                -hash_value
            )

        else:

            current_largest_kept = (
                -heap[0]
            )

            if hash_value < current_largest_kept:

                removed = -heapq.heapreplace(
                    heap,
                    -hash_value
                )

                # We do NOT remove the old hash from seen_hashes.
                # seen_hashes is only used to avoid processing the
                # same shingle value repeatedly.

    return tuple(
        sorted(
            -value
            for value in heap
        )
    )


# ============================================================
# SIMHASH
# ============================================================

def build_simhash(words):
    """
    64-bit SimHash built from word tokens.

    Useful as a second, independent similarity signal.
    """

    if not words:
        return 0

    vector = [
        0
        for _ in range(64)
    ]

    # Unigrams are intentionally used here.
    # The shingle signature already captures local word order.
    for word in words:

        h = stable_hash64(
            word
        )

        for bit in range(64):

            if h & (1 << bit):

                vector[bit] += 1

            else:

                vector[bit] -= 1

    fingerprint = 0

    for bit, value in enumerate(
        vector
    ):

        if value >= 0:

            fingerprint |= (
                1 << bit
            )

    return fingerprint


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================

def signature_overlap(
    signature_a,
    signature_b
):
    """
    Overlap coefficient between compact shingle signatures.

    1.0 = signatures completely overlap.
    """

    if (
        not signature_a
        or not signature_b
    ):
        return 0.0

    a = set(signature_a)
    b = set(signature_b)

    denominator = min(
        len(a),
        len(b)
    )

    if denominator == 0:
        return 0.0

    return (
        len(
            a.intersection(b)
        )
        / denominator
    )


def simhash_similarity(
    hash_a,
    hash_b
):

    xor = hash_a ^ hash_b

    hamming_distance = (
        xor.bit_count()
    )

    return (
        1.0
        - (
            hamming_distance / 64.0
        )
    )


def length_similarity(
    length_a,
    length_b
):

    if (
        length_a <= 0
        or length_b <= 0
    ):
        return 0.0

    return (
        min(
            length_a,
            length_b
        )
        / max(
            length_a,
            length_b
        )
    )


def title_similarity(
    title_a,
    title_b
):
    """
    Simple token-set title similarity.
    """

    if not title_a or not title_b:
        return 0.0

    words_a = set(
        title_a.split()
    )

    words_b = set(
        title_b.split()
    )

    if not words_a or not words_b:
        return 0.0

    intersection = len(
        words_a & words_b
    )

    union = len(
        words_a | words_b
    )

    return (
        intersection / union
        if union
        else 0.0
    )


# ============================================================
# PAIR CLASSIFICATION
# ============================================================

def classify_pair(
    exact_match,
    shingle_similarity,
    simhash_score,
    length_score,
    title_score
):

    # --------------------------------------------------------
    # Exact normalized text
    # --------------------------------------------------------

    if exact_match:

        return (
            EXACT_DUPLICATE,
            1.0
        )

    # --------------------------------------------------------
    # Strong near duplicate
    #
    # Require multiple signals to agree.
    # --------------------------------------------------------

    if (
        shingle_similarity >= 0.90
        and simhash_score >= 0.90
        and length_score >= 0.90
    ):

        confidence = (
            0.50 * shingle_similarity
            + 0.25 * simhash_score
            + 0.20 * length_score
            + 0.05 * title_score
        )

        return (
            PROBABLE_DUPLICATE,
            confidence
        )

    # --------------------------------------------------------
    # Review zone
    # --------------------------------------------------------

    if (
        (
            shingle_similarity >= 0.70
            and length_score >= 0.80
        )
        or
        (
            shingle_similarity >= 0.80
            and simhash_score >= 0.80
        )
    ):

        confidence = (
            0.50 * shingle_similarity
            + 0.25 * simhash_score
            + 0.20 * length_score
            + 0.05 * title_score
        )

        return (
            REVIEW,
            confidence
        )

    confidence = (
        0.50 * shingle_similarity
        + 0.25 * simhash_score
        + 0.20 * length_score
        + 0.05 * title_score
    )

    return (
        DISTINCT,
        confidence
    )


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
# PASS 1
#
# Identify only same-case candidate groups.
# ============================================================

def find_candidate_groups():

    print(
        "\nPASS 1: Finding same-case candidate groups..."
    )

    case_members = defaultdict(
        list
    )

    total_sc = 0

    for record in load_jsonl(
        CPT_FILE
    ):

        if (
            record.get("source_type")
            != "supreme_court"
        ):
            continue

        total_sc += 1

        document_id = safe_text(
            record.get("id")
        )

        case_key = extract_case_key(
            document_id
        )

        if case_key:

            case_members[
                case_key
            ].append(
                document_id
            )

    candidate_groups = {
        key: ids
        for key, ids in case_members.items()
        if len(ids) > 1
    }

    candidate_ids = set()

    for ids in candidate_groups.values():

        candidate_ids.update(
            ids
        )

    print(
        "Supreme Court documents:",
        f"{total_sc:,}"
    )

    print(
        "Candidate case groups:",
        f"{len(candidate_groups):,}"
    )

    print(
        "Documents requiring comparison:",
        f"{len(candidate_ids):,}"
    )

    return (
        candidate_groups,
        candidate_ids
    )


# ============================================================
# PASS 2
#
# Build compact fingerprints only for documents that belong to
# duplicate case-key groups.
# ============================================================

def build_candidate_fingerprints(
    candidate_ids
):

    print(
        "\nPASS 2: Building document fingerprints..."
    )

    fingerprints = {}

    processed = 0

    for record in load_jsonl(
        CPT_FILE
    ):

        document_id = safe_text(
            record.get("id")
        )

        if document_id not in candidate_ids:
            continue

        text = record.get(
            "text",
            ""
        )

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

        normalized = (
            normalize_for_duplicate_detection(
                text
            )
        )

        words = get_words(
            normalized
        )

        word_count = len(
            words
        )

        character_count = len(
            normalized
        )

        signature = build_signature(
            words
        )

        simhash = build_simhash(
            words
        )

        exact_hash = normalized_sha256(
            normalized
        )

        normalized_title = (
            normalize_title(
                title
            )
        )

        fingerprints[
            document_id
        ] = {
            "id":
                document_id,

            "title":
                title,

            "normalized_title":
                normalized_title,

            "source_file":
                source_file,

            "word_count":
                word_count,

            "character_count":
                character_count,

            "exact_hash":
                exact_hash,

            "signature":
                signature,

            "simhash":
                simhash
        }

        processed += 1

        if processed % 250 == 0:

            print(
                "Fingerprinted:",
                f"{processed:,}/"
                f"{len(candidate_ids):,}"
            )

        # Explicitly release the biggest temporary objects.
        del normalized
        del words

    print(
        "Fingerprints created:",
        f"{len(fingerprints):,}"
    )

    return fingerprints


# ============================================================
# PASS 3
#
# Compare all pairs inside each same-case group.
# ============================================================

def compare_candidate_groups(
    candidate_groups,
    fingerprints
):

    print(
        "\nPASS 3: Comparing candidate pairs..."
    )

    pair_rows = []

    group_rows = []

    classification_counts = Counter()

    total_pairs = 0

    groups_with_probable_duplicates = 0
    groups_with_exact_duplicates = 0
    groups_requiring_review = 0

    for group_number, (
        case_key,
        document_ids
    ) in enumerate(
        sorted(
            candidate_groups.items()
        ),
        start=1
    ):

        available_ids = [
            document_id
            for document_id in document_ids
            if document_id in fingerprints
        ]

        group_pair_rows = []

        # ----------------------------------------------------
        # Pairwise comparison only inside this case.
        # ----------------------------------------------------

        for i in range(
            len(available_ids)
        ):

            for j in range(
                i + 1,
                len(available_ids)
            ):

                id_a = available_ids[i]
                id_b = available_ids[j]

                a = fingerprints[id_a]
                b = fingerprints[id_b]

                exact_match = (
                    a["exact_hash"]
                    == b["exact_hash"]
                )

                shingle_score = (
                    signature_overlap(
                        a["signature"],
                        b["signature"]
                    )
                )

                simhash_score = (
                    simhash_similarity(
                        a["simhash"],
                        b["simhash"]
                    )
                )

                word_length_score = (
                    length_similarity(
                        a["word_count"],
                        b["word_count"]
                    )
                )

                char_length_score = (
                    length_similarity(
                        a["character_count"],
                        b["character_count"]
                    )
                )

                # Average both length measurements.
                combined_length_score = (
                    (
                        word_length_score
                        + char_length_score
                    )
                    / 2.0
                )

                title_score = (
                    title_similarity(
                        a[
                            "normalized_title"
                        ],
                        b[
                            "normalized_title"
                        ]
                    )
                )

                (
                    classification,
                    confidence
                ) = classify_pair(
                    exact_match,
                    shingle_score,
                    simhash_score,
                    combined_length_score,
                    title_score
                )

                classification_counts[
                    classification
                ] += 1

                total_pairs += 1

                row = {
                    "case_key":
                        case_key,

                    "classification":
                        classification,

                    "confidence":
                        round(
                            confidence,
                            4
                        ),

                    "id_a":
                        id_a,

                    "id_b":
                        id_b,

                    "title_a":
                        a["title"],

                    "title_b":
                        b["title"],

                    "source_file_a":
                        a["source_file"],

                    "source_file_b":
                        b["source_file"],

                    "words_a":
                        a["word_count"],

                    "words_b":
                        b["word_count"],

                    "characters_a":
                        a[
                            "character_count"
                        ],

                    "characters_b":
                        b[
                            "character_count"
                        ],

                    "exact_normalized_match":
                        exact_match,

                    "shingle_similarity":
                        round(
                            shingle_score,
                            4
                        ),

                    "simhash_similarity":
                        round(
                            simhash_score,
                            4
                        ),

                    "length_similarity":
                        round(
                            combined_length_score,
                            4
                        ),

                    "title_similarity":
                        round(
                            title_score,
                            4
                        )
                }

                pair_rows.append(
                    row
                )

                group_pair_rows.append(
                    row
                )

        # ----------------------------------------------------
        # Group classification summary
        # ----------------------------------------------------

        group_classes = Counter(
            row["classification"]
            for row in group_pair_rows
        )

        if (
            group_classes[
                EXACT_DUPLICATE
            ] > 0
        ):
            groups_with_exact_duplicates += 1

        if (
            group_classes[
                PROBABLE_DUPLICATE
            ] > 0
        ):
            groups_with_probable_duplicates += 1

        if (
            group_classes[
                REVIEW
            ] > 0
        ):
            groups_requiring_review += 1

        highest_confidence = max(
            (
                row["confidence"]
                for row in group_pair_rows
            ),
            default=0.0
        )

        group_rows.append({
            "case_key":
                case_key,

            "document_count":
                len(
                    available_ids
                ),

            "pair_count":
                len(
                    group_pair_rows
                ),

            "exact_duplicate_pairs":
                group_classes[
                    EXACT_DUPLICATE
                ],

            "probable_near_duplicate_pairs":
                group_classes[
                    PROBABLE_DUPLICATE
                ],

            "review_pairs":
                group_classes[
                    REVIEW
                ],

            "likely_distinct_pairs":
                group_classes[
                    DISTINCT
                ],

            "highest_confidence":
                highest_confidence,

            "document_ids":
                " | ".join(
                    available_ids
                )
        })

        if group_number % 250 == 0:

            print(
                "Groups compared:",
                f"{group_number:,}/"
                f"{len(candidate_groups):,}"
            )

    # Sort most suspicious pairs first.
    class_priority = {
        EXACT_DUPLICATE: 0,
        PROBABLE_DUPLICATE: 1,
        REVIEW: 2,
        DISTINCT: 3
    }

    pair_rows.sort(
        key=lambda row: (
            class_priority[
                row["classification"]
            ],
            -row["confidence"]
        )
    )

    group_rows.sort(
        key=lambda row: (
            -row[
                "exact_duplicate_pairs"
            ],
            -row[
                "probable_near_duplicate_pairs"
            ],
            -row[
                "review_pairs"
            ],
            -row[
                "highest_confidence"
            ]
        )
    )

    stats = {
        "total_pairs":
            total_pairs,

        "classification_counts":
            dict(
                classification_counts
            ),

        "groups_with_exact_duplicates":
            groups_with_exact_duplicates,

        "groups_with_probable_duplicates":
            groups_with_probable_duplicates,

        "groups_requiring_review":
            groups_requiring_review
    }

    return (
        pair_rows,
        group_rows,
        stats
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("SUPREME COURT NEAR-DUPLICATE AUDIT")
    print("=" * 72)

    if not CPT_FILE.exists():

        raise FileNotFoundError(
            f"CPT corpus not found: "
            f"{CPT_FILE}"
        )

    QUALITY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # PASS 1
    # ========================================================

    (
        candidate_groups,
        candidate_ids
    ) = find_candidate_groups()

    if not candidate_groups:

        print(
            "\nNo duplicate case-key groups found."
        )

        return

    # ========================================================
    # PASS 2
    # ========================================================

    fingerprints = (
        build_candidate_fingerprints(
            candidate_ids
        )
    )

    # ========================================================
    # PASS 3
    # ========================================================

    (
        pair_rows,
        group_rows,
        stats
    ) = compare_candidate_groups(
        candidate_groups,
        fingerprints
    )

    # ========================================================
    # SAVE PAIR REPORT
    # ========================================================

    write_csv(
        PAIR_REPORT_FILE,
        pair_rows,
        [
            "case_key",
            "classification",
            "confidence",
            "id_a",
            "id_b",
            "title_a",
            "title_b",
            "source_file_a",
            "source_file_b",
            "words_a",
            "words_b",
            "characters_a",
            "characters_b",
            "exact_normalized_match",
            "shingle_similarity",
            "simhash_similarity",
            "length_similarity",
            "title_similarity"
        ]
    )

    # ========================================================
    # SAVE GROUP REPORT
    # ========================================================

    write_csv(
        GROUP_REPORT_FILE,
        group_rows,
        [
            "case_key",
            "document_count",
            "pair_count",
            "exact_duplicate_pairs",
            "probable_near_duplicate_pairs",
            "review_pairs",
            "likely_distinct_pairs",
            "highest_confidence",
            "document_ids"
        ]
    )

    # ========================================================
    # BUILD SUMMARY
    # ========================================================

    summary = {
        "input_file":
            str(CPT_FILE),

        "candidate_case_groups":
            len(
                candidate_groups
            ),

        "candidate_documents":
            len(
                candidate_ids
            ),

        "fingerprinted_documents":
            len(
                fingerprints
            ),

        "pairs_compared":
            stats[
                "total_pairs"
            ],

        "classification_counts":
            stats[
                "classification_counts"
            ],

        "groups_with_exact_duplicates":
            stats[
                "groups_with_exact_duplicates"
            ],

        "groups_with_probable_duplicates":
            stats[
                "groups_with_probable_duplicates"
            ],

        "groups_requiring_review":
            stats[
                "groups_requiring_review"
            ],

        "thresholds": {
            "signature_size":
                SIGNATURE_SIZE,

            "shingle_size":
                SHINGLE_SIZE,

            "probable_near_duplicate": {
                "minimum_shingle_similarity":
                    0.90,

                "minimum_simhash_similarity":
                    0.90,

                "minimum_length_similarity":
                    0.90
            },

            "review": {
                "rule_1":
                    (
                        "shingle >= 0.70 "
                        "and length >= 0.80"
                    ),

                "rule_2":
                    (
                        "shingle >= 0.80 "
                        "and simhash >= 0.80"
                    )
            }
        },

        "report_files": {
            "pair_report":
                str(
                    PAIR_REPORT_FILE
                ),

            "group_report":
                str(
                    GROUP_REPORT_FILE
                )
        },

        "important_note": (
            "This is an audit only. "
            "No source documents were deleted."
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
    # TERMINAL SUMMARY
    # ========================================================

    counts = Counter(
        row["classification"]
        for row in pair_rows
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "NEAR-DUPLICATE AUDIT COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "\nCandidate case groups:",
        f"{len(candidate_groups):,}"
    )

    print(
        "Candidate documents:",
        f"{len(candidate_ids):,}"
    )

    print(
        "Pairs compared:",
        f"{stats['total_pairs']:,}"
    )

    print(
        "\nPAIR CLASSIFICATIONS"
    )

    print(
        "-" * 72
    )

    print(
        "Exact duplicates:",
        f"{counts[EXACT_DUPLICATE]:,}"
    )

    print(
        "Probable near duplicates:",
        f"{counts[PROBABLE_DUPLICATE]:,}"
    )

    print(
        "Review:",
        f"{counts[REVIEW]:,}"
    )

    print(
        "Likely distinct:",
        f"{counts[DISTINCT]:,}"
    )

    print(
        "\nGROUPS"
    )

    print(
        "-" * 72
    )

    print(
        "Groups containing exact duplicates:",
        f"{stats['groups_with_exact_duplicates']:,}"
    )

    print(
        "Groups containing probable duplicates:",
        f"{stats['groups_with_probable_duplicates']:,}"
    )

    print(
        "Groups requiring review:",
        f"{stats['groups_requiring_review']:,}"
    )

    print(
        "\nTOP 20 MOST SUSPICIOUS PAIRS"
    )

    print(
        "-" * 72
    )

    suspicious = [
        row
        for row in pair_rows
        if row["classification"]
        != DISTINCT
    ]

    for rank, row in enumerate(
        suspicious[:20],
        start=1
    ):

        print(
            f"\n{rank}. "
            f"{row['case_key']} | "
            f"{row['classification']}"
        )

        print(
            f"   Confidence: "
            f"{row['confidence']:.4f}"
        )

        print(
            f"   Shingle: "
            f"{row['shingle_similarity']:.4f}"
            f" | SimHash: "
            f"{row['simhash_similarity']:.4f}"
            f" | Length: "
            f"{row['length_similarity']:.4f}"
            f" | Title: "
            f"{row['title_similarity']:.4f}"
        )

        print(
            "   A:",
            row["id_a"]
        )

        print(
            "   B:",
            row["id_b"]
        )

    print(
        "\nREPORTS"
    )

    print(
        "-" * 72
    )

    print(
        "Pair report:",
        PAIR_REPORT_FILE
    )

    print(
        "Group report:",
        GROUP_REPORT_FILE
    )

    print(
        "Summary:",
        SUMMARY_FILE
    )

    print(
        "\nIMPORTANT: "
        "No records were removed."
    )


if __name__ == "__main__":
    main()