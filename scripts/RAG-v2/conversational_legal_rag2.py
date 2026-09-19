import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer

import conversational_ilsic_mapper as mapper


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Projects\LLM_training"
)

LEGAL_INDEX_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "data"
    / "rag"
    / "v2"
)

LEGAL_DOCS_FILE = (
    LEGAL_INDEX_DIR
    / "legal_documents.jsonl"
)

LEGAL_EMBEDDINGS_FILE = (
    LEGAL_INDEX_DIR
    / "legal_embeddings.npy"
)

LEGAL_OFFSETS_FILE = (
    LEGAL_INDEX_DIR
    / "legal_documents_offsets.npy"
)

LEGAL_AUTHORITY_CATALOG_FILE = (
    LEGAL_INDEX_DIR
    / "legal_authority_catalog.json"
)

# Authority-aware retrieval settings.
AUTHORITY_CATALOG_VERSION = 2
AUTHORITY_SEED_TOP_PER_ACT = 7
AUTHORITY_EXACT_SECTION_BONUS = 0.070
AUTHORITY_RESTRICTED_BONUS = 0.040
AUTHORITY_HEADING_BONUS = 0.055
AUTHORITY_HEADING_SEEDS_PER_ACT = 8
MAX_AUTHORITY_TARGETS = 8

EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)

LLM_MODEL = (
    "qwen/qwen3-32b"
)


# ============================================================
# TUNE EXISTING ILSIC MODULE
# ============================================================

# Smaller reranking pool helps GPT return complete JSON.
mapper.PRE_RERANK_SCENARIOS = 10
mapper.FINAL_ANALOGOUS_SCENARIOS = 5

# Intake should not interrogate the user endlessly.
MAX_INTAKE_FOLLOWUPS = 2

# Later research stages can still ask important questions.
MAX_TOTAL_FOLLOWUPS = 6

MAX_RESEARCH_ITERATIONS = 4


# ============================================================
# LEGAL RETRIEVAL SETTINGS
# ============================================================

LEGAL_TOP_PER_QUERY = 30

LEGAL_FUSED_POOL = 70

# LEGAL_RERANK_CANDIDATES = 14
LEGAL_RERANK_CANDIDATES = 8

FINAL_SOURCE_K = 6

# When strict verification yields no final sources, keep a small set of
# still-relevant retrieved passages for a clearly labelled preliminary answer.
PRELIMINARY_SOURCE_K = 4
PRELIMINARY_MIN_LEGAL_SCORE = 1

# Passage verification should remain strict about legal meaning, but not fail
# merely because an LLM copied a quote with slightly different punctuation or
# wording. These thresholds are used only to recover a real sentence/clause
# from the retrieved passage; they do not invent legal propositions.
ANCHOR_SEQUENCE_THRESHOLD = 0.72
ANCHOR_TOKEN_COVERAGE_THRESHOLD = 0.78

# After case-law verification, search the Central Acts corpus again for the
# primary statutory text behind an accepted legal rule.
PRIMARY_AUTHORITY_TOP_PER_RULE = 8
PRIMARY_AUTHORITY_MAX_CANDIDATES = 12
PRIMARY_AUTHORITY_MAX_SOURCES = 4

LEGAL_RRF_K = 60

LEGAL_PREVIEW_CHARS = 650

FINAL_SOURCE_CHARS = 1400


# ============================================================
# LLM TOKEN SETTINGS
# ============================================================

HYPOTHESIS_MAX_TOKENS = 650

# LEGAL_RERANK_MAX_TOKENS = 1900
LEGAL_RERANK_MAX_TOKENS = 3200

DRAFT_ANSWER_MAX_TOKENS = 750

CLAIM_AUDIT_MAX_TOKENS = 1200

FINAL_ANSWER_MAX_TOKENS = 850

# Preliminary answers are intentionally shorter and more cautious because
# their source passages did not pass the strict final-verification threshold.
PRELIMINARY_ANSWER_MAX_TOKENS = 700


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(value):

    text = str(
        value or ""
    )

    text = text.replace(
        "\u00a0",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_match_text(value):

    value = normalize_text(
        value
    ).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def shorten(
    text,
    limit
):

    text = normalize_text(
        text
    )

    if len(text) <= limit:
        return text

    return (
        text[:limit]
        + "..."
    )


# ============================================================
# RESILIENT STRUCTURED-OUTPUT HELPERS
# ============================================================

def strip_json_code_fences(text):
    """Remove common markdown fences without otherwise rewriting JSON."""

    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"\s*```$",
            "",
            text
        )

    return text.strip()


def recover_complete_objects_from_array(
    raw_text,
    array_key
):
    """
    Recover only COMPLETE top-level objects from a named JSON array.

    This is deliberately conservative. If the model is truncated halfway
    through the next object, that unfinished object is ignored while every
    earlier complete object is preserved.
    """

    text = strip_json_code_fences(raw_text)

    match = re.search(
        rf'"{re.escape(array_key)}"\s*:\s*\[',
        text
    )

    if not match:
        return [], True

    array_start = text.find(
        "[",
        match.start()
    )

    if array_start < 0:
        return [], True

    items = []
    item_start = None
    brace_depth = 0
    bracket_depth = 1
    in_string = False
    escaped = False
    array_closed = False

    index = array_start + 1

    while index < len(text):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False

            index += 1
            continue

        if char == '"':
            in_string = True
            index += 1
            continue

        if item_start is None:
            if char == "{":
                item_start = index
                brace_depth = 1
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    array_closed = True
                    break

            index += 1
            continue

        # We are inside one top-level object from the target array.
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1

            if brace_depth == 0:
                candidate = text[
                    item_start:index + 1
                ]

                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        items.append(parsed)
                except Exception:
                    # Complete-looking but invalid object: fail closed by
                    # dropping just this object, not the earlier ones.
                    pass

                item_start = None

        index += 1

    partial = not array_closed or item_start is not None

    return items, partial


def parse_json_array_resilient(
    raw_text,
    array_key
):
    """
    Parse a normal JSON response when possible; otherwise salvage complete
    objects from the requested array.

    Returns: (items, partial_recovery)
    """

    text = strip_json_code_fences(raw_text)

    # First try the response exactly as returned.
    try:
        data = json.loads(text)
        items = data.get(array_key, []) if isinstance(data, dict) else []
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)], False
    except Exception:
        pass

    # Then try extracting a complete outer JSON object if the model wrapped
    # it in a short prose prefix/suffix.
    first = text.find("{")
    last = text.rfind("}")

    if first >= 0 and last > first:
        try:
            data = json.loads(text[first:last + 1])
            items = data.get(array_key, []) if isinstance(data, dict) else []
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)], False
        except Exception:
            pass

    # Finally preserve only complete objects from a truncated array.
    return recover_complete_objects_from_array(
        text,
        array_key
    )


# ============================================================
# FALLBACK SCENARIO SUMMARY
# ============================================================

def build_fallback_scenario(
    original_problem,
    followups
):

    parts = [
        original_problem
    ]

    for item in followups:

        question = normalize_text(
            item.get(
                "question",
                ""
            )
        )

        answer = normalize_text(
            item.get(
                "answer",
                ""
            )
        )

        if question and answer:

            parts.append(
                f"{question} "
                f"Answer: {answer}"
            )

    result = " ".join(
        parts
    )

    # BGE has a 512-token input limit.
    return shorten(
        result,
        1800
    )


# ============================================================
# LEGAL CORPUS BYTE OFFSETS
# ============================================================

def build_or_load_offsets(
    expected_rows
):
    """
    The legal corpus has ~710K JSONL records.

    Loading every JSON object into RAM is unnecessary.

    We create a small array containing the byte offset of
    each JSONL record. Then we can directly seek to only
    the 50-100 retrieved records we need.
    """

    if LEGAL_OFFSETS_FILE.exists():

        offsets = np.load(
            LEGAL_OFFSETS_FILE,
            mmap_mode="r"
        )

        if len(offsets) == expected_rows:

            print(
                "Legal document offsets:",
                f"{len(offsets):,}"
            )

            return offsets

        print(
            "Existing offset file does not match "
            "embedding row count."
        )

        print(
            "Rebuilding offsets..."
        )

    print()
    print(
        "Building legal-document byte offsets..."
    )

    print(
        "This happens only once."
    )

    offsets = []

    with open(
        LEGAL_DOCS_FILE,
        "rb"
    ) as file:

        while True:

            position = file.tell()

            line = file.readline()

            if not line:
                break

            if not line.strip():
                continue

            offsets.append(
                position
            )

    offsets = np.asarray(
        offsets,
        dtype=np.int64
    )

    if len(offsets) != expected_rows:

        raise ValueError(
            "Legal document / embedding mismatch.\n"
            f"Documents: {len(offsets):,}\n"
            f"Embedding rows: {expected_rows:,}"
        )

    np.save(
        LEGAL_OFFSETS_FILE,
        offsets
    )

    print(
        "Offsets saved:"
    )

    print(
        LEGAL_OFFSETS_FILE
    )

    print(
        "Rows:",
        f"{len(offsets):,}"
    )

    return np.load(
        LEGAL_OFFSETS_FILE,
        mmap_mode="r"
    )


# ============================================================
# READ LEGAL DOCUMENTS BY VECTOR INDEX
# ============================================================

def read_legal_documents(
    indexes,
    offsets
):

    documents = {}

    with open(
        LEGAL_DOCS_FILE,
        "rb"
    ) as file:

        for index in indexes:

            index = int(
                index
            )

            file.seek(
                int(
                    offsets[index]
                )
            )

            raw = file.readline()

            try:

                document = json.loads(
                    raw.decode(
                        "utf-8"
                    )
                )

            except Exception:

                continue

            documents[
                index
            ] = document

    return documents


# ============================================================
# LEGAL DOCUMENT FIELD HELPERS
# ============================================================

def get_metadata(
    document
):

    metadata = document.get(
        "metadata",
        {}
    )

    if isinstance(
        metadata,
        dict
    ):

        return metadata

    return {}


def first_value(
    document,
    keys
):

    metadata = get_metadata(
        document
    )

    for key in keys:

        value = document.get(
            key
        )

        if value not in {
            None,
            ""
        }:

            return value

    for key in keys:

        value = metadata.get(
            key
        )

        if value not in {
            None,
            ""
        }:

            return value

    return None


def get_document_text(
    document
):

    keys = [
        "text",
        "content",
        "chunk_text",
        "page_content",
        "body"
    ]

    value = first_value(
        document,
        keys
    )

    if isinstance(
        value,
        str
    ):

        return normalize_text(
            value
        )

    # Last-resort fallback.
    return normalize_text(
        json.dumps(
            document,
            ensure_ascii=False,
            default=str
        )
    )


def get_source_type(
    document
):

    value = first_value(
        document,
        [
            "source_type",
            "type",
            "document_type",
            "category"
        ]
    )

    if value:

        return normalize_text(
            value
        ).lower()

    text = get_document_text(
        document
    ).lower()

    if (
        "supreme court of india"
        in text
    ):

        return "supreme_court"

    if (
        "constitution of india"
        in text
    ):

        return "constitution"

    return "unknown"


def get_source_title(
    document
):

    value = first_value(
        document,
        [
            "title",
            "act_title",
            "act_name",
            "case_name",
            "case_title",
            "name",
            "citation",
            "source"
        ]
    )

    if value:

        return normalize_text(
            value
        )

    return "Untitled legal source"


def get_section(
    document
):
    """
    Return the most specific provision locator available in the corpus.

    Older corpus rows mostly use `section`, but schedules, articles,
    rules and orders can appear under different metadata keys. Treating
    all of them as locators improves exact provision retrieval without
    inventing provision numbers.
    """
    value = first_value(
        document,
        [
            "section",
            "section_number",
            "article",
            "article_number",
            "order",
            "order_number",
            "rule",
            "rule_number",
            "schedule_entry",
            "schedule_item",
            "provision",
            "item_number"
        ]
    )

    if value:
        return normalize_text(value)

    return ""


def get_document_key(
    document,
    index
):

    value = first_value(
        document,
        [
            "document_id",
            "doc_id",
            "case_id",
            "source_id",
            "file_id"
        ]
    )

    if value:

        return str(
            value
        )

    title = get_source_title(
        document
    )

    section = get_section(
        document
    )

    if (
        title
        != "Untitled legal source"
    ):

        return (
            f"{title}|{section}"
        )

    return (
        f"row-{index}"
    )


def source_label(
    document
):

    source_type = get_source_type(
        document
    )

    title = get_source_title(
        document
    )

    section = get_section(
        document
    )

    pieces = [
        source_type,
        title
    ]

    if section:

        pieces.append(
            section
        )

    return " | ".join(
        pieces
    )


# ============================================================
# AUTHORITY CATALOG + CURRENT-LAW STATUS
# ============================================================

CURRENT_LAW_SUCCESSORS = [
    {
        "historical_title": "The Indian Penal Code, 1860",
        "successor_title": "Bharatiya Nyaya Sanhita, 2023",
        "effective_from": "2024-07-01"
    },
    {
        "historical_title": "The Code of Criminal Procedure, 1973",
        "successor_title": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "effective_from": "2024-07-01"
    },
    {
        "historical_title": "The Indian Evidence Act, 1872",
        "successor_title": "Bharatiya Sakshya Adhiniyam, 2023",
        "effective_from": "2024-07-01"
    }
]


def get_heading(document):
    value = first_value(
        document,
        [
            "heading",
            "section_heading",
            "provision_heading"
        ]
    )

    if value:
        return normalize_text(value)

    return ""


def normalize_act_key(value):
    text = normalize_text(value).lower()

    # Remove metadata that sometimes appears in titles.
    text = re.sub(
        r"\blast updated\b.*$",
        "",
        text
    )

    text = re.sub(
        r"\bthe\b",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def normalize_section_key(value):
    text = normalize_text(value).lower()
    text = text.replace("sec.", "section")
    text = text.replace("sec ", "section ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_provision_suffix(authority_hint):
    """
    Convert a potentially over-specific GPT hint such as:

      Limitation Act, 1963 - Section 61 (...)

    into the Act-level hint:

      Limitation Act, 1963

    We deliberately do not trust model-generated section numbers.
    """
    text = normalize_text(authority_hint)

    match = re.search(
        r"\b(?:sections?|secs?\.?|articles?|orders?|rules?)\b",
        text,
        flags=re.IGNORECASE
    )

    if match:
        text = text[:match.start()]

    text = re.sub(
        r"\s*[-–—:;,]+\s*$",
        "",
        text
    )

    return normalize_text(text)


def extract_requested_sections(authority_hint):
    """
    Parse section numbers only for validation against the local corpus.
    Invalid numbers are ignored rather than used as search truth.
    """
    text = normalize_text(authority_hint)

    match = re.search(
        r"\b(?:sections?|secs?\.?)\b(.+)$",
        text,
        flags=re.IGNORECASE
    )

    if not match:
        return []

    tail = match.group(1)

    values = re.findall(
        r"\b\d+[A-Za-z]{0,3}\b",
        tail
    )

    result = []
    seen = set()

    for value in values:
        key = normalize_section_key(
            f"Section {value}"
        )

        if key not in seen:
            seen.add(key)
            result.append(key)

    return result


def current_law_status_for_title(title):
    key = normalize_act_key(title)

    for item in CURRENT_LAW_SUCCESSORS:
        historical_key = normalize_act_key(
            item["historical_title"]
        )

        if key == historical_key:
            return {
                "is_replaced": True,
                "historical_title": item[
                    "historical_title"
                ],
                "successor_title": item[
                    "successor_title"
                ],
                "effective_from": item[
                    "effective_from"
                ]
            }

    return {
        "is_replaced": False,
        "historical_title": "",
        "successor_title": "",
        "effective_from": ""
    }



LEGAL_RETRIEVAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on",
    "by", "with", "from", "under", "where", "when", "whether", "what",
    "which", "that", "this", "is", "are", "be", "being", "been", "as",
    "act", "law", "legal", "india", "indian", "section", "article",
    "rule", "order", "provision", "provisions", "claim", "claims"
}


def retrieval_terms(value):
    tokens = normalize_match_text(value).split()
    return {
        token
        for token in tokens
        if len(token) >= 3
        and token not in LEGAL_RETRIEVAL_STOPWORDS
    }


def lexical_concept_score(concepts, heading):
    """Score a provision heading against issue-specific concepts."""
    heading_norm = normalize_match_text(heading)
    heading_terms = retrieval_terms(heading_norm)

    if not heading_terms:
        return 0.0

    best = 0.0

    for concept in concepts or []:
        concept_norm = normalize_match_text(concept)
        concept_terms = retrieval_terms(concept_norm)

        if not concept_terms:
            continue

        overlap = len(concept_terms & heading_terms)
        recall = overlap / max(1, len(concept_terms))
        precision = overlap / max(1, len(heading_terms))

        phrase_bonus = 0.0
        if concept_norm and concept_norm in heading_norm:
            phrase_bonus = 0.60
        elif heading_norm and heading_norm in concept_norm:
            phrase_bonus = 0.35

        score = (
            0.65 * recall
            + 0.25 * precision
            + phrase_bonus
        )

        best = max(best, score)

    return min(best, 1.0)


def find_heading_seed_rows(act_entry, concepts):
    """
    Find provision rows whose *headings* lexically match the exact issue.

    Dense retrieval is still used, but this heading layer helps surface
    provisions such as schedule entries about money lent, acknowledgments,
    repayment, damages, documentary proof, etc., instead of generic
    provisions that merely contain words like "enforcement".
    """
    candidates = []

    for heading, rows in act_entry.get("headings", {}).items():
        score = lexical_concept_score(concepts, heading)

        if score < 0.18:
            continue

        for row in rows:
            candidates.append((float(score), int(row), heading))

    candidates.sort(reverse=True)

    result = []
    seen = set()

    for score, row, heading in candidates:
        if row in seen:
            continue
        seen.add(row)
        result.append({
            "row": row,
            "score": score,
            "heading": heading
        })

        if len(result) >= AUTHORITY_HEADING_SEEDS_PER_ACT:
            break

    return result

def build_or_load_authority_catalog(expected_rows):
    """
    Build a compact title/provision index for central Acts.

    Version 2 indexes section/article/rule/order locators plus provision
    headings. The heading index is used for hybrid semantic + lexical
    provision discovery inside a validated Act.
    """
    expected_size = LEGAL_DOCS_FILE.stat().st_size

    if LEGAL_AUTHORITY_CATALOG_FILE.exists():
        try:
            with open(
                LEGAL_AUTHORITY_CATALOG_FILE,
                "r",
                encoding="utf-8"
            ) as file:
                catalog = json.load(file)

            metadata = catalog.get("metadata", {})

            if (
                metadata.get("catalog_version") == AUTHORITY_CATALOG_VERSION
                and metadata.get("row_count") == expected_rows
                and metadata.get("legal_docs_size") == expected_size
            ):
                print(
                    "Authority catalog Acts:",
                    f"{len(catalog.get('acts', [])):,}"
                )
                return catalog

        except Exception:
            pass

    print()
    print("Building statute-title / provision authority catalog...")
    print("This is a one-time scan of the 710K JSONL corpus.")

    acts = {}

    with open(
        LEGAL_DOCS_FILE,
        "rb"
    ) as file:
        row_index = 0

        while True:
            raw = file.readline()

            if not raw:
                break

            if not raw.strip():
                continue

            try:
                document = json.loads(
                    raw.decode("utf-8")
                )
            except Exception:
                row_index += 1
                continue

            source_type = get_source_type(document)

            if source_type == "central_act":
                title = get_source_title(document)
                act_key = normalize_act_key(title)

                if act_key:
                    entry = acts.setdefault(
                        act_key,
                        {
                            "key": act_key,
                            "title": title,
                            "rows": [],
                            "sections": {},
                            "headings": {}
                        }
                    )

                    entry["rows"].append(row_index)

                    locator = get_section(document)
                    locator_key = normalize_section_key(locator)

                    if locator_key:
                        entry["sections"].setdefault(
                            locator_key,
                            []
                        ).append(row_index)

                    heading = get_heading(document)
                    heading_key = normalize_match_text(heading)

                    if heading_key:
                        entry["headings"].setdefault(
                            heading_key,
                            []
                        ).append(row_index)

            row_index += 1

    if row_index != expected_rows:
        raise ValueError(
            "Authority catalog row mismatch.\n"
            f"JSONL rows: {row_index:,}\n"
            f"Embedding rows: {expected_rows:,}"
        )

    catalog = {
        "metadata": {
            "catalog_version": AUTHORITY_CATALOG_VERSION,
            "row_count": expected_rows,
            "legal_docs_size": expected_size
        },
        "acts": sorted(
            acts.values(),
            key=lambda item: item["title"]
        )
    }

    with open(
        LEGAL_AUTHORITY_CATALOG_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            catalog,
            file,
            ensure_ascii=False
        )

    print(
        "Authority catalog saved:",
        LEGAL_AUTHORITY_CATALOG_FILE
    )
    print(
        "Acts indexed:",
        f"{len(catalog['acts']):,}"
    )

    return catalog

def catalog_index(catalog):
    return {
        item["key"]: item
        for item in catalog.get("acts", [])
    }


def match_act_hint(authority_hint, authority_catalog):
    """
    Match an Act-level hint to the corpus catalog.

    We accept an exact normalized title first. Fuzzy matching is only
    used for strong near-matches so that generic words like "recovery"
    do not accidentally select a special banking statute.
    """
    act_name = strip_provision_suffix(
        authority_hint
    )
    hint_key = normalize_act_key(act_name)

    if not hint_key:
        return None

    index = catalog_index(
        authority_catalog
    )

    if hint_key in index:
        return index[hint_key]

    hint_tokens = set(
        hint_key.split()
    )

    best = None
    best_score = 0.0

    for key, item in index.items():
        key_tokens = set(key.split())

        if not key_tokens:
            continue

        overlap = len(
            hint_tokens & key_tokens
        ) / max(
            1,
            len(hint_tokens | key_tokens)
        )

        sequence = SequenceMatcher(
            None,
            hint_key,
            key
        ).ratio()

        # Years are useful disambiguators. If the hint contains a year,
        # require the candidate to contain the same year.
        hint_years = set(
            re.findall(r"\b\d{4}\b", hint_key)
        )
        key_years = set(
            re.findall(r"\b\d{4}\b", key)
        )

        if (
            hint_years
            and
            key_years
            and
            hint_years != key_years
        ):
            continue

        score = (
            0.60 * overlap
            +
            0.40 * sequence
        )

        if score > best_score:
            best_score = score
            best = item

    if best_score >= 0.82:
        return best

    return None


def has_specific_provision_reference(text):
    text = normalize_text(text)

    return bool(
        re.search(
            r"\b(?:section|sec\.?|article|order|rule)\s+"
            r"(?:\d+|[ivxlcdm]+)\b",
            text,
            flags=re.IGNORECASE
        )
    )


def clean_hypotheses(hypotheses):
    """
    Normalize issue plans into ACT-SPECIFIC research targets.

    The key change is that queries/concepts are no longer cross-applied
    to every Act mentioned in the same hypothesis. Each Act gets its own
    concepts, preventing e.g. Contract Act concepts from steering CPC or
    Limitation Act retrieval.
    """
    cleaned = []

    for item in hypotheses or []:
        if not isinstance(item, dict):
            continue

        label = normalize_text(item.get("label", ""))
        why = normalize_text(item.get("why_research_it", ""))
        act_targets = []

        raw_targets = item.get("authority_targets", [])

        if isinstance(raw_targets, list) and raw_targets:
            for target in raw_targets[:4]:
                if not isinstance(target, dict):
                    continue

                act = strip_provision_suffix(
                    normalize_text(target.get("act", ""))
                )

                if not act:
                    continue

                queries = []
                for query in target.get("research_queries", []):
                    query = normalize_text(query)
                    if query and not has_specific_provision_reference(query):
                        queries.append(query)

                concepts = []
                for concept in target.get("provision_concepts", []):
                    concept = normalize_text(concept)
                    if concept:
                        concepts.append(concept)

                if not concepts and label:
                    concepts.append(label)

                act_targets.append({
                    "act": act,
                    "raw_act": normalize_text(target.get("act", "")),
                    "research_queries": list(dict.fromkeys(queries))[:4],
                    "provision_concepts": list(dict.fromkeys(concepts))[:5]
                })

        # Backward-compatible fallback for older planner output.
        if not act_targets:
            broad_hints = []
            raw_hints = []
            for hint in item.get("authority_hints", []):
                raw = normalize_text(hint)
                if not raw:
                    continue
                raw_hints.append(raw)
                broad = strip_provision_suffix(raw)
                if broad:
                    broad_hints.append(broad)

            queries = []
            for query in item.get("research_queries", []):
                query = normalize_text(query)
                if query and not has_specific_provision_reference(query):
                    queries.append(query)

            concepts = [
                normalize_text(value)
                for value in item.get("provision_concepts", [])
                if normalize_text(value)
            ]
            if not concepts and label:
                concepts = [label]

            for idx, act in enumerate(broad_hints[:3]):
                act_targets.append({
                    "act": act,
                    "raw_act": raw_hints[idx] if idx < len(raw_hints) else act,
                    "research_queries": queries[:3],
                    "provision_concepts": concepts[:4]
                })

        cleaned.append({
            "label": label,
            "why_research_it": why,
            "authority_targets": act_targets[:4]
        })

    return cleaned[:4]


def prepare_authority_targets(
    hypotheses,
    authority_catalog
):
    """
    Validate each ACT-SPECIFIC research target against the local corpus.

    Each validated Act keeps only the queries/concepts assigned to that
    Act. It also receives lexical heading seeds for exact provision
    discovery.
    """
    targets_by_key = {}
    validation = []

    for hypothesis_index, hypothesis in enumerate(
        hypotheses,
        start=1
    ):
        for plan in hypothesis.get("authority_targets", []):
            raw_hint = normalize_text(plan.get("raw_act") or plan.get("act"))
            broad_hint = normalize_text(plan.get("act", ""))

            match = match_act_hint(
                broad_hint,
                authority_catalog
            )

            if not match:
                validation.append({
                    "hint": raw_hint,
                    "status": "UNMATCHED",
                    "matched_title": "",
                    "invalid_sections": [],
                    "current_status": None
                })
                continue

            requested_sections = extract_requested_sections(raw_hint)
            exact_rows = []
            invalid_sections = []

            for section_key in requested_sections:
                rows = match.get("sections", {}).get(section_key)
                if rows:
                    exact_rows.extend(rows)
                else:
                    invalid_sections.append(section_key)

            current_status = current_law_status_for_title(match["title"])

            target = targets_by_key.setdefault(
                match["key"],
                {
                    "act_key": match["key"],
                    "title": match["title"],
                    "rows": match.get("rows", []),
                    "exact_rows": [],
                    "heading_seeds": [],
                    "concepts": [],
                    "research_queries": [],
                    "hypotheses": [],
                    "current_status": current_status
                }
            )

            target["exact_rows"].extend(exact_rows)

            for concept in plan.get("provision_concepts", []):
                concept = normalize_text(concept)
                if concept and concept not in target["concepts"]:
                    target["concepts"].append(concept)

            for query in plan.get("research_queries", []):
                query = normalize_text(query)
                if query and query not in target["research_queries"]:
                    target["research_queries"].append(query)

            h_label = f"H{hypothesis_index}"
            if h_label not in target["hypotheses"]:
                target["hypotheses"].append(h_label)

            validation.append({
                "hint": raw_hint,
                "status": "MATCHED",
                "matched_title": match["title"],
                "invalid_sections": invalid_sections,
                "current_status": current_status
            })

            # Current successor target, only if actually in local corpus.
            if current_status.get("is_replaced", False):
                successor_match = match_act_hint(
                    current_status["successor_title"],
                    authority_catalog
                )

                if successor_match:
                    successor = targets_by_key.setdefault(
                        successor_match["key"],
                        {
                            "act_key": successor_match["key"],
                            "title": successor_match["title"],
                            "rows": successor_match.get("rows", []),
                            "exact_rows": [],
                            "heading_seeds": [],
                            "concepts": [],
                            "research_queries": [],
                            "hypotheses": [],
                            "current_status": current_law_status_for_title(
                                successor_match["title"]
                            )
                        }
                    )

                    for concept in plan.get("provision_concepts", []):
                        concept = normalize_text(concept)
                        if concept and concept not in successor["concepts"]:
                            successor["concepts"].append(concept)
                    for query in plan.get("research_queries", []):
                        query = normalize_text(query)
                        if query and query not in successor["research_queries"]:
                            successor["research_queries"].append(query)
                    successor["hypotheses"].append(f"H{hypothesis_index}-CURRENT")

                    validation.append({
                        "hint": current_status["successor_title"],
                        "status": "CURRENT_SUCCESSOR_MATCHED",
                        "matched_title": successor_match["title"],
                        "invalid_sections": [],
                        "current_status": current_law_status_for_title(
                            successor_match["title"]
                        )
                    })
                else:
                    validation.append({
                        "hint": current_status["successor_title"],
                        "status": "CURRENT_SUCCESSOR_NOT_IN_CORPUS",
                        "matched_title": "",
                        "invalid_sections": [],
                        "current_status": None
                    })

    targets = list(targets_by_key.values())[:MAX_AUTHORITY_TARGETS]
    catalog_by_key = catalog_index(authority_catalog)

    for target in targets:
        target["exact_rows"] = sorted(set(target["exact_rows"]))
        act_entry = catalog_by_key.get(target["act_key"], {})
        target["heading_seeds"] = find_heading_seed_rows(
            act_entry,
            target.get("concepts", []) + target.get("research_queries", [])
        )

    return targets, validation


def display_authority_validation(
    targets,
    validation
):
    print()
    print("=" * 80)
    print("AUTHORITY-HINT VALIDATION / ACT TARGETING")
    print("=" * 80)

    if not validation:
        print("\nNo Act-level authority hints to validate.")
        return

    for item in validation:
        print()
        print(
            f"[{item['status']}] "
            f"{item['hint']}"
        )

        if item.get("matched_title"):
            print(
                "Matched corpus Act:",
                item["matched_title"]
            )

        if item.get("invalid_sections"):
            print(
                "Ignored unverified section hint(s):",
                ", ".join(
                    item["invalid_sections"]
                )
            )

        current_status = item.get(
            "current_status"
        )

        if (
            current_status
            and
            current_status.get(
                "is_replaced",
                False
            )
        ):
            print(
                "Historical Act warning:",
                f"replaced by "
                f"{current_status['successor_title']} "
                f"from {current_status['effective_from']}"
            )

    if targets:
        print()
        print("Validated Act targets:")

        for target in targets:
            print(
                "  -",
                target["title"]
            )


# ============================================================
# DETERMINISTIC ISSUE FALLBACKS
# ============================================================

def build_deterministic_fallback_hypotheses(
    original_problem,
    followups,
    normalized_scenario
):
    """
    Deterministic rescue plan for common loan/debt relationships.

    This is NOT a legal conclusion. It only supplies broad Acts and issue
    concepts to the corpus retriever when the LLM planner is truncated or
    unavailable. No section numbers are guessed here.

    Important relationship split:
    - bank + co-borrower / joint borrower
    - bank + guarantor / surety
    - private person-to-person loan

    This prevents a bank co-borrower dispute from being routed through the
    earlier informal-friend-loan research plan.
    """

    facts = build_fallback_scenario(
        original_problem,
        followups
    )

    combined = normalize_match_text(
        f"{facts} {normalized_scenario}"
    )

    loan_signals = (
        "loan",
        "borrow",
        "borrowed",
        "lent",
        "lend",
        "repay",
        "repayment",
        "debt",
        "money owed",
        "emi"
    )

    if not any(signal in combined for signal in loan_signals):
        return []

    bank_signals = (
        "bank",
        "bank loan",
        "personal loan",
        "loan account",
        "demand letter",
        "demand notice",
        "lender bank",
        "financial institution"
    )

    co_borrower_signals = (
        "co borrower",
        "co-borrower",
        "coborrower",
        "joint borrower",
        "joint-borrower",
        "co applicant",
        "co-applicant"
    )

    guarantor_signals = (
        "guarantor",
        "guarantee",
        "surety"
    )

    has_bank_relationship = any(
        signal in combined
        for signal in bank_signals
    )

    is_co_borrower = any(
        signal in combined
        for signal in co_borrower_signals
    )

    is_guarantor = any(
        signal in combined
        for signal in guarantor_signals
    )

    # --------------------------------------------------------
    # BANK + CO-BORROWER
    # --------------------------------------------------------
    if has_bank_relationship and is_co_borrower:

        return [
            {
                "label": "Deterministic bank co-borrower research fallback",
                "why_research_it": (
                    "Fallback authority targeting for a bank loan where the "
                    "user is described as a co-borrower or joint borrower."
                ),
                "authority_targets": [
                    {
                        "act": "Indian Contract Act, 1872",
                        "raw_act": "Indian Contract Act, 1872",
                        "research_queries": [
                            "joint borrower contractual liability to lender",
                            "performance and liability under joint promises"
                        ],
                        "provision_concepts": [
                            "joint promises and joint contractual obligations",
                            "liability arising from a loan agreement",
                            "performance of promises by joint promisors"
                        ]
                    },
                    {
                        "act": "Limitation Act, 1963",
                        "raw_act": "Limitation Act, 1963",
                        "research_queries": [
                            "limitation for lender claim under loan agreement",
                            "acknowledgment or payment affecting debt limitation"
                        ],
                        "provision_concepts": [
                            "limitation for contractual money claim",
                            "acknowledgment of liability",
                            "part payment of debt"
                        ]
                    },
                    {
                        "act": "Code of Civil Procedure, 1908",
                        "raw_act": "Code of Civil Procedure, 1908",
                        "research_queries": [
                            "civil money claim against borrower and co-borrower",
                            "procedure for contractual debt recovery suit"
                        ],
                        "provision_concepts": [
                            "joinder of defendants in money claim",
                            "institution of civil money claim"
                        ]
                    }
                ]
            }
        ]

    # --------------------------------------------------------
    # BANK + GUARANTOR / SURETY
    # --------------------------------------------------------
    if has_bank_relationship and is_guarantor:

        return [
            {
                "label": "Deterministic bank guarantor research fallback",
                "why_research_it": (
                    "Fallback authority targeting for a bank loan where the "
                    "user is described as a guarantor or surety."
                ),
                "authority_targets": [
                    {
                        "act": "Indian Contract Act, 1872",
                        "raw_act": "Indian Contract Act, 1872",
                        "research_queries": [
                            "surety liability to creditor under guarantee",
                            "extent of guarantor liability"
                        ],
                        "provision_concepts": [
                            "contract of guarantee",
                            "surety liability",
                            "extent of liability of guarantor"
                        ]
                    },
                    {
                        "act": "Limitation Act, 1963",
                        "raw_act": "Limitation Act, 1963",
                        "research_queries": [
                            "limitation for creditor claim against guarantor",
                            "acknowledgment affecting limitation of debt"
                        ],
                        "provision_concepts": [
                            "limitation in guarantee claim",
                            "acknowledgment of liability"
                        ]
                    },
                    {
                        "act": "Code of Civil Procedure, 1908",
                        "raw_act": "Code of Civil Procedure, 1908",
                        "research_queries": [
                            "civil recovery claim against borrower and guarantor"
                        ],
                        "provision_concepts": [
                            "civil money claim against multiple defendants"
                        ]
                    }
                ]
            }
        ]

    # --------------------------------------------------------
    # PRIVATE PERSON-TO-PERSON LOAN
    # --------------------------------------------------------
    no_written = any(
        phrase in combined
        for phrase in (
            "no written",
            "no agreement",
            "no letter",
            "never have a letter",
            "without written",
            "oral loan"
        )
    )

    evidence_concepts = (
        [
            "proof of an oral loan transaction",
            "proof of an oral promise to repay money"
        ]
        if no_written
        else [
            "documentary proof of a loan transaction",
            "proof of written repayment terms"
        ]
    )

    return [
        {
            "label": "Deterministic private-loan research fallback",
            "why_research_it": (
                "Fallback authority targeting for a private person-to-person "
                "loan dispute."
            ),
            "authority_targets": [
                {
                    "act": "Indian Contract Act, 1872",
                    "raw_act": "Indian Contract Act, 1872",
                    "research_queries": [
                        "agreement to repay borrowed money",
                        "breach by failure to repay loan"
                    ],
                    "provision_concepts": [
                        "validity of agreement to repay money",
                        "consequences of breach of repayment promise"
                    ]
                },
                {
                    "act": "Limitation Act, 1963",
                    "raw_act": "Limitation Act, 1963",
                    "research_queries": [
                        "limitation for money lent with fixed repayment date",
                        "limitation for recovery of money lent"
                    ],
                    "provision_concepts": [
                        "money lent payable after a fixed period",
                        "acknowledgment or part payment of debt"
                    ]
                },
                {
                    "act": "Code of Civil Procedure, 1908",
                    "raw_act": "Code of Civil Procedure, 1908",
                    "research_queries": [
                        "civil money claim for unpaid loan",
                        "plaint seeking recovery of money"
                    ],
                    "provision_concepts": [
                        "institution of a civil money claim",
                        "particulars required in a money claim"
                    ]
                },
                {
                    "act": "Indian Evidence Act, 1872",
                    "raw_act": "Indian Evidence Act, 1872",
                    "research_queries": [
                        "proof of private loan and repayment promise"
                    ],
                    "provision_concepts": evidence_concepts
                }
            ]
        }
    ]

def merge_missing_fallback_targets(
    hypotheses,
    fallback_hypotheses
):
    """Add only Act targets the planner failed to return."""

    hypotheses = list(hypotheses or [])

    existing_act_keys = set()

    for hypothesis in hypotheses:
        for target in hypothesis.get("authority_targets", []):
            key = normalize_act_key(
                target.get("act", "")
            )
            if key:
                existing_act_keys.add(key)

    missing_targets = []

    for fallback in fallback_hypotheses or []:
        for target in fallback.get("authority_targets", []):
            key = normalize_act_key(
                target.get("act", "")
            )
            if not key or key in existing_act_keys:
                continue

            existing_act_keys.add(key)
            missing_targets.append(target)

    if missing_targets:
        hypotheses.append({
            "label": "Deterministic authority fallback",
            "why_research_it": (
                "Added because the research planner did not return all "
                "core authority classes for this private-loan scenario."
            ),
            "authority_targets": missing_targets
        })

    return hypotheses


# ============================================================
# RESEARCH HYPOTHESIS FALLBACK
# ============================================================

def generate_research_hypotheses(
    original_problem,
    followups,
    normalized_scenario,
    candidate_verification,
    client
):
    """
    Generate compact Act-specific research targets.

    Unlike the older version, this function does not require a perfectly
    closed JSON object. Complete targets are recovered from a truncated
    response and then supplemented by deterministic issue fallbacks.
    """

    conversation = mapper.format_conversation(
        original_problem,
        followups
    )

    verification_text = shorten(
        json.dumps(
            candidate_verification,
            ensure_ascii=False
        ),
        2600
    )

    system_prompt = """
You are a LEGAL RESEARCH TARGET PLANNER for an Indian legal RAG
prototype.

You are NOT giving the final legal answer.

Return only broad ACT-SPECIFIC retrieval targets.

STRICT RULES:
- DO NOT guess section, article, order or rule numbers.
- Preserve the actual party relationship and transaction.
- Do not turn a private loan into a bank/secured-finance dispute.
- Do not force criminal law merely because money is unpaid.
- Do not invent facts.
- Each target is ONE Act only.
- Use at most 5 targets.
- Each target: at most 2 short research queries and 3 short concepts.
- Prefer Contract / Limitation / Procedure / Evidence concepts when the
  facts genuinely require them.

Return JSON only in this compact shape:

{
  "targets": [
    {
      "act": "Act name only",
      "queries": ["short query", "short query"],
      "concepts": ["exact rule to locate", "exact rule to locate"]
    }
  ]
}
"""

    user_prompt = f"""
USER FACTS:
{conversation}

NORMALIZED SCENARIO:
{normalized_scenario}

PREVIOUS CANDIDATE-LAW VERIFICATION:
{verification_text}

Return compact Act-specific research targets only.
"""

    planner_hypotheses = []
    planner_partial = False

    try:
        raw = mapper.groq_text(
            client=client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=HYPOTHESIS_MAX_TOKENS,
            reasoning_effort="low"
        )

        targets, planner_partial = parse_json_array_resilient(
            raw,
            "targets"
        )

        # Backward-compatible salvage if a model returns the older schema.
        if not targets:
            old_hypotheses, old_partial = parse_json_array_resilient(
                raw,
                "hypotheses"
            )

            if old_hypotheses:
                planner_hypotheses = clean_hypotheses(
                    old_hypotheses
                )
                planner_partial = old_partial

        if targets:
            pseudo_hypotheses = []

            for number, target in enumerate(
                targets[:5],
                start=1
            ):
                if not isinstance(target, dict):
                    continue

                act = strip_provision_suffix(
                    normalize_text(
                        target.get("act", "")
                    )
                )

                if not act:
                    continue

                queries = target.get(
                    "queries",
                    target.get("research_queries", [])
                )
                concepts = target.get(
                    "concepts",
                    target.get("provision_concepts", [])
                )

                if not isinstance(queries, list):
                    queries = []
                if not isinstance(concepts, list):
                    concepts = []

                pseudo_hypotheses.append({
                    "label": f"Research target {number}: {act}",
                    "why_research_it": "",
                    "authority_targets": [
                        {
                            "act": act,
                            "raw_act": normalize_text(
                                target.get("act", act)
                            ),
                            "research_queries": [
                                normalize_text(value)
                                for value in queries[:2]
                                if normalize_text(value)
                                and not has_specific_provision_reference(value)
                            ],
                            "provision_concepts": [
                                normalize_text(value)
                                for value in concepts[:3]
                                if normalize_text(value)
                            ]
                        }
                    ]
                })

            planner_hypotheses = clean_hypotheses(
                pseudo_hypotheses
            )

    except Exception as exc:
        print()
        print(
            "Research planner API call failed; "
            "using deterministic authority fallback."
        )
        print(
            "Planner error:",
            exc
        )

    if planner_partial and planner_hypotheses:
        print()
        print(
            "Research planner JSON was incomplete, but "
            f"{sum(len(h.get('authority_targets', [])) for h in planner_hypotheses)} "
            "complete Act target(s) were recovered."
        )

    fallback = build_deterministic_fallback_hypotheses(
        original_problem,
        followups,
        normalized_scenario
    )

    merged = merge_missing_fallback_targets(
        planner_hypotheses,
        fallback
    )

    if fallback:
        planner_acts = {
            normalize_act_key(target.get("act", ""))
            for hypothesis in planner_hypotheses
            for target in hypothesis.get("authority_targets", [])
            if normalize_act_key(target.get("act", ""))
        }

        merged_acts = {
            normalize_act_key(target.get("act", ""))
            for hypothesis in merged
            for target in hypothesis.get("authority_targets", [])
            if normalize_act_key(target.get("act", ""))
        }

        added = len(merged_acts - planner_acts)

        if added:
            print()
            print(
                "Deterministic fallback added",
                added,
                "missing Act target(s)."
            )

    return merged


# ============================================================
# DISPLAY HYPOTHESES
# ============================================================

def display_hypotheses(
    hypotheses
):
    print()
    print("=" * 80)
    print("UNVERIFIED LEGAL RESEARCH HYPOTHESES")
    print("=" * 80)

    if not hypotheses:
        print("\nNo additional hypotheses.")
        return

    for number, item in enumerate(hypotheses, start=1):
        print()
        print(f"[H{number}] {item.get('label', '')}")
        print("Why:", item.get("why_research_it", ""))

        for target in item.get("authority_targets", []):
            print("Act target:", target.get("act", ""))
            concepts = target.get("provision_concepts", [])
            if concepts:
                print("  Provision concepts:")
                for concept in concepts:
                    print("    -", concept)
            queries = target.get("research_queries", [])
            if queries:
                print("  Research queries:")
                for query in queries:
                    print("    -", query)


# ============================================================
# BUILD LEGAL CORPUS QUERIES
# ============================================================

def build_legal_queries(
    original_problem,
    normalized_scenario,
    kept_authorities,
    hypotheses,
    authority_targets
):
    queries = []

    def add_query(text, weight, query_type, act_key=None):
        text = normalize_text(text)
        if not text:
            return

        key = (text.lower(), act_key or "")
        for existing in queries:
            if (existing["text"].lower(), existing.get("act_key", "")) == key:
                if weight > existing["weight"]:
                    existing["weight"] = float(weight)
                    existing["type"] = query_type
                return

        queries.append({
            "text": text,
            "weight": float(weight),
            "type": query_type,
            "act_key": act_key or ""
        })

    for item in kept_authorities:
        add_query(item.get("authority", ""), 2.1, "kept_authority")

    # Use only validated, Act-specific queries. This prevents a hypothesis
    # that names several Acts from cross-applying every concept to every Act.
    for target in authority_targets:
        act_key = target["act_key"]

        for query in target.get("research_queries", []):
            add_query(query, 1.75, "hypothesis_query", act_key=act_key)

        for concept in target.get("concepts", []):
            add_query(concept, 1.90, "provision_concept", act_key=act_key)

        add_query(
            target["title"],
            0.55,
            "validated_authority_title",
            act_key=act_key
        )

        current_status = target.get("current_status", {})
        if current_status.get("is_replaced", False):
            add_query(
                current_status.get("successor_title", ""),
                1.9,
                "current_successor_name"
            )

    # General factual queries remain available, but with lower weight.
    add_query(normalized_scenario, 0.95, "normalized_scenario")
    add_query(original_problem, 0.85, "original_problem")

    return queries[:24]


# ============================================================
# DISPLAY LEGAL QUERIES
# ============================================================

def display_legal_queries(
    queries
):

    print()
    print(
        "=" * 80
    )

    print(
        "LEGAL CORPUS SEARCH QUERIES"
    )

    print(
        "=" * 80
    )

    for number, item in enumerate(
        queries,
        start=1
    ):

        print(
            f"Q{number} "
            f"[{item['type']}] "
            f"weight={item['weight']:.1f}"
        )

        print(
            " ",
            item["text"]
        )


# ============================================================
# BATCH SEARCH 710K LEGAL EMBEDDINGS
# ============================================================

def search_legal_corpus(
    queries,
    embedding_model,
    legal_embeddings,
    offsets,
    authority_targets=None
):
    if not queries:
        return []

    authority_targets = authority_targets or []

    prepared_queries = [
        mapper.QUERY_PREFIX + item["text"]
        for item in queries
    ]

    query_vectors = embedding_model.encode(
        prepared_queries,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype(np.float32)

    print()
    print("Scanning 710K legal vectors...")

    similarity_matrix = legal_embeddings @ query_vectors.T

    fused = defaultdict(float)
    best_similarity = defaultdict(lambda: -1.0)
    query_hits = defaultdict(set)
    authority_seed_bonus = defaultdict(float)
    authority_seed_labels = defaultdict(list)
    heading_seed_score = defaultdict(float)
    heading_seed_label = defaultdict(str)

    # Global dense retrieval.
    for query_index, query in enumerate(queries):
        scores = similarity_matrix[:, query_index]
        top_k = min(LEGAL_TOP_PER_QUERY, len(scores))
        indexes = np.argpartition(scores, -top_k)[-top_k:]
        indexes = indexes[np.argsort(scores[indexes])[::-1]]

        for rank, index in enumerate(indexes, start=1):
            index = int(index)
            similarity = float(scores[index])
            fused[index] += query["weight"] / (LEGAL_RRF_K + rank)
            best_similarity[index] = max(best_similarity[index], similarity)
            query_hits[index].add(query_index)

    # Act-restricted semantic retrieval + provision-heading seeds.
    for target in authority_targets:
        act_rows = np.asarray(target.get("rows", []), dtype=np.int64)

        if act_rows.size == 0:
            continue

        related_query_indexes = [
            index
            for index, query in enumerate(queries)
            if query.get("act_key") == target.get("act_key")
            and query["type"] in {
                "hypothesis_query",
                "provision_concept",
                "validated_authority_title"
            }
        ]

        if related_query_indexes:
            vectors = query_vectors[related_query_indexes]
            restricted_scores = legal_embeddings[act_rows] @ vectors.T
            row_scores = (
                restricted_scores
                if restricted_scores.ndim == 1
                else restricted_scores.max(axis=1)
            )

            top_k = min(AUTHORITY_SEED_TOP_PER_ACT, len(row_scores))
            if top_k > 0:
                local_indexes = np.argpartition(row_scores, -top_k)[-top_k:]
                local_indexes = local_indexes[
                    np.argsort(row_scores[local_indexes])[::-1]
                ]

                for local_rank, local_index in enumerate(local_indexes, start=1):
                    row_index = int(act_rows[local_index])
                    similarity = float(row_scores[local_index])
                    bonus = AUTHORITY_RESTRICTED_BONUS / (
                        1.0 + 0.15 * (local_rank - 1)
                    )
                    authority_seed_bonus[row_index] = max(
                        authority_seed_bonus[row_index], bonus
                    )
                    if target["title"] not in authority_seed_labels[row_index]:
                        authority_seed_labels[row_index].append(target["title"])
                    fused[row_index] += bonus
                    best_similarity[row_index] = max(
                        best_similarity[row_index], similarity
                    )
                    query_hits[row_index].add(f"authority:{target['act_key']}")

        # Heading-based exact provision seeds. These are independent of
        # semantic similarity and reward headings that directly match the
        # issue concepts assigned to this Act.
        for seed in target.get("heading_seeds", []):
            row_index = int(seed["row"])
            score = float(seed["score"])
            bonus = AUTHORITY_HEADING_BONUS * score
            fused[row_index] += bonus
            authority_seed_bonus[row_index] = max(
                authority_seed_bonus[row_index], bonus
            )
            heading_seed_score[row_index] = max(
                heading_seed_score[row_index], score
            )
            heading_seed_label[row_index] = seed.get("heading", "")
            if target["title"] not in authority_seed_labels[row_index]:
                authority_seed_labels[row_index].append(target["title"])
            query_hits[row_index].add(f"heading:{target['act_key']}")

        # Exact provision numbers are accepted only when the local catalog
        # confirms that the provision actually exists in that Act.
        for row_index in target.get("exact_rows", []):
            row_index = int(row_index)
            bonus = AUTHORITY_EXACT_SECTION_BONUS
            fused[row_index] += bonus
            authority_seed_bonus[row_index] = max(
                authority_seed_bonus[row_index], bonus
            )
            if target["title"] not in authority_seed_labels[row_index]:
                authority_seed_labels[row_index].append(target["title"])
            query_hits[row_index].add(f"exact:{target['act_key']}")

    ranked_indexes = sorted(
        fused.keys(),
        key=lambda index: (
            fused[index],
            heading_seed_score[index],
            authority_seed_bonus[index],
            best_similarity[index]
        ),
        reverse=True
    )[:LEGAL_FUSED_POOL]

    documents = read_legal_documents(ranked_indexes, offsets)

    authority_phrases = [
        normalize_match_text(target["title"])
        for target in authority_targets
        if target.get("title")
    ]

    results = []

    for index in ranked_indexes:
        document = documents.get(index)
        if not document:
            continue

        text = get_document_text(document)
        normalized_text = normalize_match_text(text)
        lexical_bonus = 0.0

        for phrase in authority_phrases:
            if phrase and phrase in normalized_text:
                lexical_bonus = max(lexical_bonus, 0.008)

        current_status = current_law_status_for_title(
            get_source_title(document)
        )

        results.append({
            "index": index,
            "document": document,
            "text": text,
            "source_type": get_source_type(document),
            "title": get_source_title(document),
            "section": get_section(document),
            "heading": get_heading(document),
            "document_key": get_document_key(document, index),
            "fused_score": float(fused[index]),
            "best_similarity": float(best_similarity[index]),
            "query_hits": len(query_hits[index]),
            "lexical_bonus": lexical_bonus,
            "authority_seed_bonus": float(authority_seed_bonus[index]),
            "authority_seed_labels": authority_seed_labels[index],
            "heading_seed_score": float(heading_seed_score[index]),
            "heading_seed_label": heading_seed_label[index],
            "current_law_status": current_status,
            "retrieval_score": float(fused[index] + lexical_bonus)
        })

    results.sort(
        key=lambda item: (
            item["retrieval_score"],
            item["heading_seed_score"],
            item["authority_seed_bonus"],
            item["best_similarity"]
        ),
        reverse=True
    )

    return diversify_sources(results)


# ============================================================
# SOURCE DIVERSIFICATION
# ============================================================

def diversify_sources(
    results
):

    selected = []

    seen_documents = set()

    type_counts = defaultdict(
        int
    )

    type_limits = {
        "supreme_court": 6,
        "central_act": 8,
        "constitution": 4,
        "unknown": 5
    }

    for item in results:

        key = item[
            "document_key"
        ]

        if key in seen_documents:
            continue

        source_type = item[
            "source_type"
        ]

        limit = type_limits.get(
            source_type,
            5
        )

        if (
            type_counts[
                source_type
            ]
            >= limit
        ):

            continue

        seen_documents.add(
            key
        )

        type_counts[
            source_type
        ] += 1

        selected.append(
            item
        )

        if (
            len(selected)
            >= LEGAL_RERANK_CANDIDATES
        ):

            break

    return selected


# ============================================================
# DISPLAY RAW LEGAL SOURCES
# ============================================================

def display_legal_candidates(
    results
):

    print()
    print("=" * 80)
    print("710K LEGAL CORPUS CANDIDATES")
    print("=" * 80)

    if not results:
        print("\nNo legal sources retrieved.")
        return

    for number, item in enumerate(
        results,
        start=1
    ):
        seed_marker = (
            " | ACT-TARGETED"
            if item.get(
                "authority_seed_bonus",
                0.0
            ) > 0
            else ""
        )

        print()
        print(
            f"[D{number}] "
            f"type={item['source_type']}"
            f" | similarity={item['best_similarity']:.4f}"
            f" | query_hits={item['query_hits']}"
            f"{seed_marker}"
        )

        print(
            source_label(
                item["document"]
            )
        )

        if item.get(
            "authority_seed_labels"
        ):
            print(
                "Targeted via:",
                "; ".join(
                    item[
                        "authority_seed_labels"
                    ]
                )
            )

        current_status = item.get(
            "current_law_status",
            {}
        )

        if current_status.get(
            "is_replaced",
            False
        ):
            print(
                "CURRENT-LAW WARNING:",
                f"historical source replaced by "
                f"{current_status['successor_title']} "
                f"from {current_status['effective_from']}"
            )

        print(
            shorten(
                item["text"],
                420
            )
        )


# ============================================================
# BUILD LEGAL RERANKER INPUT
# ============================================================

def build_legal_reranker_input(
    candidates
):

    blocks = []

    for number, item in enumerate(
        candidates,
        start=1
    ):
        current_status = item.get(
            "current_law_status",
            {}
        )

        if current_status.get(
            "is_replaced",
            False
        ):
            status_text = (
                "HISTORICAL / REPLACED: "
                f"successor is "
                f"{current_status['successor_title']} "
                f"effective "
                f"{current_status['effective_from']}"
            )
        else:
            status_text = "No deterministic replacement flag."

        targeted_text = (
            ", ".join(
                item.get(
                    "authority_seed_labels",
                    []
                )
            )
            or
            "No"
        )

        blocks.append(
            f"[D{number}]\n"
            f"Source type: {item['source_type']}\n"
            f"Title: {item['title']}\n"
            f"Section/provision: {item['section']}\n"
            f"Heading: {item.get('heading', '')}\n"
            f"Act-targeted retrieval: {targeted_text}\n"
            f"Current-law status: {status_text}\n"
            f"Text:\n"
            f"{shorten(item['text'], LEGAL_PREVIEW_CHARS)}"
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# TOLERANT PASSAGE-ANCHOR RECOVERY
# ============================================================

def _anchor_tokens(value):
    return [
        token
        for token in normalize_match_text(value).split()
        if token
    ]


def _ordered_token_coverage(needle_tokens, haystack_tokens):
    """
    Fraction of needle tokens that can be found in order in the haystack.
    This is deliberately stricter than unordered keyword overlap.
    """
    if not needle_tokens or not haystack_tokens:
        return 0.0

    matched = 0
    cursor = 0

    for token in needle_tokens:
        while cursor < len(haystack_tokens) and haystack_tokens[cursor] != token:
            cursor += 1

        if cursor >= len(haystack_tokens):
            break

        matched += 1
        cursor += 1

    return matched / max(1, len(needle_tokens))


def _candidate_passage_clauses(source_text):
    """
    Return actual contiguous clauses/sentences from the source passage.
    The returned string is always copied from the retrieved passage.
    """
    text = normalize_text(source_text)

    if not text:
        return []

    pieces = re.split(
        r"(?<=[\.\?\!;:])\s+|\s+[–—]\s+",
        text
    )

    clauses = []

    for piece in pieces:
        piece = normalize_text(piece)

        if len(_anchor_tokens(piece)) < 3:
            continue

        clauses.append(piece)

    # Keep a whole-passage candidate for short statutory provisions.
    if len(_anchor_tokens(text)) <= 80:
        clauses.append(text)

    return list(dict.fromkeys(clauses))


def resolve_passage_quote_anchor(
    source_text,
    requested_anchor,
    proposition
):
    """
    Resolve an LLM-proposed anchor to an ACTUAL contiguous clause from the
    retrieved passage.

    Exact normalized containment still wins. If the model paraphrases the
    anchor slightly, use lexical/sequence similarity to recover the closest
    real source clause. If no sufficiently close clause exists, return "".

    This relaxes formatting brittleness without relaxing legal support.
    """
    source_text = normalize_text(source_text)
    requested_anchor = normalize_text(requested_anchor)
    proposition = normalize_text(proposition)

    source_norm = normalize_match_text(source_text)
    anchor_norm = normalize_match_text(requested_anchor)

    if not source_text:
        return ""

    # Existing strict path.
    if anchor_norm and anchor_norm in source_norm:
        return requested_anchor

    anchor_tokens = _anchor_tokens(requested_anchor)
    prop_tokens = _anchor_tokens(proposition)

    query_norm = anchor_norm or normalize_match_text(proposition)
    query_tokens = anchor_tokens or prop_tokens

    if len(query_tokens) < 3:
        return ""

    best_clause = ""
    best_score = 0.0
    best_coverage = 0.0
    best_sequence = 0.0

    for clause in _candidate_passage_clauses(source_text):
        clause_norm = normalize_match_text(clause)
        clause_tokens = _anchor_tokens(clause)

        if not clause_norm or not clause_tokens:
            continue

        sequence = SequenceMatcher(
            None,
            query_norm,
            clause_norm
        ).ratio()

        ordered_coverage = _ordered_token_coverage(
            query_tokens,
            clause_tokens
        )

        query_set = set(query_tokens)
        clause_set = set(clause_tokens)

        token_coverage = (
            len(query_set & clause_set)
            /
            max(1, len(query_set))
        )

        coverage = max(
            ordered_coverage,
            token_coverage
        )

        # Proposition overlap is only a supporting signal. It cannot rescue a
        # clause with almost no overlap with the requested anchor/query.
        prop_overlap = 0.0
        if prop_tokens:
            prop_set = set(prop_tokens)
            prop_overlap = (
                len(prop_set & clause_set)
                /
                max(1, len(prop_set))
            )

        score = (
            0.52 * sequence
            + 0.38 * coverage
            + 0.10 * prop_overlap
        )

        if score > best_score:
            best_score = score
            best_clause = clause
            best_coverage = coverage
            best_sequence = sequence

    if (
        best_clause
        and (
            best_sequence >= ANCHOR_SEQUENCE_THRESHOLD
            or (
                best_coverage >= ANCHOR_TOKEN_COVERAGE_THRESHOLD
                and best_sequence >= 0.45
            )
        )
    ):
        return best_clause

    return ""


# ============================================================
# LEGAL SOURCE VERIFIER / RERANKER
# ============================================================

def _recover_verifier_source_objects(
    raw_text
):
    """
    Secondary verifier-specific recovery.

    `parse_json_array_resilient()` already preserves complete objects from a
    truncated `results` array. This helper adds one more recovery path for the
    specific failure mode where ONE result object contains malformed JSON
    (for example: `"score": 1"`), while later D-items are still valid.

    Each D-item is decoded independently. A malformed item is dropped without
    destroying valid items before or after it.
    """

    text = strip_json_code_fences(
        raw_text
    )

    starts = [
        match.start()
        for match in re.finditer(
            r'\{\s*"source_id"\s*:\s*"D\d+"',
            text,
            flags=re.IGNORECASE
        )
    ]

    if not starts:
        return []

    decoder = json.JSONDecoder()
    recovered = []

    for index, start in enumerate(
        starts
    ):
        end = (
            starts[index + 1]
            if index + 1 < len(starts)
            else len(text)
        )

        chunk = text[
            start:end
        ].lstrip()

        try:
            item, _ = decoder.raw_decode(
                chunk
            )
        except Exception:
            # Fail closed for this individual source only.
            continue

        if isinstance(
            item,
            dict
        ):
            recovered.append(
                item
            )

    return recovered


def _parse_legal_verifier_response(
    raw_text
):
    """
    Parse the legal-source verifier without making the whole verification
    stage depend on perfectly formatted JSON.

    Returns:
        results,
        partial_recovery,
        needs_more_information,
        next_question

    Recovery policy:
    1. Accept fully valid JSON normally.
    2. Otherwise use the shared resilient `results` array recovery.
    3. Also decode each D-item independently so one malformed source object
       does not destroy later valid source objects.
    4. Merge recovered objects by source_id.
    5. Never invent a missing verifier result.
    """

    text = strip_json_code_fences(
        raw_text
    )

    full_data = None

    # --------------------------------------------------------
    # 1. Normal strict JSON parse.
    # --------------------------------------------------------
    try:
        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            dict
        ):
            full_data = parsed

    except Exception:
        pass

    # --------------------------------------------------------
    # 2. Sometimes a short prefix/suffix surrounds valid JSON.
    # --------------------------------------------------------
    if full_data is None:

        first = text.find(
            "{"
        )

        last = text.rfind(
            "}"
        )

        if (
            first >= 0
            and last > first
        ):
            try:
                parsed = json.loads(
                    text[
                        first:last + 1
                    ]
                )

                if isinstance(
                    parsed,
                    dict
                ):
                    full_data = parsed

            except Exception:
                pass

    # --------------------------------------------------------
    # 3. Recover `results` even if outer JSON is incomplete.
    # --------------------------------------------------------
    array_results, array_partial = (
        parse_json_array_resilient(
            raw_text,
            "results"
        )
    )

    source_results = (
        _recover_verifier_source_objects(
            raw_text
        )
    )

    merged_results = []
    seen_source_ids = set()

    # Prefer the shared array parser first, then add independently
    # recovered D-items that were not already preserved.
    for item in (
        list(array_results)
        + list(source_results)
    ):

        if not isinstance(
            item,
            dict
        ):
            continue

        source_id = normalize_text(
            item.get(
                "source_id",
                ""
            )
        ).upper()

        if (
            not source_id
            or source_id in seen_source_ids
        ):
            continue

        seen_source_ids.add(
            source_id
        )

        merged_results.append(
            item
        )

    # If the full JSON parsed cleanly, its top-level fields are trusted.
    # Otherwise recover only simple scalar fields when they are explicitly
    # present. Missing fields remain conservative defaults.
    needs_more_information = False
    next_question = ""

    if full_data is not None:

        needs_more_information = bool(
            full_data.get(
                "needs_more_information",
                False
            )
        )

        next_question = normalize_text(
            full_data.get(
                "next_question",
                ""
            )
        )

        full_results = full_data.get(
            "results",
            []
        )

        if isinstance(
            full_results,
            list
        ):
            # A valid full JSON response is authoritative. Still normalize it
            # through the same merge logic so duplicate source IDs cannot
            # create unstable results.
            merged_results = []
            seen_source_ids = set()

            for item in full_results:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                source_id = normalize_text(
                    item.get(
                        "source_id",
                        ""
                    )
                ).upper()

                if (
                    not source_id
                    or source_id in seen_source_ids
                ):
                    continue

                seen_source_ids.add(
                    source_id
                )

                merged_results.append(
                    item
                )

            array_partial = False

    else:

        needs_match = re.search(
            r'"needs_more_information"\s*:\s*(true|false)',
            text,
            flags=re.IGNORECASE
        )

        if needs_match:
            needs_more_information = (
                needs_match.group(1).lower()
                == "true"
            )

        question_match = re.search(
            r'"next_question"\s*:\s*"((?:\\.|[^"\\])*)"',
            text,
            flags=re.IGNORECASE
        )

        if question_match:
            try:
                next_question = normalize_text(
                    json.loads(
                        '"'
                        + question_match.group(1)
                        + '"'
                    )
                )
            except Exception:
                next_question = ""

    partial_recovery = (
        full_data is None
        or array_partial
    )

    return (
        merged_results,
        partial_recovery,
        needs_more_information,
        next_question
    )


def _coerce_verifier_score(
    value
):
    """
    Convert ordinary JSON numeric/string scores safely to 0..3.
    Does not infer a score when no clear digit is present.
    """

    try:
        return max(
            0,
            min(
                int(value),
                3
            )
        )

    except Exception:
        match = re.fullmatch(
            r"\s*([0-3])(?:\s*/\s*3)?\s*",
            str(
                value or ""
            )
        )

        if not match:
            return 0

        return int(
            match.group(1)
        )


def rerank_legal_sources(
    original_problem,
    followups,
    normalized_scenario,
    hypotheses,
    candidates,
    client
):
    conversation = mapper.format_conversation(
        original_problem,
        followups
    )

    hypothesis_text = shorten(
        json.dumps(
            hypotheses,
            ensure_ascii=False
        ),
        3000
    )

    source_text = (
        build_legal_reranker_input(
            candidates
        )
    )

    system_prompt = """
You are the PASSAGE-SCOPE AUTHORITY VERIFIER for an Indian legal RAG
system.

You receive the user's facts, unverified research hypotheses and exact
retrieved passages. The passages are the ONLY legal evidence you may use.

Your task is to separate:
- directly applicable authority,
- materially useful authority with a scope condition,
- background material,
- irrelevant material.

SCORE EACH SOURCE:
3 = DIRECT: the excerpt itself states a rule/holding that directly addresses
    the user's legal relationship or transaction.

2 = MATERIAL / LIMITED: the excerpt states a legal rule that materially helps
    answer the user's issue, but application depends on a contract term, party
    status, timing fact, or other condition not fully established yet.

1 = BACKGROUND: related vocabulary, a general discussion, analogy, or a
    materially different transaction/regime that should not be used as a
    final legal rule.

0 = IRRELEVANT.

IMPORTANT BALANCE:
- Do NOT downgrade a genuinely relevant statutory rule to 1 merely because
  every application fact is not yet known. Use score 2 and state the missing
  condition in scope_limit.
- Do NOT require the passage to use exactly the user's wording.
- Do NOT use model memory to repair a missing rule.
- Do NOT convert a case-specific fact pattern into a general rule unless the
  excerpt actually states that rule.

For every score 2 or 3, provide ONE narrow safe_proposition:
- proposition: the narrow legal proposition the excerpt actually supports;
- quote_anchor: copy the shortest exact phrase or sentence you can from the
  supplied excerpt that supports it;
- scope_limit: identify any factual/doctrinal condition that prevents
  overgeneralization. Use "" when none is apparent.

For score 0 or 1, safe_propositions MUST be [].

STRICT LEGAL ENTAILMENT RULES:
- A case mentioning a loan does NOT establish a general right to a remedy.
- A case about mortgage/security does NOT automatically govern an unsecured
  personal-loan dispute.
- A guarantor/surety rule does NOT automatically apply to a co-borrower.
- A co-borrower rule does NOT automatically apply to a guarantor.
- A passage about payment restarting limitation does NOT establish written
  acknowledgment unless the excerpt itself says so.
- A general limitation provision does NOT establish a specific limitation
  period or accrual date unless the excerpt states it.
- A statute heading alone is not enough; use the provided text.
- Special bank/SARFAESI/recovery regimes require matching facts.
- A historical/replaced statute cannot independently support a present-day
  rule as current law.
- Do not repair missing propositions using pretrained knowledge.

STRICT OUTPUT RULES:
- Return JSON only. No markdown and no prose outside JSON.
- Use valid JSON syntax: no trailing commas and no stray quotation marks.
- Evaluate every supplied D-source exactly once.
- Keep each "reason" to at most 20 words.
- Return at most ONE safe_proposition per source.
- Keep each "scope_limit" to at most 20 words.
- Do not repeat the full source passage.

Return exactly this shape:
{
  "results": [
    {
      "source_id": "D1",
      "score": 2,
      "reason": "...",
      "safe_propositions": [
        {
          "proposition": "...",
          "quote_anchor": "exact phrase or sentence from excerpt",
          "scope_limit": "..."
        }
      ]
    }
  ],
  "needs_more_information": false,
  "next_question": ""
}
"""

    user_prompt = f"""
USER FACTS:
{conversation}

NORMALIZED SCENARIO:
{normalized_scenario}

UNVERIFIED RESEARCH HYPOTHESES:
{hypothesis_text}

LEGAL CORPUS CANDIDATES:
{source_text}

Evaluate passage-level relevance and applicability. Preserve materially useful
authorities as score 2 with a clear scope_limit instead of rejecting them only
because one application fact is unresolved.

Return compact, valid JSON only.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    # Use raw text + resilient parsing here instead of the older
    # all-or-nothing mapper.groq_json() path.
    # --------------------------------------------------------
    raw = mapper.groq_text(
        client=client,
        messages=messages,
        max_completion_tokens=
            LEGAL_RERANK_MAX_TOKENS,
        reasoning_effort="low"
    )

    (
        raw_results,
        partial_recovery,
        needs_more_information,
        next_question
    ) = _parse_legal_verifier_response(
        raw
    )

    # If absolutely nothing usable was recoverable, make one fresh attempt.
    # A failure of the retry does not erase anything because there was
    # nothing usable to erase in the first response.
    if not raw_results:

        print()
        print(
            "Legal source verifier returned no usable "
            "source objects."
        )
        print(
            "Retrying verifier once with strict JSON formatting..."
        )

        retry_messages = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n\nFINAL FORMAT REMINDER:\n"
                    + "Return one compact valid JSON object only. "
                    + "Do not use markdown. Do not omit the results array."
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        try:
            retry_raw = mapper.groq_text(
                client=client,
                messages=retry_messages,
                max_completion_tokens=
                    LEGAL_RERANK_MAX_TOKENS,
                reasoning_effort="low"
            )

            (
                retry_results,
                retry_partial,
                retry_needs_more_information,
                retry_next_question
            ) = _parse_legal_verifier_response(
                retry_raw
            )

            if retry_results:
                raw_results = (
                    retry_results
                )
                partial_recovery = (
                    retry_partial
                )
                needs_more_information = (
                    retry_needs_more_information
                )
                next_question = (
                    retry_next_question
                )

        except Exception as retry_exc:

            print(
                "Legal source verifier retry failed:",
                retry_exc
            )

    candidate_map = {
        f"D{number}": candidate
        for number, candidate in enumerate(
            candidates,
            start=1
        )
    }

    # Keep only known D-IDs and make duplicate source results deterministic.
    filtered_results = []
    seen_result_ids = set()

    for result in raw_results:

        if not isinstance(
            result,
            dict
        ):
            continue

        source_id = normalize_text(
            result.get(
                "source_id",
                ""
            )
        ).upper()

        if (
            source_id not in candidate_map
            or source_id in seen_result_ids
        ):
            continue

        seen_result_ids.add(
            source_id
        )

        filtered_results.append(
            result
        )

    raw_results = filtered_results

    if partial_recovery:

        missing_ids = [
            source_id
            for source_id in candidate_map
            if source_id
            not in seen_result_ids
        ]

        print()
        print(
            "Legal source verifier JSON was incomplete or malformed."
        )
        print(
            "Recovered complete verifier result(s):",
            len(
                raw_results
            ),
            "of",
            len(
                candidate_map
            )
        )

        if missing_ids:
            print(
                "Unrecoverable source result(s) will fail closed:",
                ", ".join(
                    missing_ids
                )
            )

    score_map = {}

    for result in raw_results:

        source_id = normalize_text(
            result.get(
                "source_id",
                ""
            )
        ).upper()

        candidate = candidate_map.get(
            source_id
        )

        if not candidate:
            continue

        score = _coerce_verifier_score(
            result.get(
                "score",
                0
            )
        )

        valid_props = []

        propositions = result.get(
            "safe_propositions",
            []
        )

        if not isinstance(
            propositions,
            list
        ):
            propositions = []

        # The verifier is explicitly asked for at most one proposition.
        # Keeping only the first valid anchored proposition further limits
        # accidental over-generalization.
        for proposition in propositions:

            if not isinstance(
                proposition,
                dict
            ):
                continue

            prop_text = normalize_text(
                proposition.get(
                    "proposition",
                    ""
                )
            )

            requested_anchor = normalize_text(
                proposition.get(
                    "quote_anchor",
                    ""
                )
            )

            scope = normalize_text(
                proposition.get(
                    "scope_limit",
                    ""
                )
            )

            if not prop_text:
                continue

            resolved_anchor = (
                resolve_passage_quote_anchor(
                    source_text=candidate.get(
                        "text",
                        ""
                    ),
                    requested_anchor=
                        requested_anchor,
                    proposition=
                        prop_text
                )
            )

            if not resolved_anchor:
                continue

            valid_props.append({
                "proposition":
                    prop_text,
                "quote_anchor":
                    resolved_anchor,
                "scope_limit":
                    scope
            })

            break

        # A high verifier score is not enough by itself. Score 2/3 survives
        # only when at least one proposition can be anchored back to the
        # retrieved passage.
        if (
            score >= 2
            and not valid_props
        ):
            score = 1

        score_map[
            source_id
        ] = {
            "score":
                score,
            "reason":
                normalize_text(
                    result.get(
                        "reason",
                        ""
                    )
                ),
            "safe_propositions":
                valid_props
        }

    reranked = []

    for number, candidate in enumerate(
        candidates,
        start=1
    ):

        source_id = (
            f"D{number}"
        )

        evaluation = score_map.get(
            source_id,
            {
                "score": 0,
                "reason": (
                    "No complete passage-scoped verifier "
                    "result was recoverable for this source."
                ),
                "safe_propositions": []
            }
        )

        item = dict(
            candidate
        )

        item[
            "source_id"
        ] = source_id

        item[
            "legal_score"
        ] = evaluation[
            "score"
        ]

        item[
            "legal_reason"
        ] = evaluation[
            "reason"
        ]

        item[
            "safe_propositions"
        ] = evaluation[
            "safe_propositions"
        ]

        current_status = item.get(
            "current_law_status",
            {}
        )

        if (
            current_status.get(
                "is_replaced",
                False
            )
            and item[
                "legal_score"
            ] > 1
        ):

            item[
                "legal_score"
            ] = 1

            item[
                "safe_propositions"
            ] = []

            item[
                "legal_reason"
            ] = (
                item[
                    "legal_reason"
                ]
                + " Historical-source clamp: "
                + "this Act has been replaced for "
                + "present-day law by "
                + current_status.get(
                    "successor_title",
                    "a successor statute"
                )
                + ", so it cannot independently "
                + "support a current-law claim."
            )

        reranked.append(
            item
        )

    reranked.sort(
        key=lambda item: (
            item[
                "legal_score"
            ],
            len(
                item.get(
                    "safe_propositions",
                    []
                )
            ),
            item[
                "retrieval_score"
            ]
        ),
        reverse=True
    )

    selected = [
        item
        for item in reranked
        if (
            item[
                "legal_score"
            ] >= 2
            and item.get(
                "safe_propositions"
            )
        )
    ][
        :FINAL_SOURCE_K
    ]

    next_question = normalize_text(
        next_question
    )

    normalized_data = {
        "results":
            raw_results,
        "needs_more_information": (
            bool(
                needs_more_information
            )
            and bool(
                next_question
            )
        ),
        "next_question":
            next_question
    }

    return (
        reranked,
        selected,
        normalized_data
    )

def display_source_verification(
    reranked,
    selected
):
    print()
    print("=" * 80)
    print("LEGAL SOURCE VERIFICATION")
    print("=" * 80)

    selected_ids = {item["source_id"] for item in selected}

    for item in reranked:
        decision = "KEEP" if item["source_id"] in selected_ids else "REJECT"
        print()
        print(
            f"[{item['source_id']}] {decision}"
            f" | legal_score={item['legal_score']}/3"
        )
        print(source_label(item["document"]))
        print("Reason:", item["legal_reason"])

        for prop in item.get("safe_propositions", []):
            print("  Safe proposition:", prop["proposition"])
            print("  Quote anchor:", repr(prop["quote_anchor"]))
            if prop.get("scope_limit"):
                print("  Scope limit:", prop["scope_limit"])


# ============================================================
# PRIMARY STATUTORY AUTHORITY COMPLETION
# ============================================================

def collect_primary_completion_rules(
    verified_sources,
    research_sources=None
):
    """
    Build narrow rule seeds from already passage-anchored case-law propositions.

    Verified case-law sources are preferred. If none survived the final
    threshold, passage-anchored score-1 research sources may still seed a
    statutory search, but those case sources themselves are NOT promoted.
    """
    rules = []
    seen = set()

    pools = [
        (verified_sources or [], True),
        (research_sources or [], False)
    ]

    for pool, is_verified in pools:
        for item in pool:

            if item.get("source_type") != "supreme_court":
                continue

            if (
                not is_verified
                and int(item.get("legal_score", 0)) < 1
            ):
                continue

            for proposition in item.get(
                "safe_propositions",
                []
            ):
                rule_text = normalize_text(
                    proposition.get(
                        "proposition",
                        ""
                    )
                )

                if not rule_text:
                    continue

                key = normalize_match_text(rule_text)

                if not key or key in seen:
                    continue

                seen.add(key)

                rules.append({
                    "rule_id": f"R{len(rules) + 1}",
                    "proposition": rule_text,
                    "scope_limit": normalize_text(
                        proposition.get(
                            "scope_limit",
                            ""
                        )
                    ),
                    "case_title": item.get(
                        "title",
                        ""
                    ),
                    "case_source_id": item.get(
                        "source_id",
                        ""
                    ),
                    "case_was_verified": is_verified
                })

                if len(rules) >= 6:
                    return rules

        # If verified rules exist, do not dilute them with research-only rules.
        if rules and is_verified:
            break

    return rules


def _all_central_act_rows(authority_catalog):
    rows = []

    for act in authority_catalog.get(
        "acts",
        []
    ):
        rows.extend(
            int(row)
            for row in act.get(
                "rows",
                []
            )
        )

    if not rows:
        return np.asarray(
            [],
            dtype=np.int64
        )

    return np.asarray(
        sorted(set(rows)),
        dtype=np.int64
    )


def search_primary_statutory_candidates(
    rules,
    embedding_model,
    legal_embeddings,
    offsets,
    authority_catalog
):
    """
    Search only Central Act rows for the statutory text behind a case-law rule.

    No Act or section number is guessed. The query is the already anchored
    legal proposition from the retrieved case passage.
    """
    if not rules:
        return []

    central_rows = _all_central_act_rows(
        authority_catalog
    )

    if central_rows.size == 0:
        return []

    prepared_queries = [
        mapper.QUERY_PREFIX
        + rule["proposition"]
        for rule in rules
    ]

    query_vectors = embedding_model.encode(
        prepared_queries,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype(np.float32)

    scores = (
        legal_embeddings[central_rows]
        @ query_vectors.T
    )

    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    row_best = defaultdict(
        lambda: -1.0
    )
    row_rule_hits = defaultdict(set)

    for rule_index, rule in enumerate(rules):
        column = scores[:, rule_index]

        top_k = min(
            PRIMARY_AUTHORITY_TOP_PER_RULE,
            len(column)
        )

        if top_k <= 0:
            continue

        local_indexes = np.argpartition(
            column,
            -top_k
        )[-top_k:]

        local_indexes = local_indexes[
            np.argsort(
                column[local_indexes]
            )[::-1]
        ]

        for local_index in local_indexes:
            row_index = int(
                central_rows[
                    int(local_index)
                ]
            )

            similarity = float(
                column[
                    int(local_index)
                ]
            )

            row_best[row_index] = max(
                row_best[row_index],
                similarity
            )

            row_rule_hits[row_index].add(
                rule["rule_id"]
            )

    ranked_rows = sorted(
        row_best.keys(),
        key=lambda row: (
            len(row_rule_hits[row]),
            row_best[row]
        ),
        reverse=True
    )[:PRIMARY_AUTHORITY_MAX_CANDIDATES * 2]

    documents = read_legal_documents(
        ranked_rows,
        offsets
    )

    results = []
    seen_documents = set()

    rule_map = {
        rule["rule_id"]: rule
        for rule in rules
    }

    for row_index in ranked_rows:
        document = documents.get(
            row_index
        )

        if not document:
            continue

        if get_source_type(document) != "central_act":
            continue

        document_key = get_document_key(
            document,
            row_index
        )

        if document_key in seen_documents:
            continue

        seen_documents.add(
            document_key
        )

        text = get_document_text(
            document
        )

        heading = get_heading(
            document
        )

        title = get_source_title(
            document
        )

        hit_ids = sorted(
            row_rule_hits[row_index]
        )

        lexical_bonus = 0.0

        for rule_id in hit_ids:
            rule = rule_map.get(
                rule_id,
                {}
            )

            rule_terms = retrieval_terms(
                rule.get(
                    "proposition",
                    ""
                )
            )

            candidate_terms = retrieval_terms(
                f"{title} {heading} {shorten(text, 900)}"
            )

            if rule_terms:
                overlap = (
                    len(
                        rule_terms
                        & candidate_terms
                    )
                    /
                    max(
                        1,
                        len(rule_terms)
                    )
                )

                lexical_bonus = max(
                    lexical_bonus,
                    0.08 * overlap
                )

        current_status = (
            current_law_status_for_title(
                title
            )
        )

        results.append({
            "index": row_index,
            "document": document,
            "text": text,
            "source_type": "central_act",
            "title": title,
            "section": get_section(document),
            "heading": heading,
            "document_key": document_key,
            "best_similarity": float(
                row_best[row_index]
            ),
            "retrieval_score": float(
                row_best[row_index]
                + lexical_bonus
            ),
            "current_law_status": current_status,
            "primary_completion_for": hit_ids,
            "primary_completion_candidate": True
        })

        if (
            len(results)
            >= PRIMARY_AUTHORITY_MAX_CANDIDATES
        ):
            break

    results.sort(
        key=lambda item: (
            len(
                item.get(
                    "primary_completion_for",
                    []
                )
            ),
            item["retrieval_score"]
        ),
        reverse=True
    )

    return results


def build_primary_completion_input(
    rules,
    candidates
):
    rule_blocks = []

    for rule in rules:
        rule_blocks.append(
            f"[{rule['rule_id']}]\n"
            f"Case-derived proposition: {rule['proposition']}\n"
            f"Scope limit: {rule.get('scope_limit', '')}\n"
            f"Case: {rule.get('case_title', '')}"
        )

    candidate_blocks = []

    for number, item in enumerate(
        candidates,
        start=1
    ):
        current_status = item.get(
            "current_law_status",
            {}
        )

        if current_status.get(
            "is_replaced",
            False
        ):
            status_text = (
                "HISTORICAL / REPLACED: successor is "
                f"{current_status.get('successor_title', '')}"
            )
        else:
            status_text = (
                "No deterministic replacement flag."
            )

        candidate_blocks.append(
            f"[P{number}]\n"
            f"Act: {item.get('title', '')}\n"
            f"Section/provision: {item.get('section', '')}\n"
            f"Heading: {item.get('heading', '')}\n"
            f"Candidate for rules: "
            f"{', '.join(item.get('primary_completion_for', []))}\n"
            f"Current-law status: {status_text}\n"
            f"TEXT:\n"
            f"{shorten(item.get('text', ''), FINAL_SOURCE_CHARS)}"
        )

    return (
        "\n\n".join(rule_blocks),
        "\n\n".join(candidate_blocks)
    )


def verify_primary_statutory_candidates(
    rules,
    candidates,
    client
):
    """
    Verify whether a retrieved Central Act passage is actually the primary
    statutory basis for an already anchored case-law rule.
    """
    if not rules or not candidates:
        return []

    rule_text, candidate_text = (
        build_primary_completion_input(
            rules,
            candidates
        )
    )

    system_prompt = """
You are the PRIMARY STATUTORY AUTHORITY VERIFIER for an Indian legal RAG
system.

You receive:
1. narrow legal propositions already anchored in retrieved Supreme Court
   passages; and
2. Central Act passages retrieved specifically to find the statutory basis
   for those propositions.

Use ONLY the supplied Act passages.

Score each Act passage:
3 = PRIMARY MATCH: the statute directly states substantially the same legal
    rule as one or more case-derived propositions.
2 = MATERIAL STATUTORY SUPPORT: the statute supplies an important part of the
    rule but has a narrower condition or scope that must be preserved.
1 = RELATED: same topic but not the statutory basis for the proposition.
0 = IRRELEVANT.

For score 2 or 3:
- matched_rule_ids must list only supplied R1/R2/... IDs.
- provide at least one safe_proposition.
- quote_anchor should copy the shortest exact phrase/sentence possible from
  the Act passage.
- scope_limit must preserve any statutory condition.
- do not infer application to the user's facts here; identify the statute's
  rule only.
- do not invent a section number. Use the supplied metadata only.
- a historical/replaced statute cannot independently support a present-day
  current-law proposition.

Return JSON only:
{
  "results": [
    {
      "source_id": "P1",
      "score": 3,
      "matched_rule_ids": ["R1"],
      "reason": "...",
      "safe_propositions": [
        {
          "proposition": "...",
          "quote_anchor": "exact words from Act passage",
          "scope_limit": "..."
        }
      ]
    }
  ]
}
"""

    user_prompt = f"""
CASE-DERIVED RULES:
{rule_text}

CENTRAL ACT CANDIDATES:
{candidate_text}

Identify only genuine primary statutory matches.
"""

    data = mapper.groq_json(
        client=client,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_completion_tokens=1400,
        required_keys=["results"],
        label="Primary statutory authority verifier"
    )

    raw_results = data.get(
        "results",
        []
    )

    if not isinstance(
        raw_results,
        list
    ):
        raw_results = []

    candidate_map = {
        f"P{number}": item
        for number, item in enumerate(
            candidates,
            start=1
        )
    }

    valid_rule_ids = {
        rule["rule_id"]
        for rule in rules
    }

    accepted = []

    for result in raw_results:

        if not isinstance(
            result,
            dict
        ):
            continue

        source_id = normalize_text(
            result.get(
                "source_id",
                ""
            )
        ).upper()

        candidate = candidate_map.get(
            source_id
        )

        if not candidate:
            continue

        try:
            score = max(
                0,
                min(
                    int(
                        result.get(
                            "score",
                            0
                        )
                    ),
                    3
                )
            )
        except Exception:
            score = 0

        matched_rule_ids = [
            normalize_text(
                value
            ).upper()
            for value in result.get(
                "matched_rule_ids",
                []
            )
            if normalize_text(
                value
            ).upper()
            in valid_rule_ids
        ]

        valid_props = []

        for proposition in result.get(
            "safe_propositions",
            []
        ) or []:

            if not isinstance(
                proposition,
                dict
            ):
                continue

            prop_text = normalize_text(
                proposition.get(
                    "proposition",
                    ""
                )
            )

            requested_anchor = normalize_text(
                proposition.get(
                    "quote_anchor",
                    ""
                )
            )

            scope = normalize_text(
                proposition.get(
                    "scope_limit",
                    ""
                )
            )

            if not prop_text:
                continue

            resolved_anchor = (
                resolve_passage_quote_anchor(
                    source_text=candidate.get(
                        "text",
                        ""
                    ),
                    requested_anchor=requested_anchor,
                    proposition=prop_text
                )
            )

            if not resolved_anchor:
                continue

            valid_props.append({
                "proposition": prop_text,
                "quote_anchor": resolved_anchor,
                "scope_limit": scope
            })

        if score >= 2 and (
            not valid_props
            or not matched_rule_ids
        ):
            score = 1

        current_status = candidate.get(
            "current_law_status",
            {}
        )

        if (
            current_status.get(
                "is_replaced",
                False
            )
            and score > 1
        ):
            score = 1
            valid_props = []

        if score < 2:
            continue

        item = dict(
            candidate
        )

        item["source_id"] = source_id
        item["legal_score"] = score
        item["legal_reason"] = normalize_text(
            result.get(
                "reason",
                ""
            )
        )
        item["safe_propositions"] = valid_props
        item["primary_completion_for"] = (
            matched_rule_ids
        )
        item["primary_authority_completed"] = True

        accepted.append(
            item
        )

    accepted.sort(
        key=lambda item: (
            item["legal_score"],
            len(
                item.get(
                    "primary_completion_for",
                    []
                )
            ),
            item["retrieval_score"]
        ),
        reverse=True
    )

    return accepted[
        :PRIMARY_AUTHORITY_MAX_SOURCES
    ]


def merge_primary_authorities(
    selected_sources,
    primary_sources
):
    """
    Put primary statutory authority before supporting case law while
    deduplicating exact document/provision rows.
    """
    merged = []
    seen = set()

    for item in (
        list(primary_sources or [])
        + list(selected_sources or [])
    ):
        key = item.get(
            "document_key"
        ) or (
            f"{item.get('title', '')}|"
            f"{item.get('section', '')}"
        )

        if key in seen:
            continue

        seen.add(key)
        merged.append(item)

        if len(merged) >= FINAL_SOURCE_K:
            break

    return merged


def display_primary_authority_completion(
    rules,
    primary_sources
):
    print()
    print("=" * 80)
    print("PRIMARY AUTHORITY COMPLETION")
    print("=" * 80)

    if not rules:
        print(
            "\nNo passage-anchored case-law rule required "
            "statutory completion."
        )
        return

    print()
    print("Case-law rules searched:")

    for rule in rules:
        print(
            f"  [{rule['rule_id']}] "
            f"{rule['proposition']}"
        )

    if not primary_sources:
        print()
        print(
            "No Central Act passage was verified as the "
            "primary statutory basis for these rules."
        )
        return

    print()
    print("Verified primary statutory authorities:")

    for item in primary_sources:
        print()
        print(
            f"- {item.get('title', '')}"
            f" | {item.get('section', '')}"
        )
        print(
            "  Completes:",
            ", ".join(
                item.get(
                    "primary_completion_for",
                    []
                )
            )
        )

        for proposition in item.get(
            "safe_propositions",
            []
        ):
            print(
                "  Statutory proposition:",
                proposition.get(
                    "proposition",
                    ""
                )
            )
            print(
                "  Quote anchor:",
                repr(
                    proposition.get(
                        "quote_anchor",
                        ""
                    )
                )
            )


def complete_primary_statutory_authorities(
    selected_sources,
    reranked_sources,
    embedding_model,
    legal_embeddings,
    legal_offsets,
    authority_catalog,
    client
):
    """
    Return (merged_selected_sources, rules, accepted_primary_sources).
    """
    research_seed_sources = [
        item
        for item in reranked_sources or []
        if item.get(
            "source_type"
        ) == "supreme_court"
        and int(
            item.get(
                "legal_score",
                0
            )
        ) >= 1
        and item.get(
            "safe_propositions"
        )
    ][:4]

    rules = collect_primary_completion_rules(
        verified_sources=selected_sources,
        research_sources=research_seed_sources
    )

    if not rules:
        return (
            list(selected_sources or []),
            [],
            []
        )

    candidates = (
        search_primary_statutory_candidates(
            rules=rules,
            embedding_model=embedding_model,
            legal_embeddings=legal_embeddings,
            offsets=legal_offsets,
            authority_catalog=authority_catalog
        )
    )

    primary_sources = (
        verify_primary_statutory_candidates(
            rules=rules,
            candidates=candidates,
            client=client
        )
    )

    merged = merge_primary_authorities(
        selected_sources=selected_sources,
        primary_sources=primary_sources
    )

    return (
        merged,
        rules,
        primary_sources
    )


# ============================================================
# PRELIMINARY RETRIEVED-ANSWER FALLBACK
# ============================================================

def select_preliminary_sources(
    reranked_sources
):
    """
    Preserve useful retrieval when the strict verifier finds no source strong
    enough for the fully verified answer path.

    A preliminary source is NOT promoted to verified status. It remains a
    research-level passage and is used only to produce a cautious answer.
    """

    preliminary = []

    for item in reranked_sources or []:
        try:
            score = int(item.get("legal_score", 0))
        except Exception:
            score = 0

        if score < PRELIMINARY_MIN_LEGAL_SCORE:
            continue

        if not normalize_text(item.get("text", "")):
            continue

        preliminary.append(item)

        if len(preliminary) >= PRELIMINARY_SOURCE_K:
            break

    return preliminary


def build_preliminary_sources(
    selected
):
    """
    Build source context for a user-facing preliminary retrieved answer.

    Unlike build_final_sources(), this function deliberately does NOT call the
    passages verified propositions. The passages failed the strict final
    verification threshold, so the model only receives the raw excerpt plus
    the verifier's caution and current-law status.
    """

    blocks = []

    for number, item in enumerate(selected, start=1):
        item["final_source_id"] = f"S{number}"

        current_status = item.get(
            "current_law_status",
            {}
        )

        if current_status.get("is_replaced", False):
            status_text = (
                "HISTORICAL / REPLACED SOURCE. "
                f"Successor: {current_status.get('successor_title', '')}; "
                f"effective from {current_status.get('effective_from', '')}. "
                "Do not present this historical Act as current law."
            )
        else:
            status_text = "No deterministic replacement flag."

        blocks.append(
            f"[S{number}]\n"
            f"Source type: {item.get('source_type', '')}\n"
            f"Title: {item.get('title', '')}\n"
            f"Section/provision: {item.get('section', '')}\n"
            f"Heading: {item.get('heading', '')}\n"
            f"Verification status: PRELIMINARY / NOT FULLY VERIFIED\n"
            f"Verifier caution: {item.get('legal_reason', '')}\n"
            f"Current-law status: {status_text}\n"
            f"RETRIEVED PASSAGE:\n"
            f"{shorten(item.get('text', ''), FINAL_SOURCE_CHARS)}"
        )

    return "\n\n".join(blocks)


def generate_preliminary_answer(
    original_problem,
    followups,
    selected,
    client
):
    """
    Generate a useful answer from relevant retrieved passages when no passage
    survives the strict verified-answer threshold.

    This keeps the prototype informative without silently reclassifying weak
    evidence as verified law.
    """

    conversation = mapper.format_conversation(
        original_problem,
        followups
    )

    sources = build_preliminary_sources(
        selected
    )

    system_prompt = """
You are the PRELIMINARY RETRIEVED-ANSWER WRITER for an Indian legal RAG
prototype.

The supplied passages were relevant enough to retain for research, but they
did NOT pass the system's strict final-verification threshold.

Your job is to give the user the most useful answer that can be responsibly
stated from these retrieved passages alone.

STRICT RULES:
1. Use ONLY the supplied retrieved passages. Do not use pretrained legal
   memory to fill gaps.
2. User facts may be restated without citations.
3. Every legal or procedural proposition must cite one or more supplied
   source IDs in exact ASCII form [S1], [S2], etc.
4. Do not convert a related passage into a definitive rule for the user's
   facts unless the passage itself expressly supports that application.
5. Do not invent a limitation period, accrual date, remedy, notice
   requirement, jurisdiction rule, interest entitlement, criminal theory,
   evidentiary consequence, or filing procedure.
6. If a source is historical/replaced, do not state it as current law. You
   may say that the retrieved historical material discusses the point, while
   noting that current-law verification is still required.
7. Prefer cautious language such as:
   - "The retrieved material indicates..."
   - "This passage may be relevant because..."
   - "The current retrieval has not yet verified..."
8. Do not mention similarity scores, reranking, vector search, agents,
   internal pipeline stages, JSON, verifier scores, or debugging details.
9. Do not say merely "insufficient retrieval" if the supplied passages allow
   you to explain something useful. Answer the supported part and identify
   only the unresolved part briefly.
10. Keep the response concise and understandable to a non-lawyer.

Return only the user-facing answer.
"""

    user_prompt = f"""
USER CONVERSATION:

{conversation}

RELEVANT RETRIEVED PASSAGES (PRELIMINARY, NOT FULLY VERIFIED):

{sources}

Write a cautious but useful retrieved answer. Cite every legal proposition.
"""

    answer = mapper.groq_text(
        client=client,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_completion_tokens=PRELIMINARY_ANSWER_MAX_TOKENS,
        reasoning_effort="low"
    )

    return normalize_source_citations(answer)


def build_deterministic_preliminary_answer(
    selected
):
    """
    Last-resort user-facing fallback if preliminary prose generation fails.

    It reports only what was actually retrieved and why it may matter, without
    turning the verifier's research note into a definitive legal conclusion.
    """

    if not selected:
        return (
            "The current search did not return a sufficiently relevant legal "
            "passage to provide even a preliminary retrieved answer."
        )

    lines = [
        "Based on the legal materials retrieved for your situation, the "
        "following sources appear potentially relevant, but they have not "
        "yet passed the prototype's strict final-verification threshold:"
    ]

    for number, item in enumerate(selected, start=1):
        reason = normalize_text(item.get("legal_reason", ""))
        if reason:
            lines.append(
                f"- {reason} [S{number}]"
            )
        else:
            lines.append(
                f"- {source_label(item['document'])} may be relevant to the "
                f"issue raised. [S{number}]"
            )

    lines.append("")
    lines.append(
        "These points should be treated as preliminary research rather than a "
        "fully verified legal conclusion."
    )

    return "\n".join(lines)


# ============================================================
# FINAL SOURCE CONTEXT
# ============================================================

def build_final_sources(
    selected
):
    blocks = []

    for number, item in enumerate(selected, start=1):
        item["final_source_id"] = f"S{number}"

        propositions = item.get("safe_propositions", [])
        prop_lines = []

        for idx, proposition in enumerate(propositions, start=1):
            prop_lines.append(
                f"Verified proposition {idx}: {proposition.get('proposition', '')}\n"
                f"Exact quote anchor: {proposition.get('quote_anchor', '')}\n"
                f"Scope limit: {proposition.get('scope_limit', '')}"
            )

        blocks.append(
            f"[S{number}]\n"
            f"Source type: {item['source_type']}\n"
            f"Title: {item['title']}\n"
            f"Section/provision: {item['section']}\n"
            f"Heading: {item.get('heading', '')}\n"
            f"PASSAGE-SCOPED VERIFIED PROPOSITIONS:\n"
            f"{chr(10).join(prop_lines)}\n"
            f"SOURCE TEXT:\n"
            f"{shorten(item['text'], FINAL_SOURCE_CHARS)}"
        )

    return "\n\n".join(blocks)


# ============================================================
# DRAFT GROUNDED ANSWER
# ============================================================

def normalize_source_citations(text):

    text = str(
        text or ""
    )

    # GPT sometimes changes [S1] into decorative Unicode
    # brackets. Normalize them before auditing.
    text = re.sub(
        r"[【\[]\s*S(\d+)\s*[】\]]",
        lambda match: f"[S{match.group(1)}]",
        text
    )

    return text.strip()


def generate_draft_answer(
    original_problem,
    followups,
    selected,
    client
):

    conversation = (
        mapper.format_conversation(
            original_problem,
            followups
        )
    )

    sources = build_final_sources(
        selected
    )

    system_prompt = """
You are the DRAFT answer writer for an Indian legal RAG
prototype.

Use ONLY the PASSAGE-SCOPED VERIFIED PROPOSITIONS supplied for legal claims.
The source text is provided for context, but you may NOT broaden a verified proposition beyond its stated scope limit.

The user's own facts may be restated without citations.

Every statement about law, legal remedy, limitation, procedure,
evidence, jurisdiction, interest, damages, court powers, notices,
deadlines, criminal liability or civil liability must be supported
by one or more supplied sources and must cite them using the exact
ASCII form [S1], [S2], etc.

Do NOT use pretrained legal memory to fill gaps.

Do NOT state that:
- a notice is required;
- a particular court has jurisdiction;
- a particular limitation period applies;
- interest can be claimed;
- a document proves a contract;
- a particular procedure is available;
unless the supplied source passage actually establishes that point.

If an important issue is not contained in a PASSAGE-SCOPED VERIFIED PROPOSITION,
say that the current retrieval has not yet verified it. Do not infer a broader rule from a case's surrounding facts or from a statute mentioned in passing.

When both a Central Act source and a Supreme Court source support the same
legal point, present the Central Act / section first as the primary authority,
then use the judgment as supporting interpretation or application. Explicitly
name the Act and section/provision from the supplied source metadata.

Keep the draft concise. Avoid tables.

Return prose only.
"""

    user_prompt = f"""
USER CONVERSATION:

{conversation}

VERIFIED RETRIEVED SOURCES:

{sources}

Write a concise source-grounded draft answer.
"""

    draft = mapper.groq_text(

        client=client,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        max_completion_tokens=(
            DRAFT_ANSWER_MAX_TOKENS
        ),

        reasoning_effort="low"
    )

    return normalize_source_citations(
        draft
    )


# ============================================================
# CLAIM ↔ CITATION AUDITOR
# ============================================================

def audit_draft_claims(
    original_problem,
    followups,
    selected,
    draft,
    client
):
    """
    Audit legal claims against the retrieved passages.

    The response parser is intentionally partial-output tolerant: complete
    claim objects are preserved even if the model is truncated while
    writing a later claim. Any unreturned claim is excluded from the
    allow-list, which is fail-closed behavior.
    """

    conversation = mapper.format_conversation(
        original_problem,
        followups
    )

    sources = build_final_sources(
        selected
    )

    system_prompt = """
You are the CLAIM-CITATION AUDITOR for an Indian legal RAG system.

Use ONLY the supplied source passages and their passage-scoped verified
propositions. Do not use your own legal knowledge.

Audit every material LEGAL proposition in the answer. Do not audit simple
restatements of user facts.

VERDICTS:
SUPPORTED = source directly supports the proposition.
PARTIAL = source supports only a narrower proposition.
UNSUPPORTED = source does not establish it, citation is absent, or the
citation is wrong.

SAFE CLAIM RULES:
- SUPPORTED: safe_claim is a concise directly supported proposition.
- PARTIAL: safe_claim narrows to only what is directly supported.
- UNSUPPORTED: safe_claim is empty.
- Never broaden beyond a verified proposition or its scope limit.
- Never invent source IDs.
- If the supported claim cites a Central Act source and the draft correctly
  names its Act title and section/provision from the supplied metadata, keep
  that Act + provision identity in safe_claim so the final answer explicitly
  shows the relevant law.

Return compact JSON only:
{
  "claims": [
    {
      "claim_id": "C1",
      "claim": "short claim text",
      "verdict": "SUPPORTED",
      "supported_by": ["S1"],
      "safe_claim": "short directly supported proposition",
      "reason": "short reason"
    }
  ]
}
"""

    user_prompt = f"""
USER FACTS:
{conversation}

ANSWER TO AUDIT:
{draft}

AVAILABLE SOURCES:
{sources}

Audit every material legal proposition. Return compact JSON only.
"""

    try:
        raw = mapper.groq_text(
            client=client,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_completion_tokens=CLAIM_AUDIT_MAX_TOKENS,
            reasoning_effort="low"
        )

        claims, audit_partial = parse_json_array_resilient(
            raw,
            "claims"
        )

    except Exception as exc:
        print()
        print(
            "WARNING: Claim-citation audit API call failed."
        )
        print(
            "Audit error:",
            exc
        )
        return {
            "claims": [],
            "audit_failed": True,
            "audit_partial": False
        }

    if not claims:
        print()
        print(
            "WARNING: No complete claim-audit objects could be parsed."
        )
        return {
            "claims": [],
            "audit_failed": True,
            "audit_partial": audit_partial
        }

    if audit_partial:
        print()
        print(
            "Claim-audit JSON was incomplete; preserving "
            f"{len(claims)} complete claim decision(s)."
        )
        print(
            "Any unreturned claim is treated as unsupported by omission."
        )

    valid_source_ids = {
        f"S{number}"
        for number in range(
            1,
            len(selected) + 1
        )
    }

    cleaned = []

    for number, item in enumerate(
        claims,
        start=1
    ):
        if not isinstance(item, dict):
            continue

        verdict = normalize_text(
            item.get(
                "verdict",
                "UNSUPPORTED"
            )
        ).upper()

        if verdict not in {
            "SUPPORTED",
            "PARTIAL",
            "UNSUPPORTED"
        }:
            verdict = "UNSUPPORTED"

        supported_by = item.get(
            "supported_by",
            []
        )

        if not isinstance(supported_by, list):
            supported_by = []

        supported_by = [
            normalize_text(source).upper()
            for source in supported_by
            if normalize_text(source).upper()
            in valid_source_ids
        ]

        safe_claim = normalize_text(
            item.get(
                "safe_claim",
                ""
            )
        )

        # Fail closed. Nothing reaches the final writer unless both the
        # wording and at least one valid source survive the audit.
        if verdict == "UNSUPPORTED":
            safe_claim = ""
            supported_by = []
        elif not supported_by or not safe_claim:
            verdict = "UNSUPPORTED"
            safe_claim = ""
            supported_by = []

        cleaned.append({
            "claim_id": normalize_text(
                item.get(
                    "claim_id",
                    f"C{number}"
                )
            ),
            "claim": normalize_text(
                item.get(
                    "claim",
                    ""
                )
            ),
            "cited_sources": list(supported_by),
            "verdict": verdict,
            "supported_by": supported_by,
            "safe_claim": safe_claim,
            "reason": normalize_text(
                item.get(
                    "reason",
                    ""
                )
            )
        })

    return {
        "claims": cleaned,
        "audit_failed": False,
        "audit_partial": audit_partial
    }


def display_claim_audit(
    audit
):

    print()
    print(
        "=" * 80
    )

    print(
        "CLAIM-CITATION AUDIT"
    )

    print(
        "=" * 80
    )

    if audit.get(
        "audit_failed",
        False
    ):

        print()
        print(
            "Structured audit failed. "
            "The final rewriter will use "
            "fail-closed source checking."
        )

        return

    if audit.get(
        "audit_partial",
        False
    ):
        print()
        print(
            "Partial structured audit recovered. "
            "Only completed claim decisions can enter the allow-list."
        )

    claims = audit.get(
        "claims",
        []
    )

    if not claims:

        print()
        print(
            "No material legal claims were "
            "returned by the auditor."
        )

        return

    counts = defaultdict(
        int
    )

    for item in claims:

        counts[
            item["verdict"]
        ] += 1

    print()
    print(
        "Supported:",
        counts["SUPPORTED"]
    )

    print(
        "Partial:",
        counts["PARTIAL"]
    )

    print(
        "Unsupported:",
        counts["UNSUPPORTED"]
    )

    for item in claims:

        if item[
            "verdict"
        ] == "SUPPORTED":

            continue

        print()
        print(
            f"[{item['claim_id']}] "
            f"{item['verdict']}"
        )

        print(
            "Claim:",
            item["claim"]
        )

        print(
            "Reason:",
            item["reason"]
        )

        if (
            item["verdict"] == "PARTIAL"
            and
            item.get("safe_claim")
        ):
            print(
                "Safe narrowed claim:",
                item["safe_claim"]
            )


# ============================================================
# FINAL AUDITED ANSWER
# ============================================================

def get_allowed_audit_claims(
    audit
):
    """
    Convert the first claim audit into a strict allow-list.
    """

    if audit.get(
        "audit_failed",
        False
    ):
        return []

    allowed = []

    for item in audit.get(
        "claims",
        []
    ):
        if item.get(
            "verdict"
        ) not in {
            "SUPPORTED",
            "PARTIAL"
        }:
            continue

        safe_claim = normalize_text(
            item.get(
                "safe_claim",
                ""
            )
        )
        supported_by = item.get(
            "supported_by",
            []
        )

        if (
            not safe_claim
            or
            not supported_by
        ):
            continue

        allowed.append({
            "claim_id": item.get(
                "claim_id",
                ""
            ),
            "safe_claim": safe_claim,
            "supported_by": supported_by,
            "original_verdict": item.get(
                "verdict",
                ""
            )
        })

    return allowed


def rewrite_audited_answer(
    original_problem,
    followups,
    selected,
    draft,
    audit,
    client
):
    """
    Final prose generation from the allow-list only.

    The original draft and its rejected claims are intentionally NOT
    passed to the final writer.
    """

    conversation = (
        mapper.format_conversation(
            original_problem,
            followups
        )
    )

    allowed_claims = (
        get_allowed_audit_claims(
            audit
        )
    )

    if not allowed_claims:
        return (
            "Based on the current retrieval, I do not yet have "
            "enough source-supported legal propositions to give a "
            "grounded legal answer. The system should retrieve "
            "stronger authorities before making a legal claim.\n\n"
            "Note: the research corpus may contain historical law; "
            "current legal status should be verified before real-world "
            "reliance."
        )

    allowed_text = json.dumps(
        allowed_claims,
        ensure_ascii=False,
        indent=2
    )

    system_prompt = """
You are the FINAL ALLOW-LIST ANSWER WRITER for an Indian legal RAG
prototype.

You receive:
- the user's facts;
- an explicit ALLOW-LIST of legal claims that passed claim-citation
  auditing.

CRITICAL RULE:
You may state ONLY the legal propositions contained in the
ALLOW-LIST.

You are NOT allowed to derive additional legal conclusions from the
user's facts or from your own legal knowledge.

STRICT RULES:

1. User facts may be restated without citations.
2. Every legal proposition must be a faithful paraphrase of one
   safe_claim from the ALLOW-LIST.
3. Use ONLY the source IDs listed in that safe_claim's supported_by
   field.
4. Do not add a new remedy, limitation trigger, limitation period,
   notice requirement, jurisdiction rule, interest rule, evidentiary
   consequence, procedure, criminal theory or civil procedure unless
   it appears explicitly in a safe_claim.
5. Do not infer that a right to sue accrued on a particular date
   unless that exact proposition is present in the ALLOW-LIST.
6. Do not say that the user can file a particular suit or go to a
   particular court unless that proposition is present in the
   ALLOW-LIST.
7. Do not mention unsupported draft claims; they are intentionally
   absent from your input.
8. Use exact ASCII citations such as [S1], [S2].
9. When an allowed claim contains a Central Act title and section/provision,
   state that law explicitly; do not replace it with a vague phrase such as
   "the law says".
10. Prefer primary statutory authority before supporting case law when both
    appear in the allowed claims.
11. Keep the answer concise and conversational.
12. Add one sentence saying that issues not covered by the allowed
    claims remain unverified by the current retrieval.
13. End with one short note that the research corpus may contain
    historical law and current legal status should be verified before
    real-world reliance.

Return only the final user-facing prose.
"""

    user_prompt = f"""
USER FACTS:

{conversation}

ALLOWED LEGAL CLAIMS:

{allowed_text}

Write the final answer using only the allowed legal claims.
"""

    final_answer = mapper.groq_text(
        client=client,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_completion_tokens=(
            FINAL_ANSWER_MAX_TOKENS
        ),
        reasoning_effort="low"
    )

    return normalize_source_citations(
        final_answer
    )


def build_fail_closed_answer(
    audit
):
    """
    Deterministic fallback used if the model's final rewrite fails the
    second grounding audit.
    """

    allowed_claims = (
        get_allowed_audit_claims(
            audit
        )
    )

    if not allowed_claims:
        return (
            "The current retrieved sources do not yet support enough "
            "legal propositions to give a grounded answer. Stronger "
            "legal retrieval is needed before the system should make "
            "a legal recommendation.\n\n"
            "Note: the research corpus may contain historical law; "
            "current legal status should be verified before real-world "
            "reliance."
        )

    lines = [
        "Based strictly on the legal propositions verified by the "
        "retrieved sources:"
    ]

    for item in allowed_claims:
        citations = "".join(
            f"[{source}]"
            for source in item[
                "supported_by"
            ]
        )
        lines.append(
            f"- {item['safe_claim']} {citations}"
        )

    lines.append("")
    lines.append(
        "The current retrieved sources do not yet verify any additional "
        "legal proposition beyond the points above, so the system is "
        "not adding further procedural, jurisdictional, limitation, "
        "interest, notice, or remedy claims from general model knowledge."
    )
    lines.append("")
    lines.append(
        "Note: the research corpus may contain historical law; current "
        "legal status should be verified before real-world reliance."
    )

    return "\n".join(
        lines
    )


def final_answer_passes_audit(
    audit
):
    if audit.get(
        "audit_failed",
        False
    ):
        return False

    # A partial SECOND audit cannot prove that every final legal claim was
    # checked, so final enforcement must fail closed.
    if audit.get(
        "audit_partial",
        False
    ):
        return False

    claims = audit.get(
        "claims",
        []
    )

    if not claims:
        return False

    for item in claims:
        if item.get(
            "verdict"
        ) != "SUPPORTED":
            return False

    return True


def validate_final_citation_ids(
    answer,
    selected
):
    """
    Deterministic final guardrail: the model may only cite S1..Sn.
    """

    valid_ids = {
        f"S{number}"
        for number in range(
            1,
            len(selected) + 1
        )
    }

    cited_ids = {
        f"S{match}"
        for match in re.findall(
            r"\[S(\d+)\]",
            answer
        )
    }

    invalid_ids = sorted(
        cited_ids
        - valid_ids
    )

    return invalid_ids


# ============================================================
# DISPLAY FINAL SOURCE LIST
# ============================================================

def display_final_source_list(
    selected,
    answer=None
):
    print()
    print(
        "-" * 80
    )
    print(
        "SOURCES CITED IN FINAL ANSWER"
    )
    print(
        "-" * 80
    )

    used_ids = None

    if answer is not None:
        used_ids = {
            f"S{match}"
            for match in re.findall(
                r"\[S(\d+)\]",
                answer
            )
        }

    shown = 0

    for number, item in enumerate(
        selected,
        start=1
    ):
        source_id = f"S{number}"

        if (
            used_ids is not None
            and
            source_id not in used_ids
        ):
            continue

        shown += 1
        print(
            f"[{source_id}] "
            f"{source_label(item['document'])}"
        )

    if shown == 0:
        print(
            "No source citation appears in the final answer."
        )


# ============================================================
# RUN ONE END-TO-END CASE
# ============================================================
# RUN ONE END-TO-END CASE
# ============================================================

def run_case(
    embedding_model,
    ilsic_scenario_map,
    ilsic_chunks,
    ilsic_embeddings,
    legal_embeddings,
    legal_offsets,
    authority_catalog,
    client
):

    print()
    print(
        "=" * 80
    )

    original_problem = input(
        "\nDescribe your legal problem "
        "(or 'exit'): "
    ).strip()

    if original_problem.lower() in {
        "exit",
        "quit"
    }:

        return False

    if not original_problem:

        return True

    followups = []

    asked_question_keys = set()

    total_followups = 0

    research_iteration = 0

    while (
        research_iteration
        <
        MAX_RESEARCH_ITERATIONS
    ):

        research_iteration += 1

        print()
        print(
            "=" * 80
        )

        print(
            f"RESEARCH ITERATION "
            f"{research_iteration}"
        )

        print(
            "=" * 80
        )

        # ====================================================
        # 1. CONVERSATIONAL INTAKE
        # ====================================================

        while True:

            print()
            print(
                "[1/8] Understanding the facts..."
            )

            analysis = mapper.analyze_intake(

                original_problem=
                    original_problem,

                followups=
                    followups,

                client=
                    client,

                questions_already_asked=
                    total_followups
            )

            status = normalize_text(
                analysis.get(
                    "status",
                    ""
                )
            ).upper()

            if (
                status == "ASK"
                and
                total_followups
                < MAX_INTAKE_FOLLOWUPS
            ):

                question = normalize_text(
                    analysis.get(
                        "next_question",
                        ""
                    )
                )

                asked = (
                    mapper.ask_followup(
                        question=
                            question,

                        followups=
                            followups,

                        asked_question_keys=
                            asked_question_keys
                    )
                )

                if asked:

                    total_followups += 1

                    continue

            break

        normalized_scenario = (
            normalize_text(
                analysis.get(
                    "normalized_scenario",
                    ""
                )
            )
        )

        retrieval_queries = (
            analysis.get(
                "retrieval_queries",
                []
            )
        )

        if not normalized_scenario:

            normalized_scenario = (
                build_fallback_scenario(
                    original_problem,
                    followups
                )
            )

            retrieval_queries = []

        print()
        print(
            "=" * 80
        )

        print(
            "SCENARIO UNDERSTOOD"
        )

        print(
            "=" * 80
        )

        print(
            normalized_scenario
        )

        # ====================================================
        # 2. ILSIC SCENARIO RETRIEVAL
        # ====================================================

        search_queries = (
            mapper.build_search_queries(
                original_problem=
                    original_problem,

                normalized_scenario=
                    normalized_scenario,

                retrieval_queries=
                    retrieval_queries
            )
        )

        print()
        print(
            "[2/8] Searching analogous "
            "ILSIC scenarios..."
        )

        candidates = (
            mapper.retrieve_candidate_scenarios(
                search_queries=
                    search_queries,

                embedding_model=
                    embedding_model,

                scenario_map=
                    ilsic_scenario_map,

                chunks=
                    ilsic_chunks,

                embeddings=
                    ilsic_embeddings
            )
        )

        mapper.display_raw_retrieval(
            candidates
        )

        # ====================================================
        # 3. FACTUAL ANALOGY RERANK
        # ====================================================

        print()
        print(
            "[3/8] Checking factual analogy..."
        )

        (
            reranked_scenarios,
            accepted_scenarios
        ) = (
            mapper.rerank_analogous_scenarios(
                original_problem=
                    original_problem,

                followups=
                    followups,

                normalized_scenario=
                    normalized_scenario,

                candidates=
                    candidates,

                client=
                    client
            )
        )

        mapper.display_reranking(
            reranked_scenarios,
            accepted_scenarios
        )

        # ====================================================
        # 4. DATASET CANDIDATE LAW VERIFICATION
        # ====================================================

        print()
        print(
            "[4/8] Evaluating scenario-mapped "
            "candidate laws..."
        )

        if accepted_scenarios:

            candidate_laws = (
                mapper.aggregate_candidate_laws(
                    accepted_scenarios
                )
            )

            mapper.display_candidate_laws(
                candidate_laws
            )

            # ------------------------------------------------
            # FAIL-SAFE:
            #
            # ILSIC is only a candidate-law generator.
            # A formatting/API failure here must never prevent
            # the pipeline from reaching hypothesis generation
            # and the actual 710K legal corpus.
            # ------------------------------------------------

            try:

                candidate_verification = (
                    mapper.verify_candidate_laws(
                        original_problem=
                            original_problem,

                        followups=
                            followups,

                        normalized_scenario=
                            normalized_scenario,

                        accepted=
                            accepted_scenarios,

                        candidate_laws=
                            candidate_laws,

                        client=
                            client
                    )
                )

            except Exception as exc:

                print()
                print(
                    "WARNING:"
                )

                print(
                    "Candidate-law verification failed."
                )

                print(
                    "The system will continue using the "
                    "legal-research hypothesis fallback."
                )

                print(
                    "Verifier error:",
                    exc
                )

                candidate_verification = {
                    "candidate_laws": [],
                    "needs_more_information": False,
                    "next_question": ""
                }

            mapper.display_verification(
                candidate_verification
            )

        else:

            candidate_laws = []

            candidate_verification = {
                "candidate_laws": [],
                "needs_more_information": False,
                "next_question": ""
            }

            print()
            print(
                "No sufficiently analogous "
                "ILSIC scenarios survived."
            )

        # ----------------------------------------------------
        # Dataset verifier can ask another question.
        # ----------------------------------------------------

        if (
            candidate_verification.get(
                "needs_more_information",
                False
            )
            and
            total_followups
            < MAX_TOTAL_FOLLOWUPS
        ):

            question = normalize_text(
                candidate_verification.get(
                    "next_question",
                    ""
                )
            )

            if question:

                asked = (
                    mapper.ask_followup(
                        question=
                            question,

                        followups=
                            followups,

                        asked_question_keys=
                            asked_question_keys
                    )
                )

                if asked:

                    total_followups += 1

                    print()
                    print(
                        "Re-running research with "
                        "the new fact..."
                    )

                    continue

        kept_authorities = (
            mapper.get_kept_authorities(
                candidate_verification
            )
        )

        # ====================================================
        # 5. LEGAL-HYPOTHESIS FALLBACK
        # ====================================================

        print()
        print(
            "[5/8] Generating controlled "
            "legal-research hypotheses..."
        )

        try:

            hypotheses = (
                generate_research_hypotheses(
                    original_problem=
                        original_problem,

                    followups=
                        followups,

                    normalized_scenario=
                        normalized_scenario,

                    candidate_verification=
                        candidate_verification,

                    client=
                        client
                )
            )

        except Exception as exc:

            # The corpus search can still proceed using
            # surviving verified candidate authorities plus
            # the normalized/original scenario queries.

            print()
            print(
                "WARNING:"
            )

            print(
                "Research-hypothesis generation failed."
            )

            print(
                "Continuing with direct legal-corpus "
                "retrieval."
            )

            print(
                "Hypothesis error:",
                exc
            )

            hypotheses = (
                build_deterministic_fallback_hypotheses(
                    original_problem=original_problem,
                    followups=followups,
                    normalized_scenario=normalized_scenario
                )
            )

            if hypotheses:
                print(
                    "Deterministic authority fallback activated."
                )

        display_hypotheses(
            hypotheses
        )

        authority_targets, authority_validation = (
            prepare_authority_targets(
                hypotheses=
                    hypotheses,
                authority_catalog=
                    authority_catalog
            )
        )

        display_authority_validation(
            authority_targets,
            authority_validation
        )

        # ====================================================
        # 6. SEARCH 710K LEGAL CORPUS
        # ====================================================

        legal_queries = (
            build_legal_queries(
                original_problem=
                    original_problem,

                normalized_scenario=
                    normalized_scenario,

                kept_authorities=
                    kept_authorities,

                hypotheses=
                    hypotheses,

                authority_targets=
                    authority_targets
            )
        )

        display_legal_queries(
            legal_queries
        )

        print()
        print(
            "[6/8] Searching 710K "
            "legal corpus..."
        )

        legal_candidates = (
            search_legal_corpus(
                queries=
                    legal_queries,

                embedding_model=
                    embedding_model,

                legal_embeddings=
                    legal_embeddings,

                offsets=
                    legal_offsets,

                authority_targets=
                    authority_targets
            )
        )

        display_legal_candidates(
            legal_candidates
        )

        if not legal_candidates:

            print()
            print(
                "No legal-corpus sources were "
                "retrieved."
            )

            return True

        # ====================================================
        # 7. LEGAL SOURCE VERIFICATION
        # ====================================================

        print()
        print(
            "[7/8] Verifying retrieved "
            "legal authorities..."
        )

        try:

            (
                reranked_sources,
                selected_sources,
                source_verification
            ) = rerank_legal_sources(

                original_problem=
                    original_problem,

                followups=
                    followups,

                normalized_scenario=
                    normalized_scenario,

                hypotheses=
                    hypotheses,

                candidates=
                    legal_candidates,

                client=
                    client
            )

        except Exception as exc:

            # Do not convert unverified retrieved passages
            # into a final answer merely because the verifier
            # failed. Fail closed, but do not crash.

            print()
            print(
                "WARNING:"
            )

            print(
                "Legal-source verification failed."
            )

            print(
                "No retrieved source will be treated as "
                "verified in this iteration."
            )

            print(
                "Authority-verifier error:",
                exc
            )

            reranked_sources = []
            selected_sources = []

            source_verification = {
                "results": [],
                "needs_more_information": False,
                "next_question": ""
            }

        display_source_verification(
            reranked_sources,
            selected_sources
        )

        # ----------------------------------------------------
        # Legal research itself may discover that one fact
        # is crucial, e.g. when repayment became due.
        # ----------------------------------------------------

        if (
            source_verification.get(
                "needs_more_information",
                False
            )
            and
            total_followups
            < MAX_TOTAL_FOLLOWUPS
        ):

            question = normalize_text(
                source_verification.get(
                    "next_question",
                    ""
                )
            )

            if question:

                asked = (
                    mapper.ask_followup(
                        question=
                            question,

                        followups=
                            followups,

                        asked_question_keys=
                            asked_question_keys
                    )
                )

                if asked:

                    total_followups += 1

                    print()
                    print(
                        "That fact may change the "
                        "legal analysis."
                    )

                    print(
                        "Re-running the research..."
                    )

                    continue

        # ====================================================
        # 7.5 PRIMARY STATUTORY AUTHORITY COMPLETION
        # ====================================================

        print()
        print(
            "[7.5/8] Completing case-law rules with "
            "primary statutory authority..."
        )

        try:
            (
                selected_sources,
                completion_rules,
                primary_sources
            ) = complete_primary_statutory_authorities(
                selected_sources=selected_sources,
                reranked_sources=reranked_sources,
                embedding_model=embedding_model,
                legal_embeddings=legal_embeddings,
                legal_offsets=legal_offsets,
                authority_catalog=authority_catalog,
                client=client
            )

            display_primary_authority_completion(
                completion_rules,
                primary_sources
            )

        except Exception as exc:
            print()
            print(
                "Primary-authority completion failed; "
                "continuing with the already verified sources."
            )
            print(
                "Primary-authority error:",
                exc
            )

        # ====================================================
        # PRELIMINARY RETRIEVED-ANSWER FALLBACK
        # ====================================================

        if not selected_sources:

            preliminary_sources = (
                select_preliminary_sources(
                    reranked_sources
                )
            )

            if preliminary_sources:

                print()
                print(
                    "[8/8] No source passed the strict final threshold; "
                    "building a cautious answer from the best relevant "
                    "retrieved passages..."
                )

                try:
                    preliminary_answer = (
                        generate_preliminary_answer(
                            original_problem=
                                original_problem,

                            followups=
                                followups,

                            selected=
                                preliminary_sources,

                            client=
                                client
                        )
                    )

                except Exception as exc:
                    print()
                    print(
                        "Preliminary answer generation failed; using a "
                        "deterministic retrieved-source summary."
                    )
                    print(
                        "Preliminary-answer error:",
                        exc
                    )

                    preliminary_answer = (
                        build_deterministic_preliminary_answer(
                            preliminary_sources
                        )
                    )

                invalid_citations = (
                    validate_final_citation_ids(
                        preliminary_answer,
                        preliminary_sources
                    )
                )

                # A preliminary answer should cite the retrieved passages it
                # relies on. If the model returns invalid source IDs, replace
                # it with the deterministic source-grounded fallback instead
                # of showing malformed citations.
                cited_preliminary_ids = set(
                    re.findall(
                        r"\[S(\d+)\]",
                        preliminary_answer
                    )
                )

                if invalid_citations or not cited_preliminary_ids:
                    preliminary_answer = (
                        build_deterministic_preliminary_answer(
                            preliminary_sources
                        )
                    )

                print()
                print(
                    "=" * 80
                )
                print(
                    "FINAL PRELIMINARY RETRIEVED LEGAL ANSWER"
                )
                print(
                    "=" * 80
                )
                print()
                print(
                    preliminary_answer.strip()
                )
                print()
                print(
                    "Note: this answer uses relevant retrieved passages "
                    "that did not pass the prototype's strict final-"
                    "verification threshold. It is shown as preliminary "
                    "research, not as a fully verified legal conclusion."
                )

                display_final_source_list(
                    preliminary_sources,
                    answer=preliminary_answer
                )

                return True

            # Nothing even reached the preliminary relevance threshold.
            print()
            print(
                "=" * 80
            )
            print(
                "INSUFFICIENT LEGAL RETRIEVAL"
            )
            print(
                "=" * 80
            )
            print()
            print(
                "The current search did not retrieve a sufficiently "
                "relevant legal passage to produce even a preliminary "
                "answer."
            )

            return True

        # ====================================================
        # 8. CLAIM-CITATION AUDIT + FINAL ANSWER
        # ====================================================

        print()
        print(
            "[8/8] Auditing final legal claims "
            "against citations..."
        )

        draft = generate_draft_answer(

            original_problem=
                original_problem,

            followups=
                followups,

            selected=
                selected_sources,

            client=
                client
        )

        audit = audit_draft_claims(

            original_problem=
                original_problem,

            followups=
                followups,

            selected=
                selected_sources,

            draft=
                draft,

            client=
                client
        )

        display_claim_audit(
            audit
        )

        answer = rewrite_audited_answer(

            original_problem=
                original_problem,

            followups=
                followups,

            selected=
                selected_sources,

            draft=
                draft,

            audit=
                audit,

            client=
                client
        )

        # ----------------------------------------------------
        # SECOND / ENFORCEMENT AUDIT
        # ----------------------------------------------------
        final_audit = audit_draft_claims(

            original_problem=
                original_problem,

            followups=
                followups,

            selected=
                selected_sources,

            draft=
                answer,

            client=
                client
        )

        if final_answer_passes_audit(
            final_audit
        ):
            print()
            print(
                "FINAL GROUNDING ENFORCEMENT: PASSED"
            )
        else:
            print()
            print(
                "FINAL GROUNDING ENFORCEMENT: FAILED"
            )
            print(
                "The final rewrite reintroduced or overstated "
                "at least one legal claim."
            )
            print(
                "Switching to deterministic fail-closed answer."
            )
            answer = build_fail_closed_answer(
                audit
            )

        invalid_citations = (
            validate_final_citation_ids(
                answer,
                selected_sources
            )
        )

        print()
        print(
            "=" * 80
        )

        print(
            "FINAL VERIFIED CONVERSATIONAL ANSWER"
        )

        print(
            "=" * 80
        )

        if invalid_citations:

            print()
            print(
                "Final answer withheld because "
                "it contained invalid source IDs:"
            )

            print(
                ", ".join(
                    invalid_citations
                )
            )

            print()
            print(
                "No unverified final answer will "
                "be shown."
            )

        else:

            print()
            print(
                answer.strip()
            )

        display_final_source_list(
            selected_sources,
            answer=answer
        )

        return True

    print()
    print(
        "Maximum research iterations reached."
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "MULTI-STAGE CONVERSATIONAL "
        "INDIAN LEGAL RAG"
    )

    print(
        "Intake → ILSIC → Issue Planning → Act Validation → "
        "710K Corpus → Authority Verification → Primary Statute Completion "
        "→ Claim Audit → Answer"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    required_files = [
        mapper.SCENARIOS_FILE,
        mapper.CHUNKS_FILE,
        mapper.EMBEDDINGS_FILE,
        LEGAL_DOCS_FILE,
        LEGAL_EMBEDDINGS_FILE
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing:\n"
                f"{path}"
            )

    # ========================================================
    # ILSIC INDEX
    # ========================================================

    (
        ilsic_scenario_map,
        ilsic_chunks,
        ilsic_embeddings
    ) = mapper.load_ilsic_index()

    # ========================================================
    # LEGAL EMBEDDINGS
    # ========================================================

    print()
    print(
        "Loading 710K legal embeddings..."
    )

    legal_embeddings = np.load(
        LEGAL_EMBEDDINGS_FILE,
        mmap_mode="r"
    )

    print(
        "Legal embeddings:",
        legal_embeddings.shape
    )

    legal_offsets = (
        build_or_load_offsets(
            expected_rows=
                legal_embeddings.shape[0]
        )
    )

    authority_catalog = (
        build_or_load_authority_catalog(
            expected_rows=
                legal_embeddings.shape[0]
        )
    )

    # ========================================================
    # BGE MODEL
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Embedding device:",
        device
    )

    print(
        "Loading BGE-small..."
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL,
            device=device
        )
    )

    embedding_model.max_seq_length = 512

    # ========================================================
    # GROQ
    # ========================================================

    print(
        "Creating Groq client..."
    )

    client = Groq(
        api_key=
            mapper.GROQ_API_KEY
    )

    print()
    print(
        "END-TO-END SYSTEM READY"
    )

    # ========================================================
    # INTERACTIVE LOOP
    # ========================================================

    while True:

        keep_running = run_case(

            embedding_model=
                embedding_model,

            ilsic_scenario_map=
                ilsic_scenario_map,

            ilsic_chunks=
                ilsic_chunks,

            ilsic_embeddings=
                ilsic_embeddings,

            legal_embeddings=
                legal_embeddings,

            legal_offsets=
                legal_offsets,

            authority_catalog=
                authority_catalog,

            client=
                client
        )

        if not keep_running:

            print()
            print(
                "Closing system."
            )

            break


if __name__ == "__main__":
    main()