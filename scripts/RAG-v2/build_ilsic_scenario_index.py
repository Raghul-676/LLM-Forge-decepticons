import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

PROJECT_ROOT = Path(
    r"D:\Projects\LLM_training"
)

ILSIC_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ILSIC-dataset"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "data"
    / "rag"
    / "ilsic"
)

SCENARIOS_FILE = (
    OUTPUT_DIR
    / "ilsic_train_scenarios.jsonl"
)

CHUNKS_FILE = (
    OUTPUT_DIR
    / "ilsic_train_chunks.jsonl"
)

EMBEDDINGS_FILE = (
    OUTPUT_DIR
    / "ilsic_train_embeddings.npy"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "ilsic_index_summary.json"
)


# ------------------------------------------------------------
# CHUNKING
# ------------------------------------------------------------

# We deliberately use smaller chunks here because the user may
# ask a short question while an ILSIC scenario may be very long.
CHUNK_WORDS = 180

OVERLAP_WORDS = 40

MIN_CHUNK_WORDS = 15

BATCH_SIZE = 64


# ============================================================
# FIND TRAIN DATASET
# ============================================================

def find_layman_train_file():
    """
    Locate the uncompressed FT-Layman-train.jsonl.

    Your extracted dataset currently has a structure such as:

    Layman-new-dataset/
        FT-Layman-train.jsonl/
            FT-Layman-train.jsonl

    This function handles that automatically.
    """

    candidates = list(
        ILSIC_ROOT.rglob(
            "FT-Layman-train.jsonl"
        )
    )

    candidates = [
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() == ".jsonl"
    ]

    if not candidates:

        raise FileNotFoundError(
            "Could not find uncompressed "
            "FT-Layman-train.jsonl under:\n"
            f"{ILSIC_ROOT}"
        )

    # Prefer file inside Layman-new-dataset.
    preferred = [
        path
        for path in candidates
        if "Layman-new-dataset"
        in str(path)
    ]

    if preferred:
        return preferred[0]

    return candidates[0]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(
        text
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


# ============================================================
# STATUTE NORMALIZATION
# ============================================================

def normalize_statutes(answer):

    if not isinstance(
        answer,
        list
    ):

        return []

    statutes = []

    seen = set()

    for item in answer:

        statute = normalize_text(
            item
        )

        if not statute:
            continue

        key = statute.lower()

        if key in seen:
            continue

        seen.add(
            key
        )

        statutes.append(
            statute
        )

    return statutes


# ============================================================
# SCENARIO CHUNKING
# ============================================================

def chunk_words(
    text,
    chunk_words=CHUNK_WORDS,
    overlap_words=OVERLAP_WORDS
):

    words = text.split()

    if not words:

        return []

    # Short scenario = one chunk.
    if len(words) <= chunk_words:

        return [
            text
        ]

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_words,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if (
            len(chunk.split())
            >= MIN_CHUNK_WORDS
        ):

            chunks.append(
                chunk
            )

        if end >= len(words):
            break

        next_start = (
            end
            - overlap_words
        )

        if next_start <= start:

            next_start = end

        start = next_start

    return chunks


# ============================================================
# LOAD + CLEAN ILSIC
# ============================================================

def build_clean_scenarios(
    train_file
):

    scenarios = []

    skipped_invalid = 0

    duplicate_scenarios = 0

    # If identical scenario text appears more than once,
    # merge its mapped statutes.
    by_text = {}

    with open(
        train_file,
        "r",
        encoding="utf-8"
    ) as f:

        for raw_line in f:

            raw_line = raw_line.strip()

            if not raw_line:
                continue

            try:

                record = json.loads(
                    raw_line
                )

            except json.JSONDecodeError:

                skipped_invalid += 1
                continue

            instruction = normalize_text(
                record.get(
                    "instruction",
                    ""
                )
            )

            statutes = normalize_statutes(
                record.get(
                    "answer",
                    []
                )
            )

            if (
                not instruction
                or not statutes
            ):

                skipped_invalid += 1
                continue

            dedup_key = (
                instruction.lower()
            )

            if dedup_key in by_text:

                duplicate_scenarios += 1

                existing = by_text[
                    dedup_key
                ]

                existing_set = {
                    item.lower()
                    for item in existing[
                        "statutes"
                    ]
                }

                for statute in statutes:

                    if (
                        statute.lower()
                        not in existing_set
                    ):

                        existing[
                            "statutes"
                        ].append(
                            statute
                        )

                        existing_set.add(
                            statute.lower()
                        )

                continue

            scenario = {
                "scenario_id": None,
                "scenario": instruction,
                "statutes": statutes
            }

            by_text[
                dedup_key
            ] = scenario

            scenarios.append(
                scenario
            )

    # Stable IDs after deduplication.
    for index, scenario in enumerate(
        scenarios
    ):

        scenario[
            "scenario_id"
        ] = f"ilsic_train_{index:06d}"

    return (
        scenarios,
        skipped_invalid,
        duplicate_scenarios
    )


# ============================================================
# BUILD RETRIEVAL CHUNKS
# ============================================================

def build_chunks(
    scenarios
):

    chunks = []

    chunk_index = 0

    for scenario in scenarios:

        scenario_chunks = chunk_words(
            scenario[
                "scenario"
            ]
        )

        for local_index, chunk_text in enumerate(
            scenario_chunks
        ):

            chunks.append({
                "chunk_index":
                    chunk_index,

                "chunk_id":
                    (
                        f"{scenario['scenario_id']}"
                        f"_chunk_{local_index:03d}"
                    ),

                "scenario_id":
                    scenario[
                        "scenario_id"
                    ],

                "chunk_number":
                    local_index,

                "text":
                    chunk_text
            })

            chunk_index += 1

    return chunks


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(
    path,
    records
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# BUILD EMBEDDINGS
# ============================================================

def build_embeddings(
    chunks
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nEmbedding device:",
        device
    )

    print(
        "Loading embedding model:",
        EMBEDDING_MODEL
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device
    )

    model.max_seq_length = 512

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "\nEmbedding scenario chunks..."
    )

    print(
        "Chunks:",
        f"{len(texts):,}"
    )

    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        np.float32
    )

    elapsed = (
        time.time()
        - start_time
    )

    return (
        embeddings,
        elapsed,
        device
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "ILSIC SCENARIO → STATUTE INDEX BUILDER"
    )

    print(
        "=" * 100
    )

    print(
        "\nDataset root:"
    )

    print(
        ILSIC_ROOT
    )

    # --------------------------------------------------------
    # LOCATE DATASET
    # --------------------------------------------------------

    train_file = (
        find_layman_train_file()
    )

    print(
        "\nUsing training dataset:"
    )

    print(
        train_file
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # LOAD + CLEAN
    # --------------------------------------------------------

    print(
        "\n[1/4] Loading and cleaning "
        "ILSIC layman scenarios..."
    )

    (
        scenarios,
        skipped_invalid,
        duplicate_scenarios
    ) = build_clean_scenarios(
        train_file
    )

    print(
        "Clean unique scenarios:",
        f"{len(scenarios):,}"
    )

    print(
        "Invalid/skipped:",
        f"{skipped_invalid:,}"
    )

    print(
        "Exact duplicate scenarios merged:",
        f"{duplicate_scenarios:,}"
    )

    # --------------------------------------------------------
    # CHUNK
    # --------------------------------------------------------

    print(
        "\n[2/4] Chunking long scenarios..."
    )

    chunks = build_chunks(
        scenarios
    )

    print(
        "Scenario chunks:",
        f"{len(chunks):,}"
    )

    if not chunks:

        raise RuntimeError(
            "No retrieval chunks were produced."
        )

    # --------------------------------------------------------
    # SAVE CLEAN DATA
    # --------------------------------------------------------

    print(
        "\n[3/4] Saving cleaned mapping data..."
    )

    save_jsonl(
        SCENARIOS_FILE,
        scenarios
    )

    save_jsonl(
        CHUNKS_FILE,
        chunks
    )

    print(
        "Scenarios:",
        SCENARIOS_FILE
    )

    print(
        "Chunks:",
        CHUNKS_FILE
    )

    # --------------------------------------------------------
    # EMBED
    # --------------------------------------------------------

    print(
        "\n[4/4] Building BGE embeddings..."
    )

    (
        embeddings,
        elapsed,
        device
    ) = build_embeddings(
        chunks
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    # --------------------------------------------------------
    # VERIFY NORMALIZATION
    # --------------------------------------------------------

    sample_size = min(
        1000,
        len(embeddings)
    )

    norms = np.linalg.norm(
        embeddings[
            :sample_size
        ],
        axis=1
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    scenario_word_counts = [
        len(
            scenario[
                "scenario"
            ].split()
        )
        for scenario in scenarios
    ]

    statutes_total = sum(
        len(
            scenario[
                "statutes"
            ]
        )
        for scenario in scenarios
    )

    unique_statutes = sorted({
        statute
        for scenario in scenarios
        for statute in scenario[
            "statutes"
        ]
    })

    summary = {
        "embedding_model":
            EMBEDDING_MODEL,

        "dataset":
            str(
                train_file
            ),

        "scenarios":
            len(
                scenarios
            ),

        "scenario_chunks":
            len(
                chunks
            ),

        "statute_assignments":
            statutes_total,

        "unique_statutes":
            len(
                unique_statutes
            ),

        "skipped_invalid":
            skipped_invalid,

        "duplicates_merged":
            duplicate_scenarios,

        "chunk_words":
            CHUNK_WORDS,

        "overlap_words":
            OVERLAP_WORDS,

        "embedding_shape":
            list(
                embeddings.shape
            ),

        "embedding_dtype":
            str(
                embeddings.dtype
            ),

        "mean_embedding_norm":
            float(
                norms.mean()
            ),

        "min_embedding_norm":
            float(
                norms.min()
            ),

        "max_embedding_norm":
            float(
                norms.max()
            ),

        "average_scenario_words":
            (
                sum(
                    scenario_word_counts
                )
                /
                len(
                    scenario_word_counts
                )
                if scenario_word_counts
                else 0
            ),

        "maximum_scenario_words":
            (
                max(
                    scenario_word_counts
                )
                if scenario_word_counts
                else 0
            ),

        "embedding_device":
            device,

        "elapsed_seconds":
            elapsed,

        "output_files": {
            "scenarios":
                str(
                    SCENARIOS_FILE
                ),

            "chunks":
                str(
                    CHUNKS_FILE
                ),

            "embeddings":
                str(
                    EMBEDDINGS_FILE
                )
        }
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ILSIC SCENARIO INDEX COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "\nScenarios:",
        f"{len(scenarios):,}"
    )

    print(
        "Scenario chunks:",
        f"{len(chunks):,}"
    )

    print(
        "Statute assignments:",
        f"{statutes_total:,}"
    )

    print(
        "Unique mapped statutes:",
        f"{len(unique_statutes):,}"
    )

    print(
        "\nEmbedding shape:",
        embeddings.shape
    )

    print(
        "dtype:",
        embeddings.dtype
    )

    print(
        f"Mean norm: "
        f"{norms.mean():.6f}"
    )

    print(
        f"Elapsed: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "\nOUTPUT FILES"
    )

    print(
        "-" * 100
    )

    print(
        "Scenario mapping:"
    )

    print(
        SCENARIOS_FILE
    )

    print(
        "\nRetrieval chunks:"
    )

    print(
        CHUNKS_FILE
    )

    print(
        "\nEmbeddings:"
    )

    print(
        EMBEDDINGS_FILE
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()