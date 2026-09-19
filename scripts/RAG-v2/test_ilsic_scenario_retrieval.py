import json
from pathlib import Path
from collections import defaultdict, Counter

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

INDEX_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "data"
    / "rag"
    / "ilsic"
)

SCENARIOS_FILE = (
    INDEX_DIR
    / "ilsic_train_scenarios.jsonl"
)

CHUNKS_FILE = (
    INDEX_DIR
    / "ilsic_train_chunks.jsonl"
)

EMBEDDINGS_FILE = (
    INDEX_DIR
    / "ilsic_train_embeddings.npy"
)


# BGE query prefix.
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

# Retrieve many chunks first.
TOP_CHUNKS = 50

# Then collapse those chunks into unique ILSIC scenarios.
TOP_SCENARIOS = 10

# Display this many candidate statute sections.
TOP_STATUTES = 15


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# LOAD INDEX
# ============================================================

def load_index():

    print(
        "Loading ILSIC scenarios..."
    )

    scenarios = load_jsonl(
        SCENARIOS_FILE
    )

    print(
        "Loading ILSIC chunks..."
    )

    chunks = load_jsonl(
        CHUNKS_FILE
    )

    print(
        "Loading embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r"
    )

    if len(chunks) != embeddings.shape[0]:

        raise ValueError(
            "Chunk / embedding count mismatch.\n"
            f"Chunks: {len(chunks)}\n"
            f"Embeddings: {embeddings.shape[0]}"
        )

    # Fast scenario lookup.
    scenario_map = {
        scenario["scenario_id"]:
            scenario
        for scenario in scenarios
    }

    print(
        "\nScenarios:",
        f"{len(scenarios):,}"
    )

    print(
        "Chunks:",
        f"{len(chunks):,}"
    )

    print(
        "Embeddings:",
        embeddings.shape
    )

    return (
        scenarios,
        scenario_map,
        chunks,
        embeddings
    )


# ============================================================
# EMBED USER QUERY
# ============================================================

def embed_query(
    query,
    model
):

    query_text = (
        QUERY_PREFIX
        + query
    )

    embedding = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding.astype(
        np.float32
    )


# ============================================================
# SEARCH CHUNKS
# ============================================================

def search_chunks(
    query,
    model,
    chunks,
    embeddings
):

    query_embedding = embed_query(
        query,
        model
    )

    # All stored embeddings are normalized.
    # Query is normalized.
    #
    # Dot product = cosine similarity.
    scores = (
        embeddings
        @ query_embedding
    )

    top_k = min(
        TOP_CHUNKS,
        len(scores)
    )

    indexes = np.argpartition(
        scores,
        -top_k
    )[-top_k:]

    indexes = indexes[
        np.argsort(
            scores[indexes]
        )[::-1]
    ]

    results = []

    for rank, index in enumerate(
        indexes,
        start=1
    ):

        results.append({
            "chunk_index":
                int(index),

            "rank":
                rank,

            "similarity":
                float(
                    scores[index]
                ),

            "chunk":
                chunks[
                    int(index)
                ]
        })

    return results


# ============================================================
# COLLAPSE CHUNKS INTO UNIQUE SCENARIOS
# ============================================================

def aggregate_scenarios(
    chunk_results,
    scenario_map
):
    """
    One ILSIC scenario may have multiple chunks.

    We do NOT want the same long scenario to appear 5 times.

    For every scenario:
      - keep its best similarity;
      - count how many chunks matched;
      - remember the best chunk.
    """

    grouped = {}

    for result in chunk_results:

        chunk = result[
            "chunk"
        ]

        scenario_id = chunk[
            "scenario_id"
        ]

        similarity = result[
            "similarity"
        ]

        if scenario_id not in grouped:

            grouped[
                scenario_id
            ] = {
                "scenario_id":
                    scenario_id,

                "best_similarity":
                    similarity,

                "matching_chunks":
                    1,

                "best_chunk":
                    chunk["text"],

                "best_chunk_rank":
                    result["rank"]
            }

        else:

            grouped[
                scenario_id
            ][
                "matching_chunks"
            ] += 1

            if (
                similarity
                >
                grouped[
                    scenario_id
                ][
                    "best_similarity"
                ]
            ):

                grouped[
                    scenario_id
                ][
                    "best_similarity"
                ] = similarity

                grouped[
                    scenario_id
                ][
                    "best_chunk"
                ] = chunk[
                    "text"
                ]

                grouped[
                    scenario_id
                ][
                    "best_chunk_rank"
                ] = result[
                    "rank"
                ]

    results = []

    for scenario_id, data in grouped.items():

        scenario = scenario_map.get(
            scenario_id
        )

        if scenario is None:
            continue

        results.append({
            **data,

            "scenario":
                scenario[
                    "scenario"
                ],

            "statutes":
                scenario[
                    "statutes"
                ]
        })

    # Primary ranking:
    # highest semantic similarity.
    #
    # Secondary ranking:
    # scenarios with multiple matching chunks.
    results.sort(
        key=lambda item: (
            item[
                "best_similarity"
            ],
            item[
                "matching_chunks"
            ]
        ),
        reverse=True
    )

    return results[
        :TOP_SCENARIOS
    ]


# ============================================================
# AGGREGATE STATUTES
# ============================================================

def aggregate_statutes(
    scenario_results
):
    """
    If several similar scenarios point to the same statute,
    that statute becomes more interesting.

    We use:
      1. number of scenario votes;
      2. accumulated similarity;
      3. maximum similarity.
    """

    statute_data = defaultdict(
        lambda: {
            "votes": 0,
            "similarity_sum": 0.0,
            "max_similarity": 0.0,
            "scenario_ids": []
        }
    )

    for result in scenario_results:

        similarity = result[
            "best_similarity"
        ]

        for statute in result[
            "statutes"
        ]:

            data = statute_data[
                statute
            ]

            data[
                "votes"
            ] += 1

            data[
                "similarity_sum"
            ] += similarity

            data[
                "max_similarity"
            ] = max(
                data[
                    "max_similarity"
                ],
                similarity
            )

            data[
                "scenario_ids"
            ].append(
                result[
                    "scenario_id"
                ]
            )

    ranked = []

    for statute, data in statute_data.items():

        ranked.append({
            "statute":
                statute,

            "votes":
                data[
                    "votes"
                ],

            "similarity_sum":
                data[
                    "similarity_sum"
                ],

            "max_similarity":
                data[
                    "max_similarity"
                ],

            "scenario_ids":
                data[
                    "scenario_ids"
                ]
        })

    ranked.sort(
        key=lambda item: (
            item[
                "votes"
            ],
            item[
                "similarity_sum"
            ],
            item[
                "max_similarity"
            ]
        ),
        reverse=True
    )

    return ranked


# ============================================================
# SHORTEN DISPLAY TEXT
# ============================================================

def shorten(
    text,
    limit=850
):

    text = str(
        text
    ).strip()

    if len(text) <= limit:
        return text

    return (
        text[:limit]
        + "..."
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    user_query,
    scenario_results,
    statute_results
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "USER QUERY"
    )

    print(
        "=" * 100
    )

    print(
        user_query
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TOP SIMILAR ILSIC SCENARIOS"
    )

    print(
        "=" * 100
    )

    if not scenario_results:

        print(
            "\nNo scenarios retrieved."
        )

        return

    for number, result in enumerate(
        scenario_results,
        start=1
    ):

        print(
            f"\n[{number}] "
            f"similarity="
            f"{result['best_similarity']:.4f}"
            f" | matching_chunks="
            f"{result['matching_chunks']}"
        )

        print(
            f"Scenario ID: "
            f"{result['scenario_id']}"
        )

        print(
            "\nScenario:"
        )

        print(
            shorten(
                result[
                    "scenario"
                ]
            )
        )

        print(
            "\nMapped statutes:"
        )

        for statute in result[
            "statutes"
        ]:

            print(
                "  -",
                statute
            )

        print(
            "-" * 100
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AGGREGATED CANDIDATE STATUTES"
    )

    print(
        "=" * 100
    )

    if not statute_results:

        print(
            "\nNo candidate statutes."
        )

        return

    for number, item in enumerate(
        statute_results[
            :TOP_STATUTES
        ],
        start=1
    ):

        print(
            f"\n[{number}] "
            f"votes={item['votes']} "
            f"| max_similarity="
            f"{item['max_similarity']:.4f}"
        )

        print(
            item[
                "statute"
            ]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "ILSIC SCENARIO → STATUTE RETRIEVAL TEST"
    )

    print(
        "=" * 100
    )

    (
        scenarios,
        scenario_map,
        chunks,
        embeddings
    ) = load_index()

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

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
        "Loading BGE-small..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device
    )

    model.max_seq_length = 512

    print(
        "\nILSIC scenario retrieval ready."
    )

    # --------------------------------------------------------
    # INTERACTIVE LOOP
    # --------------------------------------------------------

    while True:

        print(
            "\n"
            + "=" * 100
        )

        query = input(
            "\nEnter citizen legal scenario "
            "(or 'exit'): "
        ).strip()

        if not query:

            continue

        if query.lower() in {
            "exit",
            "quit"
        }:

            print(
                "\nClosing ILSIC retrieval test."
            )

            break

        # ----------------------------------------------------
        # SEARCH CHUNKS
        # ----------------------------------------------------

        chunk_results = search_chunks(
            query,
            model,
            chunks,
            embeddings
        )

        # ----------------------------------------------------
        # COLLAPSE TO UNIQUE SCENARIOS
        # ----------------------------------------------------

        scenario_results = (
            aggregate_scenarios(
                chunk_results,
                scenario_map
            )
        )

        # ----------------------------------------------------
        # MAP SCENARIOS → STATUTES
        # ----------------------------------------------------

        statute_results = (
            aggregate_statutes(
                scenario_results
            )
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display_results(
            query,
            scenario_results,
            statute_results
        )


if __name__ == "__main__":
    main()