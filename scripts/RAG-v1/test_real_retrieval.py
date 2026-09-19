from pathlib import Path
import argparse
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

CORPUS_FILE = Path(
    "data/rag/v1/legal_documents.jsonl"
)

EMBEDDINGS_FILE = Path(
    "data/rag/v1/legal_embeddings.npy"
)

TOP_K = 8

# Recommended query instruction for BGE retrieval models.
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


def load_documents():
    documents = []

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            if line.strip():
                documents.append(
                    json.loads(line)
                )

    return documents


def source_label(metadata):

    source_type = metadata.get(
        "source_type",
        "unknown"
    )

    if source_type == "central_act":

        return (
            f"{metadata.get('title', 'Unknown Act')} | "
            f"{metadata.get('section', 'Unknown Section')}"
        )

    if source_type == "constitution":

        return (
            "Constitution of India | "
            f"PDF page {metadata.get('pdf_page')}"
        )

    if source_type == "supreme_court":

        title = metadata.get(
            "title",
            "Unknown Case"
        )

        citation = metadata.get(
            "citation"
        )

        if citation:
            return (
                f"Supreme Court | {title} | "
                f"{citation.strip()}"
            )

        return (
            f"Supreme Court | {title}"
        )

    return source_type


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "What protection does the Constitution "
            "provide for life and personal liberty?"
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K
    )

    args = parser.parse_args()

    query = args.query
    top_k = args.top_k

    print("=" * 80)
    print("RAG PROTOTYPE V1 - REAL RETRIEVAL")
    print("=" * 80)

    print("\nLoading legal corpus...")

    documents = load_documents()

    print(
        "Documents:",
        len(documents)
    )

    print("\nLoading saved embeddings...")

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    print(
        "Embeddings shape:",
        embeddings.shape
    )

    if len(documents) != embeddings.shape[0]:
        raise ValueError(
            "Document count and embedding count do not match."
        )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    model.max_seq_length = 512

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    print("\nQUERY:")
    print(query)

    retrieval_query = (
        QUERY_PREFIX + query
    )

    query_embedding = model.encode(
        retrieval_query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Since document and query vectors are normalized,
    # dot product == cosine similarity.
    scores = embeddings @ query_embedding

    if top_k > len(documents):
        top_k = len(documents)

    # Efficiently identify top candidates without sorting
    # all 55k entries first.
    candidate_indexes = np.argpartition(
        scores,
        -top_k
    )[-top_k:]

    ranked_indexes = candidate_indexes[
        np.argsort(
            scores[candidate_indexes]
        )[::-1]
    ]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print(f"TOP {top_k} RETRIEVAL RESULTS")
    print("=" * 80)

    for rank, index in enumerate(
        ranked_indexes,
        start=1
    ):

        document = documents[int(index)]

        metadata = document.get(
            "metadata",
            {}
        )

        text = document.get(
            "text",
            ""
        )

        print("\n" + "-" * 80)

        print(
            f"RANK {rank}"
        )

        print(
            f"Score: "
            f"{float(scores[index]):.4f}"
        )

        print(
            "Document ID:",
            document.get("id")
        )

        print(
            "Source type:",
            metadata.get("source_type")
        )

        print(
            "Source:",
            source_label(metadata)
        )

        if metadata.get("source_file"):
            print(
                "Source file:",
                metadata.get("source_file")
            )

        print(
            "Chunk:",
            metadata.get("chunk_index"),
            "/",
            metadata.get("chunk_count")
        )

        print("\nTEXT PREVIEW:")

        # Terminal preview only.
        # Full text is still stored in legal_documents.jsonl.
        print(
            text[:1200]
        )

        if len(text) > 1200:
            print("... [TRUNCATED PREVIEW]")

    print("\n" + "=" * 80)
    print("RETRIEVAL COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()