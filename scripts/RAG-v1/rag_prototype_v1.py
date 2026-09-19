import os
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

CORPUS_FILE = Path(
    "data/rag/v1/legal_documents.jsonl"
)

EMBEDDINGS_FILE = Path(
    "data/rag/v1/legal_embeddings.npy"
)

# Retrieve more candidates first.
RETRIEVAL_TOP_K = 10

# Only send the best diversified chunks to Qwen.
CONTEXT_TOP_K = 6

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN was not found in .env"
    )


# ============================================================
# LOAD CORPUS
# ============================================================

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


# ============================================================
# SOURCE DISPLAY
# ============================================================

def source_label(metadata):

    source_type = metadata.get(
        "source_type",
        "unknown"
    )

    if source_type == "central_act":

        title = metadata.get(
            "title",
            "Unknown Act"
        )

        section = metadata.get(
            "section",
            "Unknown section"
        )

        return f"{title}, {section}"

    if source_type == "constitution":

        page = metadata.get(
            "pdf_page"
        )

        return (
            f"Constitution of India, "
            f"PDF page {page}"
        )

    if source_type == "supreme_court":

        title = metadata.get(
            "title"
        ) or "Unknown Supreme Court case"

        citation = metadata.get(
            "citation"
        )

        if citation:
            citation = " ".join(
                citation.split()
            )

            return (
                f"Supreme Court of India, "
                f"{title}, {citation}"
            )

        return (
            f"Supreme Court of India, "
            f"{title}"
        )

    return str(source_type)


# ============================================================
# USED FOR DIVERSITY
# ============================================================

def source_group(document):

    metadata = document["metadata"]

    source_type = metadata.get(
        "source_type"
    )

    if source_type == "supreme_court":

        return (
            "supreme_court",
            metadata.get("document_id")
            or metadata.get("source_file")
        )

    if source_type == "central_act":

        return (
            "central_act",
            metadata.get("document_id")
            or document["id"]
        )

    if source_type == "constitution":

        return (
            "constitution",
            metadata.get("pdf_page")
        )

    return (
        source_type,
        document["id"]
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    embedding_model,
    embeddings,
    documents
):

    retrieval_query = (
        QUERY_PREFIX + query
    )

    query_embedding = embedding_model.encode(
        retrieval_query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Both document and query vectors are normalized,
    # therefore dot product = cosine similarity.
    scores = (
        embeddings @ query_embedding
    )

    candidate_indexes = np.argpartition(
        scores,
        -RETRIEVAL_TOP_K
    )[-RETRIEVAL_TOP_K:]

    candidate_indexes = candidate_indexes[
        np.argsort(
            scores[candidate_indexes]
        )[::-1]
    ]

    candidates = []

    for index in candidate_indexes:

        document = documents[int(index)]

        candidates.append({
            "document": document,
            "score": float(
                scores[index]
            )
        })

    # --------------------------------------------------------
    # Basic diversity:
    # Avoid allowing one Supreme Court judgment
    # to occupy almost the entire context.
    # --------------------------------------------------------

    selected = []

    group_counts = Counter()

    for result in candidates:

        document = result["document"]

        group = source_group(
            document
        )

        source_type = document[
            "metadata"
        ].get("source_type")

        # At most 2 chunks from the same SC judgment.
        if (
            source_type == "supreme_court"
            and group_counts[group] >= 2
        ):
            continue

        selected.append(result)

        group_counts[group] += 1

        if len(selected) >= CONTEXT_TOP_K:
            break

    return selected


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_blocks = []

    for number, result in enumerate(
        results,
        start=1
    ):

        document = result[
            "document"
        ]

        metadata = document[
            "metadata"
        ]

        label = source_label(
            metadata
        )

        legal_status = metadata.get(
            "legal_status"
        )

        extra = ""

        if legal_status:
            extra = (
                f"\nLegal status metadata: "
                f"{legal_status}"
            )

        block = (
            f"[S{number}]\n"
            f"Source: {label}\n"
            f"Source type: "
            f"{metadata.get('source_type')}"
            f"{extra}\n\n"
            f"{document['text']}"
        )

        context_blocks.append(
            block
        )

    return "\n\n" + (
        "\n\n".join(
            context_blocks
        )
    )


# ============================================================
# QWEN GENERATION
# ============================================================

def generate_answer(
    query,
    context,
    client
):

    system_prompt = """
You are an Indian legal research assistant operating inside
a retrieval-augmented generation system.

You must follow these rules:

1. Answer using ONLY the supplied retrieved sources.
2. Do not invent any statute, section, judgment, citation,
   legal rule, date, or fact.
3. Cite factual/legal propositions using source labels such
   as [S1], [S2], etc.
4. A citation must support the statement immediately before it.
5. If the retrieved material is insufficient, explicitly say so.
6. Distinguish constitutional text, statutes, and judgments.
7. Some statutory material may be historical or have an
   unverified current legal status. Do not describe such material
   as current law unless the supplied source establishes that.
8. Do not give a definitive legal conclusion when important facts
   are missing.
9. Keep the response clear and concise.
"""

    user_prompt = f"""
USER QUESTION:

{query}


RETRIEVED LEGAL SOURCES:

{context}


TASK:

Answer the user's question based only on the retrieved legal
sources above.

Use citations like [S1] and [S2].

If these sources do not provide enough information for a reliable
answer, explain what is missing.
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

    response = (
        client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=700,
            temperature=0.1
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INDIAN LEGAL RAG - PROTOTYPE V1")
    print("=" * 80)

    # --------------------------------------------------------
    # Load knowledge base
    # --------------------------------------------------------

    print("\nLoading legal documents...")

    documents = load_documents()

    print(
        "Documents:",
        len(documents)
    )

    print(
        "Loading saved embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    print(
        "Embeddings:",
        embeddings.shape
    )

    if (
        embeddings.shape[0]
        != len(documents)
    ):
        raise ValueError(
            "Embedding/document count mismatch"
        )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Embedding device:",
        device
    )

    print(
        "Loading embedding model..."
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL,
            device=device
        )
    )

    embedding_model.max_seq_length = 512

    # --------------------------------------------------------
    # Qwen API
    # --------------------------------------------------------

    print(
        "Creating Qwen API client..."
    )

    client = InferenceClient(
        provider="featherless-ai",
        api_key=HF_TOKEN
    )

    print("\nPrototype ready.")

    # --------------------------------------------------------
    # Interactive loop
    # --------------------------------------------------------

    while True:

        print("\n" + "=" * 80)

        query = input(
            "\nEnter legal question "
            "(or type 'exit'): "
        ).strip()

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit"
        }:
            print(
                "\nClosing Prototype v1."
            )
            break

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        print(
            "\nSearching legal corpus..."
        )

        results = retrieve(
            query,
            embedding_model,
            embeddings,
            documents
        )

        print(
            "\nRetrieved sources:"
        )

        for number, result in enumerate(
            results,
            start=1
        ):

            metadata = result[
                "document"
            ]["metadata"]

            print(
                f"\n[S{number}] "
                f"score="
                f"{result['score']:.4f}"
            )

            print(
                source_label(
                    metadata
                )
            )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context = build_context(
            results
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        print(
            "\nSending retrieved "
            "context to Qwen..."
        )

        try:

            answer = generate_answer(
                query,
                context,
                client
            )

        except Exception as exc:

            print(
                "\nQwen API error:"
            )

            print(
                type(exc).__name__
            )

            print(exc)

            continue

        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        print(
            "\n" + "=" * 80
        )

        print(
            "RAG ANSWER"
        )

        print(
            "=" * 80
        )

        print(answer)

        print(
            "\n" + "-" * 80
        )

        print(
            "RETRIEVED SOURCE MAP"
        )

        print(
            "-" * 80
        )

        for number, result in enumerate(
            results,
            start=1
        ):

            metadata = result[
                "document"
            ]["metadata"]

            print(
                f"[S{number}] "
                f"{source_label(metadata)}"
            )


if __name__ == "__main__":
    main()