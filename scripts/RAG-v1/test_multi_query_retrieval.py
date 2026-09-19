import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

CORPUS_FILE = Path(
    "data/rag/v1/legal_documents.jsonl"
)

EMBEDDINGS_FILE = Path(
    "data/rag/v1/legal_embeddings.npy"
)

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

TOP_PER_QUERY = 10
FINAL_TOP_K = 10


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in .env")


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


def extract_json(text):

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found."
        )

    return json.loads(
        text[start:end + 1]
    )


def create_search_plan(query, client):

    system_prompt = """
You are a query-understanding component for an Indian legal
retrieval system.

Do not answer the legal problem.

Convert the informal user query into formal search queries for
retrieving Indian statutes and judgments.

Important rules:

- The first-person speaker is the user.
- Do not reverse parties.
- Do not invent facts.
- Do not invent statute numbers or case citations.
- Ordinary refusal to repay borrowed money should primarily be
  treated as civil debt recovery / money recovery / contractual
  obligation.
- Do not classify ordinary non-repayment as criminal unless the
  user states facts suggesting deception, forgery,
  misappropriation, threats, or another criminal allegation.

Return valid JSON only:

{
  "problem_summary": "...",
  "user_role": "...",
  "other_party_role": "...",
  "primary_legal_area": "...",
  "search_queries": [
    "...",
    "...",
    "...",
    "..."
  ]
}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ],
        max_tokens=400,
        temperature=0.0
    )

    return extract_json(
        response.choices[0].message.content
    )


def search_one_query(
    query,
    model,
    embeddings
):

    query_embedding = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    scores = embeddings @ query_embedding

    indexes = np.argpartition(
        scores,
        -TOP_PER_QUERY
    )[-TOP_PER_QUERY:]

    indexes = indexes[
        np.argsort(scores[indexes])[::-1]
    ]

    return [
        (
            int(index),
            float(scores[index])
        )
        for index in indexes
    ]


def source_label(metadata):

    source_type = metadata.get(
        "source_type"
    )

    if source_type == "central_act":

        return (
            f"{metadata.get('title')} | "
            f"{metadata.get('section')}"
        )

    if source_type == "constitution":

        return (
            "Constitution of India | "
            f"PDF page {metadata.get('pdf_page')}"
        )

    if source_type == "supreme_court":

        return (
            "Supreme Court | "
            f"{metadata.get('title')}"
        )

    return str(source_type)


def main():

    print("=" * 80)
    print("RAG V1.1 - MULTI-QUERY RETRIEVAL")
    print("=" * 80)

    print("\nLoading corpus...")

    documents = load_documents()

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    print("Documents:", len(documents))
    print("Embeddings:", embeddings.shape)

    if len(documents) != embeddings.shape[0]:
        raise ValueError(
            "Document/embedding mismatch."
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Embedding device:", device)

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device
    )

    embedding_model.max_seq_length = 512

    client = InferenceClient(
        provider="featherless-ai",
        api_key=HF_TOKEN
    )

    query = input(
        "\nEnter citizen legal query: "
    ).strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    # --------------------------------------------------------
    # Query planning
    # --------------------------------------------------------

    print("\nCreating search plan...")

    plan = create_search_plan(
        query,
        client
    )

    print("\nSEARCH PLAN")

    print(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False
        )
    )

    search_queries = plan.get(
        "search_queries",
        []
    )

    if not search_queries:
        search_queries = [query]

    # Also search the original query.
    all_queries = [query] + search_queries

    print("\nQueries used for retrieval:")

    for i, q in enumerate(
        all_queries,
        start=1
    ):
        print(f"{i}. {q}")

    # --------------------------------------------------------
    # Search every query
    # --------------------------------------------------------

    merged = defaultdict(
        lambda: {
            "best_score": -1.0,
            "score_sum": 0.0,
            "hits": 0,
            "matched_queries": []
        }
    )

    for search_query in all_queries:

        results = search_one_query(
            search_query,
            embedding_model,
            embeddings
        )

        for index, score in results:

            item = merged[index]

            item["best_score"] = max(
                item["best_score"],
                score
            )

            item["score_sum"] += score
            item["hits"] += 1

            item["matched_queries"].append(
                search_query
            )

    # --------------------------------------------------------
    # Fusion score
    #
    # Rewards:
    # - high semantic score
    # - appearing for multiple queries
    # --------------------------------------------------------

    ranked = []

    for index, data in merged.items():

        average_score = (
            data["score_sum"]
            / data["hits"]
        )

        fusion_score = (
            data["best_score"]
            + 0.03 * (data["hits"] - 1)
        )

        ranked.append({
            "index": index,
            "fusion_score": fusion_score,
            "best_score":
                data["best_score"],
            "average_score":
                average_score,
            "hits":
                data["hits"],
            "matched_queries":
                data["matched_queries"]
        })

    ranked.sort(
        key=lambda x: x["fusion_score"],
        reverse=True
    )

    ranked = ranked[
        :FINAL_TOP_K
    ]

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MULTI-QUERY RETRIEVAL RESULTS")
    print("=" * 80)

    for rank, result in enumerate(
        ranked,
        start=1
    ):

        document = documents[
            result["index"]
        ]

        metadata = document[
            "metadata"
        ]

        print("\n" + "-" * 80)

        print(f"RANK {rank}")

        print(
            f"Fusion score: "
            f"{result['fusion_score']:.4f}"
        )

        print(
            f"Best similarity: "
            f"{result['best_score']:.4f}"
        )

        print(
            "Query hits:",
            result["hits"]
        )

        print(
            "Source type:",
            metadata.get("source_type")
        )

        print(
            "Source:",
            source_label(metadata)
        )

        print(
            "\nTEXT PREVIEW:"
        )

        text = document.get(
            "text",
            ""
        )

        print(
            text[:1000]
        )

        if len(text) > 1000:
            print(
                "... [TRUNCATED]"
            )

    print("\n" + "=" * 80)
    print("MULTI-QUERY RETRIEVAL COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()