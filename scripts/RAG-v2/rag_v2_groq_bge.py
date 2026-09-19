import os
import re
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

# This script is:
# PROJECT_ROOT/scripts/RAG-v2/rag_v2_groq_bge.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "scripts"
    / "data"
)

CORPUS_FILE = (
    DATA_ROOT
    / "rag"
    / "v2"
    / "legal_documents.jsonl"
)

EMBEDDINGS_FILE = (
    DATA_ROOT
    / "rag"
    / "v2"
    / "legal_embeddings.npy"
)


# ------------------------------------------------------------
# BGE SETTINGS
# ------------------------------------------------------------

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

# Retrieve 30 chunks PER query from BGE.
TOP_PER_QUERY = 30

# After Reciprocal Rank Fusion, keep a larger temporary pool.
# Diversification happens AFTER this.
FUSED_POOL_SIZE = 40

# Maximum number of diverse candidates sent to GPT-OSS.
RERANK_CANDIDATES = 12

# Only strongest sources reach final answer generation.
#
# 4 is deliberate because your current Groq tier has an
# 8,000 tokens-per-minute limit.
FINAL_CONTEXT_K = 4

# Reciprocal Rank Fusion constant.
RRF_K = 60

# Characters of each candidate shown to GPT-OSS reranker.
#
# Previously this was 2200.
# 700 greatly reduces the reranker request size.
RERANK_PREVIEW_CHARS = 700


# ------------------------------------------------------------
# GROQ TOKEN / RETRY SETTINGS
# ------------------------------------------------------------

PLANNER_MAX_TOKENS = 350
RERANK_MAX_TOKENS = 500
FINAL_MAX_TOKENS = 700

# If multiple requests together hit the rolling TPM limit,
# wait and retry once.
RATE_LIMIT_RETRY_SECONDS = 65


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:

    raise RuntimeError(
        "GROQ_API_KEY not found in .env"
    )


# ============================================================
# GROQ CALL WITH ONE RATE-LIMIT RETRY
# ============================================================

def groq_chat(
    client,
    *,
    messages,
    reasoning_effort,
    temperature,
    max_completion_tokens
):
    """
    Make one Groq request.

    If our individual request is small enough but the rolling
    tokens-per-minute allowance has already been consumed by
    earlier calls, wait once and retry.
    """

    for attempt in range(2):

        try:

            return client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                reasoning_effort=reasoning_effort,
                include_reasoning=False,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens
            )

        except Exception as exc:

            message = str(exc).lower()

            is_rate_limit = (
                "rate_limit_exceeded" in message
                or
                "tokens per minute" in message
                or
                "tpm" in message
            )

            if (
                not is_rate_limit
                or attempt == 1
            ):

                raise

            print(
                "\nGroq rate limit reached."
            )

            print(
                f"Waiting {RATE_LIMIT_RETRY_SECONDS} seconds "
                "and retrying once..."
            )

            time.sleep(
                RATE_LIMIT_RETRY_SECONDS
            )

    raise RuntimeError(
        "Groq request failed."
    )


# ============================================================
# LOAD RAG CORPUS
# ============================================================

def load_documents():

    documents = []

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            documents.append(
                json.loads(line)
            )

    return documents


# ============================================================
# JSON EXTRACTION FROM GPT RESPONSE
# ============================================================

def extract_json(text):

    if not text:

        raise ValueError(
            "Model returned an empty response."
        )

    text = text.strip()

    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```JSON",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if (
        start == -1
        or end == -1
    ):

        raise ValueError(
            "No JSON object found in model response."
        )

    json_text = text[
        start:end + 1
    ]

    try:

        return json.loads(
            json_text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Invalid JSON returned by model:\n"
            + json_text
        ) from exc


# ============================================================
# SOURCE LABEL
# ============================================================

def source_label(metadata):

    source_type = metadata.get(
        "source_type",
        "unknown"
    )

    # --------------------------------------------------------
    # CENTRAL ACT
    # --------------------------------------------------------

    if source_type == "central_act":

        title = (
            metadata.get("title")
            or
            "Unknown Act"
        )

        section = (
            metadata.get("section")
            or
            "Unknown Section"
        )

        return (
            f"{title}, {section}"
        )

    # --------------------------------------------------------
    # CONSTITUTION
    # --------------------------------------------------------

    if source_type == "constitution":

        page = metadata.get(
            "pdf_page"
        )

        return (
            "Constitution of India, "
            f"PDF page {page}"
        )

    # --------------------------------------------------------
    # SUPREME COURT
    # --------------------------------------------------------

    if source_type == "supreme_court":

        title = (
            metadata.get("title")
            or
            metadata.get("document_id")
            or
            "Unknown Supreme Court case"
        )

        citation = metadata.get(
            "citation"
        )

        if citation:

            citation = " ".join(
                str(citation).split()
            )

            return (
                "Supreme Court of India, "
                f"{title}, {citation}"
            )

        return (
            "Supreme Court of India, "
            f"{title}"
        )

    return str(
        source_type
    )


# ============================================================
# ORIGINAL DOCUMENT ID
# ============================================================

def get_original_document_id(
    document
):
    """
    All RAG chunks contain their original document_id.

    This allows us to stop one long Supreme Court judgment from
    occupying many reranker positions.
    """

    metadata = document.get(
        "metadata",
        {}
    )

    return str(
        metadata.get("document_id")
        or document.get("document_id")
        or document.get("id")
    )


# ============================================================
# QUERY PLANNER
# ============================================================

def create_search_plan(
    user_query,
    client
):

    # Deliberately shorter than the previous prompt to reduce TPM.
    system_prompt = """
You are the query planner for an Indian legal RAG system.

Do NOT answer the legal problem.

Understand the facts and create exactly FOUR concise search queries
for an Indian legal database containing statutes, the Constitution
and Supreme Court judgments.

Rules:

1. Never invent facts.
2. Never reverse the parties.
3. Do not automatically turn a civil dispute into a criminal one.
4. Do NOT guess an Act, section, article, rule, case or citation
   unless the USER explicitly named it.
5. Search using legal concepts and factual relationships.
6. Never invent legislation.

Example:

User:
"My friend borrowed money and refuses to repay."

Good queries:
- private loan repayment civil recovery creditor debtor
- recovery of money lent between private individuals
- contractual obligation to repay borrowed money
- civil remedies for non repayment of personal loan

Return JSON ONLY:

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

    response = groq_chat(
        client,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_query
            }
        ],

        reasoning_effort="low",

        temperature=0.0,

        max_completion_tokens=
            PLANNER_MAX_TOKENS
    )

    raw = (
        response
        .choices[0]
        .message
        .content
    )

    plan = extract_json(
        raw
    )

    search_queries = plan.get(
        "search_queries"
    )

    if not isinstance(
        search_queries,
        list
    ):

        raise ValueError(
            "Planner did not return "
            "search_queries as a list."
        )

    search_queries = [
        str(query).strip()
        for query in search_queries
        if str(query).strip()
    ]

    if len(search_queries) != 4:

        raise ValueError(
            "Planner must return exactly "
            "4 search queries. "
            f"Received {len(search_queries)}."
        )

    plan[
        "search_queries"
    ] = search_queries

    return plan


# ============================================================
# BGE SEARCH FOR ONE QUERY
# ============================================================

def search_one_query(
    query,
    embedding_model,
    embeddings
):

    retrieval_query = (
        QUERY_PREFIX
        + query
    )

    query_embedding = (
        embedding_model.encode(
            retrieval_query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        .astype(
            np.float32
        )
    )

    # Both document and query vectors are normalized.
    # Dot product therefore acts as cosine similarity.
    scores = (
        embeddings
        @ query_embedding
    )

    top_k = min(
        TOP_PER_QUERY,
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
            "index":
                int(index),

            "score":
                float(
                    scores[index]
                ),

            "rank":
                rank,

            "query":
                query
        })

    return results


# ============================================================
# MULTI-QUERY RETRIEVAL + RRF
# ============================================================

def multi_query_retrieve(
    original_query,
    search_queries,
    embedding_model,
    embeddings
):

    # Search:
    # 1 original user query
    # +
    # 4 GPT-created legal search queries
    all_queries = [
        original_query
    ]

    all_queries.extend(
        search_queries
    )

    merged = defaultdict(
        lambda: {
            "rrf_score": 0.0,
            "best_similarity": -1.0,
            "query_hits": 0,
            "matched_queries": []
        }
    )

    for query in all_queries:

        results = search_one_query(
            query,
            embedding_model,
            embeddings
        )

        for result in results:

            index = result[
                "index"
            ]

            rank = result[
                "rank"
            ]

            similarity = result[
                "score"
            ]

            merged[
                index
            ][
                "rrf_score"
            ] += (
                1.0
                /
                (
                    RRF_K
                    + rank
                )
            )

            merged[
                index
            ][
                "best_similarity"
            ] = max(
                merged[
                    index
                ][
                    "best_similarity"
                ],
                similarity
            )

            merged[
                index
            ][
                "query_hits"
            ] += 1

            merged[
                index
            ][
                "matched_queries"
            ].append(
                query
            )

    candidates = []

    for index, data in merged.items():

        candidates.append({
            "index":
                index,

            "rrf_score":
                data[
                    "rrf_score"
                ],

            "best_similarity":
                data[
                    "best_similarity"
                ],

            "query_hits":
                data[
                    "query_hits"
                ],

            "matched_queries":
                data[
                    "matched_queries"
                ]
        })

    candidates.sort(
        key=lambda item: (
            item[
                "rrf_score"
            ],
            item[
                "best_similarity"
            ]
        ),
        reverse=True
    )

    # IMPORTANT:
    # Do NOT cut to 12 yet.
    #
    # We keep 40 so that diversification has enough different
    # judgments/statutes to choose from.
    return candidates[
        :FUSED_POOL_SIZE
    ]


# ============================================================
# SOURCE / DOCUMENT DIVERSIFICATION
# ============================================================

def diversify_candidates(
    candidates,
    documents,
    max_candidates=
        RERANK_CANDIDATES
):
    """
    Keep at most one chunk from each original legal document.

    Example:

    If Supreme Court Judgment A contributes 7 high-ranking chunks,
    only its strongest one is retained for the reranker.

    This gives GPT-OSS a broader set of legal authorities.
    """

    diversified = []

    seen_documents = set()

    for candidate in candidates:

        document = documents[
            candidate["index"]
        ]

        document_id = (
            get_original_document_id(
                document
            )
        )

        if document_id in seen_documents:

            continue

        seen_documents.add(
            document_id
        )

        diversified.append(
            candidate
        )

        if (
            len(diversified)
            >= max_candidates
        ):

            break

    return diversified


# ============================================================
# BUILD SMALL RERANKER INPUT
# ============================================================

def build_reranker_candidates(
    candidates,
    documents
):

    blocks = []

    for number, candidate in enumerate(
        candidates,
        start=1
    ):

        document = documents[
            candidate["index"]
        ]

        metadata = document.get(
            "metadata",
            {}
        )

        text = str(
            document.get(
                "text",
                ""
            )
        )

        # CRITICAL TPM REDUCTION:
        #
        # Old version:
        #     text[:2200]
        #
        # New version:
        #     text[:700]
        preview = text[
            :RERANK_PREVIEW_CHARS
        ]

        legal_status = metadata.get(
            "legal_status"
        )

        status_text = (
            str(legal_status)
            if legal_status
            else "not specified"
        )

        block = (
            f"[C{number}]\n"

            f"Type: "
            f"{metadata.get('source_type')}\n"

            f"Source: "
            f"{source_label(metadata)}\n"

            f"Status: "
            f"{status_text}\n"

            f"Similarity: "
            f"{candidate['best_similarity']:.4f}\n"

            f"Text:\n"
            f"{preview}"
        )

        blocks.append(
            block
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# GPT-OSS APPLICABILITY RERANKER
# ============================================================

def rerank_candidates(
    user_query,
    candidates,
    documents,
    client
):

    candidate_text = (
        build_reranker_candidates(
            candidates,
            documents
        )
    )

    # Shorter prompt = lower TPM usage.
    system_prompt = """
You rerank retrieved passages for an Indian legal RAG system.

Do NOT answer the user's question.

Score legal applicability:

3 = directly applicable rule/remedy/holding
2 = materially useful legal principle
1 = weak/background similarity
0 = irrelevant or wrong legal context

Important:

- Semantic similarity is NOT legal applicability.
- Do not reward keyword overlap alone.
- Check whether the parties and transaction actually match.
- Banking/financial-institution law is not automatically applicable
  to a private loan.
- Tax, tenancy or secured-finance law is not relevant merely because
  it contains words like debt, loan, money or recovery.
- Failure to repay money is not automatically criminal.
- A definition section normally does NOT establish a legal remedy.
- Do not infer a right to sue merely because a statute defines
  "loan" or "debt".
- Prefer directly applicable statutes and Supreme Court reasoning.
- If nothing is sufficiently useful, return an empty list.

Return JSON ONLY:

{
  "ranked": [
    {
      "candidate_id": "C1",
      "relevance": 3,
      "reason": "short reason"
    }
  ]
}
"""

    user_prompt = f"""
USER PROBLEM:

{user_query}

CANDIDATES:

{candidate_text}

Rank only genuinely applicable sources.
"""

    response = groq_chat(
        client,

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

        reasoning_effort="low",

        temperature=0.0,

        max_completion_tokens=
            RERANK_MAX_TOKENS
    )

    raw = (
        response
        .choices[0]
        .message
        .content
    )

    data = extract_json(
        raw
    )

    ranked_items = data.get(
        "ranked",
        []
    )

    if not isinstance(
        ranked_items,
        list
    ):

        raise ValueError(
            "Reranker field 'ranked' "
            "must be a list."
        )

    selected = []

    used_indexes = set()

    for item in ranked_items:

        candidate_id = str(
            item.get(
                "candidate_id",
                ""
            )
        ).strip()

        if not candidate_id.startswith(
            "C"
        ):

            continue

        try:

            position = (
                int(
                    candidate_id[1:]
                )
                - 1
            )

        except ValueError:

            continue

        if not (
            0
            <= position
            < len(candidates)
        ):

            continue

        try:

            relevance = int(
                item.get(
                    "relevance",
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            relevance = 0

        # Only score 2 or 3 reaches final generation.
        if relevance < 2:

            continue

        corpus_index = (
            candidates[
                position
            ][
                "index"
            ]
        )

        if corpus_index in used_indexes:

            continue

        selected.append({
            "index":
                corpus_index,

            "relevance":
                relevance,

            "reason":
                str(
                    item.get(
                        "reason",
                        ""
                    )
                ).strip(),

            "retrieval":
                candidates[
                    position
                ]
        })

        used_indexes.add(
            corpus_index
        )

        if (
            len(selected)
            >= FINAL_CONTEXT_K
        ):

            break

    return selected


# ============================================================
# BUILD FINAL CONTEXT
# ============================================================

def build_final_context(
    selected,
    documents
):

    blocks = []

    source_map = []

    for number, item in enumerate(
        selected,
        start=1
    ):

        document = documents[
            item["index"]
        ]

        metadata = document.get(
            "metadata",
            {}
        )

        source_id = (
            f"S{number}"
        )

        label = source_label(
            metadata
        )

        source_type = metadata.get(
            "source_type"
        )

        legal_status = metadata.get(
            "legal_status"
        )

        source_version = metadata.get(
            "source_version"
        )

        additional_metadata = []

        if legal_status:

            additional_metadata.append(
                "Legal-status metadata: "
                + str(
                    legal_status
                )
            )

        if source_version:

            additional_metadata.append(
                "Source-version metadata: "
                + str(
                    source_version
                )
            )

        extra = ""

        if additional_metadata:

            extra = (
                "\n"
                + "\n".join(
                    additional_metadata
                )
            )

        # Unlike reranking, the final generator receives the
        # full retrieved RAG chunk for the selected sources.
        block = (
            f"[{source_id}]\n"

            f"Source: "
            f"{label}\n"

            f"Source type: "
            f"{source_type}"
            f"{extra}\n\n"

            f"{document.get('text', '')}"
        )

        blocks.append(
            block
        )

        source_map.append({
            "source_id":
                source_id,

            "label":
                label,

            "source_type":
                source_type,

            "relevance":
                item[
                    "relevance"
                ],

            "reranker_reason":
                item[
                    "reason"
                ]
        })

    return (
        "\n\n".join(
            blocks
        ),
        source_map
    )


# ============================================================
# CITATION NORMALIZATION
# ============================================================

def normalize_answer_citations(
    answer
):

    if not answer:

        return answer

    # 【S1†L2-L4】 -> [S1]
    answer = re.sub(
        r"【S(\d+)†L\d+(?:-L?\d+)?】",
        r"[S\1]",
        answer
    )

    # 【S1†L2】 -> [S1]
    answer = re.sub(
        r"【S(\d+)†L\d+】",
        r"[S\1]",
        answer
    )

    # 【S1】 -> [S1]
    answer = re.sub(
        r"【S(\d+)】",
        r"[S\1]",
        answer
    )

    return answer


# ============================================================
# FINAL GROUNDED ANSWER
# ============================================================

def generate_final_answer(
    user_query,
    context,
    client
):

    # Shorter prompt than previous version to conserve TPM,
    # while keeping the critical grounding rules.
    system_prompt = """
You are an Indian legal RAG assistant.

Use ONLY the supplied retrieved legal sources for legal claims.

Rules:

1. Do not invent statutes, sections, articles, cases, holdings,
   remedies, limitation periods or procedures.
2. Do not use unsupported pretrained legal knowledge.
3. Support each material legal claim with [S1], [S2], etc.
4. Use ONLY citation format [S1], [S2], etc.
5. Never create fake line citations.
6. Do not infer a remedy merely from a definition section.
7. Do not claim someone can sue "under" an Act unless the retrieved
   text actually supports that proposition.
8. Do not infer interest, damages, notice requirements, limitation,
   jurisdiction or criminal liability unless the sources support it.
9. Do not apply special banking, tax, tenancy or secured-finance law
   to private parties without source support.
10. If a source is historical/unverified, say so.
11. If sources are inadequate, state:
    "The retrieved sources are insufficient to answer this reliably."
12. Do not invent facts.

Use this structure where appropriate:

**Nature of the issue**

**Relevant legal principles**

**Application to the facts**

**Possible next steps / missing information**
"""

    user_prompt = f"""
USER QUESTION:

{user_query}

RETRIEVED SOURCES:

{context}

Answer only from those sources.
"""

    response = groq_chat(
        client,

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

        reasoning_effort="medium",

        temperature=0.0,

        max_completion_tokens=
            FINAL_MAX_TOKENS
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return normalize_answer_citations(
        answer
    )


# ============================================================
# DISPLAY FUSED RETRIEVAL RESULTS
# ============================================================

def display_candidate_summary(
    candidates,
    documents,
    title="Top fused retrieval candidates:"
):

    print(
        f"\n{title}"
    )

    display_count = min(
        10,
        len(candidates)
    )

    for number, candidate in enumerate(
        candidates[
            :display_count
        ],
        start=1
    ):

        document = documents[
            candidate["index"]
        ]

        metadata = document.get(
            "metadata",
            {}
        )

        print(
            f"\n[C{number}] "
            f"similarity="
            f"{candidate['best_similarity']:.4f}"
            f" | hits="
            f"{candidate['query_hits']}"
            f" | rrf="
            f"{candidate['rrf_score']:.4f}"
        )

        print(
            source_label(
                metadata
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "INDIAN LEGAL RAG V2"
    )

    print(
        "BGE-SMALL + GPT-OSS-120B"
    )

    print(
        "=" * 80
    )

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "RAG corpus:",
        CORPUS_FILE
    )

    print(
        "Embeddings:",
        EMBEDDINGS_FILE
    )

    # ========================================================
    # VERIFY FILES
    # ========================================================

    if not CORPUS_FILE.exists():

        raise FileNotFoundError(
            f"RAG corpus not found:\n"
            f"{CORPUS_FILE}"
        )

    if not EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(
            f"Embedding file not found:\n"
            f"{EMBEDDINGS_FILE}"
        )

    # ========================================================
    # LOAD CORPUS
    # ========================================================

    print(
        "\nLoading legal corpus..."
    )

    documents = load_documents()

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    print(
        "Loading embeddings..."
    )

    # mmap avoids eagerly copying the entire ~1.016 GiB matrix
    # into normal process RAM.
    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r"
    )

    print(
        "Documents:",
        f"{len(documents):,}"
    )

    print(
        "Embeddings:",
        embeddings.shape
    )

    if (
        len(documents)
        != embeddings.shape[0]
    ):

        raise ValueError(
            "Corpus and embedding count mismatch.\n"
            f"Documents: "
            f"{len(documents):,}\n"
            f"Embeddings: "
            f"{embeddings.shape[0]:,}"
        )

    if embeddings.shape[1] != 384:

        raise ValueError(
            "Expected BGE-small embedding "
            f"dimension 384, got "
            f"{embeddings.shape[1]}."
        )

    # ========================================================
    # LOAD BGE
    # ========================================================

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
        "Loading BGE embedding model..."
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL,
            device=device
        )
    )

    embedding_model.max_seq_length = (
        512
    )

    # ========================================================
    # GROQ
    # ========================================================

    print(
        "Creating Groq client..."
    )

    client = Groq(
        api_key=GROQ_API_KEY
    )

    print(
        "\nRAG v2 ready."
    )

    print(
        "\nRetrieval configuration:"
    )

    print(
        f"  BGE top/query: "
        f"{TOP_PER_QUERY}"
    )

    print(
        f"  Fused pool: "
        f"{FUSED_POOL_SIZE}"
    )

    print(
        f"  Diverse reranker candidates: "
        f"{RERANK_CANDIDATES}"
    )

    print(
        f"  Final sources: "
        f"{FINAL_CONTEXT_K}"
    )

    # ========================================================
    # INTERACTIVE LOOP
    # ========================================================

    while True:

        print(
            "\n"
            + "=" * 80
        )

        user_query = input(
            "\nEnter legal question "
            "(or 'exit'): "
        ).strip()

        if not user_query:

            continue

        if user_query.lower() in {
            "exit",
            "quit"
        }:

            print(
                "\nClosing RAG v2."
            )

            break

        try:

            # =================================================
            # STEP 1
            # QUERY PLANNING
            # =================================================

            print(
                "\n[1/4] Creating legal search plan..."
            )

            plan = create_search_plan(
                user_query,
                client
            )

            print(
                "\nSearch plan:"
            )

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

            # =================================================
            # STEP 2
            # BGE RETRIEVAL
            # =================================================

            print(
                "\n[2/4] Searching "
                "710K-chunk legal corpus..."
            )

            fused_candidates = (
                multi_query_retrieve(
                    user_query,
                    search_queries,
                    embedding_model,
                    embeddings
                )
            )

            print(
                "Fused candidate chunks:",
                len(
                    fused_candidates
                )
            )

            display_candidate_summary(
                fused_candidates,
                documents
            )

            # =================================================
            # DIVERSIFICATION
            # =================================================

            candidates = (
                diversify_candidates(
                    fused_candidates,
                    documents,
                    max_candidates=
                        RERANK_CANDIDATES
                )
            )

            print(
                "\nAfter document diversification:"
            )

            print(
                f"{len(candidates)} "
                "different legal documents "
                "will be sent to GPT-OSS."
            )

            # =================================================
            # STEP 3
            # GPT-OSS RERANKING
            # =================================================

            print(
                "\n[3/4] Reranking for "
                "actual legal applicability..."
            )

            selected = (
                rerank_candidates(
                    user_query,
                    candidates,
                    documents,
                    client
                )
            )

            # -------------------------------------------------
            # NOTHING RELIABLE RETRIEVED
            # -------------------------------------------------

            if not selected:

                print(
                    "\n"
                    + "=" * 80
                )

                print(
                    "INSUFFICIENT RETRIEVAL"
                )

                print(
                    "=" * 80
                )

                print(
                    "\nThe retrieved sources are "
                    "insufficient to answer this reliably."
                )

                print(
                    "\nNo source received enough "
                    "legal-applicability confidence."
                )

                print(
                    "\nThe system intentionally refuses "
                    "to manufacture an answer."
                )

                continue

            # -------------------------------------------------
            # DISPLAY SELECTED SOURCES
            # -------------------------------------------------

            print(
                "\nSelected legal sources:"
            )

            for number, item in enumerate(
                selected,
                start=1
            ):

                document = documents[
                    item["index"]
                ]

                metadata = document.get(
                    "metadata",
                    {}
                )

                print(
                    f"\n[S{number}] "
                    f"relevance="
                    f"{item['relevance']}"
                )

                print(
                    source_label(
                        metadata
                    )
                )

                print(
                    "Reranker reason:",
                    item[
                        "reason"
                    ]
                )

            # =================================================
            # STEP 4
            # FINAL GROUNDED ANSWER
            # =================================================

            (
                context,
                source_map
            ) = build_final_context(
                selected,
                documents
            )

            print(
                "\n[4/4] Generating "
                "grounded answer..."
            )

            answer = (
                generate_final_answer(
                    user_query,
                    context,
                    client
                )
            )

            print(
                "\n"
                + "=" * 80
            )

            print(
                "GROUNDED RAG ANSWER"
            )

            print(
                "=" * 80
            )

            print(
                answer
            )

            print(
                "\n"
                + "-" * 80
            )

            print(
                "SOURCE MAP"
            )

            print(
                "-" * 80
            )

            for source in source_map:

                print(
                    f"[{source['source_id']}] "
                    f"{source['label']}"
                )

                print(
                    f"    relevance="
                    f"{source['relevance']}"
                )

                print(
                    "    reranker="
                    f"{source['reranker_reason']}"
                )

        except Exception as exc:

            print(
                "\n"
                + "=" * 80
            )

            print(
                "ERROR"
            )

            print(
                "=" * 80
            )

            print(
                type(exc).__name__
            )

            print(
                exc
            )


if __name__ == "__main__":
    main()