import os
import json
import re
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

PROJECT_ROOT = Path(r"D:\Projects\LLM_training")

ILSIC_INDEX_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "data"
    / "rag"
    / "ilsic"
)

SCENARIOS_FILE = (
    ILSIC_INDEX_DIR
    / "ilsic_train_scenarios.jsonl"
)

CHUNKS_FILE = (
    ILSIC_INDEX_DIR
    / "ilsic_train_chunks.jsonl"
)

EMBEDDINGS_FILE = (
    ILSIC_INDEX_DIR
    / "ilsic_train_embeddings.npy"
)


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

LLM_MODEL = "openai/gpt-oss-120b"

FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b"
]


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

# Retrieve a reasonably broad set from every representation.
TOP_CHUNKS_PER_QUERY = 60

# Before GPT reranking.
PRE_RERANK_SCENARIOS = 14

# After GPT factual-analogy reranking.
FINAL_ANALOGOUS_SCENARIOS = 5

# Maximum search representations:
#
# 1. original user wording
# 2. normalized scenario
# 3-4. GPT expansions
MAX_SEARCH_QUERIES = 4

MAX_CANDIDATE_LAWS = 10

SCENARIO_PREVIEW_CHARS = 500

VERIFY_SCENARIO_PREVIEW_CHARS = 350


# ============================================================
# CONVERSATION SETTINGS
# ============================================================

MAX_TOTAL_FOLLOWUPS = 5

MAX_RESEARCH_ITERATIONS = 3


# ============================================================
# GROQ SETTINGS
# ============================================================

INTAKE_MAX_TOKENS = 550

RERANK_MAX_TOKENS = 800

VERIFY_MAX_TOKENS = 1400

JSON_REPAIR_MAX_TOKENS = 800

RATE_LIMIT_WAIT_SECONDS = 65


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY was not found in "
        f"{PROJECT_ROOT / '.env'}"
    )


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(value):
    text = str(value or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def shorten(text, limit):
    text = normalize_text(text)

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def normalize_question(text):
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json_object(text):
    """
    Robustly find the first valid JSON object in model output.

    Handles:
    - plain JSON
    - ```json fenced JSON
    - extra prose around JSON
    """

    if not text:
        raise ValueError(
            "Model returned an empty response."
        )

    cleaned = text.strip()

    cleaned = cleaned.replace(
        "```json",
        ""
    )

    cleaned = cleaned.replace(
        "```JSON",
        ""
    )

    cleaned = cleaned.replace(
        "```",
        ""
    )

    cleaned = cleaned.strip()

    # --------------------------------------------------------
    # FIRST TRY: whole response
    # --------------------------------------------------------

    try:
        value = json.loads(cleaned)

        if isinstance(value, dict):
            return value

    except Exception:
        pass

    # --------------------------------------------------------
    # SECOND TRY:
    # scan for first decodable JSON object
    # --------------------------------------------------------

    decoder = json.JSONDecoder()

    for index, char in enumerate(cleaned):

        if char != "{":
            continue

        try:
            value, _ = decoder.raw_decode(
                cleaned[index:]
            )

            if isinstance(value, dict):
                return value

        except Exception:
            continue

    raise ValueError(
        "No valid JSON object found "
        "in model response."
    )


# ============================================================
# GROQ TEXT CALL
# ============================================================

def groq_text(
    client,
    messages,
    max_completion_tokens,
    reasoning_effort="low"
):
    """
    Make one Groq call with automatic model fallback and rate-limit handling.
    """
    models_to_try = [LLM_MODEL] + [m for m in FALLBACK_MODELS if m != LLM_MODEL]
    last_exc = None

    for model in models_to_try:
        for attempt in range(2):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_completion_tokens": max_completion_tokens
                }
                if "gpt-oss" in model:
                    kwargs["reasoning_effort"] = reasoning_effort
                    kwargs["include_reasoning"] = False

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return content or ""

            except Exception as exc:
                last_exc = exc
                error_text = str(exc).lower()

                rate_limited = any(
                    phrase in error_text
                    for phrase in [
                        "rate_limit",
                        "tokens per minute",
                        "tokens per day",
                        "tpd",
                        "tpm",
                        "429"
                    ]
                )

                if "tokens per day" in error_text or "tpd" in error_text:
                    # Daily token quota exhausted for this model - immediately try fallback model
                    print(f"[Groq Fallback] Model {model} daily token quota exhausted. Switching to fallback model...")
                    break

                if not rate_limited or attempt == 1:
                    break

                print(f"[Groq] Rate limit reached for {model}. Retrying in 4 seconds...")
                time.sleep(4)

    if last_exc:
        raise last_exc
    raise RuntimeError("All Groq model attempts failed.")


# ============================================================
# JSON CALL WITH AUTOMATIC REPAIR
# ============================================================

def groq_json(
    client,
    messages,
    max_completion_tokens,
    required_keys=None,
    label="JSON response"
):
    """
    First ask normally.

    If GPT-OSS returns malformed/empty JSON,
    make one small repair request instead of
    crashing the program.
    """

    raw = groq_text(
        client=client,
        messages=messages,
        max_completion_tokens=(
            max_completion_tokens
        ),
        reasoning_effort="low"
    )

    try:
        data = extract_json_object(raw)

        if required_keys:

            missing = [
                key
                for key in required_keys
                if key not in data
            ]

            if missing:
                raise ValueError(
                    f"Missing keys: {missing}"
                )

        return data

    except Exception as first_error:

        print()
        print(
            f"{label} was not valid JSON."
        )

        print(
            "Attempting automatic JSON repair..."
        )

        # Prevent an unexpectedly huge malformed
        # response from bloating the repair call.
        raw_preview = raw[:5000]

        repair_system = """
Return ONLY one valid JSON object.

Do not use Markdown.
Do not use code fences.
Do not explain anything outside the JSON.
Do not invent information that was not present in the
original response.
"""

        repair_user = f"""
The previous model response was supposed to be JSON but
could not be parsed.

Repair it into one valid JSON object.

BROKEN RESPONSE:

{raw_preview}
"""

        repaired_raw = groq_text(
            client=client,
            messages=[
                {
                    "role": "system",
                    "content": repair_system
                },
                {
                    "role": "user",
                    "content": repair_user
                }
            ],
            max_completion_tokens=(
                JSON_REPAIR_MAX_TOKENS
            ),
            reasoning_effort="low"
        )

        try:
            data = extract_json_object(
                repaired_raw
            )

            if required_keys:

                missing = [
                    key
                    for key in required_keys
                    if key not in data
                ]

                if missing:
                    raise ValueError(
                        f"Missing keys after repair: "
                        f"{missing}"
                    )

            return data

        except Exception as second_error:

            raise ValueError(
                f"{label} failed twice.\n"
                f"First error: {first_error}\n"
                f"Repair error: {second_error}\n"
                f"Original response preview:\n"
                f"{raw_preview}"
            )


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):
    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# LOAD ILSIC INDEX
# ============================================================

def load_ilsic_index():

    print(
        "\nLoading ILSIC scenarios..."
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
        "Loading ILSIC embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r"
    )

    if (
        len(chunks)
        != embeddings.shape[0]
    ):

        raise ValueError(
            "Chunk / embedding mismatch.\n"
            f"Chunks: {len(chunks)}\n"
            f"Embeddings: "
            f"{embeddings.shape[0]}"
        )

    scenario_map = {
        item["scenario_id"]: item
        for item in scenarios
    }

    print(
        "Scenarios:",
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
        scenario_map,
        chunks,
        embeddings
    )


# ============================================================
# CONVERSATION FORMATTER
# ============================================================

def format_conversation(
    original_problem,
    followups
):

    lines = [
        "INITIAL USER PROBLEM:",
        original_problem
    ]

    if followups:

        lines.append(
            "\nFOLLOW-UP INFORMATION:"
        )

    for number, item in enumerate(
        followups,
        start=1
    ):

        lines.append(
            f"\nQuestion {number}: "
            f"{item['question']}"
        )

        lines.append(
            f"User answer {number}: "
            f"{item['answer']}"
        )

    return "\n".join(lines)


# ============================================================
# INTAKE AGENT
# ============================================================

def analyze_intake(
    original_problem,
    followups,
    client,
    questions_already_asked
):

    conversation = format_conversation(
        original_problem,
        followups
    )

    system_prompt = """
You are the conversational intake agent for an Indian legal
research prototype.

Your task is NOT to give legal advice and NOT to decide what
statute applies.

Your task is to obtain enough factual information to search
for analogous legal situations.

Ask at most ONE question at a time.

Ask a question only when the missing fact could materially
change:
- the legal relationship;
- civil versus criminal characterization;
- a legal remedy;
- an important factual element;
- the kind of statute that should be researched.

Never invent facts.

For loan or money disputes:
- do not assume fraud;
- do not assume dishonest intention;
- do not assume entrustment;
- do not assume cheque dishonour;
- do not assume a bank or financial institution is involved.

When enough facts exist, create a concise factual
normalization.

IMPORTANT FOR RETRIEVAL:
Preserve concrete party relationships.

For example:
"friend lent money to friend who has not repaid"
is better than merely saying
"personal loan default".

Do not turn a private loan into a bank loan.

Generate at most TWO additional retrieval queries.
They must preserve the known party types and facts.

Do NOT name an Act, statute, section or court case.

Return JSON only.

ASK schema:

{
  "status": "ASK",
  "known_facts": ["..."],
  "missing_material_facts": ["..."],
  "next_question": "...",
  "normalized_scenario": "",
  "retrieval_queries": []
}

READY schema:

{
  "status": "READY",
  "known_facts": ["..."],
  "missing_material_facts": ["..."],
  "next_question": "",
  "normalized_scenario": "...",
  "retrieval_queries": [
    "...",
    "..."
  ]
}
"""

    user_prompt = f"""
FACTS:

{conversation}

Questions already asked:
{questions_already_asked}

Decide whether another material factual question is required.

Return JSON only.
"""

    data = groq_json(
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
            INTAKE_MAX_TOKENS
        ),
        required_keys=["status"],
        label="Intake response"
    )

    status = normalize_text(
        data.get("status")
    ).upper()

    if status not in {
        "ASK",
        "READY"
    }:
        raise ValueError(
            f"Invalid intake status: {status}"
        )

    return data


# ============================================================
# BUILD SEARCH REPRESENTATIONS
# ============================================================

def build_search_queries(
    original_problem,
    normalized_scenario,
    retrieval_queries
):
    """
    CRITICAL FIX:

    Query #1 is ALWAYS the user's original wording.

    GPT-normalized language is only an additional search
    representation. It can never replace the original query.
    """

    candidates = [
        normalize_text(original_problem),
        normalize_text(normalized_scenario)
    ]

    for query in retrieval_queries or []:

        candidates.append(
            normalize_text(query)
        )

    final_queries = []

    seen = set()

    for query in candidates:

        if not query:
            continue

        key = query.lower()

        if key in seen:
            continue

        seen.add(key)
        final_queries.append(query)

        if (
            len(final_queries)
            >= MAX_SEARCH_QUERIES
        ):
            break

    return final_queries


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(
    query,
    embedding_model
):

    prepared = (
        QUERY_PREFIX
        + query
    )

    vector = embedding_model.encode(
        prepared,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return vector.astype(
        np.float32
    )


# ============================================================
# SEARCH ONE QUERY
# ============================================================

def search_one_query(
    query,
    embedding_model,
    embeddings
):

    query_vector = embed_query(
        query,
        embedding_model
    )

    scores = (
        embeddings
        @ query_vector
    )

    top_k = min(
        TOP_CHUNKS_PER_QUERY,
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
            "index": int(index),
            "rank": rank,
            "similarity": float(
                scores[index]
            )
        })

    return results


# ============================================================
# MULTI-QUERY RETRIEVAL
# ============================================================

def retrieve_candidate_scenarios(
    search_queries,
    embedding_model,
    scenario_map,
    chunks,
    embeddings
):
    """
    Important changes from previous version:

    1. Original query is query index 0.
    2. Each scenario keeps only its BEST chunk PER query.
    3. distinct_query_hits counts different queries, not chunks.
    4. Original-query similarity is the dominant ranking signal.
    5. Expansion queries can help, but cannot overpower a strong
       direct match.
    """

    scenario_hits = defaultdict(
        lambda: {
            "per_query_similarity": {},
            "per_query_rank": {},
            "best_similarity": -1.0,
            "best_chunk": "",
            "best_query_index": None
        }
    )

    for query_index, query in enumerate(
        search_queries
    ):

        results = search_one_query(
            query=query,
            embedding_model=embedding_model,
            embeddings=embeddings
        )

        for result in results:

            chunk = chunks[
                result["index"]
            ]

            scenario_id = chunk[
                "scenario_id"
            ]

            similarity = result[
                "similarity"
            ]

            rank = result[
                "rank"
            ]

            info = scenario_hits[
                scenario_id
            ]

            previous = (
                info[
                    "per_query_similarity"
                ].get(
                    query_index
                )
            )

            # Only one best chunk per scenario per query.
            if (
                previous is None
                or similarity > previous
            ):

                info[
                    "per_query_similarity"
                ][query_index] = similarity

                info[
                    "per_query_rank"
                ][query_index] = rank

            if (
                similarity
                >
                info[
                    "best_similarity"
                ]
            ):

                info[
                    "best_similarity"
                ] = similarity

                info[
                    "best_chunk"
                ] = chunk["text"]

                info[
                    "best_query_index"
                ] = query_index

    candidates = []

    for scenario_id, info in (
        scenario_hits.items()
    ):

        scenario = scenario_map.get(
            scenario_id
        )

        if not scenario:
            continue

        per_query = info[
            "per_query_similarity"
        ]

        original_similarity = (
            per_query.get(0)
        )

        best_similarity = info[
            "best_similarity"
        ]

        distinct_query_hits = len(
            per_query
        )

        # ----------------------------------------------------
        # RETRIEVAL SCORE
        #
        # Original wording dominates.
        #
        # Expansion queries can improve the score slightly,
        # but cannot promote a generic bank-loan result above
        # an excellent direct "friend loan" match.
        # ----------------------------------------------------

        if original_similarity is not None:

            expansion_gain = max(
                0.0,
                best_similarity
                - original_similarity
            )

            expansion_gain = min(
                expansion_gain,
                0.025
            )

            diversity_bonus = min(
                max(
                    distinct_query_hits - 1,
                    0
                )
                * 0.004,
                0.012
            )

            retrieval_score = (
                original_similarity
                + expansion_gain
                + diversity_bonus
            )

        else:

            # Scenario was only discovered through an expansion.
            #
            # It is still allowed, but gets a penalty because
            # it failed to appear among the original query's
            # strongest results.
            diversity_bonus = min(
                max(
                    distinct_query_hits - 1,
                    0
                )
                * 0.004,
                0.008
            )

            retrieval_score = (
                best_similarity
                - 0.035
                + diversity_bonus
            )

        candidates.append({
            "scenario_id":
                scenario_id,

            "scenario":
                scenario["scenario"],

            "statutes":
                scenario["statutes"],

            "best_chunk":
                info["best_chunk"],

            "original_similarity":
                original_similarity,

            "best_similarity":
                best_similarity,

            "distinct_query_hits":
                distinct_query_hits,

            "retrieval_score":
                float(
                    retrieval_score
                )
        })

    candidates.sort(
        key=lambda item: (
            item["retrieval_score"],
            (
                item["original_similarity"]
                if item[
                    "original_similarity"
                ] is not None
                else -1.0
            ),
            item["best_similarity"]
        ),
        reverse=True
    )

    return candidates[
        :PRE_RERANK_SCENARIOS
    ]


# ============================================================
# DISPLAY RAW RETRIEVAL
# ============================================================

def display_raw_retrieval(
    candidates
):

    print()
    print(
        "=" * 80
    )

    print(
        "ILSIC RETRIEVAL CANDIDATES"
    )

    print(
        "=" * 80
    )

    for number, item in enumerate(
        candidates,
        start=1
    ):

        original_similarity = item[
            "original_similarity"
        ]

        if original_similarity is None:
            original_text = "not in original top results"
        else:
            original_text = (
                f"{original_similarity:.4f}"
            )

        print()
        print(
            f"[C{number}] "
            f"score="
            f"{item['retrieval_score']:.4f}"
            f" | original="
            f"{original_text}"
            f" | best="
            f"{item['best_similarity']:.4f}"
            f" | query_hits="
            f"{item['distinct_query_hits']}"
        )

        print(
            shorten(
                item["scenario"],
                360
            )
        )


# ============================================================
# SCENARIO RERANKER INPUT
# ============================================================

def build_reranker_candidates(
    candidates
):

    blocks = []

    for number, item in enumerate(
        candidates,
        start=1
    ):

        blocks.append(
            f"[C{number}]\n"
            f"Scenario:\n"
            f"{shorten(item['scenario'], SCENARIO_PREVIEW_CHARS)}"
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# GPT FACTUAL-ANALOGY RERANKER
# ============================================================

def rerank_analogous_scenarios(
    original_problem,
    followups,
    normalized_scenario,
    candidates,
    client
):

    conversation = format_conversation(
        original_problem,
        followups
    )

    candidate_text = (
        build_reranker_candidates(
            candidates
        )
    )

    system_prompt = """
You are the FACTUAL ANALOGY RERANKER for an Indian legal
research system.

You are given the current user's facts and retrieved historical
layperson legal scenarios.

Your ONLY job is to determine which retrieved scenarios are
factually and legally analogous enough to help generate
candidate authorities.

Do NOT decide the final law.
Do NOT give legal advice.
Do NOT add statutes.

SCORING:

3 = STRONGLY ANALOGOUS
Same core party relationship, transaction/dispute and legal
problem.

2 = MATERIALLY ANALOGOUS
Core issue is similar, although there are some additional facts.

1 = WEAK
Shares vocabulary or a broad topic, but important legal context
or party relationship differs.

0 = IRRELEVANT
Different transaction, different legal regime, or materially
different party relationship.

CRITICAL EXAMPLES:

- A private friend-to-friend loan is NOT the same as a
  bank loan, secured loan, NPA, SARFAESI dispute or DRT case.

- A cheque case is only partially analogous when the current
  user has not said a cheque exists.

- A tenancy dispute is not analogous merely because money
  is unpaid.

- A commercial supplier dispute is weaker than a private
  friend loan when the current transaction is between friends.

Do not reward a scenario for containing extra facts that the
current user has never stated.

Return JSON only:

{
  "results": [
    {
      "candidate_id": "C1",
      "score": 3,
      "reason": "..."
    }
  ]
}

Evaluate every supplied candidate.
"""

    user_prompt = f"""
CURRENT USER FACTS:

{conversation}

NORMALIZED SCENARIO:

{normalized_scenario}

RETRIEVED CANDIDATES:

{candidate_text}

Score factual analogy only.

Return JSON only.
"""

    data = groq_json(
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
            RERANK_MAX_TOKENS
        ),
        required_keys=["results"],
        label="Scenario reranker"
    )

    raw_results = data.get(
        "results",
        []
    )

    score_map = {}

    for result in raw_results:

        candidate_id = normalize_text(
            result.get(
                "candidate_id",
                ""
            )
        ).upper()

        try:
            score = int(
                result.get(
                    "score",
                    0
                )
            )
        except Exception:
            score = 0

        score = max(
            0,
            min(
                score,
                3
            )
        )

        score_map[
            candidate_id
        ] = {
            "score": score,
            "reason": normalize_text(
                result.get(
                    "reason",
                    ""
                )
            )
        }

    reranked = []

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        candidate_id = f"C{index}"

        model_result = score_map.get(
            candidate_id,
            {
                "score": 0,
                "reason":
                    "No reranker result returned."
            }
        )

        item = dict(candidate)

        item[
            "candidate_id"
        ] = candidate_id

        item[
            "analogy_score"
        ] = model_result[
            "score"
        ]

        item[
            "analogy_reason"
        ] = model_result[
            "reason"
        ]

        reranked.append(
            item
        )

    # GPT factual score dominates.
    #
    # Retrieval similarity breaks ties.
    reranked.sort(
        key=lambda item: (
            item["analogy_score"],
            item["retrieval_score"]
        ),
        reverse=True
    )

    accepted = [
        item
        for item in reranked
        if item["analogy_score"] >= 2
    ]

    return (
        reranked,
        accepted[
            :FINAL_ANALOGOUS_SCENARIOS
        ]
    )


# ============================================================
# DISPLAY RERANKING
# ============================================================

def display_reranking(
    reranked,
    accepted
):

    print()
    print(
        "=" * 80
    )

    print(
        "GPT-OSS FACTUAL-ANALOGY RERANKING"
    )

    print(
        "=" * 80
    )

    accepted_ids = {
        item["scenario_id"]
        for item in accepted
    }

    for item in reranked:

        if (
            item["scenario_id"]
            in accepted_ids
        ):

            status = "SELECTED"

        elif (
            item["analogy_score"]
            >= 2
        ):

            # Good analogy, but excluded because
            # only the strongest N scenarios are
            # allowed to contribute statute labels.
            status = "NOT SELECTED"

        else:

            status = "REJECT"

        print()
        print(
            f"[{item['candidate_id']}] "
            f"{status}"
            f" | analogy="
            f"{item['analogy_score']}/3"
            f" | retrieval="
            f"{item['retrieval_score']:.4f}"
        )

        print(
            shorten(
                item["scenario"],
                330
            )
        )

        print(
            "Reason:",
            item["analogy_reason"]
        )

# ============================================================
# AGGREGATE LAWS FROM ACCEPTED SCENARIOS ONLY
# ============================================================

def aggregate_candidate_laws(
    accepted
):

    law_data = defaultdict(
        lambda: {
            "votes": 0,
            "weight": 0.0,
            "supporting_candidates": [],
            "supporting_scenario_ids": []
        }
    )

    for item in accepted:

        # Rerank score matters much more
        # than raw vector similarity.
        scenario_weight = (
            float(
                item["analogy_score"]
            )
            +
            float(
                item["retrieval_score"]
            )
        )

        for authority in item[
            "statutes"
        ]:

            info = law_data[
                authority
            ]

            info["votes"] += 1

            info["weight"] += (
                scenario_weight
            )

            info[
                "supporting_candidates"
            ].append(
                item["candidate_id"]
            )

            info[
                "supporting_scenario_ids"
            ].append(
                item["scenario_id"]
            )

    results = []

    for authority, info in (
        law_data.items()
    ):

        results.append({
            "authority": authority,
            **info
        })

    results.sort(
        key=lambda item: (
            item["votes"],
            item["weight"]
        ),
        reverse=True
    )

    return results[
        :MAX_CANDIDATE_LAWS
    ]


# ============================================================
# DISPLAY AGGREGATED LAWS
# ============================================================

def display_candidate_laws(
    candidate_laws
):

    print()
    print(
        "=" * 80
    )

    print(
        "CANDIDATE AUTHORITIES FROM ANALOGOUS SCENARIOS"
    )

    print(
        "=" * 80
    )

    if not candidate_laws:

        print(
            "\nNo candidate authorities."
        )

        return

    for number, item in enumerate(
        candidate_laws,
        start=1
    ):

        print()
        print(
            f"[L{number}] "
            f"votes={item['votes']}"
            f" | support="
            f"{', '.join(item['supporting_candidates'])}"
        )

        print(
            item["authority"]
        )


# ============================================================
# VERIFIER INPUT
# ============================================================

def build_verifier_input(
    accepted,
    candidate_laws
):

    scenario_blocks = []

    for item in accepted:

        scenario_blocks.append(
            f"[{item['candidate_id']}]\n"
            f"{shorten(item['scenario'], VERIFY_SCENARIO_PREVIEW_CHARS)}"
        )

    law_blocks = []

    for number, item in enumerate(
        candidate_laws,
        start=1
    ):

        law_blocks.append(
            f"[L{number}] "
            f"{item['authority']}\n"
            f"Supported by analogous scenarios: "
            f"{', '.join(item['supporting_candidates'])}"
        )

    return (
        "\n\n".join(scenario_blocks),
        "\n\n".join(law_blocks)
    )


# ============================================================
# CANDIDATE-LAW VERIFIER
# ============================================================

# def verify_candidate_laws(
#     original_problem,
#     followups,
#     normalized_scenario,
#     accepted,
#     candidate_laws,
#     client
# ):

#     if not candidate_laws:

#         return {
#             "candidate_laws": [],
#             "needs_more_information": False,
#             "next_question": ""
#         }

#     conversation = format_conversation(
#         original_problem,
#         followups
#     )

#     (
#         scenario_text,
#         law_text
#     ) = build_verifier_input(
#         accepted,
#         candidate_laws
#     )

#     system_prompt = """
# You are the CANDIDATE AUTHORITY VERIFIER for an Indian legal
# research system.

# The supplied statute sections come from analogous historical
# dataset scenarios.

# They are NOT automatically applicable law.

# Your task is only to decide whether each candidate authority
# is worth searching in the authoritative legal corpus.

# Decisions:

# KEEP
# There is enough factual basis for this legal theory to justify
# retrieving and verifying the authority.

# REJECT
# The authority depends on facts or a legal relationship that the
# current user does not have.

# NEEDS_FACT
# The authority could become relevant, but one specific material
# fact is missing.

# STRICT RULES:

# - Simple non-payment of a loan does not by itself prove cheating.
# - Do not infer dishonest intention at the beginning of a
#   transaction unless facts support that inference.
# - Do not infer criminal breach of trust merely because money
#   was lent and not repaid.
# - Do not infer cheque dishonour without a cheque.
# - Do not infer banking, secured finance, tenancy, employment,
#   matrimonial or other special relationships.
# - Dataset voting is NOT proof of applicability.
# - Do not add any authority that is not in the supplied list.
# - Historical IPC/CrPC or other old-code labels may be useful
#   research clues, but must later undergo current-law
#   verification before any final answer.
# - You are not giving the final legal answer.

# If one missing fact is genuinely necessary to distinguish a
# potentially important theory, ask exactly ONE concise question.

# Return JSON only:

# {
#   "candidate_laws": [
#     {
#       "law_id": "L1",
#       "authority": "...",
#       "decision": "KEEP",
#       "reason": "...",
#       "current_law_check_required": true
#     }
#   ],
#   "needs_more_information": false,
#   "next_question": ""
# }

# Return one evaluation for every supplied candidate authority.
# """

#     user_prompt = f"""
# CURRENT USER FACTS:

# {conversation}

# NORMALIZED SCENARIO:

# {normalized_scenario}

# ACCEPTED ANALOGOUS SCENARIOS:

# {scenario_text}

# CANDIDATE AUTHORITIES:

# {law_text}

# Evaluate the candidate authorities.

# Return JSON only.
# """

#     data = groq_json(
#         client=client,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt
#             },
#             {
#                 "role": "user",
#                 "content": user_prompt
#             }
#         ],
#         max_completion_tokens=(
#             VERIFY_MAX_TOKENS
#         ),
#         required_keys=[
#             "candidate_laws",
#             "needs_more_information",
#             "next_question"
#         ],
#         label="Candidate-law verifier"
#     )

#     return data

def verify_candidate_laws(
    original_problem,
    followups,
    normalized_scenario,
    accepted,
    candidate_laws,
    client
):

    if not candidate_laws:

        return {
            "candidate_laws": [],
            "needs_more_information": False,
            "next_question": ""
        }

    conversation = format_conversation(
        original_problem,
        followups
    )

    (
        scenario_text,
        law_text
    ) = build_verifier_input(
        accepted,
        candidate_laws
    )

    system_prompt = """
You are the CANDIDATE AUTHORITY VERIFIER for an Indian legal
research system.

The supplied statute sections come from analogous historical
dataset scenarios.

They are NOT automatically applicable law.

Your task is only to decide whether each candidate authority
is worth searching in the authoritative legal corpus.

Decisions:

KEEP
There is enough factual basis for this legal theory to justify
retrieving and verifying the authority.

REJECT
The authority depends on facts or a legal relationship that the
current user does not have.

NEEDS_FACT
The authority could become relevant, but one specific material
fact is missing.

STRICT RULES:

- Simple non-payment of a loan does not by itself prove cheating.
- Do not infer dishonest intention at the beginning of a
  transaction unless facts support that inference.
- Do not infer criminal breach of trust merely because money
  was lent and not repaid.
- Do not infer cheque dishonour without a cheque.
- Do not infer banking, secured finance, tenancy, employment,
  matrimonial or other special relationships.
- Dataset voting is NOT proof of applicability.
- Do not add any authority that is not in the supplied list.
- Historical IPC/CrPC or other old-code labels may be useful
  research clues, but must later undergo current-law
  verification before any final answer.
- You are not giving the final legal answer.

IMPORTANT OUTPUT RULE:

You MUST return candidate_laws.

The fields needs_more_information and next_question are helpful,
but if there is no useful follow-up question simply use:

"needs_more_information": false,
"next_question": ""

Return JSON only:

{
  "candidate_laws": [
    {
      "law_id": "L1",
      "authority": "...",
      "decision": "KEEP",
      "reason": "...",
      "current_law_check_required": true
    }
  ],
  "needs_more_information": false,
  "next_question": ""
}

Return one evaluation for every supplied candidate authority.
"""

    user_prompt = f"""
CURRENT USER FACTS:

{conversation}

NORMALIZED SCENARIO:

{normalized_scenario}

ACCEPTED ANALOGOUS SCENARIOS:

{scenario_text}

CANDIDATE AUTHORITIES:

{law_text}

Evaluate the candidate authorities.

Return JSON only.
"""

    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # candidate_laws is the ONLY mandatory key.
    #
    # The previous version crashed if GPT correctly evaluated
    # all laws but forgot the two bookkeeping fields at the end.
    # --------------------------------------------------------

    data = groq_json(
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
            VERIFY_MAX_TOKENS
        ),
        required_keys=[
            "candidate_laws"
        ],
        label="Candidate-law verifier"
    )

    # ========================================================
    # NORMALIZE MODEL OUTPUT
    # ========================================================

    evaluations = data.get(
        "candidate_laws",
        []
    )

    if not isinstance(
        evaluations,
        list
    ):

        evaluations = []

    cleaned_evaluations = []

    returned_ids = set()

    for item in evaluations:

        if not isinstance(
            item,
            dict
        ):

            continue

        law_id = normalize_text(
            item.get(
                "law_id",
                ""
            )
        ).upper()

        authority = normalize_text(
            item.get(
                "authority",
                ""
            )
        )

        decision = normalize_text(
            item.get(
                "decision",
                ""
            )
        ).upper()

        reason = normalize_text(
            item.get(
                "reason",
                ""
            )
        )

        if decision not in {
            "KEEP",
            "REJECT",
            "NEEDS_FACT"
        }:

            decision = "UNREVIEWED"

        if law_id:

            returned_ids.add(
                law_id
            )

        cleaned_evaluations.append({
            "law_id":
                law_id,

            "authority":
                authority,

            "decision":
                decision,

            "reason":
                reason,

            "current_law_check_required":
                bool(
                    item.get(
                        "current_law_check_required",
                        True
                    )
                )
        })

    # ========================================================
    # MAKE SURE EVERY INPUT LAW HAS A RESULT
    # ========================================================

    for index, candidate in enumerate(
        candidate_laws,
        start=1
    ):

        expected_id = (
            f"L{index}"
        )

        if expected_id in returned_ids:
            continue

        cleaned_evaluations.append({
            "law_id":
                expected_id,

            "authority":
                candidate.get(
                    "authority",
                    ""
                ),

            "decision":
                "UNREVIEWED",

            "reason":
                (
                    "The verifier did not return "
                    "an evaluation for this candidate, "
                    "so it is not being accepted."
                ),

            "current_law_check_required":
                True
        })

    # ========================================================
    # OPTIONAL TOP-LEVEL FIELDS
    # ========================================================

    needs_more_information = bool(
        data.get(
            "needs_more_information",
            False
        )
    )

    next_question = normalize_text(
        data.get(
            "next_question",
            ""
        )
    )

    # Do not claim another question is required unless
    # the model actually supplied one.
    if not next_question:

        needs_more_information = False

    return {
        "candidate_laws":
            cleaned_evaluations,

        "needs_more_information":
            needs_more_information,

        "next_question":
            next_question
    }


# ============================================================
# DISPLAY VERIFICATION
# ============================================================

def display_verification(
    verification
):

    print()
    print(
        "=" * 80
    )

    print(
        "CANDIDATE-LAW VERIFICATION"
    )

    print(
        "=" * 80
    )

    results = verification.get(
        "candidate_laws",
        []
    )

    if not results:

        print(
            "\nNo candidate authorities survived."
        )

        return

    for item in results:

        decision = normalize_text(
            item.get(
                "decision",
                "UNKNOWN"
            )
        ).upper()

        print()
        print(
            f"[{item.get('law_id', '?')}] "
            f"{decision}"
        )

        print(
            item.get(
                "authority",
                ""
            )
        )

        print(
            "Reason:",
            item.get(
                "reason",
                ""
            )
        )

        if item.get(
            "current_law_check_required",
            False
        ):

            print(
                "Current-law verification: REQUIRED"
            )


# ============================================================
# EXTRACT KEPT AUTHORITIES
# ============================================================

def get_kept_authorities(
    verification
):

    kept = []

    for item in verification.get(
        "candidate_laws",
        []
    ):

        decision = normalize_text(
            item.get(
                "decision",
                ""
            )
        ).upper()

        if decision == "KEEP":

            kept.append(
                item
            )

    return kept


# ============================================================
# ASK USER SAFELY
# ============================================================

def ask_followup(
    question,
    followups,
    asked_question_keys
):

    question = normalize_text(
        question
    )

    if not question:
        return False

    key = normalize_question(
        question
    )

    if key in asked_question_keys:

        print()
        print(
            "The model attempted to repeat an "
            "earlier question, so no additional "
            "follow-up will be asked."
        )

        return False

    print()
    print(
        "I need one more detail:"
    )

    print(
        question
    )

    answer = input(
        "\nYour answer: "
    ).strip()

    followups.append({
        "question": question,
        "answer": answer
    })

    asked_question_keys.add(
        key
    )

    return True


# ============================================================
# RUN ONE LEGAL CONVERSATION
# ============================================================

def run_case(
    embedding_model,
    scenario_map,
    chunks,
    embeddings,
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

    # ========================================================
    # OUTER LOOP
    #
    # Intake -> retrieval -> rerank -> verify.
    #
    # If verifier finds one material missing fact:
    # ask user and restart.
    # ========================================================

    while (
        research_iteration
        < MAX_RESEARCH_ITERATIONS
    ):

        research_iteration += 1

        # ====================================================
        # INTAKE LOOP
        # ====================================================

        while True:

            print()
            print(
                "[1/4] Analyzing facts..."
            )

            analysis = analyze_intake(
                original_problem=(
                    original_problem
                ),
                followups=followups,
                client=client,
                questions_already_asked=(
                    total_followups
                )
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
                < MAX_TOTAL_FOLLOWUPS
            ):

                question = normalize_text(
                    analysis.get(
                        "next_question",
                        ""
                    )
                )

                asked = ask_followup(
                    question=question,
                    followups=followups,
                    asked_question_keys=(
                        asked_question_keys
                    )
                )

                if asked:
                    total_followups += 1
                    continue

            # READY or question limit reached.
            break

        # ====================================================
        # NORMALIZED SCENARIO
        # ====================================================

        normalized_scenario = normalize_text(
            analysis.get(
                "normalized_scenario",
                ""
            )
        )

        if not normalized_scenario:

            normalized_scenario = (
                format_conversation(
                    original_problem,
                    followups
                )
            )

        retrieval_queries = (
            analysis.get(
                "retrieval_queries",
                []
            )
        )

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

        known_facts = analysis.get(
            "known_facts",
            []
        )

        if known_facts:

            print()
            print(
                "Known facts:"
            )

            for fact in known_facts:

                print(
                    "  -",
                    fact
                )

        unknowns = analysis.get(
            "missing_material_facts",
            []
        )

        if unknowns:

            print()
            print(
                "Still unknown:"
            )

            for fact in unknowns:

                print(
                    "  -",
                    fact
                )

        # ====================================================
        # SEARCH QUERY REPRESENTATIONS
        # ====================================================

        search_queries = (
            build_search_queries(
                original_problem=(
                    original_problem
                ),
                normalized_scenario=(
                    normalized_scenario
                ),
                retrieval_queries=(
                    retrieval_queries
                )
            )
        )

        print()
        print(
            "Search representations:"
        )

        for index, query in enumerate(
            search_queries,
            start=1
        ):

            marker = (
                "ORIGINAL"
                if index == 1
                else "EXPANSION"
            )

            print(
                f"  Q{index} "
                f"[{marker}]: "
                f"{query}"
            )

        # ====================================================
        # RETRIEVE
        # ====================================================

        print()
        print(
            "[2/4] Searching ILSIC "
            "scenario index..."
        )

        candidates = (
            retrieve_candidate_scenarios(
                search_queries=(
                    search_queries
                ),
                embedding_model=(
                    embedding_model
                ),
                scenario_map=(
                    scenario_map
                ),
                chunks=chunks,
                embeddings=embeddings
            )
        )

        display_raw_retrieval(
            candidates
        )

        # ====================================================
        # FACTUAL ANALOGY RERANKING
        # ====================================================

        print()
        print(
            "[3/4] Checking factual analogy..."
        )

        (
            reranked,
            accepted
        ) = rerank_analogous_scenarios(
            original_problem=(
                original_problem
            ),
            followups=followups,
            normalized_scenario=(
                normalized_scenario
            ),
            candidates=candidates,
            client=client
        )

        display_reranking(
            reranked,
            accepted
        )

        # ====================================================
        # NO GOOD ANALOGUES
        # ====================================================

        if not accepted:

            print()
            print(
                "=" * 80
            )

            print(
                "NO STRONG ANALOGOUS "
                "ILSIC SCENARIOS"
            )

            print(
                "=" * 80
            )

            print(
                "The scenario dataset did not "
                "produce sufficiently analogous "
                "examples."
            )

            print(
                "The future pipeline should fall "
                "back to direct legal-corpus "
                "research rather than forcing a "
                "bad statute mapping."
            )

            return True

        # ====================================================
        # CANDIDATE LAW AGGREGATION
        # ====================================================

        candidate_laws = (
            aggregate_candidate_laws(
                accepted
            )
        )

        display_candidate_laws(
            candidate_laws
        )

        # ====================================================
        # VERIFY
        # ====================================================

        print()
        print(
            "[4/4] Verifying candidate "
            "legal theories..."
        )

        verification = (
            verify_candidate_laws(
                original_problem=(
                    original_problem
                ),
                followups=followups,
                normalized_scenario=(
                    normalized_scenario
                ),
                accepted=accepted,
                candidate_laws=(
                    candidate_laws
                ),
                client=client
            )
        )

        display_verification(
            verification
        )

        # ====================================================
        # VERIFIER WANTS ONE MORE FACT
        # ====================================================

        needs_more = bool(
            verification.get(
                "needs_more_information",
                False
            )
        )

        verifier_question = normalize_text(
            verification.get(
                "next_question",
                ""
            )
        )

        if (
            needs_more
            and
            verifier_question
            and
            total_followups
            < MAX_TOTAL_FOLLOWUPS
        ):

            print()
            print(
                "=" * 80
            )

            print(
                "VERIFIER NEEDS ONE "
                "MATERIAL FACT"
            )

            print(
                "=" * 80
            )

            asked = ask_followup(
                question=verifier_question,
                followups=followups,
                asked_question_keys=(
                    asked_question_keys
                )
            )

            if asked:

                total_followups += 1

                print()
                print(
                    "Re-running scenario "
                    "analysis and retrieval with "
                    "the new fact..."
                )

                continue

        # ====================================================
        # FINAL OUTPUT OF THIS STAGE
        # ====================================================

        kept = get_kept_authorities(
            verification
        )

        print()
        print(
            "=" * 80
        )

        print(
            "AUTHORITIES TO SEARCH IN "
            "THE 710K LEGAL CORPUS"
        )

        print(
            "=" * 80
        )

        if not kept:

            print()
            print(
                "No candidate authority has "
                "enough factual support yet."
            )

        else:

            for item in kept:

                print()
                print(
                    "-",
                    item.get(
                        "authority",
                        ""
                    )
                )

                print(
                    "  Why:",
                    item.get(
                        "reason",
                        ""
                    )
                )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "These authorities are only "
            "research candidates."
        )

        print(
            "They are NOT the final legal answer."
        )

        print(
            "Historical IPC/CrPC and other old "
            "labels must undergo current-law "
            "verification before being presented "
            "to the user."
        )

        print(
            "The next stage will retrieve the "
            "actual provision/judgment text from "
            "the 710K legal corpus."
        )

        return True

    # ========================================================
    # ITERATION LIMIT
    # ========================================================

    print()
    print(
        "Maximum conversational research "
        "iterations reached."
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
        "CONVERSATIONAL INDIAN LEGAL MAPPER"
    )

    print(
        "ILSIC + BGE + GPT-OSS-120B"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    for path in [
        SCENARIOS_FILE,
        CHUNKS_FILE,
        EMBEDDINGS_FILE
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing:\n"
                f"{path}"
            )

    # --------------------------------------------------------
    # LOAD ILSIC
    # --------------------------------------------------------

    (
        scenario_map,
        chunks,
        embeddings
    ) = load_ilsic_index()

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # GROQ CLIENT
    # --------------------------------------------------------

    print(
        "Creating Groq client..."
    )

    client = Groq(
        api_key=GROQ_API_KEY
    )

    print()
    print(
        "System ready."
    )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        keep_running = run_case(
            embedding_model=(
                embedding_model
            ),
            scenario_map=scenario_map,
            chunks=chunks,
            embeddings=embeddings,
            client=client
        )

        if not keep_running:

            print()
            print(
                "Closing system."
            )

            break


if __name__ == "__main__":
    main()