import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN was not found in .env"
    )


def extract_json(text):
    """
    Handles plain JSON or JSON wrapped in markdown fences.
    """

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in model response."
        )

    return json.loads(
        text[start:end + 1]
    )


def create_search_plan(query, client):

    system_prompt = """
You are the query-understanding component of an Indian legal
retrieval system.

You DO NOT provide legal advice.
You DO NOT determine liability.
You DO NOT invent statutes, sections, judgments, or facts.

Your only task is to convert an informal user statement into a
structured retrieval plan.

============================================================
STRICT FACT RULES
============================================================

KNOWN FACTS:
- "known_facts" may contain ONLY facts explicitly stated by the user.
- Never put unanswered questions or desired information in known_facts.
- Never assume an amount, date, agreement, payment method, location,
  evidence, motive, or repayment term unless the user stated it.

MISSING INFORMATION:
- Put important facts that were NOT supplied by the user here.
- A fact must never appear in both known_facts and missing_information.

============================================================
STRICT PARTY-ROLE RULES
============================================================

The first-person speaker ("I", "me", "my") is always the user.

Pay careful attention to the direction of actions.

Example 1:

User:
"My friend borrowed money from me."

Correct interpretation:

user_role = "lender / creditor"
other_party_role = "borrower / debtor"

Because the FRIEND borrowed FROM the USER.

Example 2:

User:
"I borrowed money from my friend."

Correct interpretation:

user_role = "borrower / debtor"
other_party_role = "lender / creditor"

Example 3:

User:
"My employer has not paid my salary."

Correct interpretation:

user_role = "employee"
other_party_role = "employer"

Never reverse the parties.

============================================================
LEGAL AREA RULES
============================================================

Non-payment or refusal to repay borrowed money should ordinarily
be treated first as a possible:

- civil dispute
- debt recovery / money recovery
- contractual obligation

Do NOT classify the matter as criminal merely because money has
not been returned.

Only include criminal-law retrieval when the stated facts
specifically indicate possible deception from the beginning,
dishonest inducement, misappropriation, forgery, threats, or
another independent criminal allegation.

============================================================
SEARCH QUERY RULES
============================================================

Generate 4 concise queries using formal legal terminology that are
useful for retrieving Indian statutes and judgments.

Do not invent statute numbers or case citations.

Prefer concepts such as:

- debt recovery
- money recovery
- creditor and debtor
- repayment of loan
- contractual obligation
- acknowledgment of debt

when supported by the user's facts.

============================================================
JURISDICTION
============================================================

Country is India.

If State/UT was not supplied:

"state": null

Do not infer it.

============================================================
OUTPUT
============================================================

Return VALID JSON ONLY.

Use exactly this structure:

{
  "jurisdiction": {
    "country": "India",
    "state": null
  },
  "problem_summary": "...",
  "user_role": "...",
  "other_party_role": "...",
  "primary_legal_area": "...",
  "secondary_legal_areas": [],
  "known_facts": [],
  "missing_information": [],
  "search_queries": []
}

Before producing the JSON, internally verify:

1. Did I reverse the parties?
2. Did I put any unstated fact into known_facts?
3. Does any item appear in both known_facts and missing_information?
4. Did I classify ordinary non-repayment as criminal without supporting facts?

Do not output this verification. Output JSON only.
"""

    user_prompt = f"""
USER QUERY:

{query}

Create the retrieval plan.

Return JSON only.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
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
        max_tokens=500,
        temperature=0.1
    )

    raw_response = (
        response
        .choices[0]
        .message
        .content
    )

    return (
        extract_json(raw_response),
        raw_response
    )


def main():

    print("=" * 80)
    print("RAG PROTOTYPE V1.1 - QUERY PLANNER TEST")
    print("=" * 80)

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

    print(
        "\nCreating retrieval plan..."
    )

    try:

        plan, raw_response = (
            create_search_plan(
                query,
                client
            )
        )

    except Exception as exc:

        print(
            "\nQUERY PLANNER FAILED"
        )

        print(
            type(exc).__name__
        )

        print(exc)

        raise

    print(
        "\n" + "=" * 80
    )

    print(
        "STRUCTURED RETRIEVAL PLAN"
    )

    print(
        "=" * 80
    )

    print(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "SEARCH QUERIES"
    )

    print(
        "=" * 80
    )

    for index, search_query in enumerate(
        plan.get(
            "search_queries",
            []
        ),
        start=1
    ):

        print(
            f"{index}. {search_query}"
        )


if __name__ == "__main__":
    main()