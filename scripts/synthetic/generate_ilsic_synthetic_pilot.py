import os
import json
import random
import time
import argparse
import re
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\Projects\LLM_training")
ILSIC_DIR = PROJECT_ROOT / "data" / "ILSIC-dataset"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "data" / "training" / "sft" / "synthetic_pilot"

MODEL = "openai/gpt-oss-120b"

# Safer default for testing. Use --limit 100 when the 10-row test looks good.
DEFAULT_LIMIT = 10
DEFAULT_SEED = 42

QUESTION_MAX_TOKENS = 500
ANSWER_MAX_TOKENS = 900
JUDGE_MAX_TOKENS = 450

# Strict JSON removes almost all malformed-JSON failures, while retry logic
# handles 429/rate-limit, transient network errors, 5xx, and empty responses.
MAX_RETRIES = 8

# Hard API pacing: at most 2 API request starts in any rolling 60-second window.
# Retries are counted too because every retry is also an API call.
MAX_API_CALLS_PER_MINUTE = 2
API_RATE_WINDOW_SECONDS = 60.0
API_RATE_SAFETY_SECONDS = 1.0

# GPT-OSS supports low/medium/high reasoning effort. Low is enough for this
# structured generation task and reduces latency/token use.
REASONING_EFFORT = "low"


# ============================================================
# STRICT JSON SCHEMAS
# ============================================================
# Groq strict structured output requires:
#   - every field listed in required
#   - additionalProperties: false on objects

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "user_question": {"type": "string"},
        "detail_level": {
            "type": "string",
            "enum": ["VAGUE", "NORMAL", "DETAILED"],
        },
    },
    "required": ["user_question", "detail_level"],
    "additionalProperties": False,
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "assistant_answer": {"type": "string"},
    },
    "required": ["assistant_answer"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "question_fidelity": {
            "type": "string",
            "enum": ["PASS", "FAIL"],
        },
        "question_naturalness": {
            "type": "string",
            "enum": ["PASS", "FAIL"],
        },
        "answer_relevance": {
            "type": "string",
            "enum": ["PASS", "FAIL"],
        },
        "answer_style": {
            "type": "string",
            "enum": ["PASS", "FAIL"],
        },
        "answer_safety": {
            "type": "string",
            "enum": ["PASS", "FAIL"],
        },
        "overall": {
            "type": "string",
            "enum": ["PASS", "FAIL"],
        },
        "reason": {"type": "string"},
    },
    "required": [
        "question_fidelity",
        "question_naturalness",
        "answer_relevance",
        "answer_style",
        "answer_safety",
        "overall",
        "reason",
    ],
    "additionalProperties": False,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(value):
    return " ".join(str(value or "").split()).strip()


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                print(
                    f"WARNING: Could not parse line {line_number} "
                    f"in {path.name}; skipping."
                )
                continue

            if isinstance(row, dict):
                rows.append(row)

    return rows


def safe_labels(value):
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]

    if value is None:
        return []

    value = normalize_text(value)
    return [value] if value else []


def safe_json_loads(text):
    """Strict Structured Outputs should make this a normal json.loads call."""
    text = str(text or "").strip()
    if not text:
        raise ValueError("Empty model response.")

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Structured model response was not a JSON object.")

    return data


# ============================================================
# FIND ILSIC TRAIN FILE
# ============================================================

def find_ilsic_train_file():
    """
    Prefer the primary Layman-new train split.
    Dev/test files are never used.
    """

    if not ILSIC_DIR.exists():
        raise FileNotFoundError(
            f"ILSIC directory not found:\n{ILSIC_DIR}"
        )

    candidates = [
        path
        for path in ILSIC_DIR.rglob("*.jsonl")
        if "train" in path.name.lower()
        and "layman" in path.name.lower()
        and "dev" not in path.name.lower()
        and "test" not in path.name.lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            "Could not find a Layman training JSONL inside:\n"
            f"{ILSIC_DIR}"
        )

    primary = [
        path
        for path in candidates
        if "new" in path.name.lower()
    ]

    return sorted(primary or candidates)[0]


# ============================================================
# REQUEST PACING / RETRY HELPERS
# ============================================================

class RequestPacer:
    """Hard rolling-window limiter: no more than N API calls per window."""

    def __init__(
        self,
        max_calls=MAX_API_CALLS_PER_MINUTE,
        window_seconds=API_RATE_WINDOW_SECONDS,
        safety_seconds=API_RATE_SAFETY_SECONDS,
    ):
        self.max_calls = max(1, int(max_calls))
        self.window_seconds = max(1.0, float(window_seconds))
        self.safety_seconds = max(0.0, float(safety_seconds))
        self.request_times = deque()

    def wait(self):
        """Wait until another API call is allowed, then record the call."""
        while True:
            now = time.monotonic()

            # Remove calls that are safely outside the rolling window.
            cutoff = now - self.window_seconds
            while self.request_times and self.request_times[0] <= cutoff:
                self.request_times.popleft()

            if len(self.request_times) < self.max_calls:
                self.request_times.append(now)
                return

            oldest = self.request_times[0]
            wait_seconds = (
                oldest
                + self.window_seconds
                + self.safety_seconds
                - now
            )

            if wait_seconds > 0:
                print(
                    f"  API pacing: {self.max_calls} calls/minute limit reached. "
                    f"Waiting {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)
            else:
                # Loop once more so expired timestamps are removed cleanly.
                time.sleep(0.05)


def get_status_code(exc):
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def get_exception_headers(exc):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    if headers is None:
        return {}

    try:
        return {str(k).lower(): str(v) for k, v in headers.items()}
    except Exception:
        return {}


def parse_duration_seconds(value):
    """
    Parse strings such as:
      2
      2.4
      7.66s
      2m59.56s
      250ms
    """

    if value is None:
        return None

    text = str(value).strip().lower()

    try:
        return max(0.0, float(text))
    except Exception:
        pass

    total = 0.0
    matched = False

    minute_match = re.search(r"([0-9.]+)\s*m(?!s)", text)
    if minute_match:
        total += float(minute_match.group(1)) * 60.0
        matched = True

    second_match = re.search(r"([0-9.]+)\s*s", text)
    if second_match:
        total += float(second_match.group(1))
        matched = True

    millisecond_match = re.search(r"([0-9.]+)\s*ms", text)
    if millisecond_match:
        total += float(millisecond_match.group(1)) / 1000.0
        matched = True

    return max(0.0, total) if matched else None


def retry_wait_seconds(exc, attempt):
    """Prefer Groq Retry-After; otherwise use reset hints or exponential backoff."""

    headers = get_exception_headers(exc)

    retry_after = parse_duration_seconds(headers.get("retry-after"))
    if retry_after is not None:
        return min(max(retry_after + 0.5, 1.0), 180.0)

    token_reset = parse_duration_seconds(headers.get("x-ratelimit-reset-tokens"))
    if token_reset is not None:
        return min(max(token_reset + 0.5, 1.0), 180.0)

    # Some Groq errors include text such as "try again in 7.66s".
    message = str(exc).lower()
    match = re.search(
        r"(?:try again in|retry after)\s*([0-9.]+)\s*(ms|s|m)?",
        message,
    )

    if match:
        amount = float(match.group(1))
        unit = match.group(2) or "s"

        if unit == "ms":
            amount /= 1000.0
        elif unit == "m":
            amount *= 60.0

        return min(max(amount + 0.5, 1.0), 180.0)

    base = min(2 ** attempt, 60)
    jitter = random.uniform(0.25, 1.25)
    return min(base + jitter, 90.0)


def is_retryable_error(exc):
    status = get_status_code(exc)

    # No HTTP status usually means a network/client-side transient error.
    if status is None:
        return True

    if status in {408, 409, 429, 498, 500, 502, 503, 504}:
        return True

    return False


def serialize_exception(exc):
    headers = get_exception_headers(exc)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "status_code": get_status_code(exc),
        "retry_after": headers.get("retry-after"),
        "ratelimit_remaining_requests": headers.get("x-ratelimit-remaining-requests"),
        "ratelimit_remaining_tokens": headers.get("x-ratelimit-remaining-tokens"),
        "ratelimit_reset_requests": headers.get("x-ratelimit-reset-requests"),
        "ratelimit_reset_tokens": headers.get("x-ratelimit-reset-tokens"),
    }


# ============================================================
# GROQ STRICT STRUCTURED OUTPUT CALL
# ============================================================

def groq_json(
    client,
    pacer,
    system_prompt,
    user_prompt,
    schema_name,
    schema,
    max_tokens,
    temperature,
):
    """
    Call Groq with strict Structured Outputs.

    For openai/gpt-oss-120b, strict:true constrains generation to the supplied
    JSON schema. Therefore malformed JSON should no longer be a normal failure
    mode.
    """

    last_error = None
    last_raw_text = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pacer.wait()

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
                max_completion_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=REASONING_EFFORT,
            )

            last_raw_text = response.choices[0].message.content
            data = safe_json_loads(last_raw_text)
            return data

        except Exception as exc:
            last_error = exc
            status = get_status_code(exc)
            retryable = is_retryable_error(exc)

            # json.JSONDecodeError / empty-content errors do not carry HTTP
            # status. They are retryable because they are unexpected transient
            # response failures under strict mode.
            if isinstance(exc, (json.JSONDecodeError, ValueError)):
                retryable = True

            error_label = f"HTTP {status}" if status is not None else type(exc).__name__
            print(
                f"  API/structured-output attempt {attempt}/{MAX_RETRIES} "
                f"failed ({error_label}): {exc}"
            )

            if attempt >= MAX_RETRIES or not retryable:
                break

            wait_seconds = retry_wait_seconds(exc, attempt)

            if status == 429:
                print(
                    f"  Rate limit reached. Waiting {wait_seconds:.1f}s "
                    "before retry..."
                )
            else:
                print(
                    f"  Waiting {wait_seconds:.1f}s before retry..."
                )

            time.sleep(wait_seconds)

    detail = serialize_exception(last_error) if last_error else {}

    if last_raw_text:
        detail["raw_model_text"] = last_raw_text[:8000]

    raise RuntimeError(
        "Groq structured-output call failed after retries. "
        + json.dumps(detail, ensure_ascii=False)
    )


# ============================================================
# STAGE 1: NATURAL USER STORY
# ============================================================

QUESTION_SYSTEM_PROMPT = r'''
You are a USER-SIMULATION WRITER for an Indian legal-assistant training dataset.

You receive a detailed legal scenario written in a formal, FIR-like, case-file,
or legal-dataset style.

Your ONLY job is to rewrite the situation as if an ordinary person in India is
explaining the problem to a legal chatbot.

Do NOT answer the legal problem.

STRICT RULES:
- Preserve the core factual situation.
- Do not materially change who did what to whom.
- Do not invent a new event, payment, injury, threat, document, witness,
  relationship, admission, or legal proceeding.
- Use simple, natural English.
- Make it sound like a real person, not a police FIR, judgment, lawyer, textbook,
  statute, exam question, or legal notice.
- Avoid Act names, section numbers, case names, offence names, and technical legal
  classifications unless an ordinary user would naturally know and mention them.
- The user may sound worried, confused, frustrated, or uncertain, but do not make
  the emotion melodramatic.
- Use first-person language whenever the scenario reasonably permits it.
- End with a natural request for legal help such as "What can I do?",
  "What should I do now?", "Can I take legal action?", or similar.
- Keep the message concise enough for a chatbot.
- Do not include an answer or legal analysis.

DETAIL LEVELS:
VAGUE:
- Keep the core problem.
- Omit some secondary details that a normal user might forget to mention.
- Never remove the central event or change the parties.

NORMAL:
- Include the main facts and useful evidence naturally.
- Moderate amount of detail.

DETAILED:
- Include most important facts from the source.
- Still sound like an ordinary person, not an FIR or case brief.
'''.strip()


def generate_user_question(
    client,
    pacer,
    original_scenario,
    detail_level,
):
    prompt = f"""
DETAIL LEVEL:
{detail_level}

ORIGINAL DETAILED SCENARIO:
{original_scenario}

Rewrite only the user's side of the story.
The output detail_level must exactly equal: {detail_level}
""".strip()

    data = groq_json(
        client=client,
        pacer=pacer,
        system_prompt=QUESTION_SYSTEM_PROMPT,
        user_prompt=prompt,
        schema_name="natural_legal_user_story",
        schema=QUESTION_SCHEMA,
        max_tokens=QUESTION_MAX_TOKENS,
        temperature=0.7,
    )

    question = normalize_text(data.get("user_question", ""))
    returned_level = normalize_text(data.get("detail_level", "")).upper()

    if len(question) < 35:
        raise ValueError("Generated user question is too short.")

    if returned_level != detail_level:
        raise ValueError(
            f"Structured output returned detail_level={returned_level!r}; "
            f"expected {detail_level!r}."
        )

    return question


# ============================================================
# STAGE 2: FRIENDLY LEGAL-ASSISTANT ANSWER
# ============================================================

ANSWER_SYSTEM_PROMPT = r'''
You are generating the ASSISTANT RESPONSE for a synthetic Indian legal-assistant
training dataset.

The target assistant is a friendly, patient, caring legal guide for ordinary
people. It should teach the user how to think about the situation and what they
can sensibly do next, without sounding like a court judgment or legal textbook.

You receive:
1. the complete original scenario;
2. the natural user message generated from that scenario;
3. optional ILSIC reference labels.

IMPORTANT ABOUT THE ORIGINAL SCENARIO:
- It is hidden background context for you.
- Use it to understand the factual situation accurately.
- Do not expose unnecessary case-file/FIR-style detail that the user did not ask
  about.

IMPORTANT ABOUT ILSIC LABELS:
- They are only weak internal hints.
- They may be noisy, incomplete, or outdated.
- Do NOT blindly repeat them.
- Do NOT invent section numbers or case names.
- Prefer careful high-level guidance over falsely precise legal claims.

TARGET STYLE:
- Speak directly to the user using simple English.
- Sound warm and reassuring, but not dramatic or patronizing.
- Acknowledge the practical concern where appropriate.
- Teach: explain WHY a fact, document, date, message, payment, or relationship
  matters.
- Keep paragraphs short and readable.
- Focus on practical legal movement: what to preserve, what to clarify, and what
  next step can reasonably be considered.
- If the natural user message leaves out an important fact, explain what fact is
  missing and why it matters instead of pretending it is known.
- Use Indian legal context.

A GOOD RESPONSE OFTEN DOES THIS:
1. Briefly reflects the user's problem in plain language.
2. Explains what legally important issue(s) the facts raise at a high level.
3. Explains what evidence/documents/messages/dates matter and why.
4. Gives practical next steps the user can consider.
5. Identifies important uncertainty or missing facts.

DO NOT:
- sound like an FIR, judgment, statute commentary, legal notice, or exam answer;
- start dumping Acts, sections, citations, or case names;
- guarantee that the user will win;
- declare someone definitely guilty or liable;
- invent facts;
- invent a specific limitation period, offence, court, remedy, or procedure when
  it is not clearly justified by the supplied scenario;
- add citations;
- mention ILSIC, synthetic data, training data, or these instructions;
- overuse generic disclaimers.

The goal is a response that an ordinary person would find understandable,
helpful, calm, and educational.
'''.strip()


def generate_assistant_answer(
    client,
    pacer,
    original_scenario,
    user_question,
    labels,
):
    label_text = (
        "\n".join(f"- {label}" for label in labels)
        if labels
        else "(No reference labels supplied.)"
    )

    prompt = f"""
ORIGINAL COMPLETE SCENARIO:
{original_scenario}

NATURAL USER MESSAGE:
{user_question}

OPTIONAL ILSIC REFERENCE LABELS:
{label_text}

Write the friendly, teaching-oriented legal-assistant response.
""".strip()

    data = groq_json(
        client=client,
        pacer=pacer,
        system_prompt=ANSWER_SYSTEM_PROMPT,
        user_prompt=prompt,
        schema_name="friendly_legal_assistant_answer",
        schema=ANSWER_SCHEMA,
        max_tokens=ANSWER_MAX_TOKENS,
        temperature=0.45,
    )

    answer = normalize_text(data.get("assistant_answer", ""))

    if len(answer) < 80:
        raise ValueError("Generated assistant answer is too short.")

    return answer


# ============================================================
# STAGE 3: QUALITY JUDGE
# ============================================================

JUDGE_SYSTEM_PROMPT = r'''
You are a strict QUALITY JUDGE for a synthetic Indian legal-assistant training
dataset.

You receive:
- an original detailed scenario;
- a synthetic natural user question;
- a synthetic assistant answer.

You are checking DATASET QUALITY. You are not being asked to write a new legal
answer.

QUESTION_FIDELITY:
PASS only if the synthetic user question preserves the core situation and does
not materially invent or contradict the original scenario. It may intentionally
omit secondary details when written as a vague/normal user message.

QUESTION_NATURALNESS:
PASS only if it sounds like a real ordinary person asking a legal chatbot rather
than an FIR, judgment, statute question, case brief, or exam question.

ANSWER_RELEVANCE:
PASS only if the answer directly helps with the user's actual situation rather
than giving generic unrelated law.

ANSWER_STYLE:
PASS only if it is friendly, calm, teaching-oriented, practical, and
understandable to a non-lawyer. It should not read mainly like a judgment,
statute note, or legal textbook.

ANSWER_SAFETY:
PASS only if the answer avoids materially invented facts, fabricated section
numbers/case names, unsupported precise deadlines, guaranteed outcomes, and
obviously unjustified legal conclusions.

IMPORTANT:
- Do not fail an answer merely because it avoids detailed statutory citations.
- Do not fail a VAGUE user story merely because it leaves out secondary facts.
- This dataset is specifically intended to teach conversational legal guidance.

OVERALL:
PASS only when all five individual checks pass.

Keep the reason brief and useful for debugging.
'''.strip()


def judge_sample(
    client,
    pacer,
    original_scenario,
    user_question,
    assistant_answer,
):
    prompt = f"""
ORIGINAL DETAILED SCENARIO:
{original_scenario}

SYNTHETIC USER QUESTION:
{user_question}

SYNTHETIC ASSISTANT ANSWER:
{assistant_answer}

Judge this training sample.
""".strip()

    data = groq_json(
        client=client,
        pacer=pacer,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=prompt,
        schema_name="synthetic_sample_quality_judgement",
        schema=JUDGE_SCHEMA,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=0.0,
    )

    keys = [
        "question_fidelity",
        "question_naturalness",
        "answer_relevance",
        "answer_style",
        "answer_safety",
        "overall",
    ]

    normalized = {}

    for key in keys:
        value = normalize_text(data.get(key, "FAIL")).upper()
        normalized[key] = value if value in {"PASS", "FAIL"} else "FAIL"

    normalized["reason"] = normalize_text(data.get("reason", ""))

    # Do not trust the model's overall field if an individual check failed.
    if not all(normalized[key] == "PASS" for key in keys[:-1]):
        normalized["overall"] = "FAIL"

    return normalized


# ============================================================
# DATASET SELECTION / RESUME
# ============================================================

def build_source_id(row, source_index):
    existing = normalize_text(row.get("id", ""))
    return existing if existing else f"ilsic_{source_index:06d}"


def choose_pilot_rows(rows, limit, seed):
    usable = []

    for index, row in enumerate(rows):
        scenario = normalize_text(row.get("instruction", ""))
        if len(scenario) >= 50:
            usable.append((index, row))

    if len(usable) < limit:
        print(
            f"WARNING: Requested {limit} rows but only "
            f"{len(usable)} usable scenarios were found."
        )
        limit = len(usable)

    rng = random.Random(seed)
    chosen = rng.sample(usable, k=limit)
    chosen.sort(key=lambda item: item[0])
    return chosen


def deterministic_detail_level(seed, source_index):
    """Stable across retries/resumes for the same source row."""
    levels = ["VAGUE", "NORMAL", "DETAILED"]
    rng = random.Random(f"{seed}:{source_index}:detail-level")
    return rng.choice(levels)


def load_processed_ids(path):
    """
    Only successful/rejected completed rows go into all_output.
    Errors are NOT considered processed, so simply re-running the script retries
    them automatically while skipping rows already completed.
    """

    if not path.exists():
        return set()

    processed = set()

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                continue

            sample_id = normalize_text(row.get("synthetic_id", ""))

            if sample_id:
                processed.add(sample_id)

    return processed


def reset_output_files(paths):
    for path in paths:
        if path.exists():
            path.unlink()
            print(f"Deleted old output: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic conversational Indian legal-assistant "
            "dataset from ILSIC using Groq strict Structured Outputs."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of ILSIC scenarios to process. Default: {DEFAULT_LIMIT}",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete existing pilot output files before starting. "
            "Do NOT use this when you want to resume failed rows."
        ),
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found in your .env file."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_file = find_ilsic_train_file()

    all_output = OUTPUT_DIR / "synthetic_pilot_all.jsonl"
    accepted_output = OUTPUT_DIR / "synthetic_pilot_accepted.jsonl"
    rejected_output = OUTPUT_DIR / "synthetic_pilot_rejected.jsonl"
    sft_output = OUTPUT_DIR / "synthetic_pilot_sft.jsonl"
    summary_output = OUTPUT_DIR / "synthetic_pilot_summary.json"
    errors_output = OUTPUT_DIR / "synthetic_pilot_errors.jsonl"

    output_files = [
        all_output,
        accepted_output,
        rejected_output,
        sft_output,
        summary_output,
        errors_output,
    ]

    if args.reset:
        reset_output_files(output_files)

    print()
    print("=" * 80)
    print("ILSIC SYNTHETIC LEGAL-ASSISTANT PILOT - STRICT JSON VERSION")
    print("=" * 80)
    print()
    print("Model:", MODEL)
    print("Strict Structured Outputs: ENABLED")
    print("Reasoning effort:", REASONING_EFFORT)
    print("API rate limit:", f"{MAX_API_CALLS_PER_MINUTE} calls per 60 seconds")
    print()
    print("Source file:")
    print(source_file)

    source_rows = read_jsonl(source_file)
    print("Source rows:", f"{len(source_rows):,}")

    chosen = choose_pilot_rows(
        rows=source_rows,
        limit=args.limit,
        seed=args.seed,
    )

    print("Pilot rows selected:", len(chosen))

    processed_ids = load_processed_ids(all_output)

    if processed_ids:
        print(
            "Completed rows already on disk:",
            len(processed_ids),
        )
        print("Errors from earlier runs will be retried automatically.")

    client = Groq(api_key=api_key)
    pacer = RequestPacer()

    accepted_count = 0
    rejected_count = 0
    error_count_this_run = 0
    skipped_count_this_run = 0
    completed_this_run = 0

    if all_output.exists():
        existing_rows = read_jsonl(all_output)
        accepted_count = sum(
            1
            for row in existing_rows
            if row.get("quality", {}).get("overall") == "PASS"
        )
        rejected_count = sum(
            1
            for row in existing_rows
            if row.get("quality", {}).get("overall") == "FAIL"
        )

    current_stage = None
    current_user_question = None
    current_assistant_answer = None

    for position, (source_index, row) in enumerate(chosen, start=1):
        source_id = build_source_id(row, source_index)
        synthetic_id = f"synthetic_{args.seed}_{source_index:06d}"

        if synthetic_id in processed_ids:
            skipped_count_this_run += 1
            print()
            print(
                f"[{position}/{len(chosen)}] {synthetic_id} "
                "already completed; skipping."
            )
            continue

        original_scenario = normalize_text(row.get("instruction", ""))
        labels = safe_labels(row.get("answer", []))
        detail_level = deterministic_detail_level(args.seed, source_index)

        current_stage = None
        current_user_question = None
        current_assistant_answer = None

        print()
        print("-" * 80)
        print(f"[{position}/{len(chosen)}] {synthetic_id}")
        print("Detail level:", detail_level)

        try:
            current_stage = "question_generation"
            print("  Stage 1/3: generating natural user story...")

            current_user_question = generate_user_question(
                client=client,
                pacer=pacer,
                original_scenario=original_scenario,
                detail_level=detail_level,
            )

            current_stage = "answer_generation"
            print("  Stage 2/3: generating friendly assistant answer...")

            current_assistant_answer = generate_assistant_answer(
                client=client,
                pacer=pacer,
                original_scenario=original_scenario,
                user_question=current_user_question,
                labels=labels,
            )

            current_stage = "quality_judging"
            print("  Stage 3/3: quality judging...")

            quality = judge_sample(
                client=client,
                pacer=pacer,
                original_scenario=original_scenario,
                user_question=current_user_question,
                assistant_answer=current_assistant_answer,
            )

            output_row = {
                "synthetic_id": synthetic_id,
                "source_id": source_id,
                "source_index": source_index,
                "detail_level": detail_level,
                "user_question": current_user_question,
                "assistant_answer": current_assistant_answer,
                "original_scenario": original_scenario,
                "ilsic_labels": labels,
                "quality": quality,
                "generator": {
                    "model": MODEL,
                    "structured_outputs": "strict",
                    "reasoning_effort": REASONING_EFFORT,
                },
            }

            append_jsonl(all_output, output_row)

            if quality["overall"] == "PASS":
                append_jsonl(accepted_output, output_row)

                append_jsonl(
                    sft_output,
                    {
                        "instruction": current_user_question,
                        "output": current_assistant_answer,
                    },
                )

                accepted_count += 1
                print("  RESULT: ACCEPTED")

            else:
                append_jsonl(rejected_output, output_row)
                rejected_count += 1
                print("  RESULT: REJECTED")
                print("  Reason:", quality.get("reason", ""))

            processed_ids.add(synthetic_id)
            completed_this_run += 1

        except Exception as exc:
            error_count_this_run += 1

            print("  RESULT: ERROR")
            print("  Stage:", current_stage)
            print("  Error:", exc)
            print("  This row was NOT marked complete and will retry on the next run.")

            append_jsonl(
                errors_output,
                {
                    "synthetic_id": synthetic_id,
                    "source_id": source_id,
                    "source_index": source_index,
                    "detail_level": detail_level,
                    "stage": current_stage,
                    "error": str(exc),
                    "partial_user_question": current_user_question,
                    "partial_assistant_answer": current_assistant_answer,
                },
            )

        summary = {
            "source_file": str(source_file),
            "model": MODEL,
            "structured_outputs": "strict",
            "reasoning_effort": REASONING_EFFORT,
            "requested_limit": args.limit,
            "seed": args.seed,
            "request_delay_seconds": args.request_delay,
            "accepted_total": accepted_count,
            "rejected_total": rejected_count,
            "completed_this_run": completed_this_run,
            "errors_this_run": error_count_this_run,
            "skipped_this_run": skipped_count_this_run,
            "output_directory": str(OUTPUT_DIR),
        }

        with open(summary_output, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    print()
    print("=" * 80)
    print("PILOT COMPLETE")
    print("=" * 80)
    print()
    print("Accepted total:", accepted_count)
    print("Rejected total:", rejected_count)
    print("Completed this run:", completed_this_run)
    print("Errors this run:", error_count_this_run)
    print("Skipped this run:", skipped_count_this_run)
    print()
    print("Outputs:")
    print("  All:", all_output)
    print("  Accepted:", accepted_output)
    print("  Rejected:", rejected_output)
    print("  SFT-ready:", sft_output)
    print("  Error log:", errors_output)
    print("  Summary:", summary_output)
    print()
    print("Resume behavior:")
    print("  - completed rows are skipped")
    print("  - error rows are retried")
    print("  - use --reset only if you intentionally want a fresh pilot")


if __name__ == "__main__":
    main()
