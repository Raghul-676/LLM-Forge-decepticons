print("SCRIPT STARTED", flush=True)

import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

print("IMPORTS COMPLETED", flush=True)

load_dotenv()

print("ENV LOADED", flush=True)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def main():
    print("MAIN STARTED", flush=True)

    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN was not found in the .env file."
        )

    print("HF TOKEN FOUND", flush=True)
    print("Model:", MODEL_NAME, flush=True)

    client = InferenceClient(
        provider="featherless-ai",
        api_key=token,
    )

    print("CLIENT CREATED", flush=True)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Indian legal research assistant. "
                "Use only the supplied context. "
                "Do not invent statutes, cases, or citations."
            ),
        },
        {
            "role": "user",
            "content": """
CONTEXT:
Article 21 of the Constitution of India states:
No person shall be deprived of his life or personal liberty
except according to procedure established by law.

QUESTION:
According only to the supplied context,
what does Article 21 protect?
""",
        },
    ]

    print("SENDING API REQUEST...", flush=True)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=150,
        temperature=0.1,
    )

    print("RESPONSE RECEIVED", flush=True)

    answer = response.choices[0].message.content

    print("\n" + "=" * 70)
    print("MODEL RESPONSE")
    print("=" * 70)
    print(answer)
    print("=" * 70)


if __name__ == "__main__":
    main()