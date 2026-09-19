import os

from dotenv import load_dotenv
from groq import Groq


MODEL_NAME = "openai/gpt-oss-120b"


def main():

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found in .env"
        )

    client = Groq(
        api_key=api_key
    )

    print("=" * 70)
    print("TESTING GROQ GPT-OSS-120B")
    print("=" * 70)

    prompt = """
A person in India lends ₹1,00,000 to a friend.
The friend acknowledges receiving the money but later refuses
to repay it.

Briefly identify the likely nature of the legal dispute.
Do not invent statute sections or case citations.
"""

    print("\nSending request...")

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        reasoning_effort="medium",

        include_reasoning=False,

        temperature=0.2,

        max_completion_tokens=600
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    print("\n" + "=" * 70)
    print("GPT-OSS-120B RESPONSE")
    print("=" * 70)

    print(answer)


if __name__ == "__main__":
    main()