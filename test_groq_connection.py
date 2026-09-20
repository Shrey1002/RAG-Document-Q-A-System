"""
Quick connectivity check for the Groq API, called directly (no 9Router).

Usage:
    pip install openai python-dotenv
    Set GROQ_API_KEY in your .env (or export it in your shell)
    python test_groq_connection.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def main() -> int:
    if not API_KEY:
        print("ERROR: GROQ_API_KEY is not set. Add it to your .env file.")
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    print(f"Base URL : {BASE_URL}")
    print(f"Model    : {MODEL}")
    print("Sending test request...\n")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Reply with exactly: OK"}
            ],
            max_tokens=100,
        )
    except Exception as exc:  # noqa: BLE001 - top-level diagnostic script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("\nCommon causes:")
        print(" - Wrong or missing GROQ_API_KEY (check for a 401)")
        print(" - Model name typo or retired model (check console.groq.com/docs/models)")
        print(" - Free-tier rate limit hit (check for a 429)")
        return 1

    content = response.choices[0].message.content
    print("SUCCESS")
    print(f"Model replied: {content!r}")
    if hasattr(response, "usage") and response.usage:
        print(f"Tokens used: {response.usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())