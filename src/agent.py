from pathlib import Path
import sys
SRC = Path(__file__).resolve().parents[1]  # ...\src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import requests
import json
from typing import Iterator

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "openai_gpt-oss-20b-MXFP4.gguf"


def ask_model(message: str) -> Iterator[str]:
    """
    Stream text chunks from a local llama-server (OpenAI-compatible API).

    Yields:
        str chunks of the model's response as they arrive.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": True,
    }

    # Adjust headers if your server expects an API key, etc.
    headers = {
        "Content-Type": "application/json",
    }

    with requests.post(
        LLAMA_SERVER_URL,
        headers=headers,
        json=payload,
        stream=True,
    ) as resp:
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue

            # Some servers prefix with "data: "
            if line.startswith(b"data: "):
                line = line[len(b"data: "):]

            # End of stream
            if line == b"[DONE]":
                break

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # OpenAI-style streaming format
            choices = data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content


### Streamlit usage with streaming display

