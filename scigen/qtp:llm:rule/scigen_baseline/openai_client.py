from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List


DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def call_chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    api_base: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1800,
    retries: int = 2,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or DEEPSEEK_API_KEY is not set.")
    base_url = (api_base or os.environ.get("OPENAI_BASE_URL") or DEFAULT_API_BASE).rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if os.environ.get("DISABLE_JSON_RESPONSE_FORMAT", "").lower() not in {"1", "true", "yes"}:
        payload["response_format"] = {"type": "json_object"}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"OpenAI-compatible API request failed: {exc}") from exc
            time.sleep(2**attempt)
    raise RuntimeError("OpenAI-compatible API request failed.")


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError("Model response did not contain a JSON object.")
