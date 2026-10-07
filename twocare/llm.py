"""Minimal Groq client (OpenAI-compatible chat completions).

Retries rate limits and transient errors, and caches responses on disk so an
evaluation run can be replayed exactly. A cache hit is keyed on the full
request, so changing the prompt, the history, or the model is a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict

from .config import ROOT, settings

API_URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_DIR = ROOT / ".cache" / "llm"
MAX_RETRIES = 8

USAGE: dict = defaultdict(lambda: {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0})


class LLMError(RuntimeError):
    pass


class ToolCallFormatError(LLMError):
    """The model produced a tool call the API could not parse."""


def chat(
    *,
    model: str,
    messages: list,
    purpose: str,
    tools: list | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    json_mode: bool = False,
    max_tokens: int = 1024,
    reasoning_effort: str = "low",
) -> dict:
    config = settings()
    if not config.groq_api_key:
        raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": config.seed if seed is None else seed,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = False
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    # Reasoning models: keep thinking out of the reply and keep token use low.
    if model.startswith("qwen/"):
        payload["reasoning_format"] = "hidden"
    elif "gpt-oss" in model:
        payload["reasoning_effort"] = reasoning_effort

    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    cache_path = CACHE_DIR / f"{key}.json"
    usage = USAGE[purpose]
    if config.cache and cache_path.exists():
        usage["cached"] += 1
        return json.loads(cache_path.read_text())

    body = _post(payload, config.groq_api_key)
    usage["calls"] += 1
    usage["prompt_tokens"] += body.get("usage", {}).get("prompt_tokens", 0)
    usage["completion_tokens"] += body.get("usage", {}).get("completion_tokens", 0)
    message = body["choices"][0]["message"]
    result = {"content": message.get("content"), "tool_calls": message.get("tool_calls") or []}
    if config.cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result))
    return result


def _post(payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode()
    for attempt in range(MAX_RETRIES):
        request = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # Groq's edge rejects the default Python user agent.
                "User-Agent": "2care-agent/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code == 400 and "tool_use_failed" in detail:
                raise ToolCallFormatError(detail[:500]) from None
            if error.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                time.sleep(_retry_delay(error, detail, attempt))
                continue
            raise LLMError(f"Groq HTTP {error.code}: {detail[:500]}") from None
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(f"Groq unreachable: {error}") from None
    raise LLMError("Groq retries exhausted")


def _retry_delay(error: urllib.error.HTTPError, detail: str, attempt: int) -> float:
    header = error.headers.get("retry-after") if error.headers else None
    if header:
        try:
            return float(header) + 0.5
        except ValueError:
            pass
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", detail)
    if match:
        return int(match.group(1) or 0) * 60 + float(match.group(2)) + 0.5
    return min(60.0, 2.0 * 2 ** attempt)


def usage_summary() -> dict:
    return {purpose: dict(values) for purpose, values in USAGE.items()}
