"""DeepSeek V4 adapter (native OpenAI-compatible Chat Completions).

Endpoint: POST {DEEPSEEK_BASE_URL}/chat/completions (default
https://api.deepseek.com/v1), DeepSeek's native OpenAI-compatible
endpoint. Thinking mode is opted in via the top-level OpenAI-shape
extension:
    "thinking": {"type": "enabled", "reasoning_effort": "high"|"low"}

Multi-turn replay contract:
- DeepSeek's docs (api-docs.deepseek.com) require that assistant turns
  WITH tool_calls echo their prior `reasoning_content` on every
  subsequent request — omission returns HTTP 400.
- Plain text assistant turns (no tool_calls) do NOT require the echo.
- We persist `{reasoning_content}` to `native_assistant_payload` and
  reproduce the gating in `_build_messages`.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Credentials + endpoint from env (set before launch). BASE_URL defaults to
# DeepSeek's official API; override to point at an OpenAI-compatible proxy.
_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
_DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

NATIVE_FORMAT = "deepseek_chat_v1"


class _RetryableDeepSeekError(RuntimeError):
    """Retryable provider error from DeepSeek /v1/chat/completions."""


def _build_messages(messages: list[dict]) -> list[dict]:
    """Rebuild OpenAI-shape messages, echoing reasoning_content only on
    tool-call assistant turns (DeepSeek's hard constraint)."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            out.append({"role": "system", "content": m.get("content") or ""})
            continue
        if role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
            continue
        if role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id") or "",
                "content": m.get("content") or "",
            })
            continue
        if role == "assistant":
            asst: dict[str, Any] = {"role": "assistant"}
            has_tcs = bool(m.get("tool_calls"))
            text = m.get("content")
            asst["content"] = text if text else (None if has_tcs else "")
            if has_tcs:
                asst["tool_calls"] = [
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": (tc.get("function") or {}).get("name"),
                            "arguments": (tc.get("function") or {}).get("arguments") or "{}",
                        },
                    }
                    for tc in m.get("tool_calls") or []
                ]
            if has_tcs:
                nap = m.get("native_assistant_payload")
                npf = m.get("native_payload_format")
                if isinstance(nap, dict) and npf == NATIVE_FORMAT:
                    rc = nap.get("reasoning_content")
                else:
                    rc = m.get("reasoning_content")
                # HARD constraint: tool-call turns require the field; emit
                # "" rather than omit when we have no captured text.
                asst["reasoning_content"] = rc or ""
            out.append(asst)
            continue
    return out


def _convert_usage(usage: dict | None) -> dict | None:
    if not usage:
        return None
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": prompt_details.get("cached_tokens"),
        "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
    }


def _from_deepseek(resp: dict) -> dict:
    choices = resp.get("choices") or []
    if not choices:
        return {
            "content": None,
            "tool_calls": None,
            "reasoning_content": None,
            "native_assistant_payload": None,
            "native_payload_format": None,
            "usage": _convert_usage(resp.get("usage")),
        }
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    reasoning_content = msg.get("reasoning_content")
    raw_tcs = msg.get("tool_calls") or []
    parsed_tcs: list[dict] = []
    for tc in raw_tcs:
        fn = tc.get("function") or {}
        args_str = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Malformed DeepSeek tool_call arguments (call_id=%s): %r",
                tc.get("id"), args_str,
            )
            args = {}
        parsed_tcs.append({
            "id": tc.get("id"),
            "name": fn.get("name"),
            "arguments": args,
        })

    native_payload = (
        {"reasoning_content": reasoning_content}
        if reasoning_content is not None
        else None
    )

    return {
        "content": content,
        "tool_calls": parsed_tcs or None,
        "reasoning_content": reasoning_content,
        "native_assistant_payload": native_payload,
        "native_payload_format": NATIVE_FORMAT if native_payload else None,
        "usage": _convert_usage(resp.get("usage")),
    }


def _http_post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {429, 500, 502, 503, 504}:
            raise _RetryableDeepSeekError(
                f"DeepSeek HTTP {exc.code}: {error_body[:500]}"
            ) from exc
        raise RuntimeError(
            f"DeepSeek HTTP {exc.code}: {error_body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise _RetryableDeepSeekError(
            f"DeepSeek connection failed: {exc}"
        ) from exc


@retry(
    retry=retry_if_exception_type(_RetryableDeepSeekError),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(7),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_deepseek_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = "deepseek-v4-pro",
    temperature: float | None = 0.0,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
) -> dict:
    """provider-native call for DeepSeek V4 ids.

    `reasoning_effort`: "on"  → thinking={type:enabled, reasoning_effort:high}
                        "off" → thinking={type:disabled}
                        None  → omit (vendor default; thinking off in v4).
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(messages),
    }
    if reasoning_effort is None:
        pass
    elif reasoning_effort == "on":
        body["thinking"] = {"type": "enabled", "reasoning_effort": "high"}
    elif reasoning_effort == "off":
        body["thinking"] = {"type": "disabled"}
    else:
        raise ValueError(
            f"reasoning_effort must be 'on' | 'off' | None, got {reasoning_effort!r}"
        )
    if tools:
        body["tools"] = tools
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed
    if response_format is not None:
        body["response_format"] = response_format

    url = f"{_DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = _http_post_json(url, body, headers, timeout=360.0)
    if isinstance(resp, dict) and resp.get("error"):
        err = resp.get("error")
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise RuntimeError(f"DeepSeek error: {msg}")
    return _from_deepseek(resp)
