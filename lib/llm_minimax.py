"""MiniMax-M2.7 adapter (OpenAI-compatible Chat Completions + reasoning_split).

Endpoint: POST {MINIMAX_BASE_URL}/chat/completions (default
https://api.minimax.io/v1, MiniMax's official OpenAI-compatible API).
Body: standard OpenAI Chat shape with `reasoning_split=true` (officially
recommended per platform.minimax.io/docs/guides/text-m2-function-call).

Multi-turn replay contract:
- MiniMax docs declare it "essential for best performance" that the
  entire `response_message`, including `reasoning_details[]`, be
  preserved on every assistant turn of the conversation history.
- `reasoning_details[]` is an array of {format, id, index, text, type}
  per element; the proxy path empirically delivers all 5 subfields
  byte-faithfully. We persist this array as `native_assistant_payload`
  and replay it on assistant echoes.
- `reasoning_split=true` directs the model to emit a clean
  `reasoning_content` string + separate `reasoning_details[]` array
  rather than embedding `<think>...</think>` inside `content`.
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
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Credentials + endpoint from env (set before launch). BASE_URL defaults to
# MiniMax's official OpenAI-compatible API; override to use a proxy.
_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
_BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")

NATIVE_FORMAT = "minimax_chat_v1"


class _RetryableMiniMaxError(RuntimeError):
    """Retryable provider error from MiniMax /v1/chat/completions."""


def _build_messages(messages: list[dict]) -> list[dict]:
    """Rebuild OpenAI-shape messages with native reasoning_details echo.

    For assistant turns that carry a MiniMax native_assistant_payload, replay
    `reasoning_content` + `reasoning_details` exactly as the model emitted
    them. For non-native assistant turns (fresh) we still include
    `reasoning_content` if present so behavior degrades gracefully.
    """
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
            nap = m.get("native_assistant_payload")
            npf = m.get("native_payload_format")
            if isinstance(nap, dict) and npf == NATIVE_FORMAT:
                if nap.get("reasoning_content") is not None:
                    asst["reasoning_content"] = nap.get("reasoning_content")
                if nap.get("reasoning_details") is not None:
                    asst["reasoning_details"] = nap.get("reasoning_details")
            else:
                rc = m.get("reasoning_content")
                if rc:
                    asst["reasoning_content"] = rc
            out.append(asst)
            continue
    return out


def _convert_usage(usage: dict | None) -> dict | None:
    if not usage:
        return None
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": prompt_details.get("cached_tokens"),
        "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
    }


def _from_minimax(resp: dict) -> dict:
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
    reasoning_details = msg.get("reasoning_details")
    raw_tcs = msg.get("tool_calls") or []
    parsed_tcs: list[dict] = []
    for tc in raw_tcs:
        fn = tc.get("function") or {}
        args_str = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Malformed MiniMax tool_call arguments (call_id=%s): %r",
                tc.get("id"), args_str,
            )
            args = {}
        parsed_tcs.append({
            "id": tc.get("id"),
            "name": fn.get("name"),
            "arguments": args,
        })

    native_payload: dict | None = None
    if reasoning_content is not None or reasoning_details is not None:
        native_payload = {
            "reasoning_content": reasoning_content,
            "reasoning_details": reasoning_details,
        }

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
            raise _RetryableMiniMaxError(
                f"MiniMax HTTP {exc.code}: {error_body[:500]}"
            ) from exc
        raise RuntimeError(
            f"MiniMax HTTP {exc.code}: {error_body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise _RetryableMiniMaxError(
            f"MiniMax connection failed: {exc}"
        ) from exc


@retry(
    retry=retry_if_exception_type(_RetryableMiniMaxError),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(7),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_minimax_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = "MiniMax-M2.7",
    temperature: float | None = 0.0,
    seed: int | None = None,
    reasoning_effort: str | None = None,  # noqa: ARG001 — M2 has no on/off dial
    response_format: dict | None = None,
) -> dict:
    """provider-native call for MiniMax-M2.7.

    `reasoning_effort` is accepted for signature parity with the
    dispatcher but ignored — M2 series has no intensity dial and
    always interleaves thinking. We always set `reasoning_split=true`.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(messages),
        "reasoning_split": True,
    }
    if tools:
        body["tools"] = tools
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed
    if response_format is not None:
        body["response_format"] = response_format

    url = f"{_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = _http_post_json(url, body, headers, timeout=360.0)
    if isinstance(resp, dict) and resp.get("error"):
        err = resp.get("error")
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise RuntimeError(f"MiniMax error: {msg}")
    return _from_minimax(resp)
