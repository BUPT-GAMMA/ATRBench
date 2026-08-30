"""OpenAI Responses API adapter.

Endpoint: POST {OPENAI_BASE_URL}/responses (default
https://api.openai.com/v1, OpenAI's official Responses API).

Prompt caching:
  - `prompt_cache_retention="24h"` extends OpenAI's default in-memory TTL
    (5-10 minutes idle, max 1 hour) to a 24-hour retention window. Lets
    cache prefixes survive the multi-hour sweep runs ATR does.
  - `prompt_cache_key` provides a routing-affinity hint (combined with
    prefix hash) so the same workload lands on the same backend shard
    and improves hit rate. Per OpenAI docs, keep each unique
    prefix+key combination below 15 requests/min to avoid cache overflow.
  - Both fields are silently no-op'd by upstream if prefix < 1024 tokens
    (OpenAI's minimum cacheable size).

The adapter honors the official OpenAI Responses multi-turn protocol —
every `function_call` is paired with `function_call_output`. We request
`include=["reasoning.encrypted_content"]` + `store=false` on every call;
responses with or without reasoning items are handled transparently (a
proxy may strip them).

Multi-turn replay:
- Persist the full returned `output[]` list as `native_assistant_payload`
  so that on the next turn we splat it back into `input[]` byte-faithfully.
  Reasoning items, when present, carry forward; the message/function_call
  items always suffice on their own.
- For P8b (multi-STU dedup), ATR's single tool message must fan out into
  one `function_call_output` per native `function_call` that shares the
  same tool name. The adapter handles this in `_to_responses_input`.
"""
from __future__ import annotations

import http.client
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
# OpenAI's official Responses API; override to use a proxy.
_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

NATIVE_FORMAT = "openai_responses_v1"


class _RetryableOpenAIError(RuntimeError):
    """Retryable provider error from OpenAI Responses."""


# ── Tool schema conversion ──────────────────────────────────────────────────

def _to_responses_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert OpenAI-Chat tool shape → Responses-flat tool shape."""
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        fn = t.get("function") or {}
        out.append({
            "type": "function",
            "name": fn.get("name") or "",
            "description": fn.get("description") or "",
            "parameters": fn.get("parameters") or {
                "type": "object",
                "properties": {},
            },
            "strict": False,
        })
    return out


# ── Message conversion ──────────────────────────────────────────────────────

def _to_responses_input(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-Chat messages → (instructions text, input[] items).

    Each system message contributes to `instructions`. Other roles become
    Responses input items. Assistant turns prefer native byte-faithful
    splat from `native_assistant_payload`; fall back to reconstruction
    when absent. fan-out: a single ATR tool message paired with N native
    function_calls of the same name produces N function_call_output items.
    """
    asst_meta: dict[int, dict] = {}
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        atr_id_to_name: dict[str, str] = {}
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            atr_id_to_name[tc.get("id") or ""] = fn.get("name") or ""
        native_calls_in_order: list[tuple[str, str]] = []
        nap = m.get("native_assistant_payload")
        npf = m.get("native_payload_format")
        if isinstance(nap, list) and npf == NATIVE_FORMAT:
            for item in nap:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "function_call":
                    native_calls_in_order.append(
                        (item.get("name") or "", item.get("call_id") or "")
                    )
        asst_meta[i] = {
            "atr_id_to_name": atr_id_to_name,
            "native_calls_in_order": native_calls_in_order,
        }

    instructions_parts: list[str] = []
    input_items: list[dict] = []
    pending_tool_msgs: list[tuple[str, str]] = []  # (tc_id, raw_content)
    pending_owner_idx: int | None = None

    def _flush_tool_buffer() -> None:
        nonlocal pending_tool_msgs, pending_owner_idx
        if not pending_tool_msgs:
            return
        meta = (
            asst_meta.get(pending_owner_idx, {})
            if pending_owner_idx is not None else {}
        )
        atr_id_to_name: dict[str, str] = meta.get("atr_id_to_name") or {}
        native_calls: list[tuple[str, str]] = meta.get("native_calls_in_order") or []

        name_to_atr: dict[str, list[str]] = {}
        atr_order: list[tuple[str, str, str]] = []  # (tc_id, name, raw)
        for tc_id, raw in pending_tool_msgs:
            nm = atr_id_to_name.get(tc_id, "")
            name_to_atr.setdefault(nm, []).append(raw)
            atr_order.append((tc_id, nm, raw))

        if native_calls:
            consumed_by_name: dict[str, int] = {}
            for nm, native_call_id in native_calls:
                lst = name_to_atr.get(nm) or []
                if not lst:
                    output = json.dumps({"status": "no_result"})
                else:
                    idx = consumed_by_name.get(nm, 0)
                    if idx < len(lst):
                        output = lst[idx]
                        consumed_by_name[nm] = idx + 1
                    else:
                        output = lst[0]
                input_items.append({
                    "type": "function_call_output",
                    "call_id": native_call_id,
                    "output": output,
                })
        else:
            for tc_id, _, raw in atr_order:
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tc_id,
                    "output": raw,
                })

        pending_tool_msgs = []
        pending_owner_idx = None

    last_assistant_idx: int | None = None
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "system":
            txt = m.get("content") or ""
            if txt:
                instructions_parts.append(txt)
            continue
        if role == "user":
            _flush_tool_buffer()
            input_items.append({
                "type": "message",
                "role": "user",
                "content": m.get("content") or "",
            })
            continue
        if role == "assistant":
            _flush_tool_buffer()
            last_assistant_idx = i
            nap = m.get("native_assistant_payload")
            npf = m.get("native_payload_format")
            if isinstance(nap, list) and npf == NATIVE_FORMAT:
                for item in nap:
                    if isinstance(item, dict):
                        input_items.append(item)
            else:
                text = m.get("content")
                if text and text.strip():
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": text,
                    })
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": fn.get("arguments") or "{}",
                    })
            continue
        if role == "tool":
            pending_tool_msgs.append(
                (m.get("tool_call_id") or "", m.get("content") or "")
            )
            pending_owner_idx = last_assistant_idx
            continue

    _flush_tool_buffer()

    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, input_items


# ── Response parsing ────────────────────────────────────────────────────────

def _convert_usage(usage: dict | None) -> dict | None:
    if not usage:
        return None
    output_details = usage.get("output_tokens_details") or {}
    input_details = usage.get("input_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": input_details.get("cached_tokens"),
        "prompt_cache_hit_tokens": input_details.get("cached_tokens"),
        "reasoning_tokens": output_details.get("reasoning_tokens"),
    }


def _from_responses(resp: dict) -> dict:
    output = resp.get("output") or []

    text_parts: list[str] = []
    tool_calls_out: list[dict] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            # content may be a string or a list of content blocks.
            c = item.get("content")
            if isinstance(c, str):
                text_parts.append(c)
            elif isinstance(c, list):
                for blk in c:
                    if not isinstance(blk, dict):
                        continue
                    txt = blk.get("text")
                    if txt:
                        text_parts.append(txt)
        elif itype == "function_call":
            args_str = item.get("arguments") or "{}"
            try:
                args = (
                    json.loads(args_str)
                    if isinstance(args_str, str) else (args_str or {})
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Malformed Responses function_call arguments (call_id=%s): %r",
                    item.get("call_id"), args_str,
                )
                args = {}
            tool_calls_out.append({
                "id": item.get("call_id") or "",
                "name": item.get("name") or "",
                "arguments": args,
            })

    content_str = "".join(text_parts) if text_parts else None
    if content_str is not None and not content_str.strip():
        content_str = None

    return {
        "content": content_str,
        "tool_calls": tool_calls_out or None,
        "reasoning_content": None,
        "native_assistant_payload": output,
        "native_payload_format": NATIVE_FORMAT,
        "usage": _convert_usage(resp.get("usage")),
    }


# ── HTTP transport with retry ───────────────────────────────────────────────

def _http_post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {429, 500, 502, 503, 504}:
            raise _RetryableOpenAIError(
                f"OpenAI Responses HTTP {exc.code}: {error_body[:500]}"
            ) from exc
        raise RuntimeError(
            f"OpenAI Responses HTTP {exc.code}: {error_body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise _RetryableOpenAIError(
            f"OpenAI Responses connection failed: {exc}"
        ) from exc
    except (
        # `urllib.urlopen` raises these BELOW the URLError wrapper when the
        # remote (proxy / OpenAI upstream) closes the TCP socket
        # mid-stream. Without this catch the error bubbles up as ATR
        # `agent LLM retry exhausted` SweepAbort, killing the cell.
        http.client.RemoteDisconnected,
        http.client.IncompleteRead,
        http.client.BadStatusLine,
        ConnectionError,   # ConnectionResetError / Aborted / Refused / BrokenPipe
    ) as exc:
        raise _RetryableOpenAIError(
            f"OpenAI Responses connection dropped: {type(exc).__name__}: {exc}"
        ) from exc
    except TimeoutError as exc:  # includes socket.timeout (alias in 3.10+)
        raise _RetryableOpenAIError(
            f"OpenAI Responses read timed out: {exc}"
        ) from exc


@retry(
    retry=retry_if_exception_type(_RetryableOpenAIError),
    wait=wait_exponential(multiplier=1, min=2, max=90),
    stop=stop_after_attempt(10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_openai_responses_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = "gpt-5.4",
    temperature: float | None = 0.0,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    response_format: dict | None = None,  # noqa: ARG001 — Responses uses different shape
) -> dict:
    """provider-native call for gpt-5.4 via Responses API.

    `reasoning_effort`: "on" → high; "off" → low; None → omit.
    """
    instructions, input_items = _to_responses_input(messages)
    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        # Extend prompt-cache TTL from default 5-10min in_memory → 24h
        # window (OpenAI Responses extended cache; proxy passes the field
        # through to upstream, echoed in response).
        # No-op silently if prefix < 1024 tokens (OpenAI minimum).
        # Not setting `prompt_cache_key` — letting OpenAI route by prefix
        # hash avoids the 15 req/min/key overflow risk that an over-broad
        # key (e.g. one per model) would trigger across a parallel sweep.
        "prompt_cache_retention": "24h",
    }
    if instructions:
        body["instructions"] = instructions

    rt_tools = _to_responses_tools(tools)
    if rt_tools:
        body["tools"] = rt_tools
        body["tool_choice"] = "auto"

    if reasoning_effort is not None:
        if reasoning_effort == "on":
            body["reasoning"] = {"effort": "high"}
        elif reasoning_effort == "off":
            body["reasoning"] = {"effort": "low"}
        else:
            raise ValueError(
                f"reasoning_effort must be 'on' | 'off' | None, got {reasoning_effort!r}"
            )

    # `temperature` and `seed` are silently dropped on Responses API for
    # reasoning models — OpenAI rejects `temperature` with HTTP 400
    # ("Unsupported parameter: 'temperature' is not supported with this
    # model") on gpt-5.x reasoning channels. gpt-5.4 is a reasoning model,
    # so blanket-dropping is safe.
    _ = temperature, seed  # accepted for dispatcher signature parity

    url = f"{_BASE_URL}/responses"
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = _http_post_json(url, body, headers, timeout=360.0)
    if isinstance(resp, dict) and resp.get("error"):
        err = resp.get("error")
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise RuntimeError(f"OpenAI Responses error: {msg}")
    return _from_responses(resp)
