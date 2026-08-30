"""Gemini 3 native :generateContent adapter.

Endpoint: POST {GEMINI_BASE_URL}/models/{model}:generateContent (default
https://generativelanguage.googleapis.com/v1beta, the path Google's
official docs use — `ai.google.dev/gemini-api/docs/...`). Auth is sent as
a Bearer token; point GEMINI_BASE_URL at an OpenAI-style proxy that
accepts Bearer, or adapt the header for Google's native x-goog-api-key.

Protocol surface preserved:
- contents[] with role="user" | "model" and parts[].
- parts elements: {"text"}, {"functionCall": {name, args, id}},
  {"functionResponse": {name, response, id}}, plus the opaque
  {"thoughtSignature": ...} byte string Gemini 3 expects echoed back
  on tool-call turns.
- Native model parts (including thoughtSignature) are persisted to
  `Message.native_assistant_payload` so byte-faithful multi-turn replay
  is structural, not thread-local.
- ATR-side STU dedup (P8b) may leave 1 ATR tool message paired with N
  native functionCall parts. The adapter detects this and fans out
  one functionResponse per native id, all sharing the same response
  content, so Gemini sees N call ↔ N response pairing.
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
# Google's official Gemini API; override to point at a compatible proxy.
_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

NATIVE_FORMAT = "gemini_v1beta"


class _RetryableGeminiError(RuntimeError):
    """Retryable provider error from Gemini :generateContent."""


# ── Conversion: OpenAI-format messages → Gemini contents ─────────────────────

def _to_gemini_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    function_declarations: list[dict] = []
    for t in tools:
        fn = t.get("function") or {}
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        function_declarations.append({
            "name": fn.get("name") or "",
            "description": fn.get("description") or "",
            "parameters": params,
        })
    return [{"functionDeclarations": function_declarations}]


def _parse_tool_response_content(raw: str) -> dict:
    """Parse a tool message content string into a dict for functionResponse.response."""
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"result": raw}
    if isinstance(obj, dict):
        return obj
    return {"result": obj}


def _to_gemini_contents(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-format messages → (systemInstruction text, contents).

    Implements fan-out: if the orchestrator deduped N STU
    tool calls to 1 ATR tool_call but `native_assistant_payload` retains
    N functionCalls of the same name, emit N functionResponse parts
    sharing the single ATR ack content.
    """
    asst_meta: dict[int, dict] = {}
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        atr_id_to_name: dict[str, str] = {}
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            atr_id_to_name[tc.get("id") or ""] = fn.get("name") or ""
        native_call_order: list[tuple[str, str]] = []
        nap = m.get("native_assistant_payload")
        npf = m.get("native_payload_format")
        if isinstance(nap, list) and npf == NATIVE_FORMAT:
            for p in nap:
                if not isinstance(p, dict):
                    continue
                fc = p.get("functionCall")
                if fc:
                    native_call_order.append(
                        (fc.get("name") or "", fc.get("id") or "")
                    )
        asst_meta[i] = {
            "atr_id_to_name": atr_id_to_name,
            "native_call_order": native_call_order,
        }

    system_text: str | None = None
    contents: list[dict] = []
    pending_tool_msgs: list[tuple[str, str]] = []  # (tc_id, raw_content)
    pending_owner_idx: int | None = None

    def _flush_tool_buffer() -> None:
        nonlocal pending_tool_msgs, pending_owner_idx
        if not pending_tool_msgs:
            return
        owner_idx = pending_owner_idx
        meta = asst_meta.get(owner_idx, {}) if owner_idx is not None else {}
        atr_id_to_name: dict[str, str] = meta.get("atr_id_to_name") or {}
        native_call_order: list[tuple[str, str]] = meta.get("native_call_order") or []

        name_to_atr: dict[str, list[dict]] = {}
        atr_order: list[tuple[str, dict]] = []
        for tc_id, raw in pending_tool_msgs:
            name = atr_id_to_name.get(tc_id, "")
            resp = _parse_tool_response_content(raw)
            name_to_atr.setdefault(name, []).append(resp)
            atr_order.append((name, resp))

        parts: list[dict] = []
        if native_call_order:
            consumed_by_name: dict[str, int] = {}
            for nm, nid in native_call_order:
                lst = name_to_atr.get(nm) or []
                if not lst:
                    resp = {"status": "no_result"}
                else:
                    idx = consumed_by_name.get(nm, 0)
                    if idx < len(lst):
                        resp = lst[idx]
                        consumed_by_name[nm] = idx + 1
                    else:
                        # Fan-out: native > ATR → reuse first ATR ack content.
                        resp = lst[0]
                fr: dict[str, Any] = {
                    "functionResponse": {"name": nm, "response": resp}
                }
                if nid:
                    fr["functionResponse"]["id"] = nid
                parts.append(fr)
        else:
            # No native echo (first-turn fallback): emit ATR-order 1:1.
            for nm, resp in atr_order:
                parts.append({"functionResponse": {"name": nm, "response": resp}})

        contents.append({"role": "user", "parts": parts})
        pending_tool_msgs = []
        pending_owner_idx = None

    last_assistant_idx: int | None = None
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "system":
            txt = m.get("content") or ""
            if txt:
                system_text = f"{system_text}\n\n{txt}" if system_text else txt
            continue
        if role == "user":
            _flush_tool_buffer()
            txt = m.get("content") or ""
            contents.append({"role": "user", "parts": [{"text": txt}]})
            continue
        if role == "assistant":
            _flush_tool_buffer()
            last_assistant_idx = i
            nap = m.get("native_assistant_payload")
            npf = m.get("native_payload_format")
            if isinstance(nap, list) and npf == NATIVE_FORMAT:
                # Byte-faithful echo: strip only the model-side `index`
                # annotation (not in the request schema); preserve
                # thoughtSignature, text, functionCall, args, id.
                parts: list[dict] = []
                for p in nap:
                    if isinstance(p, dict):
                        parts.append({k: v for k, v in p.items() if k != "index"})
                if parts:
                    contents.append({"role": "model", "parts": parts})
            else:
                # Cross-provider fallback (no native payload available).
                parts = []
                text = m.get("content")
                if text and text.strip():
                    parts.append({"text": text})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args_str = fn.get("arguments") or "{}"
                    try:
                        args_obj = (
                            json.loads(args_str)
                            if isinstance(args_str, str) else args_str
                        )
                    except (json.JSONDecodeError, TypeError):
                        args_obj = {}
                    parts.append({
                        "functionCall": {
                            "name": fn.get("name") or "",
                            "args": args_obj or {},
                        }
                    })
                if parts:
                    contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            pending_tool_msgs.append(
                (m.get("tool_call_id") or "", m.get("content") or "")
            )
            pending_owner_idx = last_assistant_idx
            continue

    _flush_tool_buffer()
    return system_text, contents


# ── Conversion: Gemini response → ATR caller dict ────────────────────────────

def _convert_usage(usage_meta: dict | None) -> dict | None:
    if not usage_meta:
        return None
    return {
        "prompt_tokens": usage_meta.get("promptTokenCount"),
        "completion_tokens": usage_meta.get("candidatesTokenCount"),
        "total_tokens": usage_meta.get("totalTokenCount"),
        "cached_tokens": usage_meta.get("cachedContentTokenCount"),
        "prompt_cache_hit_tokens": usage_meta.get("cachedContentTokenCount"),
        "reasoning_tokens": usage_meta.get("thoughtsTokenCount"),
    }


def _normalize_arg_key(key: str) -> str:
    """Normalize Gemini/proxy's occasional quoted parameter keys.

    In live Gemini 3.1 runs proxy has returned functionCall args like
    {"\"thread_id\"": "..."} instead of {"thread_id": "..."}.
    The native payload still keeps the provider bytes for replay, but the
    ATR-facing tool call must use executable schema keys.
    """
    stripped = key.strip()
    if len(stripped) < 2:
        return key
    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if isinstance(decoded, str):
        return decoded
    if stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return key


def _normalize_arg_keys(value: Any) -> Any:
    """Recursively normalize function-call argument object keys.

    Only keys are normalized. Values remain untouched so IDs, prose, and
    user-visible text are not changed by the adapter.
    """
    if isinstance(value, dict):
        return {
            _normalize_arg_key(str(k)): _normalize_arg_keys(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_normalize_arg_keys(v) for v in value]
    return value


def _from_gemini(resp: dict) -> dict:
    """Convert Gemini :generateContent response → caller dict."""
    candidates = resp.get("candidates") or []
    if not candidates:
        return {
            "content": None,
            "tool_calls": None,
            "reasoning_content": None,
            "native_assistant_payload": None,
            "native_payload_format": None,
            "usage": _convert_usage(resp.get("usageMetadata")),
        }
    parts = ((candidates[0].get("content") or {}).get("parts")) or []

    text_parts: list[str] = []
    tool_calls_out: list[dict] = []
    synthetic_idx = 0
    for p in parts:
        if not isinstance(p, dict):
            continue
        if "text" in p:
            text_parts.append(p.get("text") or "")
        fc = p.get("functionCall")
        if fc:
            name = fc.get("name") or "fn"
            given_id = fc.get("id")
            # proxy strips functionCall.id; synthesize a stable id so
            # ATR's tool_call_id pairing works downstream. The native
            # part retains its original id (None), so byte-faithful echo
            # is unaffected.
            if given_id:
                tc_id = given_id
            else:
                tc_id = f"gemini_{name}_{synthetic_idx}"
                synthetic_idx += 1
            tool_calls_out.append({
                "id": tc_id,
                "name": name,
                "arguments": _normalize_arg_keys(fc.get("args") or {}),
            })

    content_str = "".join(text_parts) if text_parts else None
    if content_str is not None and not content_str.strip():
        content_str = None

    return {
        "content": content_str,
        "tool_calls": tool_calls_out or None,
        "reasoning_content": None,
        "native_assistant_payload": parts,
        "native_payload_format": NATIVE_FORMAT,
        "usage": _convert_usage(resp.get("usageMetadata")),
    }


# ── Reasoning controls ──────────────────────────────────────────────────────

def _apply_thinking_level(
    generation_config: dict[str, Any],
    override: str | None,
) -> None:
    """Map ATR's binary reasoning override → Gemini 3 thinkingLevel.

    "on"  → "high"
    "off" → "low"   (Gemini 3 does not accept "none"; "low" is the docs-
                     accepted minimum on v1beta)
    None  → omit (vendor default)
    """
    if override is None:
        return
    if override == "on":
        level = "high"
    elif override == "off":
        level = "low"
    else:
        raise ValueError(
            f"reasoning_effort must be 'on' | 'off' | None, got {override!r}"
        )
    tc = generation_config.setdefault("thinkingConfig", {})
    tc["thinkingLevel"] = level


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
            raise _RetryableGeminiError(
                f"Gemini :generateContent HTTP {exc.code}: {error_body[:500]}"
            ) from exc
        raise RuntimeError(
            f"Gemini :generateContent HTTP {exc.code}: {error_body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise _RetryableGeminiError(
            f"Gemini :generateContent connection failed: {exc}"
        ) from exc
    except TimeoutError as exc:
        raise _RetryableGeminiError(
            f"Gemini :generateContent read timed out: {exc}"
        ) from exc


@retry(
    retry=retry_if_exception_type(_RetryableGeminiError)
    | retry_if_exception(lambda exc: isinstance(exc, urllib.error.URLError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(7),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_gemini_native_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = "gemini-3-flash-preview",
    temperature: float | None = 0.0,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
) -> dict:
    """provider-native call for Gemini 3 ids.

    Returns the caller-shape dict described in {content, tool_calls, reasoning_content,
         native_assistant_payload, native_payload_format, usage}
    """
    system_text, contents = _to_gemini_contents(messages)
    body: dict[str, Any] = {"contents": contents}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}

    g_tools = _to_gemini_tools(tools)
    if g_tools:
        body["tools"] = g_tools

    generation_config: dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if seed is not None:
        generation_config["seed"] = seed
    _apply_thinking_level(generation_config, reasoning_effort)
    if response_format and response_format.get("type") == "json_object":
        generation_config["responseMimeType"] = "application/json"
    if generation_config:
        body["generationConfig"] = generation_config

    url = f"{_BASE_URL}/models/{model}:generateContent"
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = _http_post_json(url, body, headers, timeout=360.0)
    if isinstance(resp, dict) and resp.get("error"):
        err = resp.get("error")
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise RuntimeError(f"Gemini :generateContent error: {msg}")
    return _from_gemini(resp)
