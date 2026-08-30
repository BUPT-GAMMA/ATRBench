"""Anthropic native /v1/messages adapter.

Endpoint: POST {ANTHROPIC_BASE_URL}/v1/messages (default
https://api.anthropic.com, Anthropic's official Messages API).

Validated byte-faithful behavior for claude-opus-4-7:
  - thinking blocks (incl. signature) returned alongside tool_use
  - multi-turn round-trip with prior content[] echoed back is accepted
    (no "Invalid signature in thinking block" 400)

Opus 4.7 specifics:
  - Manual extended thinking `{type:"enabled", budget_tokens:N}` is REJECTED
    with HTTP 400 on 4.7. Must use adaptive thinking `{type:"adaptive"}`
    (the only supported mode).
  - `thinking.display` defaults to "omitted" on 4.7; we request "summarized"
    so the thinking text is captured for persistence + debugging.
  - `output_config.effort` (max | xhigh | high | medium | low) controls
    thinking depth. We map ATR's binary reasoning override:
      "on"  → output_config.effort = "high"
      "off" → output_config.effort = "low" (adaptive may still skip
              thinking for trivial queries; closest analogue to "off" on
              a model that has no true thinking-disabled switch)
      None  → omit output_config (vendor default = high)

Multi-turn replay:
  - Native content[] is the source of truth: thinking + text + tool_use blocks all
    persist into `native_assistant_payload` for byte-faithful echo.
  - Anthropic requires that every assistant turn containing tool_use carry
    its prior thinking block(s) (signature) unchanged on the next request.
    We splat native_assistant_payload back into messages[] assistant turn
    content[] without modification.
  - ATR-side STU dedup (P8b): a single ATR tool message paired with N native
    tool_use blocks of the same name fans out into N tool_result blocks,
    all sharing the ATR ack content.
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
# Anthropic's official Messages API; override to use a proxy.
_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

NATIVE_FORMAT = "anthropic_messages_v1"

# Default max_tokens budget. Opus 4.7 spends both thinking + visible output
# from this single budget; 16384 leaves comfortable headroom for adaptive
# thinking on agentic multi-step tasks. Callers cannot override yet (lib.llm
# call_llm_with_tools signature has no max_tokens), so a single conservative
# default is the simplest stable contract.
_DEFAULT_MAX_TOKENS = 16384
_TAIL_ASSISTANT_CONTINUATION = (
    "Continue the task. If it is complete, call finish_session."
)


class _RetryableAnthropicError(RuntimeError):
    """Retryable provider error from proxy Anthropic passthrough."""


# ── OpenAI Chat → Anthropic content blocks ───────────────────────────────────

def _to_anthropic_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        fn = t.get("function") or {}
        out.append({
            "name": fn.get("name") or "",
            "description": fn.get("description") or "",
            "input_schema": fn.get("parameters") or {
                "type": "object", "properties": {},
            },
        })
    # Prompt-caching breakpoint:
    # marking the LAST tool with cache_control tells Anthropic to cache the
    # entire prefix [system + tools]. ATR's system prompt (~1.6k) + tools
    # schema (~8-10k) comfortably exceeds the Opus 4.7 minimum cacheable
    # prefix of 4,096 tokens. The default 5-min TTL is used: its write price
    # is cheaper than the 1h TTL, whose per-session prefix-write cost
    # dominates spend in the raw memory layer (each LS accumulates new
    # prior-transcript blocks).
    #
    # Why only one breakpoint: Anthropic caps explicit cache_control blocks
    # at 4 per request. One on last tool gives us the [system + tools]
    # prefix, which is the largest stable shared chunk. Adding a second
    # breakpoint on the last message block would also cache message history
    # but ATR's messages mutate every turn so the hit rate is marginal.
    if out:
        out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def _to_anthropic_messages(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-Chat messages → (system text, Anthropic messages[]).

    Assistant turns with `native_assistant_payload` + matching
    NATIVE_FORMAT splat content[] back byte-faithfully (preserves thinking
    + signature). Without native payload we fall back to reconstruction
    from flat tool_calls — usable only when no prior assistant turn has
    tool_use (otherwise next request will 400 on missing signature).

    ATR P8b fan-out: 1 ATR tool message paired with N native tool_use
    blocks of the same name → emit N tool_result blocks sharing the ATR ack.
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
            for blk in nap:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "tool_use":
                    native_call_order.append(
                        (blk.get("name") or "", blk.get("id") or "")
                    )
        asst_meta[i] = {
            "atr_id_to_name": atr_id_to_name,
            "native_call_order": native_call_order,
        }

    system_parts: list[str] = []
    ant_msgs: list[dict] = []
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

        name_to_atr: dict[str, list[str]] = {}
        atr_order: list[tuple[str, str, str]] = []
        for tc_id, raw in pending_tool_msgs:
            nm = atr_id_to_name.get(tc_id, "")
            name_to_atr.setdefault(nm, []).append(raw)
            atr_order.append((tc_id, nm, raw))

        results_content: list[dict] = []
        if native_call_order:
            consumed_by_name: dict[str, int] = {}
            for nm, native_id in native_call_order:
                lst = name_to_atr.get(nm) or []
                if not lst:
                    content_str = "no_result"
                else:
                    idx = consumed_by_name.get(nm, 0)
                    if idx < len(lst):
                        content_str = lst[idx]
                        consumed_by_name[nm] = idx + 1
                    else:
                        # Fan-out: native > ATR → reuse first ATR ack content.
                        content_str = lst[0]
                results_content.append({
                    "type": "tool_result",
                    "tool_use_id": native_id,
                    "content": content_str,
                })
        else:
            for tc_id, _, raw in atr_order:
                results_content.append({
                    "type": "tool_result",
                    "tool_use_id": tc_id,
                    "content": raw,
                })

        ant_msgs.append({"role": "user", "content": results_content})
        pending_tool_msgs = []
        pending_owner_idx = None

    def _has_tool_use_blocks(msg: dict) -> bool:
        return any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in (msg.get("content") or [])
        )

    last_assistant_idx: int | None = None
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "system":
            txt = m.get("content") or ""
            if txt:
                system_parts.append(txt)
            continue
        if role == "user":
            _flush_tool_buffer()
            txt = m.get("content") or ""
            ant_msgs.append({
                "role": "user",
                "content": [{"type": "text", "text": txt}] if txt else [],
            })
            continue
        if role == "assistant":
            _flush_tool_buffer()
            last_assistant_idx = i
            nap = m.get("native_assistant_payload")
            npf = m.get("native_payload_format")
            if isinstance(nap, list) and npf == NATIVE_FORMAT:
                # Byte-faithful echo: thinking + text + tool_use unchanged.
                ant_msgs.append({"role": "assistant", "content": list(nap)})
            else:
                # Cross-provider fallback. No thinking block to
                # echo; safe for first turn or non-tool-use history.
                content_blocks: list[dict] = []
                text = m.get("content")
                if text and text.strip():
                    content_blocks.append({"type": "text", "text": text})
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
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "input": args_obj or {},
                    })
                if content_blocks:
                    ant_msgs.append({
                        "role": "assistant", "content": content_blocks,
                    })
            continue
        if role == "tool":
            pending_tool_msgs.append(
                (m.get("tool_call_id") or "", m.get("content") or "")
            )
            pending_owner_idx = last_assistant_idx
            continue

    _flush_tool_buffer()
    if (
        ant_msgs
        and ant_msgs[-1].get("role") == "assistant"
        and not _has_tool_use_blocks(ant_msgs[-1])
    ):
        # Opus 4.7 rejects assistant-message prefill. ATR test sessions can
        # legitimately continue after a plain assistant text turn because the
        # user is offline and termination is via finish_session(); append a
        # request-local continuation turn instead of persisting anything.
        ant_msgs.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": _TAIL_ASSISTANT_CONTINUATION,
            }],
        })
    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, ant_msgs


# ── Anthropic response → ATR caller dict ────────────────────────────────────

def _convert_usage(usage: dict | None) -> dict | None:
    if not usage:
        return None
    return {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        "cached_tokens": usage.get("cache_read_input_tokens"),
        "prompt_cache_hit_tokens": usage.get("cache_read_input_tokens"),
        # Anthropic does not break thinking tokens out from output_tokens
        # (billed but not separately reported), so reasoning_tokens stays None.
        # cost-payoff diagnostic underreports Claude thinking spend accordingly.
        "reasoning_tokens": None,
    }


def _from_anthropic(resp: dict) -> dict:
    content = resp.get("content") or []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_out: list[dict] = []
    for blk in content:
        if not isinstance(blk, dict):
            continue
        btype = blk.get("type")
        if btype == "text":
            text_parts.append(blk.get("text") or "")
        elif btype == "thinking":
            # display="omitted" → thinking text is "" but signature carries
            # the encrypted full thinking; we capture both via native_payload.
            reasoning_parts.append(blk.get("thinking") or "")
        elif btype == "tool_use":
            tool_calls_out.append({
                "id": blk.get("id"),
                "name": blk.get("name"),
                "arguments": blk.get("input") or {},
            })

    content_str = "".join(text_parts) if text_parts else None
    if content_str is not None and not content_str.strip():
        content_str = None
    reasoning_str = "".join(reasoning_parts) if reasoning_parts else None
    if reasoning_str is not None and not reasoning_str.strip():
        reasoning_str = None

    return {
        "content": content_str,
        "tool_calls": tool_calls_out or None,
        "reasoning_content": reasoning_str,
        # Source of truth: full content[] persisted for byte-faithful multi-turn echo
        # (incl. thinking block signature). Caller MUST round-trip this
        # back as messages[].native_assistant_payload to avoid 400 on
        # subsequent tool-use turns.
        "native_assistant_payload": content,
        "native_payload_format": NATIVE_FORMAT,
        "usage": _convert_usage(resp.get("usage")),
    }


# ── HTTP transport with retry ────────────────────────────────────────────────

def _http_post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {429, 500, 502, 503, 504}:
            raise _RetryableAnthropicError(
                f"Anthropic native HTTP {exc.code}: {error_body[:500]}"
            ) from exc
        raise RuntimeError(
            f"Anthropic native HTTP {exc.code}: {error_body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise _RetryableAnthropicError(
            f"Anthropic native connection failed: {exc}"
        ) from exc
    except (
        # `urllib.urlopen` can raise these BELOW the URLError wrapper when the
        # remote upstream closes the TCP socket mid-stream, surfacing as
        # `RemoteDisconnected: Remote end closed connection without
        # response`. Without this catch they bubble up as ATR `agent LLM
        # retry exhausted` SweepAbort, killing the cell.
        http.client.RemoteDisconnected,
        http.client.IncompleteRead,
        http.client.BadStatusLine,
        ConnectionError,   # ConnectionResetError / Aborted / Refused / BrokenPipe
    ) as exc:
        raise _RetryableAnthropicError(
            f"Anthropic native connection dropped: {type(exc).__name__}: {exc}"
        ) from exc
    except TimeoutError as exc:  # includes socket.timeout (alias in 3.10+)
        raise _RetryableAnthropicError(
            f"Anthropic native read timed out: {exc}"
        ) from exc


# ── Reasoning controls ───────────────────────────────────────────────────────

def _apply_thinking(body: dict, override: str | None) -> None:
    """Map ATR's binary reasoning override → Opus 4.7 thinking config.

    Opus 4.7 only accepts adaptive thinking. We always include the
    `thinking` block; the `output_config.effort` knob shapes how often
    Claude chooses to think.
    """
    body["thinking"] = {"type": "adaptive", "display": "summarized"}
    if override is None:
        return
    if override == "on":
        body["output_config"] = {"effort": "high"}
    elif override == "off":
        body["output_config"] = {"effort": "low"}
    else:
        raise ValueError(
            f"reasoning_effort must be 'on' | 'off' | None, got {override!r}"
        )


@retry(
    retry=retry_if_exception_type(_RetryableAnthropicError),
    wait=wait_exponential(multiplier=1, min=2, max=90),
    stop=stop_after_attempt(10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_anthropic_native_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = "claude-opus-4-7",
    temperature: float | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    response_format: dict | None = None,  # noqa: ARG001 — no equivalent on /v1/messages
) -> dict:
    """provider-native call for claude-opus-4-7 via proxy
    passthrough to Anthropic /v1/messages.

    `temperature` and `seed` are accepted for dispatcher signature parity
    but ignored — adaptive thinking on Opus 4.7 manages its own sampling
    and the Anthropic API has no `seed` parameter.

    Returns caller-shape dict. `native_assistant_payload`
    carries the entire response content[] array including thinking blocks
    + signature so the caller can byte-faithfully echo it on subsequent
    turns (required for any tool_use round-trip).
    """
    del temperature, seed  # explicitly unused on Anthropic native
    system_text, ant_messages = _to_anthropic_messages(messages)
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "messages": ant_messages,
    }
    if system_text:
        # System as array-of-blocks with cache_control (default 5m TTL).
        # Placing cache_control here caches the system prefix. Combined
        # with the cache_control on the last tool we get the recommended
        # double-breakpoint pattern over [system → tools].
        # Opus 4.7 only caches prefixes >= 4096 tokens; under that it
        # silent-skips with no error and 0 cache fields.
        body["system"] = [{
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }]
    ant_tools = _to_anthropic_tools(tools)
    if ant_tools:
        body["tools"] = ant_tools
    _apply_thinking(body, reasoning_effort)

    url = f"{_BASE_URL}/v1/messages"
    headers = {
        "x-api-key": _API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    resp = _http_post_json(url, body, headers, timeout=360.0)
    if isinstance(resp, dict) and resp.get("error"):
        err = resp.get("error")
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise RuntimeError(f"Anthropic native error: {msg}")
    return _from_anthropic(resp)
