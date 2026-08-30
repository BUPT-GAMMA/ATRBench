"""LLM API client for ATR pipeline.

`call_llm_with_tools`
now dispatches per-model to provider-native adapters that speak each
upstream's official multi-turn tool-calling protocol byte-faithfully.
The unified OpenAI Chat Completions abstraction is retained for
`call_llm` / `call_llm_json` / `call_llm_yaml` (single-shot generations
where multi-turn vendor state has no observer).

Per-model native dispatch in `call_llm_with_tools`:

  - `gpt-5.4`                → lib.llm_openai_responses  (Responses API)
  - `claude-opus-4-7`        → lib.llm_anthropic_native  (/v1/messages,
                                                          adaptive thinking + signature
                                                          byte-faithful across multi-turn)
  - `gemini-3-flash-preview` → lib.llm_gemini_native      (generateContent)
  - `gemini-3.1-pro-preview` → lib.llm_gemini_native
  - `MiniMax-M2.7`           → lib.llm_minimax            (reasoning_split)
  - `qwen3.6-plus`           → lib.llm_qwen               (enable_thinking)
  - `deepseek-v4-pro`        → lib.llm_deepseek           (Chat Completions)
  - `deepseek-v4-flash`      → lib.llm_deepseek

Single-shot text generation (`call_llm`, `call_llm_json`,
`call_llm_yaml`) still walks the unified OpenAI-compatible path via
`_resolve_client`.

Reasoning policy:
  - `reasoning_effort="on"`: use each reasoning model's high-thinking
    setting.
  - `reasoning_effort="off"`: disable thinking when the provider supports a
    true off switch; Gemini 3 Flash uses its minimum supported thinking level
    (`minimal`) because the model does not accept `none`.
  - `reasoning_effort=None`: no param sent, the vendor's own default kicks in.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import logging
import os
import threading
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from openai import OpenAI, APIStatusError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class _RetryableRawLLMError(RuntimeError):
    """Retryable provider error from a raw HTTP LLM request."""


# ── Provider configs ──────────────────────────────────────────────────────────
#
# Each provider's API key + base URL come from environment variables, read
# at import time (set them before launch). BASE_URL defaults to the vendor's
# official endpoint; point <PROVIDER>_BASE_URL at an OpenAI-compatible
# proxy to route every model through one instead.
#
#   OPENAI_API_KEY    / OPENAI_BASE_URL     (gpt-5.4)
#   GEMINI_API_KEY    / GEMINI_BASE_URL     (gemini-3.x; GEMINI_CHAT_BASE_URL
#                                            drives the single-shot chat path,
#                                            default {GEMINI_BASE_URL}/openai)
#   ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL  (claude-opus-4-7)
#   DEEPSEEK_API_KEY  / DEEPSEEK_BASE_URL   (deepseek-v4-pro / -flash)
#   DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL  (qwen3.6-plus)
#   MINIMAX_API_KEY   / MINIMAX_BASE_URL    (MiniMax-M2.7)
#
# The native tool-calling adapters in lib/llm_*.py read the same vars
# independently; the config here drives the single-shot unified path
# (call_llm / call_llm_json / call_llm_yaml) via _resolve_client.

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key) or default


# model id → provider tag (drives both single-shot and native dispatch).
_MODEL_PROVIDER: dict[str, str] = {
    "gpt-5.4": "openai",
    "gemini-3-flash-preview": "gemini",
    "gemini-3.1-pro-preview": "gemini",
    "claude-opus-4-7": "anthropic",
    "deepseek-v4-pro": "deepseek",
    "deepseek-v4-flash": "deepseek",
    "qwen3.6-plus": "dashscope",
    "MiniMax-M2.7": "minimax",
}

# DeepSeek ids opt into a thinking-mode body extension in the reasoning helpers.
_DEEPSEEK_MODELS = frozenset(m for m, p in _MODEL_PROVIDER.items() if p == "deepseek")


def _chat_config(provider: str) -> tuple[str, str]:
    """(api_key, base_url) for the single-shot OpenAI-compatible Chat path.

    Gemini's chat-compatible layer lives under `/openai`; its native
    generateContent path is handled separately in lib/llm_gemini_native.py.
    """
    if provider == "openai":
        return _env("OPENAI_API_KEY"), _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if provider == "gemini":
        native = _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        return _env("GEMINI_API_KEY"), _env("GEMINI_CHAT_BASE_URL", native.rstrip("/") + "/openai")
    if provider == "anthropic":
        return _env("ANTHROPIC_API_KEY"), _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    if provider == "deepseek":
        return _env("DEEPSEEK_API_KEY"), _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    if provider == "dashscope":
        return _env("DASHSCOPE_API_KEY"), _env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if provider == "minimax":
        return _env("MINIMAX_API_KEY"), _env("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
    raise ValueError(f"Unknown provider {provider!r}")

DEFAULT_MODEL = "gpt-5.4"

GPT              = "gpt-5.4"
DEEPSEEK_V4      = "deepseek-v4-pro"
DEEPSEEK_V4F     = "deepseek-v4-flash"
GEMINI           = "gemini-3-flash-preview"
GEMINI_PRO       = "gemini-3.1-pro-preview"
QWEN             = "qwen3.6-plus"
CLAUDE_OPUS_4_7  = "claude-opus-4-7"
MINIMAX          = "MiniMax-M2.7"

# Convenience aliases for shell ergonomics. CLI users can pass `--models
# opus-4.7` and it is normalised to the canonical id before any client /
# API call. Path naming (runner/paths.py:_safe_model) sees the canonical
# form, so cell dirs are stable regardless of which alias the user typed.
_MODEL_ALIASES: dict[str, str] = {
    "claude-opus-4.7": "claude-opus-4-7",
    "opus-4.7": "claude-opus-4-7",
}


def _canonical_model(model: str | None) -> str | None:
    """Resolve user-typed alias to the canonical model name used by both
    the client router and the upstream API. Idempotent on canonical names.
    """
    if model is None:
        return None
    return _MODEL_ALIASES.get(model, model)

# ── Internal helpers ──────────────────────────────────────────────────────────

_CLIENT_CACHE: dict[str, OpenAI] = {}
_GPT_PROMPT_CACHE_RETENTION = "24h"
_PROMPT_CACHE_MAX_CHARS = 4096


# ── Runner token accounting (thread-local) ───────────────────────────────────
# token_scope(bucket, role) attributes calls to agent, user_sim, or classifier.

_token_state = threading.local()


def _current_token_scope() -> tuple[dict, str] | None:
    stack: list[tuple[dict, str]] = getattr(_token_state, "stack", [])
    return stack[-1] if stack else None


USAGE_FIELDS = (
    "prompt_tokens",          # total input (cached + uncached)
    "completion_tokens",      # total output
    "total_tokens",           # provider-reported sum
    "cached_tokens",          # OpenAI-style: prompt_tokens_details.cached_tokens
    "prompt_cache_hit_tokens",  # DeepSeek-style: usage.prompt_cache_hit_tokens
    # Reasoning model thinking-token attribution. Reported by all 5
    # native adapters (gpt-5.4 / gemini-3.x / qwen / minimax / deepseek
    # all surface this either as `output_tokens_details.reasoning_tokens`,
    # `completion_tokens_details.reasoning_tokens`, or `usageMetadata.
    # thoughtsTokenCount`). Excluding it under-counts reasoning-model
    # spend by 30-70% in the cost-payoff diagnostic.
    "reasoning_tokens",
)
# Backwards-compat alias for internal callers; new code should import USAGE_FIELDS.
_USAGE_FIELDS = USAGE_FIELDS


def _record_usage(usage: dict | None) -> None:
    """Accumulate one LLM call's usage into the current scope (if any).

    Captures both classical input/output counts and KV-cache attribution
    fields. evaluator/metrics.aggregate_token_usage mirrors the same field
    set, so anything written here flows through to per-cell eval.json and
    the sweep summary.
    """
    if usage is None:
        return
    scope = _current_token_scope()
    if scope is None:
        return
    bucket, role = scope
    sub = bucket.setdefault(role, {k: 0 for k in _USAGE_FIELDS} | {"calls": 0})
    for k in _USAGE_FIELDS:
        v = usage.get(k) or 0
        sub[k] = sub.get(k, 0) + (int(v) if v else 0)
    sub["calls"] += 1


@contextlib.contextmanager
def token_scope(bucket: dict, role: str):
    """Context manager: attribute LLM calls inside the `with` to bucket[role]."""
    stack: list = getattr(_token_state, "stack", None)
    if stack is None:
        stack = []
        _token_state.stack = stack
    stack.append((bucket, role))
    try:
        yield
    finally:
        stack.pop()


# Per-model concurrency cap shared by all runtime LLM calls.
PER_MODEL_MAX_CONCURRENCY = 30
_PER_MODEL_SEMAPHORES: dict[str, threading.Semaphore] = {}
_PER_MODEL_SEM_LOCK = threading.Lock()


def _model_semaphore(model: str) -> threading.Semaphore:
    sem = _PER_MODEL_SEMAPHORES.get(model)
    if sem is not None:
        return sem
    with _PER_MODEL_SEM_LOCK:
        sem = _PER_MODEL_SEMAPHORES.get(model)
        if sem is None:
            sem = threading.Semaphore(PER_MODEL_MAX_CONCURRENCY)
            _PER_MODEL_SEMAPHORES[model] = sem
        return sem


def _resolve_client(model: str) -> OpenAI:
    """Return a cached OpenAI client for the single-shot unified path.

    Dispatches per-model to the provider's OpenAI-compatible Chat endpoint,
    reading key + base_url from the environment (see _chat_config). One
    client is cached per provider.
    """
    provider = _MODEL_PROVIDER.get(model)
    if provider is None:
        known = sorted(_MODEL_PROVIDER)
        raise ValueError(f"Unknown model '{model}'. Available models: {known}.")
    api_key, base_url = _chat_config(provider)
    if provider not in _CLIENT_CACHE:
        # timeout=360 — gpt-5.4 on large prompts (e.g. rule_gen at ~25KB) can
        # spend 150-200s server-side when reasoning is on. 120s killed long
        # runs with silent ReadTimeout. 360s leaves headroom for outliers.
        _CLIENT_CACHE[provider] = OpenAI(api_key=api_key, base_url=base_url, timeout=360.0)
    return _CLIENT_CACHE[provider]


# ── Reasoning effort + thinking handling ────────────────────────────────────

def _normalise_deepseek_effort(effort: str) -> str:
    """Map the binary pipeline switch to DeepSeek's accepted effort value."""
    if effort != "on":
        raise ValueError(f"DeepSeek reasoning must be 'on' or 'off', got {effort!r}")
    return "high"


def _is_claude_family(model: str) -> bool:
    """Anthropic Claude models (routed via proxy). Detection by name
    prefix because proxy exposes them as `anthropic/claude-...`."""
    return model.startswith("anthropic/claude") or model.startswith("claude")


def _is_gemini_family(model: str) -> bool:
    """Google Gemini models routed through proxy."""
    return "gemini" in model


def _is_gemini_3_family(model: str) -> bool:
    """Google Gemini 3.x models — all of them accept the same
    `thinking_level` dial (minimal/low/medium/high), per
    https://ai.google.dev/gemini-api/docs/gemini-3. Covers
    `gemini-3-flash-preview` and `gemini-3.1-pro-preview` and their
    `google/`-prefixed aliases. Older 2.x / `*-image-preview` /
    vision-only variants are excluded; they have different reasoning
    surfaces."""
    canon = model.removeprefix("google/")
    return canon in {
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
    }


def _is_qwen_family(model: str) -> bool:
    return model.startswith("qwen") or model.startswith("qwen/")


def _is_minimax_family(model: str) -> bool:
    """MiniMax M2 series — interleaved-thinking models with no on/off dial.
    Detection covers the canonical `MiniMax-M2.x` casing and the
    provider-prefixed `minimax/minimax-m2.x` aliases. See
    https://platform.minimax.io/docs/guides/text-m2-function-call.
    """
    return model.startswith("MiniMax-M") or model.startswith("minimax/")


def _set_gemini_thinking_level(kwargs: dict, level: str) -> None:
    """Set Gemini's proxy-validated OpenAI-compatible thinking level."""
    eb = kwargs.setdefault("extra_body", {})
    google = eb.setdefault("extra_body", {}).setdefault("google", {})
    google.setdefault("thinking_config", {})["thinking_level"] = level


def _set_qwen_thinking(kwargs: dict, enabled: bool) -> None:
    eb = kwargs.setdefault("extra_body", {})
    eb["enable_thinking"] = bool(enabled)
    eb["chat_template_kwargs"] = {"enable_thinking": bool(enabled)}


def _set_minimax_reasoning_split(kwargs: dict) -> None:
    """MiniMax M2 series has no intensity dial — the model always interleaves
    thinking with the final response. We set `extra_body.reasoning_split=True`
    so thinking lands in a separate `reasoning_details` field rather than
    being embedded as `<think>...</think>` inside `content`. Without this,
    the orchestrator would persist thinking inline into the trajectory and
    re-feed it through memory layers / user_sim's visible context.

    Note: MiniMax docs recommend preserving `reasoning_details` across
    conversation history for best multi-turn performance. The runner does
    not currently echo it back, so MiniMax may underperform vs other
    thinking models on multi-turn agentic tasks; track as known limitation.
    """
    eb = kwargs.setdefault("extra_body", {})
    eb["reasoning_split"] = True


def _validate_reasoning_switch(value: str) -> None:
    if value not in {"on", "off"}:
        raise ValueError(f"reasoning_effort must be 'on' or 'off', got {value!r}")


def _apply_reasoning(
    kwargs: dict,
    model: str,
    override: str | None,
) -> None:
    """Attach reasoning controls to the request kwargs.

    Three sources, in priority order:

      1. Caller's explicit binary `override` ("on" / "off") always wins.
         Gemini 3.x family maps on→high and off→minimal across flash/pro/lite.
         Qwen maps on to enable_thinking=true with the provider's default
         budget, and off to enable_thinking=false. DeepSeek maps on to high
         and off to native thinking disabled. MiniMax has no on/off dial —
         see special-case handling below.

      2. Otherwise — pass nothing, vendor's own default kicks in (Claude
         default = thinking off in the tested endpoint configuration).

    MiniMax M2 special case (regardless of source): the model has no
    intensity dial and always interleaves thinking. We always set
    `reasoning_split=True` to separate thinking from `content`; the
    `override` value is ignored because there is no on/off to map to. This
    means MiniMax under --reasoning off is NOT a true thinking-disabled
    ablation — it stays a thinking model. Document this in any paper
    ablation that uses --reasoning off across the model panel.
    """
    if _is_minimax_family(model):
        _set_minimax_reasoning_split(kwargs)
        return

    if override is not None:
        _validate_reasoning_switch(override)
        if model in _DEEPSEEK_MODELS:
            if override == "off":
                kwargs.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}
            else:
                kwargs.setdefault("extra_body", {})["thinking"] = {
                    "type": "enabled",
                    "reasoning_effort": _normalise_deepseek_effort(override),
                }
        elif _is_claude_family(model):
            if override == "off":
                return
            kwargs["reasoning_effort"] = "high"
            # Anthropic constraint: temperature must be exactly 1 when
            # thinking is enabled. Override the caller's value silently —
            # without this, every call site that defaults to temp=0.0
            # would 400 when running Claude with reasoning.
            kwargs["temperature"] = 1.0
            # Anthropic constraint: max_tokens must exceed
            # thinking.budget_tokens. budget scales with effort; we floor
            # at 8192 to give "high" effort comfortable headroom.
            existing_max = kwargs.get("max_tokens")
            if existing_max is None or existing_max < 8192:
                kwargs["max_tokens"] = 8192
        elif _is_gemini_3_family(model):
            _set_gemini_thinking_level(
                kwargs, "high" if override == "on" else "minimal"
            )
        elif _is_qwen_family(model):
            _set_qwen_thinking(kwargs, enabled=(override == "on"))
        else:
            kwargs["reasoning_effort"] = "high" if override == "on" else "low"
        return

    # No override: use the vendor default.


def _is_gpt_prompt_cache_model(model: str) -> bool:
    return model.startswith("gpt") or model.startswith("openai/gpt")


def _content_cache_snippet(content: Any) -> str:
    """Normalise message/tool content into a bounded string for cache affinity."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:_PROMPT_CACHE_MAX_CHARS]
    try:
        return json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )[:_PROMPT_CACHE_MAX_CHARS]
    except (TypeError, ValueError):
        return str(content)[:_PROMPT_CACHE_MAX_CHARS]


def _build_prompt_cache_key(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> str | None:
    """Build a stable cache-affinity key for GPT-family prompt caching.

    `prompt_cache_key` is a routing hint, not the cache lookup itself. We hash
    the stable prompt prefix (leading system/user messages + tool schema) so
    repeated calls in the same workload are more likely to land on a warm cache
    shard without requiring any upper-layer session/thread id plumbed through.
    """
    if not messages:
        return None

    anchor_messages: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role") or "")
        if role in {"system", "user"}:
            anchor_messages.append({
                "role": role,
                "content": _content_cache_snippet(msg.get("content")),
            })
        if len(anchor_messages) >= 2:
            break

    if not anchor_messages:
        first = messages[0]
        anchor_messages.append({
            "role": str(first.get("role") or ""),
            "content": _content_cache_snippet(first.get("content")),
        })

    tool_sig = None
    if tools:
        tool_sig = _content_cache_snippet(tools)

    payload = {
        "model": model,
        "messages": anchor_messages,
        "tools": tool_sig,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"atr:gpt-cache:{digest}"


def _apply_prompt_cache_hints(
    kwargs: dict[str, Any],
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> None:
    """Enable GPT prompt caching transparently for existing callers."""
    if not _is_gpt_prompt_cache_model(model):
        return
    key = _build_prompt_cache_key(model, messages, tools)
    if not key:
        return
    kwargs["prompt_cache_key"] = key
    kwargs["prompt_cache_retention"] = _GPT_PROMPT_CACHE_RETENTION


def _is_prompt_cache_hint_error(exc: APIStatusError) -> bool:
    if exc.status_code != 400:
        return False
    text = str(exc).lower()
    return "prompt_cache" in text or "cache retention" in text


def _to_attr_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_attr_tree(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_tree(v) for v in value]
    return value


def _create_qwen_chat_completion_raw(kwargs: dict[str, Any]) -> Any:
    """Call proxy with raw JSON so Qwen thinking flags stay top-level.

    The OpenAI SDK's `extra_body` parameter nests these fields in a way that
    proxy/Qwen treats as thinking enabled. Raw JSON matches the curl shape
    verified against the proxy.
    """
    body = dict(kwargs)
    extra = body.pop("extra_body", None)
    if extra:
        body.update(extra)

    api_key, base_url = _chat_config("dashscope")
    url = f"{base_url.rstrip('/')}/chat/completions"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=360) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {429, 500, 502, 503, 504}:
            raise _RetryableRawLLMError(
                f"Qwen raw chat completion failed HTTP {exc.code}: {error_body}"
            ) from exc
        raise RuntimeError(
            f"Qwen raw chat completion failed HTTP {exc.code}: {error_body}"
        ) from exc
    except URLError as exc:
        raise _RetryableRawLLMError(
            f"Qwen raw chat completion connection failed: {exc}"
        ) from exc

    return _to_attr_tree(payload)


def _accumulate_stream_to_completion(stream_iter: Any, model: str) -> Any:
    """Consume an OpenAI streaming response and rebuild a non-streaming
    ChatCompletion-shaped object on the client side.

    Rationale: the gpt-5.4 endpoint behind the proxy is
    stream-first; the proxy reassembles streaming chunks into a
    non-streaming `chat.completion` JSON at egress. Under high
    concurrency the reassembly occasionally leaks the first SSE chunk
    verbatim (`Content-Type: text/event-stream`, `choices:[]`,
    `object:"chat.completion.chunk"`), which the OpenAI SDK can't
    deserialize. Keeping the streaming protocol end-to-end and doing
    reassembly client-side bypasses the proxy reassembly race.

    Accumulates: content, reasoning_content (vendor extension), tool_calls
    (by index, with concatenated function.arguments), finish_reason, and
    the final usage chunk (requires stream_options.include_usage=True).
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage_obj: Any = None
    response_id: str = ""
    model_id: str = model
    system_fingerprint: str | None = None

    for chunk in stream_iter:
        if getattr(chunk, "id", None) and not response_id:
            response_id = chunk.id
        if getattr(chunk, "model", None):
            model_id = chunk.model
        if getattr(chunk, "system_fingerprint", None):
            system_fingerprint = chunk.system_fingerprint
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            ct = getattr(delta, "content", None)
            if ct:
                content_parts.append(ct)
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning_parts.append(rc)
            d_tcs = getattr(delta, "tool_calls", None) or []
            for d_tc in d_tcs:
                idx = getattr(d_tc, "index", None) or 0
                entry = tool_calls_by_index.setdefault(idx, {
                    "id": None,
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                })
                tc_id = getattr(d_tc, "id", None)
                if tc_id:
                    entry["id"] = tc_id
                tc_type = getattr(d_tc, "type", None)
                if tc_type:
                    entry["type"] = tc_type
                fn = getattr(d_tc, "function", None)
                if fn is not None:
                    nm = getattr(fn, "name", None)
                    if nm:
                        entry["function"]["name"] = nm
                    args_chunk = getattr(fn, "arguments", None)
                    if args_chunk:
                        entry["function"]["arguments"] += args_chunk
        fr = getattr(choice, "finish_reason", None)
        if fr:
            finish_reason = fr

    tool_calls_list: list[Any] | None = None
    if tool_calls_by_index:
        tool_calls_list = []
        for _idx, entry in sorted(tool_calls_by_index.items()):
            tool_calls_list.append(SimpleNamespace(
                id=entry["id"],
                type=entry["type"],
                function=SimpleNamespace(
                    name=entry["function"]["name"],
                    arguments=entry["function"]["arguments"],
                ),
            ))

    message = SimpleNamespace(
        role="assistant",
        content="".join(content_parts) if content_parts else None,
        tool_calls=tool_calls_list,
        reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
    )
    choice_ns = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
        index=0,
    )
    return SimpleNamespace(
        id=response_id,
        model=model_id,
        object="chat.completion",
        choices=[choice_ns],
        usage=usage_obj,
        system_fingerprint=system_fingerprint,
    )


def _create_chat_completion(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    """Send a chat completion via the streaming protocol, then accumulate
    chunks client-side into a non-streaming ChatCompletion-shape object.

    See `_accumulate_stream_to_completion` for why streaming is forced
    here (proxy reassembly race / SSE-leak bypass).

    Qwen family stays on the raw-JSON urllib path
    (`_create_qwen_chat_completion_raw`) since that one already parses
    the proxy response directly without depending on proxy
    reassembly.
    """
    if _is_qwen_family(str(kwargs.get("model", ""))):
        return _create_qwen_chat_completion_raw(kwargs)

    model = str(kwargs.get("model", ""))

    def _open_stream(req_kwargs: dict[str, Any]) -> Any:
        sk = dict(req_kwargs)
        sk["stream"] = True
        existing_options = dict(sk.get("stream_options") or {})
        existing_options.setdefault("include_usage", True)
        sk["stream_options"] = existing_options
        return client.chat.completions.create(**sk)

    try:
        stream = _open_stream(kwargs)
    except APIStatusError as exc:
        if not _is_prompt_cache_hint_error(exc):
            raise
        if "prompt_cache_key" not in kwargs and "prompt_cache_retention" not in kwargs:
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("prompt_cache_key", None)
        fallback_kwargs.pop("prompt_cache_retention", None)
        logger.warning(
            "GPT prompt-cache hints rejected by provider; retrying without them (model=%s)",
            kwargs.get("model"),
        )
        stream = _open_stream(fallback_kwargs)

    return _accumulate_stream_to_completion(stream, model=model)


def _validate_chat_completion_shape(resp: Any, model: str) -> None:
    """Guard against the proxy SSE-leak failure mode.

    Under high concurrency, proxy occasionally returns the gpt-5.4
    response as an SSE chunk (`data: {..."object":"chat.completion.chunk",
    "choices":[],...}`) instead of the standard `chat.completion` JSON.
    The OpenAI SDK can't deserialize that into a ChatCompletion object
    and falls back to returning the raw response body as a `str`. The
    caller then sees `AttributeError: 'str' object has no attribute
    'choices'` on the next line.

    Observed during calibration at low frequency. Single-shot reproduction
    is rare.

    We treat both the str-fallback and the empty-`choices` shape as
    transient provider faults and raise `_RetryableRawLLMError` so
    tenacity's outer `_call_with_retry` / `_call_with_tools_retry`
    decorator re-issues the request.
    """
    if isinstance(resp, str):
        raise _RetryableRawLLMError(
            f"Provider returned raw str instead of ChatCompletion "
            f"(probable SSE-leak) for model={model!r}; "
            f"preview={resp[:200]!r}"
        )
    choices = getattr(resp, "choices", None)
    if not choices:
        raise _RetryableRawLLMError(
            f"Provider returned ChatCompletion with empty choices=[] "
            f"for model={model!r}; preview={str(resp)[:200]!r}"
        )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _RetryableRawLLMError):
        return True
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        if exc.status_code in (429, 500, 502, 503, 504):
            return True
        if exc.status_code == 400 and "not a valid model" in str(exc).lower():
            return True
    return False


@retry(
    retry=retry_if_exception_type(APIConnectionError) | retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=90),
    stop=stop_after_attempt(10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float | None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    _apply_reasoning(kwargs, model, reasoning_effort)
    _apply_prompt_cache_hints(kwargs, model, messages)
    if seed is not None:
        kwargs["seed"] = seed
    resp = _create_chat_completion(client, kwargs)
    _validate_chat_completion_shape(resp, model)
    return resp


@retry(
    retry=retry_if_exception_type(APIConnectionError) | retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=90),
    stop=stop_after_attempt(10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_with_tools_retry(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float | None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    _apply_reasoning(kwargs, model, reasoning_effort)
    if tools:
        kwargs["tools"] = tools
    if response_format is not None:
        kwargs["response_format"] = response_format
    _apply_prompt_cache_hints(kwargs, model, messages, tools)
    if seed is not None:
        kwargs["seed"] = seed
    resp = _create_chat_completion(client, kwargs)
    _validate_chat_completion_shape(resp, model)
    return resp


# ── Public API ────────────────────────────────────────────────────────────────

def call_llm(
    user_prompt: str,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float | None = 0.7,
    seed: int | None = None,
    return_usage: bool = False,
    reasoning_effort: str | None = None,
) -> Any:
    """Call LLM and return raw text response. Provider is resolved from model name.

    Args:
        seed: Optional determinism hint forwarded to the provider. Not all
            providers honour it, but OpenAI-compatible endpoints accept it.
        temperature: Sampling temperature. Pass None to omit the parameter and
            use the provider/model default.
        return_usage: If True, return {"content": str, "usage": {...}} instead
            of just the content string. Default False returns the content string.
        reasoning_effort: Binary reasoning override: "on", "off", or None.
            None (default) sends no parameter and uses the vendor default.
            Explicit values are mapped per family; see _apply_reasoning.
    """
    model = _canonical_model(model)
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    client = _resolve_client(model)
    with _model_semaphore(model):
        resp = _call_with_retry(
            client, model, messages, temperature,
            seed=seed, reasoning_effort=reasoning_effort,
        )

    content = resp.choices[0].message.content or ""
    usage = None
    if resp.usage is not None:
        prompt_details = getattr(resp.usage, "prompt_tokens_details", None)
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
            "cached_tokens": getattr(prompt_details, "cached_tokens", None),
            "prompt_cache_hit_tokens": getattr(resp.usage, "prompt_cache_hit_tokens", None),
        }
    _record_usage(usage)
    if not return_usage:
        return content
    return {"content": content, "usage": usage}


# ── Provider-native dispatch ─────────────────────────────────────

_GEMINI_NATIVE_IDS = frozenset({
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
})

_OPENAI_RESPONSES_IDS = frozenset({"gpt-5.4"})
_MINIMAX_NATIVE_IDS = frozenset({"MiniMax-M2.7"})
_QWEN_NATIVE_IDS = frozenset({"qwen3.6-plus"})
_DEEPSEEK_NATIVE_IDS = frozenset({"deepseek-v4-pro", "deepseek-v4-flash"})
# Anthropic native /v1/messages. claude-opus-4-7 requires
# `thinking: {type: "adaptive"}` (manual extended thinking returns 400).
_ANTHROPIC_NATIVE_IDS = frozenset({"claude-opus-4-7"})


def _resolve_native_adapter(model: str):
    """Return the provider-native call function for `model`, or None.

    Returns the call_<provider>_with_tools function for models in scope;
    None for any other id (unit tests, future additions, etc., which walk
    the unified single-shot path).
    """
    canon = model.removeprefix("google/")
    if canon in _GEMINI_NATIVE_IDS:
        from lib.llm_gemini_native import call_gemini_native_with_tools
        return call_gemini_native_with_tools
    if canon in _OPENAI_RESPONSES_IDS:
        from lib.llm_openai_responses import call_openai_responses_with_tools
        return call_openai_responses_with_tools
    if canon in _ANTHROPIC_NATIVE_IDS:
        from lib.llm_anthropic_native import call_anthropic_native_with_tools
        return call_anthropic_native_with_tools
    if canon in _MINIMAX_NATIVE_IDS:
        from lib.llm_minimax import call_minimax_with_tools
        return call_minimax_with_tools
    if canon in _QWEN_NATIVE_IDS:
        from lib.llm_qwen import call_qwen_with_tools
        return call_qwen_with_tools
    if canon in _DEEPSEEK_NATIVE_IDS:
        from lib.llm_deepseek import call_deepseek_with_tools
        return call_deepseek_with_tools
    return None


def call_llm_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float | None = 0.7,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
) -> dict:
    """Native function-calling entry point.

    Dispatches per-model to provider-native adapters in `lib.llm_*` so
    each upstream's official multi-turn tool-calling contract is honored
    byte-faithfully (Gemini thoughtSignature, OpenAI Responses
    function_call / function_call_output, MiniMax reasoning_details, Qwen
    reasoning_content, DeepSeek thinking).

    Args:
        messages: OpenAI-format chat messages with optional
            `native_assistant_payload` + `native_payload_format` keys on
            assistant entries for byte-faithful replay.
        tools: OpenAI-Chat tool schemas. Provider modules convert internally.
        model: Model id. Routes to lib.llm_<provider>; for ids that don't
            match any native adapter, falls back to the unified OpenAI-
            compat path (test fixtures, future additions, etc.).
        temperature: Sampling temperature. None to omit.
        reasoning_effort: "on" | "off" | None; mapped per provider in
            each adapter.

    Returns: caller-shape dict:
            {
              "content": str | None,
              "tool_calls": [{"id", "name", "arguments"}, ...] | None,
              "reasoning_content": str | None,
              "native_assistant_payload": dict | list | None,
              "native_payload_format": str | None,
              "usage": {...} | None,
            }
    """
    model = _canonical_model(model)

    native_call = _resolve_native_adapter(model)
    if native_call is not None:
        with _model_semaphore(model):
            result = native_call(
                messages=messages,
                tools=tools,
                model=model,
                temperature=temperature,
                seed=seed,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
            )
        usage = result.get("usage")
        _record_usage(usage)
        # Ensure every documented key is present so callers can rely on
        # `.get("native_assistant_payload")` without KeyError.
        result.setdefault("content", None)
        result.setdefault("tool_calls", None)
        result.setdefault("reasoning_content", None)
        result.setdefault("native_assistant_payload", None)
        result.setdefault("native_payload_format", None)
        result.setdefault("usage", usage)
        return result

    # Unified path (non-native ids; test fixtures; future models).
    client = _resolve_client(model)
    with _model_semaphore(model):
        resp = _call_with_tools_retry(
            client, model, messages, tools, temperature,
            seed=seed, reasoning_effort=reasoning_effort,
            response_format=response_format,
        )

    msg = resp.choices[0].message
    content = msg.content
    reasoning_content = getattr(msg, "reasoning_content", None)

    tool_calls_parsed: list[dict] | None = None
    raw_tc = getattr(msg, "tool_calls", None)
    if raw_tc:
        tool_calls_parsed = []
        for tc in raw_tc:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Malformed tool_call arguments (model=%s, call_id=%s): %r",
                    model, tc.id, tc.function.arguments,
                )
                args = {}
            tool_calls_parsed.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": args,
            })

    usage = None
    if resp.usage is not None:
        prompt_details = getattr(resp.usage, "prompt_tokens_details", None)
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
            "cached_tokens": getattr(prompt_details, "cached_tokens", None),
            "prompt_cache_hit_tokens": getattr(resp.usage, "prompt_cache_hit_tokens", None),
        }
    _record_usage(usage)

    return {
        "content": content,
        "tool_calls": tool_calls_parsed,
        "usage": usage,
        "reasoning_content": reasoning_content,
        "native_assistant_payload": None,
        "native_payload_format": None,
    }


def _sanitize_yaml(text: str) -> str:
    text = text.replace("、", ", ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text


def _strip_yaml_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:yaml)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def call_llm_yaml(
    user_prompt: str,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_retries: int = 3,
    seed: int | None = None,
) -> Any:
    """Call LLM and parse YAML response. Retries with nudge on parse failure."""
    for attempt in range(max_retries):
        nudge = ""
        if attempt > 0:
            nudge = (
                "\n\nIMPORTANT: Your previous response had invalid YAML. "
                "Output ONLY valid YAML. Rules: use ASCII punctuation only "
                "(no Chinese commas 、or Chinese quotes), wrap all string values "
                "containing special characters in double quotes, no extra text or markdown fencing."
            )
        raw = call_llm(user_prompt + nudge, system_prompt, model, temperature, seed=seed)
        try:
            cleaned = _sanitize_yaml(_strip_yaml_fence(raw))
            return yaml.safe_load(cleaned)
        except yaml.YAMLError as e:
            logger.warning(f"YAML parse attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise ValueError(
                    f"Failed to parse YAML after {max_retries} attempts.\nRaw output:\n{raw}"
                ) from e
    return None


def call_llm_json(
    user_prompt: str,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float | None = 0.3,
    max_retries: int = 3,
    seed: int | None = None,
    reasoning_effort: str | None = None,
) -> Any:
    """Call LLM and parse JSON response. Retries with nudge on parse failure.

    `reasoning_effort` override: pass "on" for tasks that need the model's
    high-thinking condition (e.g. test_session_gen plan-then-generate).
    """
    for attempt in range(max_retries):
        nudge = ""
        if attempt > 0:
            nudge = (
                "\n\nIMPORTANT: Your previous response had invalid JSON. "
                "Output ONLY valid JSON, no extra text, no markdown fencing."
            )
        raw = call_llm(
            user_prompt + nudge, system_prompt, model, temperature,
            seed=seed, reasoning_effort=reasoning_effort,
        )
        try:
            cleaned = _strip_json_fence(raw)
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise ValueError(
                    f"Failed to parse JSON after {max_retries} attempts.\nRaw output:\n{raw}"
                ) from e
    return None
