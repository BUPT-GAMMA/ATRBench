"""Online Router for ATR standing-rule questions.

The Router owns the speech-act decision from given one agent turn, decide whether the user-visible output asks a strict
question about a future/default/standing preference. It does not match rules
and it does not decide the user's reply.
"""
from __future__ import annotations

import logging
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from lib.llm import call_llm_json, GEMINI  # type: ignore
from _prompts import load_prompt

logger = logging.getLogger(__name__)


_DEFAULT_FEWSHOT_PATH = (
    Path(__file__).resolve().parent / "router_calibration" / "shots" / "v1.json"
)


@lru_cache(maxsize=1)
def _default_fewshot_path() -> Path | None:
    env = os.environ.get("ATR_ROUTER_FEWSHOT_PATH")
    if env:
        return Path(env)
    if _DEFAULT_FEWSHOT_PATH.exists():
        return _DEFAULT_FEWSHOT_PATH
    return None


def _resolve_fewshot_path(override: Path | str | None = None) -> Path | None:
    if override:
        return Path(override)
    return _default_fewshot_path()


def _render_fewshot(path: Path | None) -> str:
    if path is None:
        return ""
    bank = json.loads(path.read_text())
    examples = bank.get("examples") or []
    if not examples:
        return ""

    rendered: list[str] = ["## Examples"]
    for idx, ex in enumerate(examples, 1):
        reason = str(ex.get("reason") or "").strip()
        output = str(ex.get("output") or "").strip()
        is_strict = bool(ex.get("is_strict_rule_question"))
        span = ex.get("rule_question_span")
        if not is_strict:
            span = None
        result = {
            "is_strict_rule_question": is_strict,
            "rule_question_span": span,
        }
        note = str(ex.get("note") or "").strip()
        rendered.append(
            f"### Example {idx}\n"
            f"<reason>{reason}</reason>\n"
            f"<output>{output}</output>\n"
            f"Decision:\n{json.dumps(result, ensure_ascii=False)}"
            + (f"\nNote: {note}" if note else "")
        )
    return "\n\n".join(rendered)


@lru_cache(maxsize=8)
def _build_system_prompt(fewshot_path: Path | None) -> str:
    return load_prompt("router").replace(
        "{{FEW_SHOT_EXAMPLES}}", _render_fewshot(fewshot_path)
    )


_SYSTEM_PROMPT = _build_system_prompt(_resolve_fewshot_path())


def _normalize_router_result(obj: Any) -> dict[str, Any] | None:
    """Normalize Router LLM output into the runtime schema, or return None
    on any schema fault. The caller (`route_agent_turn`) promotes None to
    `SweepAbort` — Router framework
    faults must cell-abort rather than silently degrade to task route.
    """
    if not isinstance(obj, dict):
        return None

    is_strict = obj.get("is_strict_rule_question")
    if not isinstance(is_strict, bool):
        return None

    span_raw = obj.get("rule_question_span")
    if span_raw is None:
        span = None
    elif isinstance(span_raw, str):
        span = span_raw.strip() or None
    else:
        return None

    if is_strict and not span:
        # Internal LLM inconsistency: strict-true must come with a span.
        # Treated as schema fault → cell-abort upstream.
        return None

    if not is_strict:
        span = None

    return {
        "is_strict_rule_question": is_strict,
        "rule_question_span": span,
    }


def route_agent_turn(
    *,
    reason: str,
    output: str,
    model: str = GEMINI,
    seed: int | None = None,
    fewshot_path: Path | str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Classify one agent turn for the Router.

    Returns:
        {
          "is_strict_rule_question": bool,
          "rule_question_span": str | None,
        }

    The decision is made from `reason` and `output` alone — Router classifies
    the speech-act type of the current message, which does not depend on
    referent resolution from prior turns. cls handles the topic-to-rule
    matching downstream using the extracted span.

     Any Router framework fault —
    LLM transport exhaustion OR malformed schema OR LLM internal
    inconsistency — raises `SweepAbort`. The caller does not see an
    `error` field; happy-path callers always observe a well-formed
    `(is_strict_rule_question, rule_question_span)`.
    """
    user_prompt = (
        f"<reason>{reason}</reason>\n"
        f"<output>{output}</output>"
    )
    try:
        result = call_llm_json(
            user_prompt=user_prompt,
            system_prompt=_build_system_prompt(_resolve_fewshot_path(fewshot_path)),
            model=model,
            temperature=None,
        )
    except Exception as e:
        from _abort import SweepAbort
        raise SweepAbort(module="router", session_id=session_id, original=e) from e

    normalized = _normalize_router_result(result)
    if normalized is None:
        logger.warning("route_agent_turn invalid result: %r", result)
        from _abort import SweepAbort
        raise SweepAbort(
            module="router",
            session_id=session_id,
            original=ValueError(f"Router returned malformed schema: {result!r}"),
        )
    return normalized
