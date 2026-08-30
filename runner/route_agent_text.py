"""Rule matching service for Router-approved standing-rule questions.

Router has already identified a strict standing-rule question and extracted
the question span; this module only does the matching: which rule (if any) does
that span correspond to?

Output schema:
    {
        "rule_id": str | None,
        "reasoning": str,
    }
"""
from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from schemas import Rule

from lib.llm import call_llm_json, GPT  # type: ignore
from lib.benchmark_data import read_json
from _prompts import load_prompt

logger = logging.getLogger(__name__)


# Few-shot lookup priority (first non-empty wins):
#   1. `fewshot_path` argument to route_agent_text(...)
#   2. environment variable ATR_CLS_FEWSHOT_PATH
#   3. compiled-in default below
# Calibration sweeps switch shot sets via (1) or (2); production runs use (3).
# The compiled-in default points at the active shot bank in cls_calibration/.
# To promote a new version, write shots/v(N+1).json then update this constant.
_DEFAULT_FEWSHOT_PATH = (
    Path(__file__).resolve().parent / "cls_calibration" / "shots" / "v12.json"
)


def _resolve_fewshot_path(override: Path | str | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("ATR_CLS_FEWSHOT_PATH")
    if env:
        return Path(env)
    return _DEFAULT_FEWSHOT_PATH


def _render_fewshot(path: Path) -> str:
    """Render few-shot examples from a shot bank file.

    Each example includes a mini rule pool so the LLM sees the comparison
    process in the same rule_id + user-statement format as runtime
    _format_rules output.
    """
    bank = read_json(path)
    examples = bank.get("match") or []
    if not examples:
        return ""
    lines: list[str] = []
    for idx, ex in enumerate(examples, 1):
        pool_lines = []
        for r in ex.get("rule_pool", []):
            line = (
                f"  - rule_id={r['rule_id']}\n"
                f"    user statement: {r['statement']}"
            )
            pool_lines.append(line)
        rid = ex.get("rule_id")
        rid_str = f'"{rid}"' if rid else "null"
        lines.append(
            f"### Example {idx}\n"
            f"Rule pool:\n" + "\n".join(pool_lines) + "\n"
            f"Agent question: {ex['agent_text']}\n"
            f"Decision: rule_id={rid_str}, reasoning=\"{ex['reasoning']}\""
        )
    return "\n\n".join(lines)


@lru_cache(maxsize=8)
def _build_system_prompt(fewshot_path: Path) -> str:
    """Compile the cls system prompt for a given shot bank. Cached per
    `fewshot_path`; calibration sweeps that swap shot files pay the
    rendering cost only once per unique path."""
    return load_prompt("classifier").replace(
        "{{FEW_SHOT_EXAMPLES}}", _render_fewshot(fewshot_path)
    )


def _format_rules(rules: list[Rule]) -> str:
    if not rules:
        return "(empty rule pool)"
    lines = []
    for r in rules:
        line = (
            f"- rule_id={r.rule_id}\n"
            f"  user statement: {r.canonical_answer}"
        )
        lines.append(line)
    return "\n".join(lines)


def route_agent_text(
    query: str,
    rules: list[Rule],
    model: str = GPT,
    seed: int | None = None,
    fewshot_path: Path | str | None = None,
    session_id: str | None = None,
) -> dict:
    """Match a query against the rule pool.

    `query` is the Router-extracted standing-rule question span. cls receives
    just that span + the rule pool — no private reason, no persona, no session
    task — so its attention is on topic-to-rule matching, not intent
    classification.

    Returns dict with keys: rule_id, reasoning.
      - `rule_id` is `None` on a genuine φ-miss (no rule in pool matched
        the query) OR when the LLM picked an id that isn't in the pool
        (cleared, treated as miss).
      - Any cls framework fault —
        LLM transport exhaustion OR non-dict result — raises
        `SweepAbort`. The caller does not see an `error` field; happy-
        path callers always observe a well-formed `(rule_id, reasoning)`.
    """
    user_prompt = (
        f"## Rule pool\n{_format_rules(rules)}\n\n"
        f"## Rule question span\n{query}"
    )

    sys_prompt = _build_system_prompt(_resolve_fewshot_path(fewshot_path))

    def _call_cls() -> Any:
        return call_llm_json(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            model=model,
            temperature=None,
            # No `reasoning_effort` override: use the vendor default.
        )

    def _parse_result(result: Any) -> tuple[str | None, str, bool]:
        """Validate result shape + rule_id membership. Returns
        (rule_id, reasoning, is_unknown_rule_id). is_unknown_rule_id True
        means cls picked an id that isn't in the rule pool — caller may
        retry. Raises SweepAbort on non-dict (framework fault)."""
        if not isinstance(result, dict):
            from _abort import SweepAbort
            raise SweepAbort(
                module="cls",
                session_id=session_id,
                original=ValueError(
                    f"cls returned non-dict: {type(result).__name__}"
                ),
            )
        rule_id = result.get("rule_id") or None
        reasoning = result.get("reasoning", "")
        known_ids = {r.rule_id for r in rules}
        is_unknown = rule_id is not None and rule_id not in known_ids
        return rule_id, reasoning, is_unknown

    # Attempt 0
    try:
        result = _call_cls()
    except Exception as e:
        from _abort import SweepAbort
        raise SweepAbort(module="cls", session_id=session_id, original=e) from e

    rule_id, reasoning, unknown = _parse_result(result)

    # H10: cls hallucinating an id not
    # in the rule pool is a framework fault — silently coercing to miss
    # would bias hit_rate downward without any audit signal. Give the
    # model one retry chance, then SweepAbort if it still misbehaves.
    if unknown:
        logger.warning(
            "route_agent_text: cls picked unknown rule_id %r (not in pool); retrying once.",
            rule_id,
        )
        try:
            result = _call_cls()
        except Exception as e:
            from _abort import SweepAbort
            raise SweepAbort(module="cls", session_id=session_id, original=e) from e
        rule_id, reasoning, unknown = _parse_result(result)
        if unknown:
            from _abort import SweepAbort
            raise SweepAbort(
                module="cls",
                session_id=session_id,
                original=ValueError(
                    f"cls returned unknown rule_id {rule_id!r} on two consecutive "
                    f"attempts (not in pool of {len(rules)} rules)"
                ),
            )

    return {
        "rule_id": rule_id,
        "reasoning": reasoning,
    }
