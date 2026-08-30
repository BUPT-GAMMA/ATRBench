"""Tool-call matching for ATR test session evaluation.

Current schema: required_actions is always length 1 (single tool gold,
including confirm rules whose gold is a single get_user_confirmation
call). `_match_step` scans the entire trajectory and locks on the first
call whose tool name matches required.tool, then runs `_compare_args`
against the locked call's arguments.

Matching semantics:
  - "Full match" (tool name + args per compare_args) required to pass.
  - list-typed args: set(gold) == set(actual) (order-independent set
    equality).
  - scalar args: exact equality.
  - confirm rules' target_params: safe-typed subset match (see
    `_compare_target_params`).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "runner"))
sys.path.insert(0, str(_ROOT))
from schemas import SessionTrajectory, RequiredAction, ToolCall, Message
from lib.benchmark_data import tool_map  # noqa: E402
from lib.compare_utils import safe_for_compare, coerce_target_params  # noqa: E402


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    step_idx: int
    required_tool: str
    tool_found: bool
    matched_call_id: str | None = None  # id of the call we locked onto
    args_match: bool = False
    # {arg_key: [gold_value, actual_value]} — only keys that failed
    arg_mismatches: dict[str, list[Any]] = {}

    @property
    def passed(self) -> bool:
        return self.tool_found and self.args_match


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_tool_call_batches(traj: SessionTrajectory) -> list[list[ToolCall]]:
    """Return tool calls grouped by assistant message (i.e. by batch).

     first-batch + same-tool consistency requires preserving
    batch boundaries: each assistant message's `tool_calls` list is one
    batch. Flattening across batches loses the semantic boundary needed
    to detect contradictory same-tool calls within a single turn.
    """
    batches: list[list[ToolCall]] = []
    for m in traj.messages:
        if m.role == "assistant" and m.tool_calls:
            batches.append(list(m.tool_calls))
    return batches


def _extract_tool_calls(traj: SessionTrajectory) -> list[ToolCall]:
    """Flatten all tool calls in turn order. Retained for callers that
    just need a flat call list; new semantics-bearing code should use
    `_extract_tool_call_batches` instead.
    """
    calls: list[ToolCall] = []
    for batch in _extract_tool_call_batches(traj):
        calls.extend(batch)
    return calls


def _compare_args(
    required: RequiredAction,
    actual_args: dict[str, Any],
) -> tuple[bool, dict[str, list[Any]]]:
    """Compare required args against actual call arguments.

    compare_args=None is the tool_identity-only signal (mutate rule with
    action_step.param=null). In that case the step passes on tool-name
    match alone; arguments are NOT checked. Any fallback to "compare every
    arg key" would invert the intended semantics.

    Special case: `target_params` (on `get_user_confirmation`) is an
    object whose comparison uses safe-typed subset semantics — only
    fields matching the `target_tool`'s signature with safe types
    (*_id / *_ids / enum / boolean) participate, and the gold subset
    must be present in the actual. This keeps confirmation chains
    robust to oracle's natural extra fields (like reason/context)
    while still verifying the meaningful intent.

    Returns (all_passed, mismatches).
    mismatches: {key: [gold, actual]} for every failing key.
    """
    if required.compare_args is None:
        return True, {}
    mismatches: dict[str, list[Any]] = {}
    target_tool_value = required.arguments.get("target_tool")
    for key in required.compare_args:
        gold = required.arguments.get(key)
        actual = actual_args.get(key)
        if key == "target_params" and target_tool_value:
            if not _compare_target_params(gold, actual, target_tool_value):
                mismatches[key] = [gold, actual]
            continue
        if not _scalar_or_list_match(gold, actual):
            mismatches[key] = [gold, actual]
    return len(mismatches) == 0, mismatches


def _scalar_or_list_match(gold: Any, actual: Any) -> bool:
    if isinstance(gold, list):
        if not isinstance(actual, list):
            return False
        return {str(v) for v in gold} == {str(v) for v in actual}
    return gold == actual


def _compare_target_params(
    gold: Any, actual: Any, target_tool: str,
) -> bool:
    """Safe-typed subset match for confirmation-chain `target_params`.

    Coerces `actual` from JSON string → dict if needed (Gemini-style
    models tend to stringify nested object args; GPT/Claude emit native
    dicts — both are legitimate under OpenAI tool-spec `"type": "object"`).

    Then filters both gold and actual to the target_tool's safe-typed
    parameters (*_id / *_ids / enum / boolean — see
    `lib.compare_utils.safe_for_compare`), and asserts gold ⊆ actual on
    those keys (every gold key/value must be present in actual; actual
    may carry additional unrelated fields like reason/note that the LLM
    naturally adds).
    """
    actual = coerce_target_params(actual)
    gold = coerce_target_params(gold)
    if not isinstance(gold, dict) or not isinstance(actual, dict):
        return gold == actual
    sig = (tool_map().get(target_tool) or {}).get("parameters") or []
    safe_keys = {p["name"] for p in sig if safe_for_compare(p["name"], p["type"])}
    if not safe_keys:
        # Target tool has no safe-typed params — confirmation can't be
        # meaningfully bound to anything, so any dict matches (this falls
        # back to tool_identity-only semantics on this slot).
        return True
    for k in safe_keys:
        if k in gold:
            if k not in actual or actual[k] != gold[k]:
                return False
    return True


def _match_step(
    step_idx: int,
    required: RequiredAction,
    batches: list[list[ToolCall]],
) -> StepResult:
    """first-batch rule + same-tool param consistency.

    1. Locate the first batch (i.e. the first assistant message's
       tool_calls list) that contains any call to `required.tool`.
       Earlier batches without `required.tool` do not count against
       the agent.
    2. `tool_identity` (`compare_args=None`): PASS on any match within
       the batch. Extra tools in the batch are ignored.
    3. `param_id` / `param_enum` / `confirm`: run `_compare_args`
       against EVERY same-tool call in the batch. If any fails, the
       step FAILs (same-tool shotgun is rejected, e.g.
       `[delete(correct), delete(wrong)]`). Extra tools in the batch
       alongside `required.tool` are ignored.
    4. Later batches are not consulted once the first matching batch
       has decided the verdict.
    """
    for batch in batches:
        same_tool_calls = [tc for tc in batch if tc.name == required.tool]
        if not same_tool_calls:
            continue
        # First matching batch found — this batch decides the verdict.
        # tool_identity short-circuit: any match passes.
        if required.compare_args is None:
            return StepResult(
                step_idx=step_idx,
                required_tool=required.tool,
                tool_found=True,
                matched_call_id=same_tool_calls[0].id,
                args_match=True,
                arg_mismatches={},
            )
        # param / confirm: every same-tool call in this batch must pass.
        # First failing call's mismatches are reported; matched_call_id
        # points at the first same-tool call so the reader can locate
        # the batch.
        first_mismatches: dict[str, list[Any]] = {}
        all_ok = True
        for tc in same_tool_calls:
            ok, mm = _compare_args(required, tc.arguments)
            if not ok:
                all_ok = False
                if not first_mismatches:
                    first_mismatches = mm
        return StepResult(
            step_idx=step_idx,
            required_tool=required.tool,
            tool_found=True,
            matched_call_id=same_tool_calls[0].id,
            args_match=all_ok,
            arg_mismatches=first_mismatches,
        )

    # Tool never called in any batch.
    return StepResult(
        step_idx=step_idx,
        required_tool=required.tool,
        tool_found=False,
        args_match=False,
        arg_mismatches={},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_actions(
    traj: SessionTrajectory,
    required_actions: list[RequiredAction],
) -> list[StepResult]:
    """Match required_actions against trajectory tool-call batches.

    Every rule is single by schema (see runner/schemas.py): exactly one
    `RequiredAction` is matched against the full trajectory.
    semantics — see `_match_step` docstring.
    """
    if not required_actions:
        return []
    batches = _extract_tool_call_batches(traj)
    return [_match_step(idx, req, batches) for idx, req in enumerate(required_actions)]
