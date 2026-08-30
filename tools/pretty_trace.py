"""Render a session trajectory JSON into a scannable human-readable .txt.

Send-to-user architecture + LS interaction protocol:
- The trace is an honest transcript: assistant turns + tool calls + tool
  results + user messages, in turn order. Tool calls are auditable on
  their face (args + result body shown).
- Annotations are layered on top, all reading from
  `trajectory.interaction_events`:
    [route]        Under the `tool` message of a `send_to_user` ack —
                   user_sim's routing decision (rule / task).
    [cls]          Under the `tool` message of a rule-routed `send_to_user`
                   ack — classifier verdict: `hit rule=...` / `no_match`
                   / `error`.
    [off_protocol] Under an assistant **text turn** (no tool_calls,
                   non-empty content) in LS — agent leaked user-facing
                   content outside `send_to_user`. This is the only
                   off-protocol shape.
    [hook ×N]      Under an assistant block whose turn_idx had hook
                   retries fire. With successful rescue, rendered as
                   `[hook ×N → rescued]`.

Streaming vs static: the live `SessionStreamWriter` and the post-hoc
`format_trajectory` may legitimately diverge. Streaming shows the leak being
detected and rescued in real time; static rendering shows only the rescued
AGENT with a `[hook ×N → rescued]` annotation.

CLI:
    uv run python tools/pretty_trace.py <traj.json> [--eval <eval.json>]
                                                    [--out <out.txt>]
                                                    [--system]   # include system prompt

Programmatic:
    from tools.pretty_trace import format_trajectory
    text = format_trajectory(traj_dict, eval_session=session_eval_dict)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
from _constants import USER_END_MARKER  # noqa: E402

_VALUE_TRUNCATE = 100   # per-value char cap when rendering tool args / results
_RESULT_TRUNCATE = 400  # tool-result body cap
_INDENT = "       "


def _short(v: Any, cap: int = _VALUE_TRUNCATE) -> str:
    """One-liner repr of an arg value, truncated to `cap` chars."""
    if v is None:
        return "None"
    if isinstance(v, str):
        s = v.replace("\n", " ")
    elif isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False)
    else:
        s = repr(v)
    return s if len(s) <= cap else s[:cap - 1] + "…"


def _format_args(args: dict) -> str:
    """`tool(k=v, k=v)` arg list, dropping None to reduce noise."""
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        if v is None:
            continue
        parts.append(f"{k}={_short(v)}")
    return ", ".join(parts)


def _wrap_block(text: str, indent: str = _INDENT, width: int = 110) -> str:
    """Indent every line of a multi-line block by `indent`."""
    if not text:
        return ""
    lines = text.splitlines() or [""]
    return "\n".join(indent + ln if ln else indent.rstrip() for ln in lines)


def _format_route_decision(
    route: str | None,
    is_cross_session_ask: bool | None = None,
    classification: str | None = None,
    reason_has_cross_session_rule_intent: bool | None = None,
    output_asks_cross_session_rule_question: bool | None = None,
    is_strict_rule_question: bool | None = None,
    rule_question_span: str | None = None,
    router_error: str | None = None,
) -> str:
    """`[route]` annotation rendered under a `send_to_user` tool response.

    Shows the binary `route` (rule / task). Current trajectories include the
    Router's strict-question flag and span. Older trajectories still render
    their classification / bool-gate fields.
    """
    if is_strict_rule_question is not None:
        subparts = [
            f"strict={str(is_strict_rule_question).lower()}",
        ]
        if rule_question_span:
            subparts.append(f"span={_short(rule_question_span, 120)!r}")
        if router_error:
            subparts.append(f"router_error={_short(router_error, 80)!r}")
        return (
            f"{_INDENT}[route] {route or '?'}  "
            f"({'; '.join(subparts)})"
        )
    if is_cross_session_ask is not None:
        subparts = []
        if reason_has_cross_session_rule_intent is not None:
            subparts.append(
                "reason="
                f"{str(reason_has_cross_session_rule_intent).lower()}"
            )
        if output_asks_cross_session_rule_question is not None:
            subparts.append(
                "output="
                f"{str(output_asks_cross_session_rule_question).lower()}"
            )
        detail = (
            "; " + ", ".join(subparts)
            if subparts
            else ""
        )
        return (
            f"{_INDENT}[route] {route or '?'}  "
            f"(is_cross_session_ask={str(is_cross_session_ask).lower()}"
            f"{detail})"
        )
    if classification:
        return f"{_INDENT}[route] {route or '?'}  ({classification})"
    return f"{_INDENT}[route] {route or '?'}"


def _format_cls_verdict(
    *,
    verdict: str,
    rule_id: str | None,
    canonical_answer: str | None,
) -> str:
    """`[cls]` annotation rendered under a rule-routed `send_to_user`
    tool response. Verdict shapes:

      hit       →  "[cls] hit  rule=<id>  canonical_answer=\"…\""
      no_match  →  "[cls] no_match"
      error     →  "[cls] error"
    """
    if verdict == "error":
        return f"{_INDENT}[cls] error"
    if verdict == "hit":
        ans = _short(canonical_answer or "", 120)
        return (
            f"{_INDENT}[cls] hit  rule={rule_id}"
            f"  canonical_answer={ans!r}"
        )
    return f"{_INDENT}[cls] no_match"


def _format_off_protocol_marker() -> str:
    """`[off_protocol]` annotation: agent emitted a text-only turn in LS
    that bypassed `send_to_user`. Marker stands alone — no subtitle.
    """
    return f"{_INDENT}[off_protocol]"


def _format_hook_marker(n: int, rescued: bool) -> str:
    """`[hook ×N]` or `[hook ×N → rescued]` annotation.

    Args:
      n: number of `hook_appended` events for this problem turn (i.e.
        how many retries the runner spent coaxing the agent toward
        `send_to_user`). `×1` is the common single-retry case;
        higher counts mean the model was stubborn.
      rescued: True iff one of the resulting send_to_user calls had
        `was_hook_rescued=True` — the rescue ultimately succeeded.
        Renders as `→ rescued` suffix in that case.
    """
    suffix = " → rescued" if rescued else ""
    if n <= 1:
        return f"{_INDENT}[hook{suffix}]"
    return f"{_INDENT}[hook ×{n}{suffix}]"


def _format_rescued_send_marker() -> str:
    """`[rescued]` standalone marker — used in streaming only, where
    the rescue success is known after the AGENT block has already been
    written. Static rendering consolidates rescue into the AGENT
    block's `[hook ×N → rescued]` annotation.
    """
    return f"{_INDENT}[rescued]"


def _build_send_groups(events: list[dict]) -> list[dict]:
    """Walk interaction_events in order and group them into one entry per
    send_to_user call. Each group: {turn_idx, route, router fields, legacy
    gate fields, cls, was_hook_rescued}.

    The orchestrator emits in this order per logical send_to_user:
    send_event, route_decision, [cls_verdict if route=rule]. Same-turn
    duplicate send_to_user calls are dropped before persistence, so this
    grouping remains one route per send.
    """
    groups: list[dict] = []
    cur: dict | None = None
    for ev in events:
        kind = ev.get("kind")
        if kind == "send_event":
            if cur is not None:
                groups.append(cur)
            cur = {
                "turn_idx": ev.get("turn_idx"),
                "route": None,
                "is_strict_rule_question": None,
                "rule_question_span": None,
                "router_error": None,
                "is_cross_session_ask": None,
                "reason_has_cross_session_rule_intent": None,
                "output_asks_cross_session_rule_question": None,
                "classification": None,
                "cls": None,
                "was_hook_rescued": bool(ev.get("was_hook_rescued")),
            }
        elif kind == "route_decision" and cur is not None:
            cur["route"] = ev.get("route")
            cur["is_strict_rule_question"] = ev.get("is_strict_rule_question")
            cur["rule_question_span"] = ev.get("rule_question_span")
            cur["router_error"] = ev.get("router_error")
            cur["is_cross_session_ask"] = ev.get("is_cross_session_ask")
            cur["reason_has_cross_session_rule_intent"] = ev.get(
                "reason_has_cross_session_rule_intent"
            )
            cur["output_asks_cross_session_rule_question"] = ev.get(
                "output_asks_cross_session_rule_question"
            )
            cur["classification"] = ev.get("classification")
        elif kind == "cls_verdict" and cur is not None:
            cur["cls"] = ev
    if cur is not None:
        groups.append(cur)
    return groups


def _format_eval_summary(eval_session: dict | None) -> str:
    """Compact gold-check footer for test sessions; empty string if no eval."""
    if not eval_session:
        return ""
    lines: list[str] = []
    success = eval_session.get("task_success")
    sym = "✓" if success else "⨯"
    lines.append(f"gold check: {sym}  task_success={success}")
    for sr in eval_session.get("step_results", []):
        idx = sr.get("step_idx")
        tool = sr.get("required_tool")
        if not sr.get("tool_found"):
            lines.append(f"  step{idx}: {tool}  ⨯ not called")
            continue
        if not sr.get("args_match"):
            mm = sr.get("arg_mismatches") or {}
            for k, pair in mm.items():
                gold, actual = pair if isinstance(pair, list) else (pair, "?")
                lines.append(
                    f"  step{idx}: {tool}  arg `{k}` "
                    f"gold={_short(gold, 60)}  actual={_short(actual, 60)}"
                )
            if not mm:
                lines.append(f"  step{idx}: {tool}  ⨯ args_match=false (no detail)")
        else:
            lines.append(f"  step{idx}: {tool}  ✓")
    return "\n".join(lines)


def format_trajectory(
    traj: dict,
    *,
    eval_session: dict | None = None,
    include_system: bool = False,
) -> str:
    """Render a SessionTrajectory dict as a readable .txt block."""
    sid = traj.get("session_id", "?")
    stype = traj.get("session_type", "?")
    variant = traj.get("agent_variant", "?")
    term = traj.get("termination_reason", "?")
    steps = traj.get("step_count", 0)
    dur = traj.get("duration_seconds", 0.0)

    rule_id = (eval_session or {}).get("rule_id")
    # For test sessions with eval data, status follows gold check, not termination —
    # an agent that called finish_session but solved the wrong task is a fail.
    if eval_session is not None:
        sym = "✓" if eval_session.get("task_success") else "⨯"
        gold_tag = " gold-pass" if eval_session.get("task_success") else " gold-fail"
    else:
        sym = "✓" if term in ("agent_stop", "task_complete") else "⨯"
        gold_tag = ""

    head_bits = [
        f"session  : {sid}",
        f"type     : {stype}  | variant: {variant}"
        + (f"  | rule: {rule_id}" if rule_id else ""),
        f"status   : {sym}{gold_tag}  | term={term}  | steps={steps}  | dur={dur:.1f}s",
    ]
    head = "\n".join(head_bits)
    div = "─" * 80

    # Build per-turn ordered queues of send_to_user groups. Each `send_to_user` tool
    # message pops one group and renders [route] (+ [cls] if rule-routed).
    send_groups = _build_send_groups(traj.get("interaction_events") or [])
    send_queue_by_turn: dict[int, list[dict]] = {}
    for g in send_groups:
        send_queue_by_turn.setdefault(g["turn_idx"], []).append(g)
    consumed_by_turn: dict[int, int] = {}

    off_protocol_by_assistant_turn: dict[int, dict] = {}
    hook_count_by_assistant_turn: dict[int, int] = {}
    rescued_assistant_turns: set[int] = set()
    for ev in (traj.get("interaction_events") or []):
        ek = ev.get("kind")
        if ek == "off_protocol_ask":
            off_protocol_by_assistant_turn[ev.get("turn_idx")] = ev
        elif ek == "hook_appended":
            ti_ev = ev.get("turn_idx")
            hook_count_by_assistant_turn[ti_ev] = (
                hook_count_by_assistant_turn.get(ti_ev, 0) + 1
            )
        elif ek == "send_event" and ev.get("was_hook_rescued"):
            rescued_assistant_turns.add(ev.get("turn_idx"))

    last_assistant_turn: int | None = None
    last_assistant_called_send: bool = False

    body_blocks: list[str] = []
    for msg in traj.get("messages", []):
        role = msg.get("role")
        if role == "system" and not include_system:
            continue
        ti = msg.get("turn_idx", "?")
        content = (msg.get("content") or "").rstrip()
        tcs = msg.get("tool_calls") or []

        if role == "system":
            body_blocks.append(f"[t{ti:>2}] SYSTEM")
            body_blocks.append(_wrap_block(content))
            continue

        if role == "user":
            # USER_END marker is the orchestrator-stamped LS terminator;
            # render as a control event, not a conversational USER turn,
            # since runtime no longer drives state from the token (it
            # reflects mark_task_complete fired earlier in user_sim).
            if content == USER_END_MARKER:
                body_blocks.append(
                    f"[t{ti:>2}] [control] task_complete (LS ends here)"
                )
                continue
            body_blocks.append(f"[t{ti:>2}] USER")
            body_blocks.append(_wrap_block(content))
            continue

        if role == "tool":
            err = msg.get("tool_error")
            err_tag = "  ERR" if err else ""
            tcid = msg.get("tool_call_id") or ""
            tcid_short = f"  ({tcid[-8:]})" if tcid else ""
            body = _short(content, _RESULT_TRUNCATE)
            block_lines = [
                f"[t{ti:>2}] TOOL ←{err_tag}{tcid_short}",
                _wrap_block(body),
            ]
            # [route] (+ [cls]) annotations: pop the next send_to_user
            # group for the latched assistant turn. [rescued] is NOT
            # rendered here under the TOOL ack, paired with the send event.
            # consolidates onto the AGENT block as `[hook ×N → rescued]`.
            if (last_assistant_called_send
                    and last_assistant_turn is not None):
                queue = send_queue_by_turn.get(last_assistant_turn) or []
                idx = consumed_by_turn.get(last_assistant_turn, 0)
                if idx < len(queue):
                    g = queue[idx]
                    consumed_by_turn[last_assistant_turn] = idx + 1
                    block_lines.append(_format_route_decision(
                        g.get("route"),
                        g.get("is_cross_session_ask"),
                        g.get("classification"),
                        g.get("reason_has_cross_session_rule_intent"),
                        g.get("output_asks_cross_session_rule_question"),
                        g.get("is_strict_rule_question"),
                        g.get("rule_question_span"),
                        g.get("router_error"),
                    ))
                    cls_ev = g.get("cls")
                    if cls_ev is not None:
                        if cls_ev.get("cls_error"):
                            verdict = "error"
                        elif cls_ev.get("rule_id"):
                            verdict = "hit"
                        else:
                            verdict = "no_match"
                        # canonical_answer isn't on the InteractionEvent;
                        # render rule_id only (callers wanting the answer
                        # find it on the user message that follows).
                        block_lines.append(_format_cls_verdict(
                            verdict=verdict,
                            rule_id=cls_ev.get("rule_id"),
                            canonical_answer=None,
                        ))
            body_blocks.append("\n".join(block_lines))
            continue

        if role == "assistant":
            # tool calls (one or many) come first; plain text (if any) after
            head_line = f"[t{ti:>2}] AGENT"
            block_lines: list[str] = [head_line]
            for tc in tcs:
                args_repr = _format_args(tc.get("arguments") or {})
                tcid = tc.get("id") or ""
                tcid_short = f"  ({tcid[-8:]})" if tcid else ""
                block_lines.append(_INDENT + f"→ {tc['name']}({args_repr}){tcid_short}")
            if content:
                if tcs:
                    block_lines.append("")  # blank between calls and text
                block_lines.append(_wrap_block(content))
            # [off_protocol] annotation: text-only leak that was NOT
            # rescued (or hook was off). Only emitted when the
            # `off_protocol_ask` event is persisted.
            if isinstance(ti, int) and ti in off_protocol_by_assistant_turn:
                block_lines.append(_format_off_protocol_marker())
            # [hook ×N (→ rescued)?] annotation:
            # consolidates rescue success status into the same
            # AGENT-block annotation. Reader sees rescue/non-rescue
            # without scanning further.
            if isinstance(ti, int) and ti in hook_count_by_assistant_turn:
                block_lines.append(
                    _format_hook_marker(
                        hook_count_by_assistant_turn[ti],
                        rescued=(ti in rescued_assistant_turns),
                    )
                )
            body_blocks.append("\n".join(block_lines))
            # Latch state for the route/cls annotations on subsequent tool
            # messages from this assistant turn.
            last_assistant_turn = ti if isinstance(ti, int) else None
            last_assistant_called_send = any(
                tc.get("name") == "send_to_user" for tc in tcs
            )
            continue

        body_blocks.append(f"[t{ti:>2}] {role.upper()}  (unhandled role)")

    out = [head, div, "\n\n".join(body_blocks)]
    eval_block = _format_eval_summary(eval_session)
    if eval_block:
        out.append(div)
        out.append(eval_block)
    out.append("")  # trailing newline
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming writer — used by runner via on_turn callback for live tail-able
# trace files. Each session opens one writer; events append immediately so
# `tail -f traces/<sid>.txt` reflects the conversation as it unfolds.
# ─────────────────────────────────────────────────────────────────────────────


def _format_session_header(meta: dict) -> str:
    """Header block written at session start (gold check appended later)."""
    sid = meta.get("session_id", "?")
    stype = meta.get("session_type", "?")
    variant = meta.get("agent_variant", "?")
    rule_id = meta.get("rule_id")
    instr = (meta.get("instruction") or "").strip()
    head_bits = [
        f"session  : {sid}",
        f"type     : {stype}  | variant: {variant}"
        + (f"  | rule: {rule_id}" if rule_id else ""),
    ]
    if instr:
        head_bits.append(f"instruction: {_short(instr, 200)}")
    return "\n".join(head_bits) + "\n" + ("─" * 80)


def _format_event(evt: dict) -> str:
    """Render one on_turn event as a complete .txt block (no trailing newline)."""
    et = evt["type"]
    ti = evt.get("turn_idx", "?")
    if et == "agent_tool":
        lines = [f"[t{ti:>2}] AGENT"]
        for tc in evt.get("tool_calls", []):
            args_repr = _format_args(tc.get("arguments") or {})
            lines.append(_INDENT + f"→ {tc['name']}({args_repr})")
        content = (evt.get("content") or "").strip()
        if content:
            lines.append("")
            lines.append(_wrap_block(content))
        return "\n".join(lines)

    if et == "tool_response":
        err = evt.get("tool_error")
        err_tag = "  ERR" if err else ""
        body = _short((evt.get("content") or ""), _RESULT_TRUNCATE)
        return f"[t{ti:>2}] TOOL ←{err_tag}\n{_wrap_block(body)}"

    if et == "agent_text":
        content = (evt.get("content") or "").strip()
        return f"[t{ti:>2}] AGENT\n{_wrap_block(content)}"

    if et == "user_reply":
        content = (evt.get("content") or "").strip()
        # Render the orchestrator-stamped USER_END marker as a control event,
        # not a USER turn — runtime no longer drives state from this token
        # (mark_task_complete owns the phase transition). Showing it as
        # `USER` would imply another conversational turn follows; the marker
        # is in fact the LS terminator.
        if content == "###USER_END###":
            return f"[t{ti:>2}] [control] task_complete (LS ends here)"
        return f"[t{ti:>2}] USER\n{_wrap_block(content)}"

    if et == "route_decision":
        # Inline annotation under the preceding `send_to_user` tool
        # response. Shows binary route + Router details when available.
        return _format_route_decision(
            evt.get("route"),
            evt.get("is_cross_session_ask"),
            evt.get("classification"),
            evt.get("reason_has_cross_session_rule_intent"),
            evt.get("output_asks_cross_session_rule_question"),
            evt.get("is_strict_rule_question"),
            evt.get("rule_question_span"),
            evt.get("router_error"),
        )

    if et == "cls_verdict":
        # Inline annotation under a rule-routed `send_to_user` tool response.
        # `verdict` ∈ {hit, no_match, error}; rule_id and canonical_answer
        # populated on hit.
        return _format_cls_verdict(
            verdict=evt.get("verdict", "?"),
            rule_id=evt.get("rule_id"),
            canonical_answer=evt.get("canonical_answer"),
        )

    if et == "off_protocol":
        return _format_off_protocol_marker()

    if et == "hook_appended":
        # Streaming emits one `[hook]` per retry as it happens (audit
        # log). Static rendering consolidates the
        # count and rescue status into a single `[hook ×N → rescued]`
        # annotation on the AGENT block.
        return _format_hook_marker(n=1, rescued=False)

    if et == "send_event":
        # The `send_to_user` tool call is already rendered via
        # `agent_tool`; we don't duplicate the full entry. In streaming
        # we still emit a standalone `[rescued]` marker right after a
        # rescued send so a developer tailing the file can see the
        # rescue happen in real time. Static rendering does the
        # consolidation onto the AGENT block instead.
        if evt.get("was_hook_rescued"):
            return _format_rescued_send_marker()
        return None  # type: ignore[return-value]

    return f"[t{ti:>2}] {et} (unhandled event type)"


def _format_session_footer(traj: dict, eval_session: dict | None) -> str:
    """Footer appended after the session finishes (and optionally after eval)."""
    term = traj.get("termination_reason", "?")
    steps = traj.get("step_count", 0)
    dur = traj.get("duration_seconds", 0.0)
    if eval_session is not None:
        sym = "✓" if eval_session.get("task_success") else "⨯"
        gold_tag = " gold-pass" if eval_session.get("task_success") else " gold-fail"
    else:
        sym = "✓" if term in ("agent_stop", "task_complete") else "⨯"
        gold_tag = ""

    lines = ["─" * 80,
             f"end      : {sym}{gold_tag}  | term={term}  | steps={steps}  | dur={dur:.1f}s"]
    eval_block = _format_eval_summary(eval_session)
    if eval_block:
        lines.append(eval_block)
    return "\n".join(lines)


class SessionStreamWriter:
    """Append-only writer for one session's pretty trace.

    Open at session start with header(meta), call on each event with on_event,
    close with on_finish(traj_dict). After eval completes, call append_eval_footer
    to attach gold-check details. The file remains tail-friendly throughout —
    events are flushed immediately.

    Annotation events (`route_decision`, `cls_verdict`, `off_protocol`)
    are rendered tight against the preceding block (no leading blank line)
    so they read as "attached to" the turn they annotate, not as
    freestanding entries.
    """

    # Event types whose render is an inline annotation under the preceding
    # block (no leading blank line — they hug the block they annotate).
    _INLINE_ANNOTATION_TYPES = frozenset({
        "route_decision", "cls_verdict", "off_protocol",
        "hook_appended", "send_event",
    })

    def __init__(self, out_path: Path, meta: dict):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = out_path
        self._fh = open(out_path, "w", encoding="utf-8")
        # Layout invariant: every write below ends with exactly one "\n".
        # Block separators (a blank line) are written as the *leading* "\n"
        # of the next regular block. Annotation events skip the leading
        # "\n" so they hug the previous block.
        self._fh.write(_format_session_header(meta) + "\n")
        self._fh.flush()
        self._first_event = True

    def on_event(self, evt: dict) -> None:
        rendered = _format_event(evt)
        # _format_event may return None for events that should be silently
        # skipped (e.g. legacy `agent_label` events that live in
        # the trajectory but is not rendered in the trace).
        if rendered is None:
            return
        is_annotation = evt.get("type") in self._INLINE_ANNOTATION_TYPES
        # Leading separator: blank line before regular blocks (and before
        # the first event, which sits after the header). Annotations skip
        # this so they sit adjacent to the prior block.
        if self._first_event:
            self._fh.write("\n")
            self._first_event = False
        elif not is_annotation:
            self._fh.write("\n")
        self._fh.write(rendered + "\n")
        self._fh.flush()

    def on_finish(self, traj: dict) -> None:
        # Footer is its own block — leading blank line for separation.
        self._fh.write("\n" + _format_session_footer(traj, eval_session=None) + "\n")
        self._fh.flush()
        self._fh.close()


def append_eval_footer(trace_path: Path, traj: dict, eval_session: dict) -> None:
    """Re-write the trailing footer once eval is available so gold-check shows.

    The streamed footer (term/steps/dur) was written without gold info; we
    rewrite the bottom of the file with the enriched footer. Implementation:
    truncate everything from the last divider onward, then re-append.
    """
    text = trace_path.read_text(encoding="utf-8")
    div = "─" * 80
    cut = text.rfind(div)
    head = text[:cut] if cut > 0 else text
    footer = _format_session_footer(traj, eval_session=eval_session)
    trace_path.write_text(head + footer + "\n", encoding="utf-8")


def render_traj_file(
    traj_path: Path,
    *,
    eval_path: Path | None = None,
    include_system: bool = False,
) -> str:
    traj = json.loads(traj_path.read_text())
    eval_session: dict | None = None
    if eval_path and eval_path.exists():
        eval_doc = json.loads(eval_path.read_text())
        sid = traj.get("session_id")
        for sr in eval_doc.get("session_results", []):
            if sr.get("session_id") == sid:
                eval_session = sr
                break
    return format_trajectory(traj, eval_session=eval_session, include_system=include_system)


def render_runs_dir(runs_dir: Path, *, include_system: bool = False) -> list[Path]:
    """Render every traj under <runs_dir>/trajectories/ → <runs_dir>/traces/.

    Looks up <runs_dir>/eval.json when present to enrich test-session footers.
    Returns the list of written .txt paths.
    """
    traj_dir = runs_dir / "trajectories"
    if not traj_dir.is_dir():
        return []
    eval_path = runs_dir / "eval.json"
    out_dir = runs_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for tp in sorted(traj_dir.glob("*.json")):
        text = render_traj_file(tp, eval_path=eval_path, include_system=include_system)
        op = out_dir / (tp.stem + ".txt")
        op.write_text(text, encoding="utf-8")
        written.append(op)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trajectory JSON as readable text.")
    parser.add_argument("traj", type=Path, help="path to trajectory JSON OR a runs/<...>/<variant>/ dir")
    parser.add_argument("--eval", type=Path, default=None,
                        help="path to eval.json (auto-detected if traj is a runs dir)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output .txt path (defaults to stdout for single file, "
                             "or <runs_dir>/traces/ for a runs dir)")
    parser.add_argument("--system", action="store_true",
                        help="include the system prompt in the output")
    args = parser.parse_args()

    if args.traj.is_dir():
        written = render_runs_dir(args.traj, include_system=args.system)
        for p in written:
            print(p)
        if not written:
            print(f"no trajectories found under {args.traj}/trajectories/", file=sys.stderr)
            sys.exit(2)
        return

    text = render_traj_file(args.traj, eval_path=args.eval, include_system=args.system)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(args.out)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
