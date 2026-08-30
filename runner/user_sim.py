"""LLM-driven user simulator for ATR learning sessions.

Plays the user side of LS interaction as an "empty-shell user":

  - Holds task_params as the user's own task needs; reveals them on demand.
    task_params is a flat {field: concrete_value} dict. user_sim reads the
    value AND the persona narrative, then paraphrases the answer naturally —
    never parrots the raw value verbatim.
  - Has NO inner long-term preferences and NO awareness of the rule-
    acquisition channel beyond the per-turn hook. The orchestrator delivers
    rule answers via a transient hook + post-processing;
    user_sim only sees the hook and emits `<RULE_ANSWER>` when present.

Output contract
-------------------------------------
Per turn the LLM returns a free-text reply, optionally with a
`mark_task_complete()` tool call to terminate the session.

  - text only                 → "continue" with the text as user reply
  - mark_task_complete() call → "end"; any accompanying text is discarded
                                by the orchestrator (USER_END marker is
                                stamped in its place) user_sim is cls-blind: it never sees `cls_status`,
`canonical_answer`, or any Router verdict. The orchestrator decides
whether to inject the rule-answer hook on a given turn, and post-processes
the reply (substitute `<RULE_ANSWER>` with canonical_answer on cls hit or
NO_RULE_DEFLECT on cls miss/error). End-intent normalization is also
orchestrator-side.

Control event protocol
----------------------
`user_sim_reply` returns `(reply_text, intent)`:

  intent ∈ {"continue", "end", "sim_failed"}
                 — "sim_failed" is the orchestrator's terminate-with-sim_error
                   sentinel (LLM call raised after the retry budget).
                 — "end" tells the orchestrator to consider terminating LS;
                   orchestrator may normalize to continue on cls hit so the
                   rule answer is delivered before USER_END.

Design note: user_sim does NOT enforce task-success gating. Agent's stated
completion is taken at face value; evaluator is the final arbiter on
success. This avoids loops where user_sim demands fields the agent
already filled via tool calls (user_sim doesn't see the env-tool history
beyond the `[invoked X → ok]` annotations).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lib.llm import call_llm_with_tools, GPT  # type: ignore
from _prompts import load_prompt
from _constants import MARK_TASK_COMPLETE_TOOL_NAME

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("user_sim", "reply")
_OPENING_SYSTEM_PROMPT = load_prompt("user_sim", "open")
_EMPTY_REPLY_RETRIES = 2

# Opening turns should give the task assistant enough context to start, but
# must not expose the full oracle state. The first gold step is used only as a
# local selector for these user-facing anchors; tool names / full trace /
# complete task_params are never shown to the opening LLM.
_OPENING_ARG_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "search_service_providers": ("service_type", "location"),
    "search_restaurants": ("cuisine", "location"),
    "search_products": ("category",),
    "search_destinations": ("destination_constraints",),
    "search_hotels": ("location",),
    "search_flights": ("origin", "destination"),
    "search_ground_transport": ("origin", "destination"),
    "search_messages": ("sender", "folder"),
    "list_events": ("calendar",),
    "search_documents": ("doc_type", "location"),
    "list_files": ("file_type", "location"),
    "search_events": ("event_tags", "location"),
    "search_trip_stops": ("destination", "stop_tags"),
    "check_conflicts": ("start_time", "end_time"),
    # First-step tools below either have no useful search anchor or their args
    # are already the final object/content. Keep the opening at intent-level.
    "list_trackers": (),
    "list_message_folders": (),
    "list_labels": (),
    "draft_message": (),
    "send_message": (),
    "create_document": (),
    "set_reminder": (),
    "compare_products": (),
    "review_recurring_charges": (),
}

_OPENING_TOOL_INTENTS: dict[str, str] = {
    # Commerce
    "search_products": "find products to buy",
    "compare_products": "compare product options",
    "build_shopping_list": "build a shopping list",
    "place_order": "place an order",
    "modify_order": "modify an order",
    "cancel_order": "cancel an order",
    "track_order": "track an order",
    "return_order": "return an order",
    "review_recurring_charges": "review recurring charges",
    "pause_subscription": "pause a subscription",
    "resume_subscription": "resume a subscription",
    "change_subscription_plan": "change a subscription plan",
    "cancel_subscription": "cancel a subscription",
    # Communication
    "search_messages": "find or respond to a message",
    "list_labels": "review message labels",
    "list_message_folders": "review message folders",
    "set_message_priority": "set message priority",
    "archive_messages": "archive messages",
    "draft_message": "draft a message",
    "send_message": "send a message",
    "label_messages": "label messages",
    "send_draft": "send a drafted message",
    # Reservation
    "search_restaurants": "find or book a restaurant",
    "book_restaurant": "book a restaurant",
    "modify_restaurant_reservation": "modify a restaurant reservation",
    "cancel_restaurant_reservation": "cancel a restaurant reservation",
    "search_events": "find events or tickets",
    "book_event_ticket": "book event tickets",
    "modify_event_ticket": "modify event tickets",
    "cancel_event_ticket": "cancel event tickets",
    "search_service_providers": "find a service provider",
    "book_service_appointment": "book a service appointment",
    "modify_service_appointment": "modify a service appointment",
    "cancel_service_appointment": "cancel a service appointment",
    "track_reservation_updates": "track reservation updates",
    # Scheduling
    "list_events": "check calendar events",
    "check_conflicts": "check schedule availability",
    "create_event": "schedule an event",
    "reschedule_event": "reschedule an event",
    "modify_event": "modify a calendar event",
    "respond_to_event_invite": "respond to an invitation",
    "cancel_event": "cancel a calendar event",
    "set_reminder": "set a reminder",
    "track_event_updates": "track event updates",
    "find_alternative_slots": "find alternative times",
    # Travel
    "search_destinations": "compare destination options",
    "shortlist_destinations": "shortlist destinations",
    "plan_trip": "plan an itinerary",
    "search_trip_stops": "find trip stops or itinerary ideas",
    "search_flights": "find flights",
    "search_hotels": "find or book a place to stay",
    "search_ground_transport": "find ground transportation",
    "book_flight": "book a flight",
    "book_hotel": "book a place to stay",
    "book_ground_transport": "book ground transportation",
    "modify_flight_booking": "modify a flight booking",
    "modify_hotel_booking": "modify a hotel booking",
    "cancel_flight_booking": "cancel a flight booking",
    "cancel_hotel_booking": "cancel a hotel booking",
    "cancel_ground_transport_booking": "cancel ground transportation",
    "track_trip_updates": "track trip updates",
    "replan_trip": "replan a trip",
    # Workspace
    "list_files": "find files",
    "list_trackers": "review trackers",
    "list_file_folders": "review file folders",
    "search_documents": "find a document",
    "classify_files": "classify files",
    "move_files": "move files",
    "archive_files": "archive files",
    "delete_files": "delete files",
    "create_document": "create a document or note",
    "update_document": "update a document",
    "update_tracker": "update a tracker",
}

_OPENING_ARG_BLOCKLIST_EXACT = {
    "id",
    "ids",
    "name",
    "title",
    "subject",
    "body",
    "content",
    "content_brief",
    "list_name",
    "folder_name",
    "project_name",
    "destination_folder_name",
    "archive_folder_name",
    "move_folder_name",
}

_OPENING_ARG_BLOCKLIST_FRAGMENTS = (
    "_id",
    "_ids",
    "points",
    "bullets",
    "keywords",
    "payload",
)

_OPENING_DOCUMENT_DOC_TYPE_KEYS = {
    "doc_type",
    "source_doc_type",
    "template_doc_type",
    "reference_doc_type",
}

MARK_TASK_COMPLETE_TOOL = {
    "type": "function",
    "function": {
        "name": MARK_TASK_COMPLETE_TOOL_NAME,
        # "Tool surface" verbatim. The orchestrator silently
        # discards accompanying text "Text + tool
        # coexistence", but keeping the directive in the description
        # cuts wasted output tokens.
        "description": (
            "Call in its own turn (no accompanying text) after both "
            "sides have exchanged closing courtesies. The session ends "
            "with this call."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def _render_task_params(
    task_params: dict,
    references: list | None = None,
) -> str:
    """Render flat {field: value} task_params as a readable list for the sim LLM.

    Accepts both a flat value and a {value, desc} wrapper shape when reading
    episode JSON — unwraps to bare value.

    When `references` is provided and a task_param value (or any item of a
    list-typed value) matches a `references[*].id`, the matched ref's
    attributes are inlined under the field as a `└─ ...` line. This lets
    user_sim paraphrase the *referent* (a real product / event / message)
    rather than the bare id string — without which it would have to invent
    distinguishing attributes when the agent asks "which one".
    """
    ref_index: dict[str, dict] = {}
    for r in (references or []):
        rid = getattr(r, "id", None) if not isinstance(r, dict) else r.get("id")
        if not rid:
            continue
        attrs = (
            getattr(r, "attributes", None)
            if not isinstance(r, dict)
            else r.get("attributes")
        )
        ref_index[rid] = attrs or {}

    def _attr_line(rid: str) -> str | None:
        attrs = ref_index.get(rid)
        if not attrs:
            return None
        kv = ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}"
            for k, v in attrs.items()
        )
        return f"  └─ this id refers to: {kv}"

    lines: list[str] = []
    for k, val in task_params.items():
        if isinstance(val, dict) and "value" in val:
            val = val["value"]  # unwrap {value, desc} shape
        val_str = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
        lines.append(f"- `{k}`: {val_str}")
        if isinstance(val, str):
            extra = _attr_line(val)
            if extra:
                lines.append(extra)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    extra = _attr_line(item)
                    if extra:
                        lines.append(f"  · {item}: {extra[len('  └─ this id refers to: '):]}")
    return "\n".join(lines)


def _format_context_block(
    task_params: dict,
    narrative: str,
    references: list | None = None,
    reason_for_call: str | None = None,
    gold_trajectory: list | None = None,
) -> str:
    """Render per-session data for user_sim. Data only — no behavioral instructions."""
    parts = []
    if narrative:
        parts.append(f"## User background\n{narrative}")
    if reason_for_call:
        parts.append(f"## Intent\n{reason_for_call}")
    parts.append(
        "## Requirements\n"
        f"{_render_task_params(task_params, references)}"
    )
    if gold_trajectory:
        steps = []
        for i, step in enumerate(gold_trajectory, 1):
            tool = step.tool if hasattr(step, "tool") else step["tool"]
            args = step.arguments if hasattr(step, "arguments") else step["arguments"]
            args_str = json.dumps(args, ensure_ascii=False)
            steps.append(f"{i}. {tool}({args_str})")
        parts.append("## Guided path\n" + "\n".join(steps))
    return "\n\n".join(parts)


def _step_tool(step: Any) -> str | None:
    if hasattr(step, "tool"):
        return step.tool
    if isinstance(step, dict):
        return step.get("tool")
    return None


def _step_arguments(step: Any) -> dict[str, Any]:
    if hasattr(step, "arguments"):
        args = step.arguments
    elif isinstance(step, dict):
        args = step.get("arguments")
    else:
        args = None
    return args if isinstance(args, dict) else {}


def _is_safe_opening_arg_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _OPENING_ARG_BLOCKLIST_EXACT:
        return False
    return not any(fragment in lowered for fragment in _OPENING_ARG_BLOCKLIST_FRAGMENTS)


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        out: list[Any] = []
        for item in value.values():
            out.extend(_flatten_values(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_flatten_values(item))
        return out
    return [value]


def _norm_opening_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def _value_covered_by_task_params(value: Any, task_params: dict[str, Any]) -> bool:
    """Whether a first-step hint value is grounded in the user's requirements."""
    if not task_params:
        return False
    task_values: set[str] = set()
    for task_value in task_params.values():
        task_values.add(_norm_opening_value(task_value))
        for atomic in _flatten_values(task_value):
            task_values.add(_norm_opening_value(atomic))

    value_norm = _norm_opening_value(value)
    if value_norm in task_values:
        return True

    value_atoms = [_norm_opening_value(v) for v in _flatten_values(value)]
    return bool(value_atoms) and all(atom in task_values for atom in value_atoms)


def _opening_arg_covered_by_task_params(
    tool_name: str,
    arg_name: str,
    value: Any,
    task_params: dict[str, Any],
) -> bool:
    if tool_name == "search_documents" and arg_name == "doc_type":
        grounded_values = {
            _norm_opening_value(task_params[key])
            for key in _OPENING_DOCUMENT_DOC_TYPE_KEYS
            if key in task_params
        }
        return _norm_opening_value(value) in grounded_values
    return _value_covered_by_task_params(value, task_params)


def _include_opening_arg(tool_name: str, arg_name: str, args: dict[str, Any]) -> bool:
    if (
        tool_name == "search_messages"
        and arg_name == "sender"
        and _norm_opening_value(args.get("folder")) == "sent"
    ):
        return False
    return True


def build_opening_hint(
    gold_trajectory: list | None,
    task_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a sanitized first-turn hint from the first oracle step.

    This function reads `gold_trajectory[0].arguments` only as candidate
    anchors. A candidate must also be present in task_params before it can be
    shown to the opening LLM; otherwise gold-only search strategy values (for
    example calendar="invites" or doc_type="template" used only to discover
    context) would leak oracle process information into the user's first turn.
    Full task_params and references still stay out of the opening prompt.
    """
    if not gold_trajectory:
        return {}
    task_params = task_params or {}
    first = gold_trajectory[0]
    tool = _step_tool(first)
    if not tool:
        return {}
    allowed = _OPENING_ARG_ALLOWLIST.get(tool, ())
    if not allowed:
        return {}
    args = _step_arguments(first)
    return {
        key: args[key]
        for key in allowed
        if key in args and _is_safe_opening_arg_key(key)
        and _include_opening_arg(tool, key, args)
        and _opening_arg_covered_by_task_params(tool, key, args[key], task_params)
    }


def build_opening_task_kind(gold_trajectory: list | None) -> str | None:
    """Return a user-facing task type from the first oracle step.

    This exposes only the broad task family ("find flights", "draft a
    message"), not raw tool names, IDs, selected objects, or full oracle flow.
    """
    if not gold_trajectory:
        return None
    first = gold_trajectory[0]
    tool = _step_tool(first)
    if not tool:
        return None
    return _OPENING_TOOL_INTENTS.get(tool)


def _format_opening_context_block(
    reason_for_call: str | None = None,
    opening_task_kind: str | None = None,
    opening_hint: dict[str, Any] | None = None,
) -> str:
    """Render the deliberately small data surface for user_sim_open."""
    parts = []
    if reason_for_call:
        parts.append(f"## Intent\n{reason_for_call}")
    if opening_task_kind:
        parts.append(f"## Task type\n{opening_task_kind}")
    if opening_hint:
        lines = []
        for key, val in opening_hint.items():
            val_str = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
            lines.append(f"- `{key}`: {val_str}")
        parts.append("## Opening hint\n" + "\n".join(lines))
    return "\n\n".join(parts)


def user_sim_open(
    reason_for_call: str,
    task_params: dict,
    narrative: str,
    references: list | None = None,
    gold_trajectory: list | None = None,
    model: str = GPT,
    seed: int | None = None,
    session_id: str | None = None,
) -> str:
    """Generate the user's opening message for a learning session.

    The agent never sees a pre-written instruction — instead, user_sim
    synthesizes the opening from `reason_for_call` plus a sanitized hint
    derived from the first gold step. Full task_params, references, and
    guided paths are intentionally excluded from the opening prompt; they
    are used only by user_sim_reply after the agent asks follow-ups.
    """
    _ = narrative, references  # opening intentionally ignores them
    opening_task_kind = build_opening_task_kind(gold_trajectory)
    opening_hint = build_opening_hint(gold_trajectory, task_params)
    context_block = _format_opening_context_block(
        reason_for_call=reason_for_call,
        opening_task_kind=opening_task_kind,
        opening_hint=opening_hint,
    )
    messages = [
        {"role": "system", "content": _OPENING_SYSTEM_PROMPT + "\n\n" + context_block},
        {"role": "user", "content": "Start the conversation."},
    ]
    # propagate SweepAbort up to the sweep runner. An empty
    # opening is not a valid user turn — retry the same prompt; if the model
    # still returns empty after _EMPTY_REPLY_RETRIES+1 attempts, abort the
    # cell rather than inject a template fallback that would distort opening
    # style. Mirrors the retry-then-abort pattern in user_sim_reply.
    for attempt_idx in range(_EMPTY_REPLY_RETRIES + 1):
        try:
            result = call_llm_with_tools(
                messages=messages,
                tools=None,
                model=model,
                temperature=None,
            )
        except Exception as e:
            from _abort import SweepAbort
            raise SweepAbort(module="user_sim_open", session_id=session_id, original=e) from e
        reply = (result.get("content") or "").strip()
        if reply:
            return reply
        logger.warning(
            "user_sim_open returned empty content (session=%s, attempt=%d); retrying",
            session_id,
            attempt_idx + 1,
        )

    from _abort import SweepAbort
    raise SweepAbort(
        module="user_sim_open_empty_reply",
        session_id=session_id,
        original=RuntimeError(
            "user_sim_open returned empty content "
            f"on {_EMPTY_REPLY_RETRIES + 1} consecutive attempts"
        ),
    )


def _build_sim_messages(
    context_block: str,
    conversation_history: list[dict],
    agent_text: str,
    rule_hook: str | None = None,
) -> list[dict]:
    """Build OpenAI messages for the user_sim LLM call.

    Role reversal convention (standard for user simulators):
      - In the real conversation: user=human, assistant=agent.
      - In this LLM call: user=agent-side messages, assistant=user-sim replies.
    The LLM plays the user replying to the agent.

    Message layout (current turn):
        ...history...
        {role: "user",   content: agent_text}      # agent's plain output
        {role: "user",   content: rule_hook}        # optional, Router=True only

    `agent_text` is the agent's user-visible message as plain text — no
    `<output>...</output>` envelope. `rule_hook`, when
    given, is a transient `<rule_answer>...</rule_answer>` directive
    appended as a separate `role="user"` message; the hook is not
    persisted to trajectory.

    `conversation_history` should NOT include the current agent turn
    being routed — the orchestrator excludes it before calling.
    """
    msgs: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT + "\n\n" + context_block},
    ]
    for turn in conversation_history:
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            msgs.append({"role": "user", "content": content})
        elif role == "user":
            msgs.append({"role": "assistant", "content": content})
    msgs.append({"role": "user", "content": agent_text})
    if rule_hook:
        msgs.append({"role": "user", "content": rule_hook})
    return msgs


def user_sim_reply(
    agent_text: str,
    task_params: dict,
    narrative: str,
    conversation_history: list[dict],
    references: list | None = None,
    reason_for_call: str | None = None,
    gold_trajectory: list | None = None,
    rule_hook: str | None = None,
    model: str = GPT,
    seed: int | None = None,
    session_id: str | None = None,
) -> tuple[str, str]:
    """Generate a task-only user reply to the agent's STU turn; user_sim is cls-blind: it sees the agent's plain
    `output` and, on Router=True turns, a transient rule-answer hook
    referencing the question span. user_sim never sees Router/cls
    verdicts or canonical_answer; the orchestrator decides hook
    presence/content and post-processes the reply.

    Args:
        agent_text: Current agent turn's user-visible message as plain
            text. there is no `<output>...</output>`
            envelope. `reason` is not included.
        task_params: Full ground-truth task parameters.
        narrative: Persona narrative for tone/style grounding.
        conversation_history: Prior {role, content} turns from before the
            current agent turn. **Must NOT include the current agent turn**
            — the orchestrator is responsible for excluding it.
        references: LearningSession.local_env.references — used to inline
            referent attributes when a task_param value is a ref id.
        reason_for_call: User's overall intent for this session.
        gold_trajectory: Oracle tool-call sequence for guiding stuck agents.
        rule_hook: Optional transient directive appended after agent_text
            as a separate `role="user"` message. Form:
            `<rule_answer>For the question "{span}" in the assistant's
            last message, your answer is <RULE_ANSWER>. Use this exact
            token in your reply where the answer would naturally
            fit.</rule_answer>`. The orchestrator sets this on
            Router=True turns and post-processes any `<RULE_ANSWER>`
            tokens in the returned reply.
        model: LLM model identifier.
        seed: Determinism hint.
            User simulator calls do not forward this to the LLM backend; it is
            retained for runner interface compatibility.

    Returns:
        (reply_text, intent)

        intent ∈ {"continue", "end", "sim_failed"}. On sim_failed,
        reply_text is empty.
    """
    context_block = _format_context_block(
        task_params, narrative,
        references=references,
        reason_for_call=reason_for_call,
        gold_trajectory=gold_trajectory,
    )
    messages = _build_sim_messages(
        context_block,
        conversation_history,
        agent_text,
        rule_hook=rule_hook,
    )

    # exhausted retry → raise SweepAbort (not sim_failed).
    # `call_llm_with_tools` already wraps a tenacity retry (7 attempts +
    # exponential backoff) for transient provider errors. An exception
    # here means the retry budget is exhausted; sweep should abort.
    #
    # A semantic empty reply (no text and no mark_task_complete) is not a
    # valid human turn. Retry the same user_sim prompt a small number of
    # times; if the simulator remains empty, abort the cell rather than
    # appending an empty user message to the experiment transcript.
    for attempt_idx in range(_EMPTY_REPLY_RETRIES + 1):
        try:
            result = call_llm_with_tools(
                messages=messages,
                tools=[MARK_TASK_COMPLETE_TOOL],
                model=model,
                temperature=None,
            )
        except Exception as e:
            from _abort import SweepAbort
            raise SweepAbort(module="user_sim", session_id=session_id, original=e) from e

        content = (result.get("content") or "").strip()
        tool_calls = result.get("tool_calls") or []
        end_signaled = any(
            tc.get("name") == MARK_TASK_COMPLETE_TOOL_NAME for tc in tool_calls
        )
        if end_signaled:
            return content, "end"
        if content:
            return content, "continue"
        logger.warning(
            "user_sim_reply returned empty content and no mark_task_complete "
            "(session=%s, attempt=%d); retrying",
            session_id,
            attempt_idx + 1,
        )

    from _abort import SweepAbort
    raise SweepAbort(
        module="user_sim_empty_reply",
        session_id=session_id,
        original=RuntimeError(
            "user_sim returned empty content and no mark_task_complete "
            f"on {_EMPTY_REPLY_RETRIES + 1} consecutive attempts"
        ),
    )
