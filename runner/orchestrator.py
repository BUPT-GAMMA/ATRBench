"""Two-party orchestrator: agent <-> ATREnv with LLM-driven user simulator.

Send-to-user architecture + free-text user_sim +
uniform rule-answer hook. The agent uses native OpenAI function
calling. Tool schemas come from env.get_openai_schemas() PLUS one
synthetic control-plane tool prepended by this orchestrator:
  - `finish_session()`             in TS only — agent's terminator
  - `send_to_user(output, reason)` in LS only — agent's sole user-facing
                                    channel

Four components in LS:
  1. agent (under test): emits tool_calls (env tools + send_to_user) or
     plain text (off-protocol leak).
  2. Router (cheap fixed model): classifies whether the output asks a strict
     standing-rule question and extracts the question span.
  3. cls (rule classifier): runs ONLY when Router returns
     is_strict_rule_question=true. Reads the Router's rule_question_span and
     matches against the episode rule pool.
  4. user_sim (GPT, free text + optional `mark_task_complete()` tool):
     produces the natural reply and LS intent. user_sim is
     cls-blind: it sees the agent's plain output and, on Router=True
     turns, a transient rule-answer hook directing it to emit
     `<RULE_ANSWER>` for runner substitution. Reason and cls verdict are
     hidden from user_sim.

Control plane:
  agent_stop    → agent calls `finish_session()` (TS only).
  task_complete → user_sim calls `mark_task_complete()` (intent="end").
                  Orchestrator stamps a visible `###USER_END###` marker
                  on the user message and ends the LS; any text emitted
                  alongside the tool call is discarded.
  sim_failed    → user_sim LLM call raised after retries. Terminate
                  with `termination_reason="sim_error"`.

Learning session — send_to_user intercept:
  When the agent calls `send_to_user(output, reason)`, orchestrator:
    1. Records a `send_event` InteractionEvent (turn_idx, output, reason).
    2. Runs Router over `(reason, output)` and records
       route_decision(route="rule" iff is_strict_rule_question).
    3. If Router=True, runs cls on `rule_question_span` and records
       cls_verdict. Builds the rule-answer hook (uniform across hit/miss/
       error) referencing `rule_question_span`.
    4. Calls user_sim_reply with the agent's plain output and the
       optional rule_hook. Post-processes the reply by substituting
       `<RULE_ANSWER>` (canonical_answer on hit; NO_RULE_DEFLECT on
       miss/error); on hit, retries user_sim once if the token is
       missing, then falls back to append. On miss/error, missing token
       is appended directly. Asymmetric end-normalize: hit + intent=end
       → continue (deliver answer); miss/error + intent=end → respect.
       - "sim_failed" → terminate with sim_error.
    5. Emits the synthetic `_SEND_TO_USER_RESPONSE` ack as the paired
       tool message. The user's actual reply follows as a normal
       role="user" message deferred to after all tool messages of the
       assistant turn.

Off-protocol detection: only triggered on **text turns**
— assistant turns where `tool_calls` is empty and `content.strip()` is
non-empty (LS only). Tool turns (assistant turns with non-empty
tool_calls) are not off-protocol regardless of content; content
alongside tool calls is internal narration under standard tool-calling
protocol. When a text turn fires off-protocol, the lazy fallback feeds
the leak text through the same Router → cls → user_sim flow above.

Test session behavior (user offline): plain text is allowed between tool
calls; session ends when the agent calls `finish_session()` or hits
max_steps / timeout.
"""
from __future__ import annotations

import json
import re
import sys
import time
import logging
from pathlib import Path
from typing import Any, Callable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import (
    LearningSession, TestSession, SessionTrajectory,
    Message, MemoryEntry, ToolCall, Rule,
    InteractionEvent,
)
from _constants import (
    FINISH_SESSION_TOOL_NAME as _FINISH_SESSION_TOOL_NAME,
    NO_RULE_DEFLECT as _NO_RULE_DEFLECT,
    RULE_ANSWER_TOKEN as _RULE_ANSWER_TOKEN,
    SEND_TO_USER_TOOL_NAME as _SEND_TO_USER_TOOL_NAME,
    USER_END_MARKER as _USER_END_MARKER,
)
from prompt_builder import build_agent_system_prompt
from router import route_agent_turn
from route_agent_text import route_agent_text
from user_sim import user_sim_reply, user_sim_open
from runner.environment.base import ATREnv, PersonaProfile
from runner.environment.domains import build_env_for_session


class MemoryContextProvider(Protocol):
    """Duck-typed interface accepted by run_session's `memory_mgr` param.

    Any object with `get_context_string(query=None)` and `get_snapshot()`
    is acceptable. Concrete types that satisfy this in the repo:
      - CrossSessionLayer concrete subclasses (ContextLayer)
      - FrozenSnapshot (test-phase immutable view)
    """
    def get_context_string(self, query: str | None = None) -> str | None: ...
    def get_snapshot(self) -> list[MemoryEntry]: ...

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from lib.llm import (
    call_llm_with_tools,
    GPT,
    GEMINI,
    token_scope,
    _is_minimax_family,
)


# ---------------------------------------------------------------------------
# Transcript / message builders
# ---------------------------------------------------------------------------

def _model_needs_reasoning_content_replay(model: str | None) -> bool:
    """Whether the agent provider expects `reasoning_content` to be echoed
    back on prior assistant turns. Two families are in scope:

    DeepSeek V4 thinking-mode rule (per
    https://api-docs.deepseek.com/guides/thinking_mode) — HARD:
      - assistant turns WITH tool_calls: `reasoning_content` MUST be echoed
        in every subsequent request, or the API returns 400.
      - assistant turns WITHOUT tool_calls: not required.

    MiniMax M2 series interleaved-thinking rule (per
    https://platform.minimax.io/docs/guides/text-m2-function-call) — SOFT:
      - the official guidance is to preserve thinking content across EVERY
        assistant turn ("the entire response_message ... must be preserved").
        Not enforced as an HTTP error, but skipping it makes the model
        re-think from scratch each turn — measured empirically to roughly
        double per-turn reasoning_tokens.
      - We therefore echo on every assistant turn, not just tool-call turns
        (see the per-turn gate in `_build_agent_messages`).

    GPT-5.x Chat Completions does not surface any reasoning-replay
    mechanism on the protocol (officially documented limitation; use
    Responses API for that). Gemini 3 native API expects thoughtSignature
    on tool-call turns, but a passthrough proxy may strip it on both
    directions, so the client side cannot always participate. Hence those two families remain
    out-of-scope for replay even though they are reasoning models.
    """
    if not model:
        return False
    if model.startswith("deepseek-v4"):
        return True
    if _is_minimax_family(model):
        return True
    return False


def _build_agent_messages(
    trajectory: SessionTrajectory,
    system_prompt: str,
    model: str | None = None,
) -> list[dict]:
    """Convert SessionTrajectory messages into OpenAI chat format for the agent.

    Preserves tool_calls + tool pairing (tool_call_id matches), assistant entries are augmented with
    `native_assistant_payload` + `native_payload_format` (when present
    on the Message). Provider-native adapters in `lib.llm_*` consume
    these to byte-faithfully replay vendor-specific multi-turn state
    (Gemini thoughtSignature, OpenAI Responses output[] items, MiniMax
    reasoning_details, Qwen reasoning_content, DeepSeek thinking
    reasoning_content). Adapters that don't recognize the format ignore
    these fields and fall back to ATR-shape reconstruction.

    `model` is consulted only to decide whether to also include the
    `reasoning_content` mirror on assistant turns for the unified
    OpenAI-compat path; the provider-native adapters read their own
    state from `native_assistant_payload`.
    """
    replay_reasoning = _model_needs_reasoning_content_replay(model)
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for m in trajectory.messages:
        if m.role == "system":
            continue
        if m.role == "user":
            msgs.append({"role": "user", "content": m.content or ""})
        elif m.role == "assistant":
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            has_tool_calls = bool(m.tool_calls)
            meaningful_text = m.content and m.content.strip()
            if has_tool_calls:
                assistant_msg["content"] = m.content if meaningful_text else None
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            else:
                assistant_msg["content"] = m.content or ""
            if replay_reasoning:
                is_minimax = _is_minimax_family(model or "")
                if has_tool_calls or is_minimax:
                    assistant_msg["reasoning_content"] = m.reasoning_content or ""
            if m.native_assistant_payload is not None:
                assistant_msg["native_assistant_payload"] = m.native_assistant_payload
            if m.native_payload_format is not None:
                assistant_msg["native_payload_format"] = m.native_payload_format
            msgs.append(assistant_msg)
        elif m.role == "tool":
            msgs.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id or "",
                "content": m.content or "",
            })
    return msgs


# Cap on how many consecutive task-phase plain-text turns the agent may emit
# in a learning session before we force-terminate. This guards against
# degenerate loops where the agent narrates without acting in the task phase.
# Closing phase has its own one-shot guard (per Def 1) and ignores this budget.
_NON_ASK_TEXT_BUDGET = 5

# Cap on consecutive LS turns where the agent issues tool calls but every tool
# response is unproductive — i.e. the env returned tool_error=True or a
# canonical empty-result body (`{"count": 0, ...}`). Without this guard, an
# agent that keeps searching for items the persona's DB doesn't carry has no
# clean termination path (LS termination is owned by user_sim, but user_sim
# only fires when the agent emits plain text — it never sees a tool-only
# turn). Hitting this budget terminates the session as task_no_progress.
# Productive events (any non-empty result, any successful write, OR any
# acquire_user_rule call) reset the counter — only sustained failure to
# obtain new state triggers it.
_TOOL_LOOP_BUDGET = 5

# Synthetic agent-side control tool. Replaces the old `###STOP###` text
# sentinel as the canonical way for the agent to end a session. Prepended
# to the env's tool schemas in run_session — orchestrator intercepts calls
# to this name and never routes them through ATREnv (it is a control-plane
# primitive, not a domain action).
_FINISH_SESSION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": _FINISH_SESSION_TOOL_NAME,
        "description": (
            "End the current session. Use this after the task is complete or "
            "when no further progress is possible. Takes no arguments."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
# Synthetic tool response body sent back when finish_session is intercepted.
# Some LLM providers reject tool_calls that aren't paired with a tool message,
# so we emit a placeholder rather than skipping the response slot.
_FINISH_SESSION_RESPONSE = '{"status": "session_ended"}'

# Synthetic agent-side communication tool. The agent's sole
# user-facing channel during LS. The schema deliberately does NOT enumerate
# intent categories on `reason` — any structure observed there is meant to
# be an emergent capability of the LLM, not a prompt-encoded artifact.
#
# The tool was originally named `speak`; it was renamed to `send_to_user`
# because "speak" implies short conversational utterances, which framed
# weaker models (e.g. deepseek-v4-flash) into bypassing the tool whenever
# the user-facing content was a markdown status report or a result table.
# "send_to_user" is metaphor-neutral and covers any payload form.
_SEND_TO_USER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": _SEND_TO_USER_TOOL_NAME,
        "description": (
            "Sends a message to the user.\n"
            "This is the only way to address the user — any text you "
            "want them to read must go through this tool. Plain "
            "assistant text outside this tool is internal reasoning "
            "only and the user will not see it. "
            "Do not combine this tool with other tool calls in the same turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "The message the user will read.",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "A brief note describing what you are doing on "
                        "this turn, in your own words."
                    ),
                },
            },
            "required": ["output", "reason"],
        },
    },
}
# Synthetic tool ack. Real user reply (canonical_answer / user_sim text)
# follows as a separate role="user" message — see the send_to_user intercept
# block in run_session for the full two-segment sequence.
_SEND_TO_USER_RESPONSE = '{"status": "delivered"}'

# `_USER_END_MARKER` is the stable visible marker stamped onto the user
# reply when user_sim calls `mark_task_complete()`. The marker
# has no runtime control semantics — orchestrator already owns the phase
# via the intent returned by user_sim_reply — but it gives the agent a
# deterministic signal in conversation history that the closing window
# has opened. Trace + agent-visible chat keep the token shape for
# readability. Any text user_sim leaks alongside the tool call is
# discarded in favour of this marker.


# Scaffolding-hook constants. The hook is a runner-side
# intervention triggered only on text turns (no tool_calls + non-empty
# content). When triggered, the orchestrator enters an inner retry loop;
# each retry call appends this scaffolding nudge to the LLM input
# *transiently* (the text never enters trajectory.messages). On a
# successful rescue, the off-protocol assistant message is popped and
# the rescued response takes its slot.
#
# The wording is fixed in; changing it requires another revision.
_SCAFFOLDING_NOTE_TEXT = (
    "<scaffolding_note>\n"
    "Your last assistant turn was not delivered to the user because it "
    "didn't go through the send_to_user tool. Please re-emit the "
    "user-facing content via send_to_user(output, reason).\n"
    "</scaffolding_note>"
)
# Per-problem-turn retry cap. A "problem turn" = one off-protocol text
# turn. Up to this many internal retries are attempted before falling
# back to lazy routing. Each new off-protocol turn starts a
# fresh budget.
_HOOK_RETRY_CAP = 3


def _build_rule_answer_hook(rule_question_span: str | None) -> str:
    """Build the transient rule-answer directive appended to user_sim's input
    on Router=True turns.

    Form (answer-driven, span-anchored, cls-blind, self-contained):
        <rule_answer>
        Even though this question is not covered by your Requirements
        and would normally get a no-preference deflect, for this specific
        question your answer is <RULE_ANSWER>. Use this exact token once
        in your reply where the answer would naturally fit; don't write
        a substantive answer yourself. Reply to anything else in the
        assistant's message as you normally would.
        </rule_answer>

    The hook is uniform regardless of cls hit / miss / error — user_sim
    only emits `<RULE_ANSWER>`, and the orchestrator post-processes the
    token. The override clause ("Even though ... would normally get a
    no-preference deflect") makes the hook self-contained so that
    user_sim.md does not need to teach the model how to recognize this
    tag — the natural-language content itself overrides the default
    deflect rule for the named question. This mirrors's
    `<scaffolding_note>` hook for agents, which is also self-contained.

    When `rule_question_span` is missing (Router edge case), the hook
    degrades to a span-less directive that still elicits the token.
    """
    span = (rule_question_span or "").strip()
    # Match "Hook content" wording verbatim when span is
    # available. When span is absent (Router edge case — empty span on a
    # Router=True turn), degrade to the same sentence skeleton with
    # `this question` as the referent so the wording does not drift from
    # the template.
    if span:
        question_clause = f'for the question "{span}" in the assistant\'s last message'
    else:
        question_clause = "for this question in the assistant's last message"
    return (
        "<rule_answer>\n"
        f"Even though this question is not covered by your Requirements "
        f"and would normally get a no-preference deflect, {question_clause} "
        f"your answer is {_RULE_ANSWER_TOKEN}. Use this exact token once "
        "in your reply where the answer would naturally fit; don't write "
        "a substantive answer yourself. Reply to anything else in the "
        "assistant's message as you normally would.\n"
        "</rule_answer>"
    )


# Lenient RULE_ANSWER detection. user_sim (especially smaller models)
# sometimes drops the angle brackets or adds whitespace inside the
# brackets. We accept those variants here so the strict-string detection
# in the fallback path doesn't degrade an entire turn over a single
# missing `<`. Restricted to `RULE_ANSWER` (all caps, with underscore)
# so natural language like "the rule answer is..." cannot match
# (placeholder is all caps; natural English isn't).
#
# IMPORTANT: this pattern does NOT consume whitespace *outside* the
# angle brackets — `<` and `>` are required to be directly adjacent to
# the token (with optional whitespace inside). Otherwise a `<?\s*…\s*>?`
# pattern would eat the surrounding space and the substitution would
# glue the replacement directly onto the preceding word (e.g.
# `"For shopping, RULE_ANSWER."` becomes `"For shopping,…"` with no
# space after the comma).
#
# Matches: `<RULE_ANSWER>`, `RULE_ANSWER`, `< RULE_ANSWER >`,
#          `<RULE_ANSWER`, `RULE_ANSWER>`.
# Does NOT match: lowercase `rule_answer`, space-separated
# `RULE ANSWER`, concatenated `RULEANSWER`.
_RULE_ANSWER_PATTERN = re.compile(r"<\s*RULE_ANSWER\s*>?|RULE_ANSWER\s*>?")


def _has_rule_answer_token(reply: str) -> bool:
    """True if `reply` contains a RULE_ANSWER token in any tolerated form."""
    return _RULE_ANSWER_PATTERN.search(reply) is not None


def _strip_rule_answer_residue(reply: str) -> str:
    """Remove every RULE_ANSWER residue from `reply` (any tolerated form).

    Used before fallback append (P15): if the runner is about to give up
    on substitution and append `canonical_answer` / `NO_RULE_DEFLECT`,
    we first clean any leftover token chunks from the user_sim reply so
    the agent doesn't see `"...RULE_ANSWER. ...\\n\\nno strong preference..."`
    — only the natural surrounding text plus the cleanly-appended
    replacement.
    """
    cleaned = _RULE_ANSWER_PATTERN.sub("", reply)
    # Collapse the whitespace runs the removal may have left behind so
    # we don't ship an oddly double-spaced sentence to the agent.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _substitute_rule_answer_token(reply: str, replacement: str) -> str:
    """Replace one RULE_ANSWER token, adjusting capitalization and
    punctuation to fit the surrounding context.

    NO_RULE_DEFLECT and (some) canonical_answers are written as
    half-sentence fragments — lowercase leading letter, no trailing
    period — designed to fit inline after a leading phrase (e.g.
    "for that, no strong preference there — your call"). user_sim,
    however, sometimes places `<RULE_ANSWER>` at sentence start, at
    reply start, or directly adjacent to a following sentence with no
    intervening punctuation. A naive textual substitution leaves visible
    rough edges: lowercase sentence starts, two sentences glued together
    without a period, or duplicated sentence-final punctuation.

    Detection uses `_RULE_ANSWER_PATTERN` (lenient) so an LLM that
    dropped `<>` still gets its token resolved.

    Heuristics (per single substitution):
      1. **Capitalization.** If the token is at the very start of the
         reply, or the previous non-whitespace character is `.`, `!`,
         or `?`, capitalize the first letter of `replacement`. Otherwise
         (token mid-sentence after `,`, `;`, `:`, `—`, etc.), keep the
         first letter lowercase — except when the replacement begins
         with the standalone word "I", which is always uppercase.
      2. **Trailing period.** If the content immediately after the
         token is an alphabetic character (i.e. user_sim glued the next
         sentence directly onto the token), append "." to the
         replacement so the next clause reads as a new sentence.
      3. **End-of-reply terminator.** If there's no content after the
         token at all, append "." if the replacement doesn't already
         end in `.!?`.
      4. **Sentence-final dedup.** If the replacement ends with `.!?`
         and the very next character after the token is `.!?`, drop
         that following character to avoid `"... — your call.."`.
    """
    m = _RULE_ANSWER_PATTERN.search(reply)
    if m is None:
        return reply
    before = reply[:m.start()]
    after = reply[m.end():]
    # Strip ANY additional RULE_ANSWER residues from the right side
    # before computing capitalization/punctuation heuristics. user_sim
    # is supposed to emit one token, but rare doubled-token outputs
    # would otherwise (a) leave a visible `RULE_ANSWER` literal in the
    # final reply and (b) confuse the "glued next sentence" heuristic
    # below. One `re.sub` clears all leftovers in O(n) regardless of
    # how many residues there are.
    after = _RULE_ANSWER_PATTERN.sub("", after)
    after = re.sub(r"[ \t]{2,}", " ", after)
    answer = replacement.strip()
    if not answer:
        return reply

    # 1. Capitalization based on left context
    before_rstripped = before.rstrip()
    sentence_start = (not before_rstripped) or before_rstripped[-1] in ".!?"
    if sentence_start and answer[0].isalpha() and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    elif not sentence_start and answer[0].isalpha() and answer[0].isupper():
        # Preserve standalone "I" / "I'..." (English first-person pronoun)
        is_first_person_I = (
            answer == "I"
            or answer.startswith("I ")
            or answer.startswith("I'")
        )
        if not is_first_person_I:
            answer = answer[0].lower() + answer[1:]

    # 2/3. Trailing punctuation based on right context
    after_lstripped_idx = len(after) - len(after.lstrip())
    after_first_nonspace = after.lstrip()[:1]
    answer_ends_sent_punct = answer.endswith((".", "!", "?"))

    if after_first_nonspace in (".", "!", "?"):
        # 4. Sentence-final dedup
        if answer_ends_sent_punct:
            # Drop the duplicate punctuation char from `after`
            after = (
                after[:after_lstripped_idx]
                + after[after_lstripped_idx + 1:]
            )
    elif after_first_nonspace.isalpha():
        # 2. Glued next sentence — ensure a period + space between.
        if not answer.endswith((".", "!", "?", ",", ";", ":")):
            answer = answer + "."
        if not after.startswith((" ", "\n", "\t")):
            after = " " + after
    elif not after_first_nonspace:
        # 3. Nothing after token — terminate the sentence.
        if not answer_ends_sent_punct:
            answer = answer + "."

    return before + answer + after


# Inject `[invoked X → ok]` annotations into agent turns in user_sim's
# history. Required by user_sim.md "Task completion" — the prompt tells
# user_sim to look for "the final tool from your Guided path" showing
# `[invoked X → ok]` in history before considering ending the session.
# Without this signal, user_sim can only infer execution from agent text
# and tends to mark complete when agent merely confirms intent or asks
# a clarifying question.
_INJECT_TOOL_ANNOTATIONS = True
_SIM_ARG_TRUNC = 80
_SIM_ERR_TRUNC = 60


def _summarize_args(args: dict | None) -> str:
    """Render tool call args as compact JSON, truncated.

    Used only when _INJECT_TOOL_ANNOTATIONS is True.
    """
    if not args:
        return ""
    s = json.dumps(args, ensure_ascii=False)
    if len(s) <= _SIM_ARG_TRUNC:
        return s
    return s[: _SIM_ARG_TRUNC - 3] + "..."


def _summarize_error(content: str | None) -> str:
    """Render an error tool response into a short single-line tag.

    Used only when _INJECT_TOOL_ANNOTATIONS is True.
    """
    if not content:
        return "error"
    s = content.replace("\n", " ").strip()
    if not s:
        return "error"
    if len(s) <= _SIM_ERR_TRUNC:
        return f"error: {s}"
    return f"error: {s[: _SIM_ERR_TRUNC - 3]}..."


# Tools whose plumbing is hidden from user_sim's view. user_sim should see
# the conversation as a natural chat:
#   - send_to_user: the `output` arg is rendered as agent plain text; the paired
#     ack is dropped. The actual user reply (canonical_answer or user_sim
#     text) follows naturally as a role="user" entry.
#   - finish_session: TS-only runner control; if hallucinated in LS alongside
#     send_to_user, hide it rather than leaking control-plane vocabulary.
#   - get_user_confirmation: tool call + paired tool result are skipped
#     entirely; the auto-stub is invisible to the user.
_TOOLS_HIDDEN_FROM_SIM = frozenset({
    _SEND_TO_USER_TOOL_NAME,
    _FINISH_SESSION_TOOL_NAME,
    "get_user_confirmation",
})


def _extract_sim_history(
    trajectory: SessionTrajectory,
    exclude_from_turn_idx: int | None = None,
    exclude_tool_call_ids: set[str] | None = None,
) -> list[dict]:
    """Extract user/assistant turns from trajectory for user_sim context.

    Past `send_to_user` calls render as their bare `output` text — same
    shape user_sim sees for the current turn (plain user-role message;
    no `<output>...</output>` envelope). `reason` is
    intentionally omitted; the runner is the only consumer
    of that field via Router.

    Other tool calls render as bracketed annotations
        [invoked draft_message({"thread_id": "thr_x"}) → ok]
    when `_INJECT_TOOL_ANNOTATIONS=True`. `get_user_confirmation` is
    hidden entirely.

    `exclude_from_turn_idx`: when given, skip every message whose
    `turn_idx >= exclude_from_turn_idx`. Use this for off-protocol text turns.

    `exclude_tool_call_ids`: when given, skip only those tool calls and their
    paired tool messages. Use this for the current send_to_user call so same
    assistant-turn business tool annotations remain visible to user_sim.
    """
    history: list[dict] = []
    excluded_tool_ids = exclude_tool_call_ids or set()
    msgs = trajectory.messages
    if exclude_from_turn_idx is not None:
        msgs = [m for m in msgs if m.turn_idx < exclude_from_turn_idx]
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.role == "system":
            i += 1
            continue
        if m.role == "user":
            content = (m.content or "").strip()
            if content:
                history.append({"role": "user", "content": content})
            i += 1
            continue
        if m.role == "assistant":
            text = (m.content or "").strip()
            tool_summaries: list[str] = []
            j = i + 1
            for tc in (m.tool_calls or []):
                paired = (
                    j < len(msgs)
                    and msgs[j].role == "tool"
                    and msgs[j].tool_call_id == tc.id
                )

                if tc.id in excluded_tool_ids:
                    if paired:
                        j += 1
                    continue

                if tc.name in _TOOLS_HIDDEN_FROM_SIM:
                    if tc.name == _SEND_TO_USER_TOOL_NAME:
                        # STU's `output` arg is rendered as plain assistant
                        # text — same shape user_sim sees for the current
                        # turn (no `<output>...</output>` envelope under
                        #). Reason is consumed by Router upstream
                        # and hidden here.
                        output = str((tc.arguments or {}).get("output") or "").strip()
                        if output:
                            tool_summaries.append(output)
                    if paired:
                        j += 1
                    continue

                if _INJECT_TOOL_ANNOTATIONS:
                    arg_str = _summarize_args(tc.arguments)
                    arg_part = f"({arg_str})" if arg_str else ""
                    if paired:
                        if msgs[j].tool_error:
                            result_part = _summarize_error(msgs[j].content)
                        else:
                            result_part = "ok"
                        tool_summaries.append(
                            f"[invoked {tc.name}{arg_part} → {result_part}]"
                        )
                    else:
                        tool_summaries.append(
                            f"[invoked {tc.name}{arg_part}]"
                        )
                if paired:
                    j += 1
            parts: list[str] = []
            if tool_summaries:
                parts.append("\n".join(tool_summaries))
            if text and not m.tool_calls:
                parts.append(text)
            combined = "\n".join(parts).strip()
            if combined:
                history.append({"role": "assistant", "content": combined})
            i = j
            continue
        i += 1
    return history


TurnCallback = Callable[[dict[str, Any]], None]
"""Called after each turn in run_session. Receives a dict with keys:
  type: "agent_text" | "agent_tool" | "tool_response" | "user_reply"
  role: "assistant" | "tool" | "user"
  content: str | None
  tool_calls: list[dict] | None   (agent_tool only)
  tool_call_id: str | None        (tool_response only)
  turn_idx: int
"""


def run_session(
    task: LearningSession | TestSession,
    variant: str,
    memory_mgr: MemoryContextProvider,
    persona: PersonaProfile,
    model: str = GPT,
    max_steps: int = 20,
    timeout: float | None = None,
    seed: int | None = None,
    episode_rules: list[Rule] | None = None,
    on_turn: TurnCallback | None = None,
    agent_reasoning_effort: str | None = None,
    hook_enabled: bool = False,
) -> SessionTrajectory:
    """Run a single session and return the trajectory.

    Args:
        task: LearningSession (user online) or TestSession (user offline).
        variant: Agent variant — send-to-user architecture set: {default, atr,
            always_ask, oracle_target, oracle_full}. Drives only prompt-side
            nudges; tool surface is variant-independent (LS gets
            `send_to_user`, TS gets `finish_session`). Oracle variants skip
            LS (run_episode injects rules into memory and runs only TS).
        memory_mgr: Object supplying `get_context_string()` / `get_snapshot()`.
        persona: Episode-level persona profile.
        model: LLM model identifier.
        max_steps: Max steps before forced termination.
        timeout: Wall-clock timeout (seconds); None = unlimited.
        seed: Determinism hint forwarded to agent, Router, user_sim, and
            classifier LLMs.
        episode_rules: Episode-level rule pool (list[Rule]) for classifier
            routing in learning sessions. None → empty pool, all rule
            routings map to cls miss → lazy fallback reply.
        agent_reasoning_effort: Reasoning effort for the agent-under-test
            LLM call only (Router, user_sim, and classifier stay on fixed-model
            defaults so scaffolding stability is independent of agent ability).
        hook_enabled: scaffolding hook. When True (LS only),
            each off-protocol **text turn** (assistant turn with no
            tool_calls and non-empty content) starts an inner retry
            loop of up to `_HOOK_RETRY_CAP` LLM calls. Each retry call
            appends a transient <scaffolding_note> to the LLM input —
            the text is never persisted to trajectory.messages. On
            success (a retry response includes send_to_user), the
            original off-protocol assistant message is popped from
            trajectory.messages and the rescued response takes its
            slot. On retry-budget exhaustion, falls back to
            lazy routing (Router -> cls -> user_sim handles the original
            off-protocol text). Each new off-protocol turn starts a fresh 3-retry
            budget; there is no per-session cap. Tool turns (assistant
            turns with non-empty tool_calls) are never subject to the
            hook regardless of content — content there is internal
            narration per standard tool-calling protocol.
    """
    t_start = time.time()
    trajectory = SessionTrajectory(
        session_id=task.session_id,
        session_type=task.session_type,
        agent_variant=variant,
    )

    env = build_env_for_session(
        task, persona, seed=seed,
    )
    # Synthetic tool prepending — by session type:
    #   TS: finish_session (agent's only legal terminator; user offline so
    #       no one else can end the session)
    #   LS: send_to_user (agent's sole user-facing channel).
    #       oracle skips LS entirely (run_episode injects rules into memory
    #       and runs only TS).
    # Orchestrator intercepts all synthetic tools; ATREnv never sees them.
    env_schemas = env.get_openai_schemas()
    if task.session_type == "test":
        tool_schemas = [_FINISH_SESSION_TOOL_SCHEMA] + env_schemas
    else:
        tool_schemas = [_SEND_TO_USER_TOOL_SCHEMA] + env_schemas

    # Memory retrieval query: TestSession uses its self-contained instruction
    # (user offline — what's-in-the-prompt is the full intent). LearningSession
    # uses reason_for_call as the query (the user's overall intent — a closer
    # match to what cross-session memory should retrieve against than the
    # opening utterance alone).
    if task.session_type == "test":
        memory_query = task.instruction
    else:
        memory_query = task.reason_for_call
    agent_sys = build_agent_system_prompt(
        task, variant, memory_mgr.get_context_string(query=memory_query),
    )
    trajectory.messages.append(Message(role="system", content=agent_sys, turn_idx=0))

    # Kick-off:
    # - TestSession: user offline → feed self-contained instruction directly.
    # - LearningSession: user online → user_sim synthesizes the opening from
    #   reason_for_call. Agent never sees a pre-written instruction; this
    #   structurally prevents instruction-side leak of task_params.
    if task.session_type == "test":
        initial_content = (task.instruction or "").strip() or "(no instruction)"
    else:
        with token_scope(trajectory.token_usage, "user_sim"):
            initial_content = user_sim_open(
                reason_for_call=task.reason_for_call,
                task_params=task.task_params,  # type: ignore[union-attr]
                narrative=persona.narrative,
                references=task.local_env.references,  # type: ignore[union-attr]
                gold_trajectory=task.gold_trajectory,  # type: ignore[union-attr]
                model=GPT,  # pinned, see classifier comment elsewhere
                seed=seed,
                session_id=task.session_id,
            )

    trajectory.messages.append(Message(role="user", content=initial_content, turn_idx=1))
    if on_turn:
        on_turn({
            "type": "user_reply", "role": "user",
            "content": initial_content, "turn_idx": 1,
        })
    turn_idx = 2
    step = 0
    non_ask_text_count = 0

    # Episode rule pool — used by cls when Router approves a strict
    # standing-rule question. TestSession has no rule oracle; the list stays
    # here for type consistency.
    _episode_rules: list[Rule] = episode_rules or []
    _rules_by_id: dict[str, Rule] = {r.rule_id: r for r in _episode_rules}

    def _run_user_sim_for_visible_turn(
        *,
        agent_text: str,
        output: str,
        reason: str,
        assistant_turn_idx: int,
        conversation_history: list[dict],
    ) -> tuple[str, bool]:
        """Run Router -> cls -> user_sim for one user-visible LS turn.

        On Router=True turns, builds a uniform rule-answer hook and passes
        it to user_sim. user_sim is cls-blind. The orchestrator then
        post-processes the reply: substitutes `<RULE_ANSWER>` with
        canonical_answer on cls hit (retry once + append fallback) or
        NO_RULE_DEFLECT on cls miss/error (append fallback only).

        Asymmetric end-intent normalize: cls hit + sim intent=end →
        continue (deliver answer before USER_END); cls miss/error /
        Router=False + sim intent=end → respect end.

        Returns `(user_reply, should_terminate_session)`. On sim failure the
        termination reason is set to `sim_error` and the reply is empty.
        """
        with token_scope(trajectory.token_usage, "router"):
            router_result = route_agent_turn(
                reason=reason,
                output=output,
                model=GEMINI,
                seed=seed,
                session_id=task.session_id,
            )

        # under the unified sweep-abort regime,
        # `router_error` is no longer written at runtime — Router framework
        # faults `SweepAbort` upstream, so a returned result is always
        # well-formed. The InteractionEvent schema retains the field as
        # Optional; it may be absent.
        is_strict_rule_question = bool(
            router_result.get("is_strict_rule_question")
        )
        rule_question_span = router_result.get("rule_question_span")
        route = "rule" if is_strict_rule_question else "task"
        trajectory.interaction_events.append(InteractionEvent(
            turn_idx=assistant_turn_idx,
            kind="route_decision",
            route=route,
            is_strict_rule_question=is_strict_rule_question,
            rule_question_span=rule_question_span,
        ))
        if on_turn:
            on_turn({
                "type": "route_decision",
                "turn_idx": assistant_turn_idx,
                "route": route,
                "is_strict_rule_question": is_strict_rule_question,
                "rule_question_span": rule_question_span,
            })

        cls_status = "not_run"
        hit_rule: Rule | None = None
        canonical_answer: str | None = None
        if is_strict_rule_question:
            # cls framework faults `SweepAbort` upstream
            # in `route_agent_text`; the only remaining cls outcomes here
            # are hit (rule matched) and miss (rule pool exhausted OR
            # empty span — both genuine `no-rule-in-context` signals).
            cls_query = str(rule_question_span or "").strip()
            if not cls_query:
                cls_routing = {"rule_id": None}
            else:
                with token_scope(trajectory.token_usage, "classifier"):
                    cls_routing = route_agent_text(
                        query=cls_query,
                        rules=_episode_rules,
                        model=GPT,
                        seed=seed,
                        session_id=task.session_id,
                    )
            hit_rule = (
                _rules_by_id.get(cls_routing["rule_id"])
                if cls_routing.get("rule_id") else None
            )
            if hit_rule:
                cls_status = "hit"
                canonical_answer = hit_rule.canonical_answer
            else:
                cls_status = "miss"
            trajectory.interaction_events.append(InteractionEvent(
                turn_idx=assistant_turn_idx,
                kind="cls_verdict",
                rule_id=hit_rule.rule_id if hit_rule else None,
            ))
            if on_turn:
                on_turn({
                    "type": "cls_verdict",
                    "turn_idx": assistant_turn_idx,
                    "verdict": "hit" if hit_rule else "miss",
                    "rule_id": hit_rule.rule_id if hit_rule else None,
                    "canonical_answer": canonical_answer,
                })

        # Build the rule-answer hook for Router=True turns. Uniform across
        # cls hit/miss/error — user_sim is cls-blind.
        rule_hook = (
            _build_rule_answer_hook(rule_question_span)
            if is_strict_rule_question else None
        )

        def _call_user_sim() -> tuple[str, str]:
            with token_scope(trajectory.token_usage, "user_sim"):
                return user_sim_reply(
                    agent_text=agent_text,
                    task_params=task.task_params,  # type: ignore[union-attr]
                    narrative=persona.narrative,
                    conversation_history=conversation_history,
                    references=task.local_env.references,  # type: ignore[union-attr]
                    reason_for_call=getattr(task, "reason_for_call", None),
                    gold_trajectory=getattr(task, "gold_trajectory", None),
                    rule_hook=rule_hook,
                    model=GPT,
                    seed=seed,
                    session_id=task.session_id,
                )

        sim_reply, sim_intent = _call_user_sim()
        # user_sim LLM retry exhaustion now raises SweepAbort
        # (user_sim.py) before reaching here, so sim_intent == "sim_failed"
        # is unreachable from the canonical path. Defensive assertion kept
        # so any future regression is caught loudly rather than silently
        # writing termination_reason="sim_error".
        if sim_intent == "sim_failed":  # pragma: no cover
            raise RuntimeError(
                f"[{task.session_id}] sim_failed reached orchestrator; "
                f"user_sim should raise SweepAbort on retry exhaustion. "
                f"This branch indicates a regression in user_sim error "
                f"handling."
            )

        user_reply = sim_reply or ""

        # ── Token post-processing ──
        # On Router=True, decide what `<RULE_ANSWER>` substitutes to and
        # how to handle a missing token:
        #   cls hit       → canonical_answer
        #   cls miss/err  → NO_RULE_DEFLECT
        # If user_sim's reply lacks the token, retry user_sim once
        # (uniformly across hit/miss/error). Each missing-token event is
        # recorded as `rule_hook_token_missing` with the cls_status and
        # the attempt index, so adherence-rate diagnostics can be derived
        # offline. After retry, if the token is still missing the
        # replacement is appended at the end as a guaranteed-delivery
        # fallback.
        if is_strict_rule_question:
            replacement = (
                canonical_answer
                if (cls_status == "hit" and canonical_answer)
                else _NO_RULE_DEFLECT
            )

            def _record_token_missing(attempt_idx: int) -> None:
                trajectory.interaction_events.append(InteractionEvent(
                    turn_idx=assistant_turn_idx,
                    kind="rule_hook_token_missing",
                    cls_status=cls_status if cls_status in ("hit", "miss", "error") else None,  # type: ignore[arg-type]
                    attempt_idx=attempt_idx,
                ))
                if on_turn:
                    on_turn({
                        "type": "rule_hook_token_missing",
                        "turn_idx": assistant_turn_idx,
                        "cls_status": cls_status,
                        "attempt_idx": attempt_idx,
                    })

            if not _has_rule_answer_token(user_reply):
                # up to 2 retries before fallback append.
                _record_token_missing(attempt_idx=0)
                for retry_attempt in range(1, 3):  # attempts 1 and 2
                    retry_reply, retry_intent = _call_user_sim()
                    if retry_intent == "sim_failed":
                        logger.warning(
                            "[%s] user_sim token-retry %d sim_failed; "
                            "using fallback append.",
                            task.session_id, retry_attempt,
                        )
                        break
                    user_reply = retry_reply or user_reply
                    sim_intent = retry_intent
                    if _has_rule_answer_token(user_reply):
                        break
                    _record_token_missing(attempt_idx=retry_attempt)

            if _has_rule_answer_token(user_reply):
                user_reply = _substitute_rule_answer_token(
                    user_reply, replacement
                )
            else:
                # Fallback append (P15). Strip any leftover RULE_ANSWER
                # residue before appending so the agent never sees a
                # mixed reply like `"... RULE_ANSWER ... \n\n
                # no strong preference there — your call"`.
                user_reply = _strip_rule_answer_residue(user_reply)
                suffix = replacement.strip()
                if suffix:
                    user_reply = (
                        f"{user_reply.rstrip()}\n\n{suffix}".strip()
                        if user_reply.strip() else suffix
                    )

        # ── Asymmetric end-intent normalize ──
        # cls hit + intent=end → continue (give agent a chance to ack the
        #                                  delivered rule answer)
        # cls miss/err + intent=end → respect end (deflect carries no
        #                                          durable info)
        # Router=False + intent=end → respect end (no rule context)
        if sim_intent == "end" and cls_status == "hit":
            logger.info(
                "[%s] normalizing end→continue for cls hit "
                "(rule answer just delivered)",
                task.session_id,
            )
            sim_intent = "continue"

        if sim_intent == "end":
            user_reply = _USER_END_MARKER
            trajectory.termination_reason = "task_complete"
            return user_reply, True

        return user_reply, False

    # LS tool-loop counter — see _TOOL_LOOP_BUDGET docstring. Increments on a
    # turn whose tool calls are ALL unproductive; resets on any productive
    # tool result or any plain-text turn (since plain text either advances
    # via user_sim or trips the existing _NON_ASK_TEXT_BUDGET).
    non_productive_tool_turns = 0

    # Scaffolding-hook per-session state.
    #   hook_pending_rescue   — set after _retry_with_hook returns a
    #                            rescued response; consumed by the first
    #                            send_event in the rescued response,
    #                            marking was_hook_rescued=True.
    #   pending_rescued_response — the rescued LLM response dict to be
    #                              processed at the top of the next outer
    #                              loop iteration (instead of invoking the
    #                              agent fresh).
    hook_pending_rescue = False
    pending_rescued_response: dict[str, Any] | None = None

    def _retry_with_hook(
        triggering_turn: int,
        triggering_content: str | None = None,
    ) -> dict[str, Any] | None:
        """Inner retry loop — try up to `_HOOK_RETRY_CAP` times to coax
        the agent into using send_to_user.

        Each attempt builds a TRANSIENT message list (trajectory.messages
        + a synthetic user-role scaffolding note) and re-invokes the
        agent. The scaffolding text is never persisted to
        trajectory.messages — it only exists in the LLM input for that
        single retry call. Likewise, retry responses that are still
        off-protocol are discarded (not appended to trajectory).

        Returns:
          dict — the rescued response (one whose tool_calls include
                 send_to_user). The caller is expected to set
                 `pending_rescued_response` so the outer loop processes
                 it as the agent's next turn, with `hook_pending_rescue`
                 set so the resulting send_event is flagged
                 was_hook_rescued.
          None — hook_enabled is False, all retries returned without
                 send_to_user, or an LLM error occurred during retry.
                 The caller falls back to lazy routing.

        Args:
          triggering_turn: the assistant turn_idx that prompted the hook
            (matches the corresponding off_protocol_ask event when
            rescue fails); used as turn_idx on every emitted
            hook_appended InteractionEvent.
          triggering_content: the leak text that prompted the hook —
            stored on each hook_appended event's `question` field, so downstream analysis can recover what the
            agent leaked even after the original message is popped on
            successful rescue.
        """
        if not hook_enabled:
            return None
        for retry_idx in range(_HOOK_RETRY_CAP):
            trajectory.interaction_events.append(InteractionEvent(
                turn_idx=triggering_turn,
                kind="hook_appended",
                question=triggering_content,
            ))
            if on_turn:
                on_turn({
                    "type": "hook_appended",
                    "turn_idx": triggering_turn,
                    "retry_idx": retry_idx,
                })
            transient_msgs = _build_agent_messages(
                trajectory, agent_sys, model=model,
            )
            transient_msgs.append({
                "role": "user",
                "content": _SCAFFOLDING_NOTE_TEXT,
            })
            try:
                with token_scope(trajectory.token_usage, "agent"):
                    retry_resp = call_llm_with_tools(
                        messages=transient_msgs,
                        tools=tool_schemas,
                        model=model,
                        temperature=0.0,
                        seed=seed,
                        reasoning_effort=agent_reasoning_effort,
                    )
            except Exception as e:
                # hook-retry LLM retry exhaustion is also
                # an agent-channel error — raise SweepAbort instead of
                # silently falling back to lazy routing.
                from _abort import SweepAbort
                raise SweepAbort(
                    module="hook_retry", session_id=task.session_id, original=e
                ) from e
            retry_tcs = retry_resp.get("tool_calls") or []
            if any(
                tc.get("name") == _SEND_TO_USER_TOOL_NAME for tc in retry_tcs
            ):
                return retry_resp
            # Otherwise: response is still off-protocol or unhelpful.
            # Discard and continue retrying (within budget).
        return None

    def _consume_pending_rescue() -> bool:
        """Pop the pending-rescue flag for the next send_event.

        Returns True iff the next send_to_user should be marked
        was_hook_rescued. Same-turn duplicate send_to_user calls are dropped
        before persistence, so one rescued turn contributes at most one
        rescued send_event.
        """
        nonlocal hook_pending_rescue
        if hook_pending_rescue:
            hook_pending_rescue = False
            return True
        return False

    def _check_timeout() -> bool:
        return timeout is not None and (time.time() - t_start) > timeout

    while step < max_steps:
        if _check_timeout():
            trajectory.termination_reason = "timeout"
            break
        step += 1

        # ── Agent turn ──
        # If a hook retry rescued us on the previous iteration, process
        # the rescued response in this iteration instead of invoking the
        # agent fresh. Token usage for the rescue was already counted
        # inside `_retry_with_hook` via the inner token_scope.
        if pending_rescued_response is not None:
            resp = pending_rescued_response
            pending_rescued_response = None
        else:
            agent_messages = _build_agent_messages(
                trajectory, agent_sys, model=model,
            )
            # Empty-response handling: a successful call that returns no
            # tool_calls AND no content is provider weirdness (commonly
            # observed on gemini-flash after long stalls). Retry up to
            # `_EMPTY_RESPONSE_RETRIES` more times with identical params
            # before giving up; the provider often recovers on a fresh
            # call. Genuine LLM-call exceptions still propagate as
            # SweepAbort immediately (no retry loop around exceptions —
            # that lives in `lib.llm._call_with_retry`).
            _EMPTY_RESPONSE_RETRIES = 2
            resp = None
            for attempt_idx in range(_EMPTY_RESPONSE_RETRIES + 1):
                try:
                    with token_scope(trajectory.token_usage, "agent"):
                        resp = call_llm_with_tools(
                            messages=agent_messages,
                            tools=tool_schemas,
                            model=model,
                            temperature=0.0,
                            seed=seed,
                            reasoning_effort=agent_reasoning_effort,
                        )
                except Exception as e:
                    # agent LLM retry exhaustion (inside
                    # _call_with_retry) → SweepAbort. No termination_reason
                    # written, no partial trajectory.
                    from _abort import SweepAbort
                    raise SweepAbort(
                        module="agent", session_id=task.session_id, original=e
                    ) from e
                if (resp.get("tool_calls")
                        or (resp.get("content") and resp["content"].strip())):
                    if attempt_idx > 0:
                        logger.info(
                            "[agent_empty_response] session %s recovered on attempt %d",
                            task.session_id, attempt_idx + 1,
                        )
                    break
                logger.warning(
                    "[agent_empty_response] session %s attempt %d returned empty; retrying",
                    task.session_id, attempt_idx + 1,
                )
            else:
                # All attempts exhausted. This is an abnormal session end, but
                # the trajectory up to this point is still semantically useful:
                # TS scoring is based on prior tool calls, and dropping the
                # partial trajectory would turn a completed required action into
                # a missing-session failure. Persist the trace as an unclean
                # termination instead of aborting the whole cell.
                trajectory.termination_reason = "agent_empty_response"
                trajectory.termination_subreason = (
                    f"empty_after_{_EMPTY_RESPONSE_RETRIES + 1}_attempts"
                )
                logger.error(
                    "[agent_empty_response] session %s terminating after %d "
                    "consecutive empty responses; preserving partial trajectory",
                    task.session_id, _EMPTY_RESPONSE_RETRIES + 1,
                )
                break

        # Token usage is captured by token_scope("agent") above — no need
        # to also write trajectory.agent_cost (removed as a duplicate field).
        content = resp.get("content")
        raw_tcs = resp.get("tool_calls")
        reasoning_content = resp.get("reasoning_content")
        native_payload = resp.get("native_assistant_payload")
        native_payload_format = resp.get("native_payload_format")
        logger.info(
            "[debug] agent resp: content_len=%d, n_tools=%d, rc_len=%d",
            len(content or ""), len(raw_tcs or []),
            len(reasoning_content or ""),
        )

        if not raw_tcs and not (content and content.strip()):
            # Reached only via the rescued-response branch above (the retry
            # loop guarantees fresh-call responses are non-empty). Preserve the
            # trajectory for scoring/inspection instead of aborting the cell.
            trajectory.termination_reason = "agent_empty_response"
            trajectory.termination_subreason = "rescued_response_empty"
            logger.error(
                "[agent_empty_response] session %s rescued response was empty; "
                "preserving partial trajectory",
                task.session_id,
            )
            break

        # ── Tool turn: agent issued tool_calls ──
        # tool turns are never off-protocol regardless
        # of `content`. Content alongside tool_calls is internal
        # narration under standard tool-calling protocol — control
        # naturally returns to the agent after env tools execute.
        if raw_tcs:
            tool_calls: list[ToolCall] = []
            seen_send_to_user = False
            for tc in raw_tcs:
                name = tc["name"]
                if (
                    task.session_type == "learning"
                    and name == _SEND_TO_USER_TOOL_NAME
                ):
                    if seen_send_to_user:
                        # Multi-STU is not part of the current protocol. Keep
                        # the first user-facing send only; later same-turn STU
                        # calls disappear from trajectory.messages and from
                        # future agent replay. If this ever appears in real
                        # runs, revisit the protocol deliberately.
                        logger.warning(
                            "[%s] dropping extra same-turn send_to_user call "
                            "(turn_idx=%d, tool_call_id=%s)",
                            task.session_id,
                            turn_idx,
                            tc.get("id"),
                        )
                        continue
                    seen_send_to_user = True
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=name,
                        arguments=tc["arguments"],
                    )
                )
            assistant_turn_idx = turn_idx  # captured for InteractionEvents below
            trajectory.messages.append(Message(
                role="assistant", content=content,
                tool_calls=tool_calls, turn_idx=turn_idx,
                reasoning_content=reasoning_content,
                native_assistant_payload=native_payload,
                native_payload_format=native_payload_format,
            ))
            if on_turn:
                on_turn({
                    "type": "agent_tool", "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in tool_calls
                    ],
                    "turn_idx": turn_idx,
                })
            turn_idx += 1

            # Off-policy structural diagnostics (LS only). The protocol
            # still executes tools in declared order and only the first
            # send_to_user reaches trajectory.messages; these events make
            # the underlying capability signal visible rather than
            # silent. Counted from raw_tcs so the dedup above does not
            # erase the multi-STU evidence.
            if task.session_type == "learning":
                stu_count = sum(
                    1 for tc in raw_tcs
                    if tc["name"] == _SEND_TO_USER_TOOL_NAME
                )
                sibling_names = [
                    tc["name"] for tc in raw_tcs
                    if tc["name"] != _SEND_TO_USER_TOOL_NAME
                ]
                if stu_count >= 1 and sibling_names:
                    trajectory.interaction_events.append(InteractionEvent(
                        turn_idx=assistant_turn_idx,
                        kind="stu_mixed_with_tools",
                        sibling_tool_names=sibling_names,
                    ))
                    if on_turn:
                        on_turn({
                            "type": "stu_mixed_with_tools",
                            "turn_idx": assistant_turn_idx,
                            "sibling_tool_names": sibling_names,
                        })
                if stu_count >= 2:
                    trajectory.interaction_events.append(InteractionEvent(
                        turn_idx=assistant_turn_idx,
                        kind="stu_duplicate_same_turn",
                        duplicate_count=stu_count - 1,
                    ))
                    if on_turn:
                        on_turn({
                            "type": "stu_duplicate_same_turn",
                            "turn_idx": assistant_turn_idx,
                            "duplicate_count": stu_count - 1,
                        })

            finish_called = False
            sim_break = False  # set True if the post-loop sim call needs to terminate
            # Deferred user reply — populated by the post-tool-call user_sim
            # invocation so the synthetic user message (canonical_answer /
            # user_sim text / lazy fallback) appears AFTER all tool messages
            # of this assistant turn. OpenAI's tool_call ↔ tool message
            # pairing requires contiguous tool messages between the assistant
            # turn and the next role. Length is at most 1 because same-turn
            # duplicate send_to_user calls are dropped before persistence.
            deferred_user_replies: list[str] = []
            send_payload: tuple[str, str, str] | None = None
            productive_this_turn = False
            for tc in tool_calls:
                if (tc.name == _FINISH_SESSION_TOOL_NAME
                        and task.session_type == "test"):
                    # Control-plane terminator. Authorization is at the
                    # schema layer (finish_session is only injected in TS);
                    # an LS hallucination falls through to ATREnv which
                    # returns "Unknown tool" — same treatment as any
                    # unregistered tool, and avoids the agent unilaterally
                    # ending an LS (LS termination is owned by user_sim).
                    finish_called = True
                    productive_this_turn = True
                    trajectory.messages.append(Message(
                        role="tool", content=_FINISH_SESSION_RESPONSE,
                        tool_call_id=tc.id, turn_idx=turn_idx,
                        tool_error=False,
                    ))
                    if on_turn:
                        on_turn({
                            "type": "tool_response", "role": "tool",
                            "content": _FINISH_SESSION_RESPONSE,
                            "tool_call_id": tc.id,
                            "tool_error": False,
                            "turn_idx": turn_idx,
                        })
                    turn_idx += 1
                    continue

                if (tc.name == _SEND_TO_USER_TOOL_NAME
                        and task.session_type == "learning"):
                    # Speak intercept. Same-turn duplicate
                    # send_to_user calls were filtered before this loop, so
                    # this branch runs at most once per assistant turn.
                    productive_this_turn = True
                    output = str((tc.arguments or {}).get("output") or "").strip()
                    reason = str((tc.arguments or {}).get("reason") or "").strip()

                    # Mark this send as rescued iff a hook appended right
                    # before it.
                    rescued = _consume_pending_rescue()
                    trajectory.interaction_events.append(InteractionEvent(
                        turn_idx=assistant_turn_idx,
                        kind="send_event",
                        output=output,
                        reason=reason,
                        was_hook_rescued=rescued,
                    ))
                    if on_turn:
                        on_turn({
                            "type": "send_event",
                            "turn_idx": assistant_turn_idx,
                            "output": output,
                            "reason": reason,
                            "was_hook_rescued": rescued,
                        })

                    # Inline ack — kept minimal so the two-segment pattern
                    # is invariant across rule and task routing outcomes.
                    trajectory.messages.append(Message(
                        role="tool", content=_SEND_TO_USER_RESPONSE,
                        tool_call_id=tc.id, turn_idx=turn_idx,
                        tool_error=False,
                    ))
                    if on_turn:
                        on_turn({
                            "type": "tool_response", "role": "tool",
                            "content": _SEND_TO_USER_RESPONSE,
                            "tool_call_id": tc.id,
                            "tool_error": False,
                            "turn_idx": turn_idx,
                        })
                    turn_idx += 1

                    # Defer user_sim until after all tool messages for this
                    # assistant turn have been appended; OpenAI requires
                    # assistant tool_calls to be paired contiguously with
                    # tool responses before the next user message.
                    send_payload = (tc.id, output, reason)
                    continue

                tool_resp = env.get_response(tc)
                # Productivity check for tool-loop budget. A tool response is
                # "productive" when it advances state — it returned non-empty
                # data or a successful write/control acknowledgement. Failures
                # (tool_error=True) and the canonical empty-result body
                # (`{"count": 0, ...}` from `_empty_with_hint`) do NOT count.
                # If any sibling tool in this turn is productive, the whole
                # turn counts as productive (one productive call is enough).
                if not tool_resp.is_error:
                    body_text = tool_resp.content or ""
                    try:
                        body = json.loads(body_text)
                    except (json.JSONDecodeError, ValueError):
                        body = None
                    is_empty = (
                        isinstance(body, dict)
                        and body.get("count") == 0
                    )
                    if not is_empty:
                        productive_this_turn = True
                trajectory.messages.append(Message(
                    role="tool", content=tool_resp.content,
                    tool_call_id=tool_resp.id or tc.id, turn_idx=turn_idx,
                    tool_error=tool_resp.is_error,
                ))
                if on_turn:
                    on_turn({
                        "type": "tool_response", "role": "tool",
                        "content": tool_resp.content,
                        "tool_call_id": tool_resp.id or tc.id,
                        "tool_error": tool_resp.is_error,
                        "turn_idx": turn_idx,
                    })
                turn_idx += 1

            # ── send_to_user sim call after tool-response flush ─────────────
            # All tool_calls have now been processed (env tools executed,
            # tool messages appended). If the agent emitted any send_to_user
            # call in this turn, route that single logical user-facing message
            # through Router -> cls -> user_sim.
            if send_payload is not None:
                send_tool_call_id, output, reason = send_payload
                # Pass the trajectory snapshot without the current STU call
                # itself — agent_text is the plain text of that same send.
                # Keep sibling business tool calls from this assistant turn so
                # user_sim can track Guided path progress accurately.
                user_reply, sim_break = _run_user_sim_for_visible_turn(
                    agent_text=output,
                    output=output,
                    reason=reason,
                    assistant_turn_idx=assistant_turn_idx,
                    conversation_history=_extract_sim_history(
                        trajectory,
                        exclude_tool_call_ids={send_tool_call_id},
                    ),
                )
                if trajectory.termination_reason != "sim_error":
                    deferred_user_replies.append(user_reply)

            # Flush deferred user replies. At most one entry because duplicate
            # same-turn STU calls are dropped before persistence.
            for answer_text in deferred_user_replies:
                trajectory.messages.append(Message(
                    role="user", content=answer_text, turn_idx=turn_idx,
                ))
                if on_turn:
                    on_turn({
                        "type": "user_reply", "role": "user",
                        "content": answer_text, "turn_idx": turn_idx,
                    })
                turn_idx += 1

            # A send_to_user intercept set sim_break to terminate after flushing.
            if sim_break:
                break

            if finish_called:
                trajectory.termination_reason = "agent_stop"
                break

            # send_to_user intercept may have set termination_reason="task_complete"
            # (user_sim said done while answering send_to_user). Deferred reply
            # (USER_END marker) has been flushed; end now.
            if trajectory.termination_reason == "task_complete":
                break

            # LS tool-loop budget — guards against tool-only spirals. user_sim
            # only fires on send_to_user / plain text; without this, repeated env
            # tool calls with empty/error results would burn through max_steps.
            if task.session_type == "learning":
                if productive_this_turn:
                    non_productive_tool_turns = 0
                else:
                    non_productive_tool_turns += 1
                    if non_productive_tool_turns >= _TOOL_LOOP_BUDGET:
                        logger.info(
                            "[%s] LS tool-loop budget (%d) exceeded: "
                            "%d consecutive non-productive tool turns; "
                            "force-terminating as task_no_progress.",
                            task.session_id, _TOOL_LOOP_BUDGET,
                            non_productive_tool_turns,
                        )
                        trajectory.termination_reason = "task_no_progress"
                        trajectory.termination_subreason = "tool_loop"
                        break

            non_ask_text_count = 0
            continue

        # ── Text turn: agent emitted only plain text (no tool_calls) ──
        # this is the only shape that counts as an
        # off-protocol leak. The agent intended to address the user but
        # bypassed send_to_user; hook (if enabled) tries to coax it
        # back to STU; otherwise lazy fallback routes the text through
        # user_sim so the session advances.
        assistant_turn_idx = turn_idx
        trajectory.messages.append(Message(
            role="assistant", content=content, turn_idx=turn_idx,
            reasoning_content=reasoning_content,
            native_assistant_payload=native_payload,
            native_payload_format=native_payload_format,
        ))
        if on_turn:
            on_turn({
                "type": "agent_text", "role": "assistant",
                "content": content, "turn_idx": turn_idx,
            })
        turn_idx += 1

        text_turn_leak = bool(
            task.session_type == "learning"
            and content and content.strip()
        )
        if text_turn_leak:
            # Streaming audit: emit [off_protocol]
            # immediately on detection so the trace shows the leak right
            # under the AGENT block. trajectory.interaction_events
            # persistence is gated on rescue *failure* —
            # successful rescues leave no off_protocol_ask in events.
            if on_turn:
                on_turn({
                    "type": "off_protocol",
                    "turn_idx": assistant_turn_idx,
                    "content": content,
                })

            if hook_enabled:
                rescued = _retry_with_hook(
                    assistant_turn_idx, triggering_content=content,
                )
                if rescued is not None:
                    # Pop the off-protocol assistant message and rewind
                    # turn_idx so the rescued response reuses the slot.
                    # The popped turn had no tool_calls (text turn), so
                    # there are no tool messages following it — popping
                    # the last message is sufficient.
                    trajectory.messages.pop()
                    turn_idx -= 1
                    pending_rescued_response = rescued
                    hook_pending_rescue = True
                    non_productive_tool_turns = 0
                    continue
            # Hook disabled or retry budget exhausted: off-protocol is
            # real and persists. Lazy fallback below routes the leak
            # text through Router -> cls -> user_sim so the session advances.
            trajectory.interaction_events.append(InteractionEvent(
                turn_idx=assistant_turn_idx,
                kind="off_protocol_ask",
                question=content,
            ))

        non_productive_tool_turns = 0

        # Test session: agent may narrate freely between tool calls.
        # Termination is via finish_session() or max_steps; user offline
        # so no user_sim path.
        if task.session_type == "test":
            continue

        # Learning session: lazy fallback — text-leak content is fed through
        # the same Router -> cls -> user_sim path as send_to_user. Router
        # still receives a synthesized reason describing the leak; user_sim
        # sees only the leaked text via `output`.
        user_reply, sim_break = _run_user_sim_for_visible_turn(
            agent_text=content or "",
            output=content or "",
            reason=f"off-protocol user-facing text: {content or ''}",
            assistant_turn_idx=assistant_turn_idx,
            conversation_history=_extract_sim_history(
                trajectory, exclude_from_turn_idx=assistant_turn_idx,
            ),
        )
        if trajectory.termination_reason == "sim_error":
            break

        trajectory.messages.append(Message(
            role="user", content=user_reply, turn_idx=turn_idx,
        ))
        if on_turn:
            on_turn({
                "type": "user_reply", "role": "user",
                "content": user_reply, "turn_idx": turn_idx,
        })
        turn_idx += 1

        if sim_break:
            break

        non_ask_text_count += 1
        if non_ask_text_count >= _NON_ASK_TEXT_BUDGET:
            # Agent keeps narrating in task phase without acting or progressing.
            # `task_no_progress` distinguishes this degenerate-loop kill from a
            # real agent_stop; lifecycle diagnostics surface it as a fault, not
            # a clean completion.
            logger.info(
                "[%s] task-phase non-ask text budget (%d) exceeded; "
                "force-terminating as task_no_progress.",
                task.session_id, _NON_ASK_TEXT_BUDGET,
            )
            trajectory.termination_reason = "task_no_progress"
            #  LS-B sub-classification; LS-A uses
            # `tool_loop`, this branch uses `stu_loop` because it fires
            # when the agent emits non-tool text (interpreted as
            # STU/chitchat under  revised semantics).
            trajectory.termination_subreason = "stu_loop"
            break
    else:
        trajectory.termination_reason = "max_steps"

    trajectory.step_count = step
    trajectory.duration_seconds = round(time.time() - t_start, 2)
    trajectory.memory_snapshot = memory_mgr.get_snapshot()
    return trajectory
