"""Data schemas shared by the ATRBench runner and evaluator.

Flat single-step Rule schema: each rule has exactly one ActionStep
(tool + at most one param).

  check_type   CheckType
    tool_identity — rule pins which tool. `param` is null.
    param_id      — rule pins a *_id; `param` is the id field name
                    (the test session encodes 1 gold + N decoys via ref
                    attributes).
    param_enum    — rule pins an enum value; `param` is the enum field name.

  ActionStep   { tool, param }

    Two flavors of `tool`:

    1. mutate tool — any real business-domain mutate (not a read prefix,
       not get_user_confirmation). `param` is either a parameter name on
       that tool, or null (for tool_identity rules). Used by
       check_type ∈ {tool_identity, param_id, param_enum}.

    2. `get_user_confirmation` — the special "ask before doing X" tool.
       `param` here is NOT a parameter name on get_user_confirmation;
       instead it carries the mutate tool name being confirmed (e.g.
       "delete_files"). At test time the gold becomes
       `get_user_confirmation(target_tool=<param>, target_params=...)`,
       and the would-be mutate never fires (test sessions are user-
       offline — no ack arrives). check_type for these is "confirm".

Invariants (enforced by gen validator + QC):
  mutate tool: in a registered business domain, NOT a read prefix
  get_user_confirmation tool: param must name a real mutate tool
  check_type == tool_identity → param null
  check_type == param_id      → param ends with _id or _ids
  check_type == param_enum    → param's ontology type starts with `enum[`
  check_type == confirm       → tool == "get_user_confirmation",
                                param names the inner mutate tool
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Literal types
# ─────────────────────────────────────────────────────────────────────────────

CheckType = Literal[
    "tool_identity",  # which tool was chosen (alternatives present in whitelist)
    "param_id",       # which *_id selected from candidate pool (1 gold + N decoy)
    "param_enum",     # which enum-typed param value
    "confirm",        # rule wraps a mutate with get_user_confirmation prologue;
                      # action_step.tool == "get_user_confirmation",
                      # action_step.param == inner mutate tool name.
                      # Optional inner-param preference is encoded at
                      # test_session_gen time via gold_value (dict) → label's
                      # target_params; if no inner-param preference, gold_value
                      # is null and the label only carries target_tool.
]


# ─────────────────────────────────────────────────────────────────────────────
# Runtime types
# ─────────────────────────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    requestor: Literal["assistant", "user"] = "assistant"


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    tool_error: bool | None = None
    turn_idx: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Provider-specific "thinking" / reasoning trace returned alongside
    # content (currently only DeepSeek V4 thinking mode). Echoed back on
    # subsequent requests for that provider — required by DeepSeek's API
    # under tool-calls. Other providers ignore it.
    reasoning_content: str | None = None
    # vendor-native assistant payload, byte-faithful for
    # multi-turn replay. None for non-assistant messages and when no
    # native payload was captured. Self-describing via `native_payload_format`.
    native_assistant_payload: dict | list | None = None
    native_payload_format: Literal[
        "openai_responses_v1",
        "gemini_v1beta",
        "minimax_chat_v1",
        "qwen_chat_v1",
        "deepseek_chat_v1",
        "anthropic_messages_v1",
    ] | None = None


class MemoryEntry(BaseModel):
    key: str
    value: str
    source: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InteractionEvent(BaseModel):
    """One send/routing event emitted by the orchestrator during a session.

    Send-to-user architecture + free-text user_sim records:
      send_event         — agent invoked `send_to_user(output, reason)`.
                           Carries `output` and `reason`. `was_hook_rescued`
                           is True when this send immediately follows a
                           `hook_appended` event from the same agent.
      route_decision     — Router's decision for the send. The `route`
                           field is the binary downstream routing axis:
                           "rule" iff is_strict_rule_question is true (cls
                           fires), else "task". Trajectories
                           carry `is_strict_rule_question`,
                           `rule_question_span`, and any `router_error`.
                           These fields remain parseable but
                           are not populated by the runtime.
      cls_verdict        — classifier verdict for a rule-routed send: matched
                           `rule_id` (None on miss) and `cls_error` flag.
      off_protocol_ask   — agent emitted user-facing content (text-turn
                           leak) that hook didn't rescue. Recorded with
                           the violating text in `question`.
      hook_appended      — orchestrator appended a `<scaffolding_note>` to
                           the LLM input on an off-protocol text turn. One event per retry attempt.
      stu_mixed_with_tools  — LS-only diagnostic: agent emitted
                              `send_to_user` together with at least one
                              other tool call in the same assistant
                              turn. The protocol still runs the tools in
                              declared order; this event records the
                              co-occurring sibling tool names so the
                              capability signal is auditable rather than
                              silent. One event per offending turn.
      stu_duplicate_same_turn  — LS-only diagnostic: agent emitted
                              multiple `send_to_user` calls in one
                              assistant turn. The orchestrator keeps the
                              first and drops the rest to preserve the
                              tool_call ↔ tool_response pairing
                              invariant; this event records how many
                              were dropped. One event per offending
                              turn.
      rule_hook_token_missing  — diagnostic: on a Router=True
                              turn, user_sim's reply lacked the
                              `<RULE_ANSWER>` token despite the
                              rule-answer hook being injected. Carries
                              `cls_status` ∈ {hit, miss, error} and an
                              `attempt_idx` (0 for initial reply, 1 for
                              the retry). One event per failed reply
                              (so a hit turn that misses twice — initial
                              + retry — emits two events). Measures
                              user_sim adherence to the hook directive.
    """
    turn_idx: int
    kind: Literal[
        "send_event",
        "route_decision",
        "cls_verdict",
        "off_protocol_ask",
        "hook_appended",
        "stu_mixed_with_tools",
        "stu_duplicate_same_turn",
        "rule_hook_token_missing",
    ]
    output: str | None = None        # send_event: send_to_user.output
    reason: str | None = None        # send_event: send_to_user.reason
    route: Literal["rule", "task"] | None = None  # route_decision
    # route_decision fields.
    is_strict_rule_question: bool | None = None
    rule_question_span: str | None = None
    router_error: str | None = None
    # route_decision fields retained so trajectories
    # parse. Runner code does not populate them.
    is_cross_session_ask: bool | None = None
    # route_decision audit subfields. The derived gate is their conjunction.
    reason_has_cross_session_rule_intent: bool | None = None
    output_asks_cross_session_rule_question: bool | None = None
    # These fields remain parseable but are not populated by the runtime.
    classification: Literal[
        "task_clarification", "cross_session", "chitchat"
    ] | None = None
    sim_reasoning: str | None = None
    rule_id: str | None = None       # cls_verdict: matched rule (None on miss)
    cls_error: bool = False          # cls_verdict: scaffolding fault flag
    question: str | None = None      # off_protocol_ask / hook_appended: leak text
    # Scaffolding-hook flag. Set on send_event when the send
    # immediately follows a hook_appended event — i.e. the agent re-emitted
    # via send_to_user after the runner's <scaffolding_note> nudge.
    was_hook_rescued: bool = False
    # stu_mixed_with_tools: tool names co-emitted with send_to_user in the
    # same assistant turn (excluding the send_to_user itself).
    sibling_tool_names: list[str] | None = None
    # stu_duplicate_same_turn: number of extra send_to_user calls dropped
    # by the orchestrator's same-turn dedup (total send_to_user calls
    # minus the one kept).
    duplicate_count: int | None = None
    # rule_hook_token_missing: cls status at the time of the missing-token
    # event ("hit" / "miss" / "error"), and the user_sim attempt index
    # (0 = initial reply; 1 = retry reply still missing).
    cls_status: Literal["hit", "miss", "error"] | None = None
    attempt_idx: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Environment types
# ─────────────────────────────────────────────────────────────────────────────

class ReferenceObject(BaseModel):
    id: str
    type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class LocalEnv(BaseModel):
    tools: list[str]
    references: list[ReferenceObject] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Eval types
# ─────────────────────────────────────────────────────────────────────────────

class RequiredAction(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    compare_args: list[str] | None = None
    # list-typed args are matched by set equality: set(gold) == set(pred).


class TaskSuccessLabel(BaseModel):
    required_actions: list[RequiredAction] = Field(default_factory=list)


class GoldStep(BaseModel):
    """One step in a LearningSession's oracle trajectory.

    Documentary, not eval-enforcing: B2 produces this to demonstrate the
    LS is internally coherent (an agent that knew all task_params upfront
    could execute this sequence against the refs / tools to complete the
    task). Sibling concept of `RequiredAction` for TestSession, but
    without compare_args (LS is not eval'd).
    """
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Persona
# ─────────────────────────────────────────────────────────────────────────────

class RawPersona(BaseModel):
    """Source persona record from Nemotron-Personas-USA.

    `narrative` is the only field the user_sim consumes directly at runtime —
    user_sim must NOT see structured_persona (which may contain rule-adjacent
    facts) or rule_pool.

    Field naming:
      - `persona_id` is the human-readable slug used as runtime identity
        (e.g. "charlie_james"); also the dir name under data/personas/.
      - `nemotron_uuid` is the original Nemotron 32-hex uuid (no hyphens),
        kept for dedup on re-ingest and provenance.
    """
    persona_id: str
    nemotron_uuid: str = ""
    narrative: str
    general_domain: str = ""
    specific_domain: str = ""

    model_config = {"extra": "allow"}  # tolerate extra fields from source data


# ─────────────────────────────────────────────────────────────────────────────
# Rule (unified eval taxonomy)
# ─────────────────────────────────────────────────────────────────────────────

class ActionStep(BaseModel):
    """The rule's bound tool call.

    *Authoring artifact only — not read by runtime or evaluator.* test_session_gen
    (Stage C) consumes this to produce the actual eval gold in
    TestSession.labels.task_success.required_actions. After C finishes its job
    the step detail serves only as documentation of the rule's intent.

    `param` semantics depend on `tool`:
      - mutate tool: param is a parameter name on that tool, or null
        (for tool_identity rules)
      - get_user_confirmation: param is the mutate tool name being
        confirmed (e.g. "delete_files"); becomes target_tool of the
        gold confirm call at test time
    """
    tool: str
    param: str | None = None


class Rule(BaseModel):
    """A long-term user preference. Fields split into two tiers:

    Tier 1 — Runtime- and evaluator-visible (serialized to episode.json,
             consumed by runner/ and evaluator/):
        rule_id             — lookup key across Rule ↔ TestSession
        rule_text           — third-person description; metadata + rule_qc
                              tagging + few-shot display (NOT injected into
                              any agent prompt)
        canonical_answer    — UNIFIED user-voice statement: injected into
                              oracle variant prompt, used verbatim as the
                              user's reply when ATR agent asks about the
                              rule, and shown as the candidate-rule label
                              in route_agent_text classifier. Same string,
                              same surface form, three call sites — keeps
                              oracle and ATR on equal footing.
        check_type          — diagnostic label; metrics.by_check_type
                              breakdown. Tags the rule's primary
                              discriminator dimension (tool choice OR which
                              param). Does NOT drive pass/fail.

    Tier 2 — Authoring artifact (consumed by Stage C test_session_gen,
             then effectively dead for runtime/eval):
        action_step.tool   — gold tool name. Either a mutate tool, or
                             `get_user_confirmation` (special).
        action_step.param  — see ActionStep docstring; mutate tools use
                             it as a param name on the tool; the confirm
                             tool uses it as the mutate target name.

    Domain inference (used by Stage C):
        confirm rule (tool == get_user_confirmation):
            tool_map()[action_step.param]["domain"]
        otherwise:
            tool_map()[action_step.tool]["domain"]
    """
    rule_id: str
    rule_text: str
    canonical_answer: str
    check_type: CheckType
    action_step: ActionStep


# ─────────────────────────────────────────────────────────────────────────────
# LearningSession (no eval labels)
# ─────────────────────────────────────────────────────────────────────────────

class LearningSession(BaseModel):
    session_id: str
    session_type: Literal["learning"] = "learning"
    day_offset: int          # offset from episode day-1; skeleton stage sets this
    domain: str
    # Vague intent the user holds in mind — seen ONLY by user_sim, never
    # by the agent directly. user_sim generates the opening message at runtime
    # from reason_for_call (lazy disclosure: reveals direction, not specific
    # task_params). Modeled after τ²-bench's `reason_for_call` to structurally
    # eliminate instruction-side leak of task_params (the failure mode where
    # a pre-written instruction echoes specific values like cuisine="sushi").
    reason_for_call: str
    # User's full ground truth: {field_name: concrete_value}. user_sim reads
    # these values and paraphrases them naturally in replies (see user_sim.py
    # system prompt). Value type is whatever the field needs — str / int /
    # date / list / enum literal. No wrapper object, no per-field rationale.
    task_params: dict[str, Any]
    # Tools the session was designed around. Now **derived** from
    # `gold_trajectory` (unique tool names) at fill assembly time, kept on
    # the schema for downstream audit (signal_ls computation in stage D).
    # NOT a runtime whitelist — `local_env.tools` remains the whole-domain
    # list.
    expected_tools: list[str]
    # Oracle trajectory: the tool calls an agent knowing all task_params
    # upfront would make to solve the task. Documentary; not eval'd.
    # Filling this requires B2 to demonstrate the LS is internally coherent
    # (search-style steps must hit refs; id args must reference real refs;
    # value args must come from task_params or refs). Subsumes most of the
    # closed-loop validation we used to do via separate static checks.
    gold_trajectory: list[GoldStep]
    local_env: LocalEnv

    model_config = {"extra": "ignore"}


# ─────────────────────────────────────────────────────────────────────────────
# TestSession (user offline, carries eval labels)
# ─────────────────────────────────────────────────────────────────────────────

class TestLabels(BaseModel):
    task_success: TaskSuccessLabel = Field(default_factory=TaskSuccessLabel)


class RuleRef(BaseModel):
    """Rule reference stored on a TestSession for oracle variant injection.

    `canonical_answer` is the unified user-voice statement (see Rule docstring).
    `rule_text` is kept for diagnostics / few-shot display only — it is NOT
    injected into the oracle prompt (oracle reads canonical_answer to stay
    on equal footing with ATR's learning-phase reply).
    """
    rule_id: str
    rule_text: str
    canonical_answer: str


class TestSession(BaseModel):
    session_id: str
    session_type: Literal["test"] = "test"
    domain: str
    rule_id: str             # which Rule this session evaluates

    # Self-contained instruction: all non-rule-binding task params are explicit
    # (user is offline — no clarification possible). Rule-binding fields are
    # deliberately absent so the agent must fill them from memory. Static check
    # at gen time enforces wording neutrality from `rule` + ontology directly,
    # so no per-session "hidden slot" list is needed.
    instruction: str

    local_env: LocalEnv
    labels: TestLabels
    rule_ref: RuleRef | None = None  # populated at assemble time; oracle reads this

    model_config = {"extra": "ignore"}


# ─────────────────────────────────────────────────────────────────────────────
# Episode
# ─────────────────────────────────────────────────────────────────────────────

class Episode(BaseModel):
    episode_id: str
    raw_persona: RawPersona
    # A0 output — structured expansion (demographics etc.) consumed by
    # runtime to derive a minimal PersonaProfile on episode load
    # (see runner.run_episode._build_persona).
    structured_persona: dict[str, Any]
    rules: list[Rule]
    learning_sessions: list[LearningSession]  # ordered by day_offset ascending
    test_sessions: list[TestSession]          # one per rule
    # Episode metadata required by acquisition metrics:
    #   - selected_rules:    list[rule_id] whose signal LS is in trajectory
    #     (every teachable rule). Defines the "covered" axis for the
    #     8-cell breakdown.
    #   - all_test_rules:    full rule_id list (== {r.rule_id for r in rules})
    #   - K / seed / signal_count / noise_count:
    #     dataset provenance, recorded in eval.json for reproducibility.
    # Older episode JSON without metadata defaults to {} — metrics that
    # depend on `selected_rules` then degrade to zero coverage.
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


# ─────────────────────────────────────────────────────────────────────────────
# SessionTrajectory
# ─────────────────────────────────────────────────────────────────────────────

class SessionTrajectory(BaseModel):
    session_id: str
    session_type: Literal["learning", "test"]
    agent_variant: Literal[
        "default", "atr", "always_ask",
        "oracle_full", "oracle_target",
    ]
    messages: list[Message] = Field(default_factory=list)
    termination_reason: Literal[
        # Normal completion: agent called `finish_session()` (TS).
        "agent_stop",
        # LS normal completion: user_sim called `mark_task_complete()`
        # (intent="end"); orchestrator stamped USER_END and ended the
        # session.
        "task_complete",
        # Task-phase: agent kept emitting plain text without progress beyond
        # the no-progress budget — degenerate narration loop, force-stopped.
        "task_no_progress",
        # user_sim returned a `sim_failed` control event (LLM call crashed
        # or produced empty content). Distinct from agent crash.
        "sim_error",
        # Agent/provider returned no content and no tool_calls after the
        # orchestrator's empty-response retry budget. The trajectory is still
        # persisted and may be TS-scored from prior tool calls, but it is not a
        # clean termination.
        "agent_empty_response",
        "max_steps",
        "timeout",
        "error",
        # Set by `_run_one_test` (TS phase) when the per-test attempt
        # raised TWICE (try + 1 retry). Carries an empty `messages` list
        # so the evaluator scores it as `task_success=False` while still
        # counting the TS in the cell's totals — keeps the eval honest
        # under sporadic infra failures (oracle config bug, transient
        # LLM hiccups uncaught by lib/llm.py's own retries).
        "error_after_retry",
    ] = "agent_stop"
    # Optional sub-classification of termination_reason. Currently used to
    # distinguish LS-A (`tool_loop`) from LS-B (`stu_loop`) under the
    # umbrella `task_no_progress` reason — A-N2 /  — so
    # downstream analysis can separate "agent looped on env tools" from
    # "agent looped on STU chitchat".
    termination_subreason: str | None = None
    step_count: int = 0
    duration_seconds: float = 0.0
    # Per-role token accounting. Keys: "agent" | "router" | "user_sim" |
    # "classifier".
    # Each value: {prompt_tokens, completion_tokens, total_tokens, calls}.
    # This is the only cost field — agent_cost / user_cost (floats)
    # are not present; both were either never written or duplicated what's
    # already in token_usage["agent"]["total_tokens"].
    token_usage: dict[str, dict[str, int]] = Field(default_factory=dict)
    memory_snapshot: list[MemoryEntry] = Field(default_factory=list)
    # Send-to-user architecture per-turn events: send_event +
    # route_decision + cls_verdict + off_protocol_ask. Source of truth for
    # evaluator's interaction_aggregates and 8-cell coverage breakdown.
    interaction_events: list[InteractionEvent] = Field(default_factory=list)
