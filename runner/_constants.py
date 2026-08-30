"""Shared string constants for runner + evaluator.

Centralises the tool names, markers, sweep dimensions, and termination
categories that several modules previously redeclared. Drift between
those copies was a documented hazard (e.g. memory.py and orchestrator.py
both spelled out `"send_to_user"`; pipeline.py and evaluator/pipeline.py
each owned their own `_VARIANTS`).

The `Literal[...]` annotations in `runner/schemas.py` remain the type
contract for trajectory data; the string values here must stay in lockstep
with those Literals.
"""
from __future__ import annotations

# ── Synthetic / control-plane tool names ─────────────────────────────────────

SEND_TO_USER_TOOL_NAME = "send_to_user"
FINISH_SESSION_TOOL_NAME = "finish_session"
MARK_TASK_COMPLETE_TOOL_NAME = "mark_task_complete"

# Tools whose presence in the trajectory must be hidden from the
# ContextLayer memory dump (control-plane plumbing, not user-facing
# dialogue). Same set is used by the user_sim history renderer.
CONTEXT_HIDDEN_TOOL_NAMES = frozenset({
    SEND_TO_USER_TOOL_NAME,
    FINISH_SESSION_TOOL_NAME,
})

# ── Conversation markers ─────────────────────────────────────────────────────

USER_END_MARKER = "###USER_END###"
RULE_ANSWER_TOKEN = "<RULE_ANSWER>"

# Fixed deflect phrase substituted for <RULE_ANSWER> on cls miss / error
#. A constant rather than a user_sim-synthesized sentence so
# that cls-miss replies are deterministic and structurally cannot invent
# a standing preference.
NO_RULE_DEFLECT = "no strong preference there — your call"

# ── Sweep dimensions ─────────────────────────────────────────────────────────

# Mirrors `Literal[...]` on SessionTrajectory.agent_variant in schemas.py.
# The reported benchmark uses four variants (default / atr / always_ask /
# oracle); the reported "oracle" is oracle_target. oracle_full is a
# broader-context reference and is not part of the reported four.
# oracle_full   = TS-only, inject ALL ground-truth rules + canonical answers.
# oracle_target = TS-only, inject ONLY the current TS's target rule
#                 canonical_answer (the reported "oracle").
#
# The bare `oracle` name is not accepted; only the two split variants
# above are, so any stale CLI / manifest referencing `oracle` fails loudly.
VARIANTS = (
    "default", "atr", "always_ask",
    "oracle_full", "oracle_target",
)
ORACLE_VARIANTS = frozenset({"oracle_full", "oracle_target"})
# Variants that skip the LS phase entirely (oracle_*). Used by the
# evaluator's R4 LS-completeness exception and any runtime path that gates
# on "did this cell produce learning sessions at all?".
TS_ONLY_VARIANTS = frozenset({"oracle_full", "oracle_target"})
LAYER_CHOICES = ("raw", "context", "native")

# ── Termination categories ───────────────────────────────────────────────────

# Termination reasons that count as a normal session end (vs. a hard failure
# the sweep summary should flag). `agent_stop` = TS `finish_session()`,
# `task_complete` = LS user_sim `mark_task_complete()`.
CLEAN_TERMINATIONS = ("agent_stop", "task_complete")
