"""Build the agent system prompt.

Send-to-user architecture variant set (canonical, mirrors
`_constants.VARIANTS`):
{default, atr, always_ask, oracle_full, oracle_target}.

Layered design (common pieces in `base`; per-block bodies live in
`runner/prompts/agent.md`):

  base                shared: identity + id rules
  domain_<d>          per-domain id discovery + shape constraints (hand-written)
  context             cross-session memory snapshot (optional)
  mode_test/_learning protocol (TS terminates via finish_session;
                                LS terminates when user_sim calls
                                mark_task_complete())
  send_protocol       (LS only) — universal "the user only hears you
                      through the `send_to_user` tool; plain assistant text
                      never reaches them" rule. HOW only, not WHEN.
  variant_atr         (LS + variant=="atr" only) — soft permission-flavoured
                      hint about acquiring long-term rules useful in future
                      sessions. Does NOT mention `send_to_user` or `reason`.
  variant_always_ask  (LS + variant=="always_ask" only) — same value-prop hint
                      framed as an obligation. Does NOT mention `send_to_user`
                      or `reason`. Prompt-only enforcement (orchestrator does
                      not block finish_session if the agent disobeys; the
                      forced_rule_ask_compliance metric records adherence).

`default` sees neither nudge — measures spontaneous rule-ask behavior.
`oracle_full` / `oracle_target` skip LS entirely (run_episode injects rules
into memory and runs only TS), so the LS-only blocks above never apply;
oracle reuses the same prompt for TS. Both oracle and ATR see the same
rule canonical_answer when rules surface in `<context>` — keeping the
variants on equal information footing so ATR's lift measures asking-
quality + learning, not formatting.

Domain policy blocks are hand-written in agent.md — TableSpec doesn't carry
enough per-tool ID-parameter semantics to auto-render accurate prompts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _constants import VARIANTS as _CANONICAL_VARIANTS
from _prompts import load_prompt
from schemas import LearningSession, TestSession


_KNOWN_DOMAINS = frozenset({
    "commerce", "reservation", "travel",
    "communication", "scheduling", "workspace",
})


def _domain_policy_block(domain: str | None) -> str | None:
    """Return the per-domain policy block, or None when the session has no
    recognised domain (defensive — every TestSession / LearningSession must
    carry a domain in current schema).

    Domain policy blocks are hand-written in agent.md.
    """
    if not domain or domain not in _KNOWN_DOMAINS:
        return None
    return load_prompt("agent", f"domain_{domain}")


def build_agent_system_prompt(
    task: LearningSession | TestSession,
    variant: str,
    context: str | None = None,
) -> str:
    """Layout (recency-aware):

      base + domain  →  <context>  →  mode_X  [+ variant_atr (LS+atr only)]

    `<context>` can run 10K+ chars (raw transcripts of past sessions). If
    `mode_X` were placed before it, the protocol constraints get drowned by
    a long tail of "agent↔user dialogue" — observed empirically as
    test_text_violation in TS for atr/default but not oracle (oracle's
    <context> is one short canonical_answer line). Putting mode_X AFTER
    <context> keeps the hard-constraint tokens at the prompt tail where
    attention is strongest.

    The `variant_atr` block (when present) goes at the very tail so the
    ATR-relevant value-prop hint sits next to the LS protocol block — agent
    reads "you're online with the user" immediately followed by "this tool
    obtains preferences that may apply in future sessions". Section is
    loaded only for variant=="atr" in LS; default / oracle never see it.
    """
    if variant not in _CANONICAL_VARIANTS:
        raise ValueError(
            f"Unknown agent variant {variant!r}. "
            f"Allowed: {sorted(_CANONICAL_VARIANTS)}. "
            f"(the bare `oracle` is not accepted; use one of the "
            f"canonical variants explicitly.)"
        )
    blocks: list[str] = [load_prompt("agent", "base")]

    # Per-domain policy (id discovery + shape constraints).
    policy = _domain_policy_block(getattr(task, "domain", None))
    if policy is not None:
        blocks.append(policy)

    # Note: env state (objects, ids, attributes) is intentionally NOT dumped
    # into the system prompt. Agent must obtain it via search_/list_/track_
    # tool calls. This prevents the agent from
    # short-circuiting to a tool-only solve by reading env attrs off the prompt
    # — and forces it to ask the user when the task references something the
    # agent has no way to disambiguate from a tool return alone.

    if context:
        blocks.append(f"<context>\n{context}\n</context>")

    # Note: oracle no longer injects a separate <user_preference> block.
    # Instead, run_episode pre-fills the cross-session layer with each
    # rule's canonical_answer as a synthetic prior user statement, so the
    # oracle agent sees the rules through the same `<context>` channel as
    # ATR / default — keeping the comparison apples-to-apples.

    if task.session_type == "test":
        blocks.append(load_prompt("agent", "mode_test"))
    else:
        blocks.append(load_prompt("agent", "mode_learning"))
        # Universal send-to-user channel rule — present in every LS prompt
        # regardless of variant. Oracle skips LS entirely so this branch never
        # runs for it.
        blocks.append(load_prompt("agent", "send_protocol"))
        # Variant-specific WHEN nudge (LS only). atr = soft permission;
        # always_ask = strong obligation. default sees no extra block — its
        # behavior is the unprompted baseline.
        # The shared `standing_rule_def` block defines what counts as a
        # rule (category, not specific instance) and is loaded before the
        # variant block so the WHEN-nudge prose can reference it.
        if variant in ("atr", "always_ask"):
            blocks.append(load_prompt("agent", "standing_rule_def"))
        if variant == "atr":
            blocks.append(load_prompt("agent", "variant_atr"))
        elif variant == "always_ask":
            blocks.append(load_prompt("agent", "variant_always_ask"))

    return "\n\n".join(blocks)
