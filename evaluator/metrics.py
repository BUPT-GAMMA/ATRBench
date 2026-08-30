"""Metrics aggregation for ATR episode evaluation (send-to-user architecture).

Two layers:

  Test-session payoff (primary headline metric):
    ts_payoff_accuracy = success / total_test
    by_check_type   diagnostic breakdown over {tool_identity, param_id,
                    param_enum, confirm} — surfaces which axis the rule
                    schema is failing on
    by_domain       cross-domain breakdown for the reported figures

  Interaction + coverage diagnostics (computed from full LS+TS
  trajectories; oracle variant has empty LS section):

    Headline triple:
      ls_send_calls_routed_rule   # send_to_user calls Router routed to rule
      ls_cls_hits                  #     of those, classifier matched a rule
      ls_rule_ask_hit_rate         hits / classifier-classified rule asks
                                   (hits + misses; None if zero)

    8-cell coverage breakdown (covered/uncovered × hit/miss × pass/fail):
      covered_hit_pass / covered_hit_fail
      covered_miss_pass / covered_miss_fail
      uncovered_hit_pass / uncovered_hit_fail
      uncovered_miss_pass / uncovered_miss_fail

      covered = rule's signal LS is in this episode's trajectory (per
                episode.metadata.selected_rules).
      hit / miss = whether any cls_verdict event matched this rule.
      pass / fail = TS task_success for the corresponding test session.

    Other single-cell derivatives:
      ls_send_calls_total        # all send_event events
      ls_send_calls_routed_task  # send_to_user calls Router handled as task user
      ls_cls_misses               # cls_verdict with null rule_id, no error
      ls_cls_errors               # cls_verdict with cls_error=True; reported
                                  # separately from hit-rate as scaffolding fault
      ls_plain_text_leaks         # text-turn leaks before any hook outcome:
                                  # distinct LS assistant text-only user-facing
                                  # turns, rescued or not
      ls_off_protocol_asks        # text-turn leaks that weren't rescued by
                                  # hook and therefore persisted
                                  #
      ls_hook_appended            # # of hook retry attempts
                                  # across all text-turn leaks in this cell.
                                  # Only populated when hook_enabled=True.
                                  # Multiple events per leak (one per retry).
      ls_hook_rescued_sends       # # send_event with
                                  # was_hook_rescued=True
      ls_native_send_calls        # send_to_user calls emitted without hook
                                  # rescue: ls_send_calls_total -
                                  # ls_hook_rescued_sends
      stu_scaffold_rate           # ls_plain_text_leaks /
                                  # (ls_native_send_calls +
                                  #  ls_plain_text_leaks)
      hook_rescue_rate            # ls_hook_rescued_sends / ls_hook_appended.
                                  # Higher → fewer retries per rescue (more
                                  # efficient hook). None when hook never fired.
      repeated_hits               # {rule_id: hit_count}, > 1 is redundant
      tool_protocol_compliance    # final send_to_user calls /
                                  # (final send_to_user calls +
                                  #  unrescued off-protocol leaks)
                                  # Numerator counts ALL sends, native + rescued
                                  # (foundation tool was used either way).
      forced_rule_ask_compliance  # only meaningful for variant=always_ask:
                                  #   % LS sessions with routed_rule >= 1
      rule_coverage               # {hit_count, total, ratio, per_rule_hit_map}

    Lifecycle:
      termination_breakdown       {session_type: {reason: count}} — sanity
      step_stats / duration_stats {session_type: {avg, p50, p95, max, min}}
      token_usage_total           {role: {prompt, completion, total, calls}}

All computers tolerate empty inputs (return zero-shaped dicts), so an
oracle cell — which has no LS — produces a coherent metrics block with
ls_* fields all zero. 8-cell counts are zero when LS is empty.
"""
from __future__ import annotations

from statistics import median
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bucket(total: int, success: int) -> dict[str, Any]:
    return {
        "total": total,
        "success": success,
        "accuracy": round(success / total, 4) if total else None,
    }


def _stats(xs: list[float]) -> dict[str, float] | None:
    """Return {avg, min, p50, p95, max} for a non-empty list, else None."""
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    p95_idx = min(n - 1, max(0, int(round(n * 0.95)) - 1))
    return {
        "avg": round(sum(s) / n, 2),
        "min": round(s[0], 2),
        "p50": round(median(s), 2),
        "p95": round(s[p95_idx], 2),
        "max": round(s[-1], 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section: TS payoff + breakdowns (needs session_results with `domain`)
# ─────────────────────────────────────────────────────────────────────────────

def compute_session_metrics(
    session_results: list[dict],
) -> tuple[float, int, int, dict, dict]:
    """Compute (payoff_accuracy, success, total, by_check_type, by_domain).

    Each session_result must carry: task_success, check_type, domain.
    Domain is optional — sessions without it are excluded from by_domain
    but still counted in payoff_accuracy / by_check_type.
    """
    total = len(session_results)
    success = sum(1 for r in session_results if r.get("task_success"))
    payoff = round(success / total, 4) if total else 0.0

    ct_total: dict[str, int] = {}
    ct_succ: dict[str, int] = {}
    dom_total: dict[str, int] = {}
    dom_succ: dict[str, int] = {}
    dom_ct: dict[str, dict[str, dict[str, int]]] = {}
    for r in session_results:
        succ = bool(r.get("task_success"))
        ct = r.get("check_type")
        dom = r.get("domain")
        if ct:
            ct_total[ct] = ct_total.get(ct, 0) + 1
            ct_succ[ct] = ct_succ.get(ct, 0) + (1 if succ else 0)
        if dom:
            dom_total[dom] = dom_total.get(dom, 0) + 1
            dom_succ[dom] = dom_succ.get(dom, 0) + (1 if succ else 0)
            if ct:
                inner = dom_ct.setdefault(dom, {})
                buck = inner.setdefault(ct, {"total": 0, "success": 0})
                buck["total"] += 1
                buck["success"] += 1 if succ else 0

    by_check_type = {ct: _bucket(ct_total[ct], ct_succ[ct]) for ct in ct_total}
    by_domain: dict[str, dict[str, Any]] = {}
    for d, n in dom_total.items():
        bucket = _bucket(n, dom_succ[d])
        if d in dom_ct:
            bucket["by_check_type"] = {
                ct: _bucket(v["total"], v["success"])
                for ct, v in dom_ct[d].items()
            }
        by_domain[d] = bucket
    return payoff, success, total, by_check_type, by_domain


# ─────────────────────────────────────────────────────────────────────────────
# Section: lifecycle health (works on full trajectory list, both LS + TS)
# ─────────────────────────────────────────────────────────────────────────────

def compute_termination_breakdown(
    trajectories: list,
) -> dict[str, dict[str, int]]:
    """{session_type: {termination_reason: count}}.

    Clean terminations: `agent_stop` (TS — finish_session) and
    `task_complete` (LS — user_sim's mark_task_complete). Anything else
    is a sanity flag — not by itself a failure (timeouts on real LLMs
    happen), but should be visible at a glance in the sweep summary.
    """
    out: dict[str, dict[str, int]] = {}
    for t in trajectories:
        d = out.setdefault(t.session_type, {})
        d[t.termination_reason] = d.get(t.termination_reason, 0) + 1
    return out


def compute_step_duration_stats(
    trajectories: list,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """{session_type: stats} for step_count and duration_seconds."""
    steps: dict[str, list[float]] = {}
    durs: dict[str, list[float]] = {}
    for t in trajectories:
        steps.setdefault(t.session_type, []).append(float(t.step_count or 0))
        durs.setdefault(t.session_type, []).append(float(t.duration_seconds or 0.0))
    step_stats = {st: _stats(xs) for st, xs in steps.items()}
    dur_stats = {st: _stats(xs) for st, xs in durs.items()}
    return (
        {k: v for k, v in step_stats.items() if v is not None},
        {k: v for k, v in dur_stats.items() if v is not None},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section: send-to-user interaction diagnostics (LS-only, reads interaction_events)
# ─────────────────────────────────────────────────────────────────────────────

def compute_interaction_aggregates(
    ls_trajectories: list,
    episode_rules: list,
    ls_domain_map: dict[str, str] | None = None,
    rule_domain_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate send-to-user interaction signals from LS trajectories.

    Reads `interaction_events` written by the orchestrator. Event kinds:
      send_event        — every `send_to_user(output, reason)` call;
                          `was_hook_rescued` flags hook rescues
      route_decision    — Router routed the send: `route` ∈ {rule, task}
                          (`route="rule"` iff
                          `is_strict_rule_question=True`).
                          Some trajectories derive route from the
                          `is_cross_session_ask` user_sim gate.
      cls_verdict       — classifier verdict for a rule-routed send_to_user;
                          carries `rule_id` (None on miss) and `cls_error`
      off_protocol_ask  — text-turn leak (no tool_calls + non-empty
                          content in LS) that hook didn't rescue. this is the only off-protocol
                          shape it's persisted only on
                          rescue failure.
      hook_appended     — scaffolding hook retry attempt; one
                          event per retry call. May carry `question`
                          field with the leak text
      stu_mixed_with_tools  — agent emitted send_to_user together with
                              other tool calls in the same assistant
                              turn. Off-policy structural diagnostic;
                              the protocol still executes tools in
                              declared order. Carries
                              `sibling_tool_names`.
      stu_duplicate_same_turn  — agent emitted multiple send_to_user
                              calls in one assistant turn. The
                              orchestrator keeps the first; this event
                              records how many were dropped via
                              `duplicate_count`.

    When `ls_domain_map` (LS session_id → domain) and `rule_domain_map`
    (rule_id → domain) are provided, rule-routed statistics are broken
    down per domain into `ls_by_domain`. Trajectories without a known
    domain still contribute to the global counters.
    """
    ls_domain_map = ls_domain_map or {}
    rule_domain_map = rule_domain_map or {}

    n_ls = len(ls_trajectories)
    n_rules = len(episode_rules) if episode_rules else 0

    # Global counters.
    send_total = 0
    routed_rule = 0
    routed_task = 0
    cls_hits = 0
    cls_misses = 0
    cls_errors = 0
    off_protocol_asks = 0
    plain_text_leak_keys: set[tuple[str, int]] = set()
    per_rule_hits: dict[str, list[str]] = {}      # rule_id → list[ls_session_id]
    repeated_hits: dict[str, int] = {}            # rule_id → hit_count
    sessions_with_routed_rule = 0                 # forced_rule_ask_compliance
    # scaffolding-hook counters. All zero on hook=off cells.
    hook_appended_total = 0
    hook_rescued_sends = 0
    # Off-policy structural diagnostics.
    stu_mixed_turns = 0
    stu_duplicate_turns = 0
    stu_duplicate_drops = 0
    # user_sim adherence-to-token diagnostic.
    rule_hook_token_missing_total = 0
    rule_hook_token_missing_by_cls: dict[str, int] = {}

    # Per-domain accumulators (rule-routed metrics only).
    dom_n_ls: dict[str, int] = {}
    dom_routed_rule: dict[str, int] = {}
    dom_cls_hits: dict[str, int] = {}
    dom_cls_errors: dict[str, int] = {}
    dom_per_rule_hits: dict[str, dict[str, list[str]]] = {}

    for traj in ls_trajectories:
        sid = traj.session_id
        dom = ls_domain_map.get(sid)
        if dom is not None:
            dom_n_ls[dom] = dom_n_ls.get(dom, 0) + 1

        traj_routed_rule = 0
        traj_cls_hits = 0
        traj_cls_errors = 0
        traj_hit_rule_ids: list[str] = []

        for ev in (getattr(traj, "interaction_events", None) or []):
            kind = getattr(ev, "kind", None)
            if kind == "send_event":
                send_total += 1
                if getattr(ev, "was_hook_rescued", False):
                    hook_rescued_sends += 1
            elif kind == "route_decision":
                route = getattr(ev, "route", None)
                if route == "rule":
                    routed_rule += 1
                    traj_routed_rule += 1
                elif route == "task":
                    routed_task += 1
            elif kind == "cls_verdict":
                if getattr(ev, "cls_error", False):
                    cls_errors += 1
                    traj_cls_errors += 1
                    continue
                rid = getattr(ev, "rule_id", None)
                if rid:
                    cls_hits += 1
                    traj_cls_hits += 1
                    traj_hit_rule_ids.append(rid)
                else:
                    cls_misses += 1
            elif kind == "off_protocol_ask":
                off_protocol_asks += 1
                plain_text_leak_keys.add((sid, getattr(ev, "turn_idx", -1)))
            elif kind == "hook_appended":
                hook_appended_total += 1
                plain_text_leak_keys.add((sid, getattr(ev, "turn_idx", -1)))
            elif kind == "stu_mixed_with_tools":
                stu_mixed_turns += 1
            elif kind == "stu_duplicate_same_turn":
                stu_duplicate_turns += 1
                stu_duplicate_drops += int(
                    getattr(ev, "duplicate_count", 0) or 0
                )
            elif kind == "rule_hook_token_missing":
                # user_sim failed to embed the
                # <RULE_ANSWER> token in its reply. Each missing-token
                # event records the cls_status (hit/miss/error) and
                # attempt_idx (0 = initial reply, 1+ = retries).
                _bucket_key = getattr(ev, "cls_status", None) or "unknown"
                rule_hook_token_missing_by_cls[_bucket_key] = (
                    rule_hook_token_missing_by_cls.get(_bucket_key, 0) + 1
                )
                rule_hook_token_missing_total += 1

        if traj_routed_rule >= 1:
            sessions_with_routed_rule += 1
        for rid in traj_hit_rule_ids:
            per_rule_hits.setdefault(rid, []).append(sid)
            repeated_hits[rid] = repeated_hits.get(rid, 0) + 1

        if dom is not None:
            dom_routed_rule[dom] = dom_routed_rule.get(dom, 0) + traj_routed_rule
            dom_cls_hits[dom] = dom_cls_hits.get(dom, 0) + traj_cls_hits
            dom_cls_errors[dom] = dom_cls_errors.get(dom, 0) + traj_cls_errors
            for rid in traj_hit_rule_ids:
                d = dom_per_rule_hits.setdefault(dom, {})
                d.setdefault(rid, []).append(sid)

    # Coverage is meaningful only when LS phase actually ran.
    coverage_ratio = (
        round(len(per_rule_hits) / n_rules, 4)
        if (n_rules and n_ls) else None
    )

    plain_text_leaks = len(plain_text_leak_keys)

    native_send_calls = max(send_total - hook_rescued_sends, 0)
    scaffold_denominator = native_send_calls + plain_text_leaks
    stu_scaffold_rate = (
        round(plain_text_leaks / scaffold_denominator, 4)
        if scaffold_denominator else None
    )

    # tool_protocol_compliance = final send_to_user calls / (final
    # send_to_user calls + unrescued off-protocol leaks). Hook-rescued sends
    # remain in the numerator because the persisted trajectory did use the
    # foundation STU channel. Native protocol dependence is reported separately
    # as stu_scaffold_rate.
    total_communication = send_total + off_protocol_asks
    tool_protocol_compliance = (
        round(send_total / total_communication, 4)
        if total_communication else None
    )

    # forced_rule_ask_compliance = % LS sessions with ≥1 rule-routed send_to_user.
    # Only meaningful for variant=always_ask; reported for all variants for
    # symmetry. Caller surfaces only the always_ask cell value.
    forced_rule_ask_compliance = (
        round(sessions_with_routed_rule / n_ls, 4)
        if n_ls else None
    )

    rule_count_by_domain: dict[str, int] = {}
    for rid, d in rule_domain_map.items():
        rule_count_by_domain[d] = rule_count_by_domain.get(d, 0) + 1
    all_domains = set(dom_n_ls) | set(rule_count_by_domain)

    ls_by_domain: dict[str, dict[str, Any]] = {}
    for dom in sorted(all_domains):
        n_ls_d = dom_n_ls.get(dom, 0)
        rr = dom_routed_rule.get(dom, 0)
        ch = dom_cls_hits.get(dom, 0)
        ce = dom_cls_errors.get(dom, 0)
        cm = max(rr - ch - ce, 0)
        classified = ch + cm
        prh = dom_per_rule_hits.get(dom, {})
        n_rules_d = rule_count_by_domain.get(dom, 0)
        ls_by_domain[dom] = {
            "total_learning": n_ls_d,
            "ls_send_calls_routed_rule": rr,
            "ls_cls_hits": ch,
            "ls_cls_misses": cm,
            "ls_cls_classified": classified,
            "ls_rule_ask_hit_rate": (
                round(ch / classified, 4) if classified else None
            ),
            "ls_cls_errors": ce,
            "rule_coverage": {
                "hit_count": len(prh),
                "total": n_rules_d,
                "ratio": (
                    round(len(prh) / n_rules_d, 4)
                    if (n_rules_d and n_ls_d) else None
                ),
                "per_rule_hit_map": prh,
            },
        }

    cls_classified = cls_hits + cls_misses

    # hook_rescue_rate is a hook-internal efficiency diagnostic:
    # rescued sends per retry attempt. Native protocol dependence is reported
    # separately as stu_scaffold_rate.
    hook_rescue_rate = (
        round(hook_rescued_sends / hook_appended_total, 4)
        if hook_appended_total else None
    )

    return {
        "total_learning": n_ls,
        # Headline triple
        "ls_send_calls_routed_rule": routed_rule,
        "ls_cls_hits": cls_hits,
        "ls_cls_classified": cls_classified,
        "ls_rule_ask_hit_rate": (
            round(cls_hits / cls_classified, 4)
            if cls_classified else None
        ),
        # Other send / cls / off-protocol counts
        "ls_send_calls_total": send_total,
        "ls_send_calls_routed_task": routed_task,
        "ls_cls_misses": cls_misses,
        "ls_cls_errors": cls_errors,
        "ls_plain_text_leaks": plain_text_leaks,
        "ls_off_protocol_asks": off_protocol_asks,
        # scaffolding-hook diagnostics
        "ls_hook_appended": hook_appended_total,
        "ls_hook_rescued_sends": hook_rescued_sends,
        "ls_native_send_calls": native_send_calls,
        "stu_scaffold_rate": stu_scaffold_rate,
        "hook_rescue_rate": hook_rescue_rate,
        # Off-policy structural diagnostics: turns where send_to_user was
        # mixed with other tool calls, or where the agent issued multiple
        # send_to_user calls in one turn. Denominator is send_total
        # (every kept STU turn).
        "ls_stu_mixed_turns": stu_mixed_turns,
        "ls_stu_duplicate_turns": stu_duplicate_turns,
        "ls_stu_duplicate_drops": stu_duplicate_drops,
        # user_sim adherence-to-token diagnostic.
        # Total = sum over (hit/miss/error/unknown). Bucket dict exposes
        # per-cls-verdict adherence rates downstream.
        "ls_rule_hook_token_missing_total": rule_hook_token_missing_total,
        "ls_rule_hook_token_missing_by_cls": rule_hook_token_missing_by_cls,
        "stu_mixed_rate": (
            round(stu_mixed_turns / send_total, 4) if send_total else None
        ),
        "stu_duplicate_rate": (
            round(stu_duplicate_turns / send_total, 4) if send_total else None
        ),
        "repeated_hits": repeated_hits,
        # Protocol metrics
        "tool_protocol_compliance": tool_protocol_compliance,
        "forced_rule_ask_compliance": forced_rule_ask_compliance,
        # Coverage
        "rule_coverage": {
            "hit_count": len(per_rule_hits),
            "total": n_rules,
            "ratio": coverage_ratio,
            "per_rule_hit_map": per_rule_hits,
        },
        "ls_by_domain": ls_by_domain,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section: 8-cell coverage breakdown (LS coverage × b-hit × TS pass)
# ─────────────────────────────────────────────────────────────────────────────

def compute_coverage_breakdown(
    episode,
    ls_trajectories: list,
    session_results: list[dict],
) -> dict[str, int]:
    """8-cell breakdown of (LS coverage) × (rule hit/miss) × (TS pass/fail).

    For each rule in the episode:
      - covered = rule_id in episode.metadata.selected_rules (signal LS in
                  this trajectory). If metadata is absent, every rule is
                  treated as uncovered.
      - hit     = any cls_verdict InteractionEvent across ls_trajectories
                  matched this rule_id.
      - pass    = the corresponding TS session_result.task_success.

    Returns a flat dict with the 8 count fields:
        covered_hit_pass / covered_hit_fail
        covered_miss_pass / covered_miss_fail
        uncovered_hit_pass / uncovered_hit_fail
        uncovered_miss_pass / uncovered_miss_fail

    Sum of all 8 counts == |episode.rules|. The uncovered cells stay
    occupied whenever some rules are unteachable (zero signal LS), even
    though every teachable rule contributes its signal to the trajectory.
    """
    metadata = getattr(episode, "metadata", None) or {}
    covered_set: set[str] = set(metadata.get("selected_rules") or [])

    hit_set: set[str] = set()
    for traj in ls_trajectories:
        for ev in (getattr(traj, "interaction_events", None) or []):
            if getattr(ev, "kind", None) != "cls_verdict":
                continue
            if getattr(ev, "cls_error", False):
                continue  # scaffolding fault: don't count as hit OR miss
            rid = getattr(ev, "rule_id", None)
            if rid:
                hit_set.add(rid)

    # TS pass: rule_id → task_success bool, from session_results.
    ts_pass_by_rule: dict[str, bool] = {}
    for r in session_results:
        rid = r.get("rule_id")
        if rid:
            ts_pass_by_rule[rid] = bool(r.get("task_success"))

    cells = {
        "covered_hit_pass": 0,
        "covered_hit_fail": 0,
        "covered_miss_pass": 0,
        "covered_miss_fail": 0,
        "uncovered_hit_pass": 0,
        "uncovered_hit_fail": 0,
        "uncovered_miss_pass": 0,
        "uncovered_miss_fail": 0,
    }

    for rule in (episode.rules or []):
        rid = getattr(rule, "rule_id", None)
        if not rid:
            continue
        coverage = "covered" if rid in covered_set else "uncovered"
        outcome = "hit" if rid in hit_set else "miss"
        ts_pass = ts_pass_by_rule.get(rid)
        if ts_pass is None:
            # Rule has no corresponding TS result — likely missing TS
            # session (counted elsewhere as missing_test_sessions). Skip.
            continue
        result = "pass" if ts_pass else "fail"
        cells[f"{coverage}_{outcome}_{result}"] += 1

    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Section: token cost
# ─────────────────────────────────────────────────────────────────────────────

from lib.llm import USAGE_FIELDS as _LLM_USAGE_FIELDS

# Same raw-usage field set lib/llm._record_usage writes, plus the "calls"
# counter the recorder maintains alongside. Sourcing the raw fields from
# lib/llm keeps the evaluator aligned with the recorder.
_USAGE_FIELDS = _LLM_USAGE_FIELDS + ("calls",)


def aggregate_token_usage(trajectories: list) -> dict[str, dict[str, int]]:
    """Sum per-role token usage across all trajectories in the cell.

    Keys mirror what the orchestrator's `token_scope` writes:
    "agent" | "router" | "user_sim" | "classifier".

    Surfaces both raw input/output counts and KV-cache attribution.
    """
    out: dict[str, dict[str, int]] = {}
    for t in trajectories:
        for role, sub in (t.token_usage or {}).items():
            bucket = out.setdefault(role, {k: 0 for k in _USAGE_FIELDS})
            for k in _USAGE_FIELDS:
                bucket[k] += int(sub.get(k) or 0)
    return out
