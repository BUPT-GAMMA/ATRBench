"""ATR evaluator pipeline.

Two layers in this module:

  Library:
    evaluate_test_session  — score a single TestSession trajectory
    evaluate_episode       — score every TS in an episode + lifecycle diags

  CLI:
    main()                 — sweep across (persona × seed × variant ×
                             model × memory_layer); read each cell's
                             trajectories from canonical paths, write
                             eval.json + sweep summary.

Layer boundary:
    released data provides episode JSON
    runner produces trajectories
    evaluator reads both and produces eval.json + summary    ← THIS

Output (per-cell `eval.json`):
  - `session_results`    list of per-test-session dicts
  - `metrics`            full aggregation block (see evaluator/metrics.py)
                         including `missing_test_sessions` (counted as
                         hard failures in payoff_accuracy denominator).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import SessionTrajectory, TestSession, Episode, Rule
from action_match import match_actions, StepResult
from metrics import (
    aggregate_token_usage,
    compute_coverage_breakdown,
    compute_interaction_aggregates,
    compute_session_metrics,
    compute_step_duration_stats,
    compute_termination_breakdown,
)
import paths as run_paths
from _constants import (
    LAYER_CHOICES,
    ORACLE_VARIANTS as _ORACLE_VARIANTS,
    TS_ONLY_VARIANTS as _TS_ONLY_VARIANTS,
    VARIANTS,
)
from lib.benchmark_data import read_json

# `tools.pretty_trace` is imported lazily inside `_eval_one_cell` so that
# `from evaluator.pipeline import evaluate_episode` (library use) doesn't
# pull in an unrelated trace-rendering dependency.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-session evaluation
# ---------------------------------------------------------------------------

def evaluate_test_session(
    traj: SessionTrajectory,
    session: TestSession,
    rule: Rule,
) -> dict[str, Any]:
    """Evaluate one test-session trajectory against its labels + rule.

    Returns a plain dict (JSON-serializable). `domain` is included so the
    episode-level by_domain breakdown can pivot on it without re-loading
    the TestSession.
    """
    required_actions = session.labels.task_success.required_actions
    step_results: list[StepResult] = match_actions(traj, required_actions)
    task_success = bool(step_results) and all(sr.passed for sr in step_results)

    return {
        "session_id": traj.session_id,
        "rule_id": session.rule_id,
        "agent_variant": traj.agent_variant,
        "check_type": rule.check_type,
        "domain": session.domain,
        "step_results": [
            {
                "step_idx": sr.step_idx,
                "required_tool": sr.required_tool,
                "tool_found": sr.tool_found,
                "matched_call_id": sr.matched_call_id,
                "args_match": sr.args_match,
                "arg_mismatches": sr.arg_mismatches,
                "passed": sr.passed,
            }
            for sr in step_results
        ],
        "task_success": task_success,
    }


# ---------------------------------------------------------------------------
# Episode-level evaluation
# ---------------------------------------------------------------------------

def evaluate_episode(
    trajectories: list[SessionTrajectory],
    episode: Episode,
    variant_filter: str | None = None,
) -> dict[str, Any]:
    """Evaluate every test session + compute episode-level diagnostics.

    LS trajectories contribute to:
      - ATR action diagnostics (rule-routed volume / classifier hits /
        coverage / incidental task-routed sends)
      - termination_breakdown / step_stats / duration_stats (LS bucket)
      - token_usage_total

    TS trajectories contribute to all of the above plus the per-session
    `task_success` evaluation.

    Denominator policy
    ------------------
    `payoff_accuracy` divides by `len(episode.test_sessions)`, NOT by the
    number of trajectories successfully loaded. Any test session for which
    a trajectory is missing or unparseable is counted as a hard failure
    (task_success=False) and surfaced in `metrics.missing_test_sessions`.
    Earlier this function silently shrunk the denominator when trajectories
    were missing — that quietly inflated payoff_accuracy whenever a cell
    had partial data.
    """
    rule_map: dict[str, Rule] = {r.rule_id: r for r in episode.rules}
    session_map: dict[str, TestSession] = {
        s.session_id: s for s in episode.test_sessions
    }

    # Variant-scoped trajectory list. Both LS and TS pass through here.
    cell_trajs: list[SessionTrajectory] = [
        t for t in trajectories
        if not variant_filter or t.agent_variant == variant_filter
    ]
    ts_trajs = [t for t in cell_trajs if t.session_type == "test"]
    ls_trajs = [t for t in cell_trajs if t.session_type == "learning"]

    # ── Per-test-session evaluation ─────────────────────────────────────
    session_results: list[dict] = []
    present_test_sids: set[str] = set()
    for traj in ts_trajs:
        session = session_map.get(traj.session_id)
        if not session:
            logger.warning(
                "No TestSession found for trajectory %s — skipping.",
                traj.session_id,
            )
            continue
        rule = rule_map.get(session.rule_id)
        if not rule:
            logger.warning(
                "Rule %r not found in episode %s — skipping session %s.",
                session.rule_id, episode.episode_id, traj.session_id,
            )
            continue
        session_results.append(evaluate_test_session(traj, session, rule))
        present_test_sids.add(traj.session_id)

    # Force-account missing test sessions: each one becomes a synthetic
    # failure entry so `compute_session_metrics` divides by the EXPECTED
    # test count, not the loaded count. The synthetic entry is also flagged
    # so callers can surface it (the CLI below prints these explicitly).
    missing_test_sessions: list[str] = []
    for sid, session in session_map.items():
        if sid in present_test_sids:
            continue
        rule = rule_map.get(session.rule_id)
        if rule is None:
            # The session itself references an unknown rule — pre-existing
            # data inconsistency, not a runner failure. Skip rather than
            # synthesise a phantom row with no rule context.
            logger.warning(
                "Test session %s references unknown rule %r — excluded "
                "from missing-session accounting.",
                sid, session.rule_id,
            )
            continue
        missing_test_sessions.append(sid)
        session_results.append({
            "session_id": sid,
            "rule_id": session.rule_id,
            "agent_variant": variant_filter or "missing",
            "check_type": rule.check_type,
            "domain": session.domain,
            "step_results": [],
            "task_success": False,
            "missing": True,
        })

    # ── Learning-session presence accounting ────────────────────────────
    # Symmetric to TS: oracle variants legitimately have zero LS (learning
    # phase is skipped), so missing_learning_sessions is meaningful only
    # for non-oracle variants. We expose the list unconditionally and let
    # the CLI gate the strictness.
    expected_ls_sids = {ls.session_id for ls in episode.learning_sessions}
    present_ls_sids = {t.session_id for t in ls_trajs}
    # Oracle exception: TS-only variants (oracle_full / oracle_target)
    # legitimately have zero LS — the learning phase is skipped by
    # run_episode. Key membership on TS_ONLY_VARIANTS so the post-split
    # scope is correct (the same LS-completeness exception applies).
    missing_learning_sessions: list[str] = (
        []
        if variant_filter in _TS_ONLY_VARIANTS
        else sorted(expected_ls_sids - present_ls_sids)
    )

    # ── Aggregate metrics ───────────────────────────────────────────────
    payoff, success, total, by_check_type, by_domain = compute_session_metrics(
        session_results,
    )
    # Maps used by interaction diagnostics for per-domain pivot. LS carries
    # domain directly on the schema; each Rule's domain is inferred via the
    # TS that targets it (test_sessions are 1:1 with rules in the current
    # schema).
    ls_domain_map = {ls.session_id: ls.domain for ls in episode.learning_sessions}
    rule_domain_map = {ts.rule_id: ts.domain for ts in episode.test_sessions}
    interaction_diag = compute_interaction_aggregates(
        ls_trajs, episode.rules,
        ls_domain_map=ls_domain_map,
        rule_domain_map=rule_domain_map,
    )
    coverage_cells = compute_coverage_breakdown(
        episode, ls_trajs, session_results,
    )
    termination_breakdown = compute_termination_breakdown(cell_trajs)
    step_stats, duration_stats = compute_step_duration_stats(cell_trajs)
    token_usage_total = aggregate_token_usage(cell_trajs)

    metrics: dict[str, Any] = {
        # --- TS payoff (primary headline metric) ---
        "ts_payoff_accuracy": payoff,
        # back-compat alias for old plotting code that reads `payoff_accuracy`
        "payoff_accuracy": payoff,
        "total_test": total,
        "success_count": success,
        "by_check_type": by_check_type,
        "by_domain": by_domain,
        # --- Missing test sessions (counted as hard failures above) ---
        # Empty list ⇒ all expected test sessions ran. Non-empty list ⇒ the
        # cell is incomplete; payoff_accuracy already accounts for this by
        # treating the listed sids as failures, but downstream tooling
        # should usually flag the cell as not ready for reporting.
        "missing_test_sessions": missing_test_sessions,
        # --- Missing learning sessions ---
        # Empty list ⇒ either every LS produced a parseable trajectory, or
        # this is an oracle cell (no LS by design). Non-empty list ⇒ at
        # least one LS trajectory failed to load; interaction diagnostics
        # (rule-routed volume / hits / coverage) are computed on the surviving subset
        # and will under-report. CLI surfaces this analogously to
        # missing_test_sessions.
        "missing_learning_sessions": missing_learning_sessions,
        # --- Speak interaction diagnostics (LS) ---
        # Headline triple + diagnostics + protocol metrics. Spread inline so
        # downstream consumers can grep `metrics.ls_send_calls_routed_rule`
        # etc. without an extra nested level.
        **interaction_diag,
        # --- 8-cell coverage × hit/miss × pass/fail breakdown ---
        # Sum of all 8 keys equals |episode.rules| (modulo missing TS).
        # See evaluator/metrics.compute_coverage_breakdown docstring for
        # the axis semantics.
        **coverage_cells,
        # --- Lifecycle health ---
        "termination_breakdown": termination_breakdown,
        "step_stats": step_stats,
        "duration_stats": duration_stats,
        # --- Token cost ---
        "token_usage_total": token_usage_total,
    }

    return {
        "episode_id": episode.episode_id,
        "variant": variant_filter,
        "session_results": session_results,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# Local list aliases for argparse choices (argparse needs a list, not a tuple).
_VARIANTS = list(VARIANTS)
_LAYER_CHOICES = list(LAYER_CHOICES)


def _load_episode_file(path: Path) -> Episode:
    return Episode.model_validate(read_json(path))


def _load_trajectories(traj_dir: Path) -> list[SessionTrajectory]:
    trajs: list[SessionTrajectory] = []
    for p in sorted(traj_dir.glob("*.json")):
        try:
            trajs.append(SessionTrajectory.model_validate(read_json(p)))
        except Exception as e:
            logger.warning("Skipping %s: %s", p.name, e)
    return trajs


_episode_path_canonical = run_paths.episode_path


def _eval_one_cell(
    *,
    ep_path: Path,
    variant: str,
    model: str | None,
    memory_layer: str,
    seed: int | None,
    cell_id: str,
    hook_enabled: bool = False,
    write_trace_footers: bool = True,
) -> dict:
    """Evaluate one cell: read trajectories, score, write eval.json, return row."""
    t0 = time.time()
    row: dict = {
        "cell_id": cell_id,
        "episode_path": str(ep_path),
        "variant": variant,
        "model": model,
        "memory_layer": memory_layer,
        "seed": seed,
        "ok": False,
        "incomplete": False,
        "duration_sec": None,
        "error": None,
    }
    try:
        if not ep_path.exists():
            raise FileNotFoundError(f"Episode not found: {ep_path}")
        episode = _load_episode_file(ep_path)
        runs_dir = run_paths.runs_dir(
            episode.raw_persona.persona_id, episode.episode_id, variant,
            model=model, memory_layer=memory_layer, seed=seed,
            hook_enabled=hook_enabled,
        )
        traj_dir = runs_dir / "trajectories"
        if not traj_dir.exists():
            raise FileNotFoundError(
                f"Trajectory dir not found: {traj_dir}. "
                f"Run `runner.pipeline` first."
            )
        trajs = _load_trajectories(traj_dir)
        # Oracle exception keys on cell_manifest.variant (the
        # source-of-truth axis), not on the sweep CLI's variant argument.
        # They should be identical in normal operation, but reading from
        # the manifest defends against axis drift / call-site bugs.
        manifest_path = runs_dir / "cell_manifest.json"
        manifest_variant: str | None = None
        try:
            from runner.manifest import read_manifest  # type: ignore[import-not-found]
        except ImportError:
            try:
                from manifest import read_manifest  # type: ignore[no-redef]
            except ImportError:
                read_manifest = None  # type: ignore[assignment]
        if read_manifest is not None and manifest_path.exists():
            m = read_manifest(manifest_path)
            if m is not None:
                manifest_variant = m.get("variant")
        effective_variant = manifest_variant or variant
        if manifest_variant and manifest_variant != variant:
            logger.warning(
                "[%s] cell_manifest.variant=%r differs from CLI variant=%r; "
                "using cell_manifest as source of truth.",
                cell_id, manifest_variant, variant,
            )
        result = evaluate_episode(trajs, episode, variant_filter=effective_variant)
        out_path = runs_dir / "eval.json"
        # Atomic write: .tmp + os.replace.
        tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        tmp_out.replace(out_path)

        # Backfill gold-check footers on existing trace files (test sessions only).
        if write_trace_footers:
            traces_dir = runs_dir / "traces"
            if traces_dir.is_dir():
                from tools.pretty_trace import append_eval_footer  # lazy
                eval_by_sid = {sr["session_id"]: sr for sr in result.get("session_results", [])}
                for traj in trajs:
                    if traj.session_type != "test":
                        continue
                    ev = eval_by_sid.get(traj.session_id)
                    tp = traces_dir / f"{traj.session_id}.txt"
                    if ev and tp.exists():
                        append_eval_footer(tp, traj.model_dump(), ev)

        m = result["metrics"]
        tok_by_role = m.get("token_usage_total") or {}
        cell_total = sum(b["total_tokens"] for b in tok_by_role.values())
        # Cell-level input / output / cached token totals, summed across all
        # roles (agent / router / user_sim / classifier). Surface them on the row so
        # the sweep table and downstream tooling can render in/out separately
        # without re-walking token_usage_total.
        cell_input = sum(b.get("prompt_tokens", 0) for b in tok_by_role.values())
        cell_output = sum(b.get("completion_tokens", 0) for b in tok_by_role.values())
        cell_cached = sum(
            b.get("cached_tokens", 0) + b.get("prompt_cache_hit_tokens", 0)
            for b in tok_by_role.values()
        )
        missing_ts = m.get("missing_test_sessions") or []
        missing_ls = m.get("missing_learning_sessions") or []
        row.update({
            "runs_dir": str(runs_dir),
            "eval_path": str(out_path),
            "payoff_accuracy": m["payoff_accuracy"],
            "ts_payoff_accuracy": m.get("ts_payoff_accuracy", m["payoff_accuracy"]),
            "success_count": m["success_count"],
            "total_test": m["total_test"],
            "missing_test_sessions": missing_ts,
            "missing_learning_sessions": missing_ls,
            "by_check_type": m.get("by_check_type", {}),
            "by_domain": m.get("by_domain", {}),
            "total_learning": m.get("total_learning", 0),
            # Speak-architecture headline triple
            "ls_send_calls_routed_rule": m.get("ls_send_calls_routed_rule", 0),
            "ls_cls_hits": m.get("ls_cls_hits", 0),
            "ls_cls_classified": m.get("ls_cls_classified", 0),
            "ls_rule_ask_hit_rate": m.get("ls_rule_ask_hit_rate"),
            # Other send / cls / off-protocol counts
            "ls_send_calls_total": m.get("ls_send_calls_total", 0),
            "ls_send_calls_routed_task": m.get("ls_send_calls_routed_task", 0),
            "ls_cls_misses": m.get("ls_cls_misses", 0),
            "ls_cls_errors": m.get("ls_cls_errors", 0),
            "ls_plain_text_leaks": m.get("ls_plain_text_leaks", 0),
            "ls_off_protocol_asks": m.get("ls_off_protocol_asks", 0),
            "ls_hook_appended": m.get("ls_hook_appended", 0),
            "ls_hook_rescued_sends": m.get("ls_hook_rescued_sends", 0),
            "ls_native_send_calls": m.get("ls_native_send_calls", 0),
            "stu_scaffold_rate": m.get("stu_scaffold_rate"),
            "hook_rescue_rate": m.get("hook_rescue_rate"),
            "repeated_hits": m.get("repeated_hits", {}),
            "tool_protocol_compliance": m.get("tool_protocol_compliance"),
            "forced_rule_ask_compliance": m.get("forced_rule_ask_compliance"),
            "rule_coverage": m.get("rule_coverage", {}),
            # 8-cell coverage breakdown
            "covered_hit_pass": m.get("covered_hit_pass", 0),
            "covered_hit_fail": m.get("covered_hit_fail", 0),
            "covered_miss_pass": m.get("covered_miss_pass", 0),
            "covered_miss_fail": m.get("covered_miss_fail", 0),
            "uncovered_hit_pass": m.get("uncovered_hit_pass", 0),
            "uncovered_hit_fail": m.get("uncovered_hit_fail", 0),
            "uncovered_miss_pass": m.get("uncovered_miss_pass", 0),
            "uncovered_miss_fail": m.get("uncovered_miss_fail", 0),
            "termination_breakdown": m.get("termination_breakdown", {}),
            "step_stats": m.get("step_stats", {}),
            "duration_stats": m.get("duration_stats", {}),
            "token_usage": tok_by_role,
            "total_tokens": cell_total,
            "input_tokens": cell_input,
            "output_tokens": cell_output,
            "cached_tokens": cell_cached,
        })
        # Two-axis cell status:
        #   ok         — eval ran without raising (no I/O / parse / agg crash)
        #   incomplete — payoff was computed but on a partial dataset
        #                (some TS or LS trajectories missing). payoff_accuracy
        #                ALREADY treats missing TS as failures, so the number
        #                is conservative; the flag exists so downstream tooling
        #                can decide whether to surface or suppress those rows.
        #
        # The CLI maps these to exit codes — see main(): --strict turns
        # `incomplete` into a non-zero exit. With --no-strict the eval
        # completes and only true crashes (ok=False) trigger non-zero exit.
        row["incomplete"] = bool(missing_ts or missing_ls)
        row["ok"] = True
        if row["incomplete"]:
            logger.warning(
                "Cell %s: incomplete (missing %d TS, %d LS) — payoff "
                "computed against expected denominator.",
                cell_id, len(missing_ts), len(missing_ls),
            )
        cache_pct = (cell_cached / cell_input * 100) if cell_input else 0.0
        logger.info(
            "=== EVAL  %s   acc=%.1f%%  (%d/%d)  miss_ts=%d miss_ls=%d  "
            "rule=%s/%s  task=%s  off=%s  cls_err=%d  in=%d out=%d cache=%.0f%%",
            cell_id, m["payoff_accuracy"] * 100,
            m["success_count"], m["total_test"],
            len(missing_ts), len(missing_ls),
            m.get("ls_cls_hits", 0),
            m.get("ls_send_calls_routed_rule", 0),
            m.get("ls_send_calls_routed_task", 0),
            m.get("ls_off_protocol_asks", 0),
            m.get("ls_cls_errors", 0),
            cell_input, cell_output, cache_pct,
        )
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("=== FAIL  %s   %s\n%s", cell_id, row["error"], traceback.format_exc())
    finally:
        row["duration_sec"] = round(time.time() - t0, 1)
    return row


def _build_cells_sweep(args) -> list[dict]:
    cells = []
    hook_enabled = bool(getattr(args, "hook", False))
    for p in args.personas:
        for s in args.seeds:
            for v in args.variants:
                for m in args.models:
                    ep = _episode_path_canonical(p, s)
                    cell_segment = run_paths.cell_segment(
                        v, m, args.memory_layer, s,
                        hook_enabled=hook_enabled,
                    )
                    if v in _ORACLE_VARIANTS:
                        short_ep = "_oracle"
                    else:
                        short_ep = f"seed{s:03d}"
                    cell_id = (
                        f"{p}/{short_ep}/{cell_segment}"
                    )
                    cells.append({
                        "ep_path": ep, "variant": v, "model": m, "seed": s,
                        "cell_id": cell_id,
                    })
    return cells


def _build_cells_single(args) -> list[dict]:
    ep = Path(args.episode).resolve()
    seed = args.seed
    if seed is None:
        seed = run_paths.seed_from_episode_id(ep.stem)
    hook_enabled = bool(getattr(args, "hook", False))
    cell_id = f"{ep.stem}/{run_paths.cell_segment(args.variant, args.model, args.memory_layer, seed, hook_enabled=hook_enabled)}"
    return [{
        "ep_path": ep, "variant": args.variant, "model": args.model, "seed": seed,
        "cell_id": cell_id,
    }]


def _fmt_rule_hit(row: dict) -> str:
    """Display classifier outcome over classified rule asks.

    Rule-routed volume remains available as `ls_send_calls_routed_rule`;
    display keeps classifier errors visible as φ instead of folding them into
    the hit-rate denominator.
    """
    hits = row.get("ls_cls_hits") or 0
    misses = row.get("ls_cls_misses") or 0
    classified = row.get("ls_cls_classified")
    if classified is None:
        classified = hits + misses
    errors = row.get("ls_cls_errors") or 0
    if not classified and not errors:
        return "-"
    parts: list[str] = []
    if classified:
        parts.append(f"{hits}/{classified}")
    if errors:
        parts.append(f"φ{errors}")
    return "+".join(parts)


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    print()
    print(
        f"{'cell_id':<60} {'acc':>7} {'pass':>6} {'mTS':>4} {'mLS':>4} "
        f"{'hit':>9} {'cov':>7} {'task':>4} {'off':>4} {'φ!':>4} {'term!':>5} "
        f"{'sec':>6} {'in':>9} {'out':>8} {'cache%':>6}"
    )
    print("-" * 161)
    for r in rows:
        if r.get("payoff_accuracy") is not None:
            acc = f"{r['payoff_accuracy']*100:5.1f}%"
            scor = f"{r['success_count']}/{r['total_test']}"
        else:
            acc = "ERR"
            scor = "-"
        miss_ts = len(r.get("missing_test_sessions") or [])
        miss_ts_s = f"{miss_ts}!" if miss_ts else "-"
        miss_ls = len(r.get("missing_learning_sessions") or [])
        miss_ls_s = f"{miss_ls}!" if miss_ls else "-"
        ls_atr = _fmt_rule_hit(r)
        cov = (r.get("rule_coverage") or {}).get("ratio")
        cov_s = f"{cov*100:5.1f}%" if cov is not None else "-"
        a_count = r.get("ls_send_calls_routed_task") or 0
        off_count = r.get("ls_off_protocol_asks") or 0
        cls_err = r.get("ls_cls_errors") or 0
        phi_s = f"{cls_err}!" if cls_err else "-"
        term_bd = r.get("termination_breakdown") or {}
        bad = sum(
            v for st, d in term_bd.items() for k, v in d.items()
            if k not in ("agent_stop", "task_complete")
        )
        term_s = "✓" if bad == 0 else f"{bad}!"
        in_tok = r.get("input_tokens") or 0
        out_tok = r.get("output_tokens") or 0
        cached = r.get("cached_tokens") or 0
        cache_s = f"{(cached / in_tok * 100):5.1f}" if in_tok else "  -  "
        print(
            f"{r['cell_id']:<60} {acc:>7} {scor:>6} {miss_ts_s:>4} {miss_ls_s:>4} "
            f"{ls_atr:>9} {cov_s:>7} {a_count:>4} {off_count:>4} {phi_s:>4} {term_s:>5} "
            f"{r['duration_sec']:>6} {in_tok:>9} {out_tok:>8} {cache_s:>6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluator pipeline — score trajectories produced by runner. "
                    "Sweep mode: --personas/--seeds/--variants. "
                    "Single-cell mode: --episode + --variant.",
    )
    # Sweep dims
    parser.add_argument("--personas", nargs="+",
                        help="Persona IDs (sweep mode)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--variants", nargs="+", choices=_VARIANTS,
                        default=list(_VARIANTS))
    parser.add_argument("--models", nargs="+", default=[None])
    # Single-cell mode
    parser.add_argument("--episode", default=None)
    parser.add_argument("--variant", default=None, choices=_VARIANTS)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Single-cell mode: LLM/cell seed. If omitted and --episode stem "
             "contains _seedNNN, that seed is inferred to match sweep mode.",
    )
    # Cell key
    parser.add_argument("--memory-layer", default="context", choices=_LAYER_CHOICES)
    # Behavior
    parser.add_argument(
        "--strict", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "When set (default), an incomplete cell (any missing TS or LS "
            "trajectory) causes exit code 2. With --no-strict the run "
            "completes and only true crashes (parse / I/O / aggregation "
            "errors) cause non-zero exit. payoff_accuracy denominator is "
            "ALWAYS the expected test count regardless of this flag — "
            "missing TS are counted as failures. The flag controls exit "
            "code only, not the metric."
        ),
    )
    parser.add_argument("--hook", action="store_true", default=False,
                        help="Read from hook-enabled cell paths (…__hook suffix).")
    parser.add_argument("--no-trace-footers", action="store_true",
                        help="Skip backfilling gold-check footers on traces/<sid>.txt.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel cells. Eval is CPU-light, default 4.")
    # Output
    parser.add_argument("--outputs-root", default=None,
                        help="Override the output root for ALL artifacts "
                             "(cell eval.json + summary). Must match the "
                             "--outputs-root used in the upstream "
                             "runner.pipeline call, otherwise this stage "
                             "won't find the trajectories.")
    parser.add_argument("--summary-dir", default=None,
                        help="Sweep summary output dir (default: "
                             "<outputs-root>/_summary/, i.e. follows "
                             "--outputs-root if set, else outputs/_summary/).")

    args = parser.parse_args()

    if args.outputs_root:
        run_paths.set_runs_root(args.outputs_root)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if args.episode:
        if not args.variant:
            parser.error("--episode requires --variant")
        cells = _build_cells_single(args)
        mode = "single-cell"
    else:
        if not args.personas:
            parser.error("Sweep mode requires --personas (or use --episode for single-cell)")
        cells = _build_cells_sweep(args)
        mode = "sweep"

    logger.info("Evaluator pipeline (%s): %d cell(s)", mode, len(cells))

    rows: list[dict] = []
    kw = dict(
        memory_layer=args.memory_layer,
        write_trace_footers=not args.no_trace_footers,
        hook_enabled=bool(getattr(args, "hook", False)),
    )
    if args.workers <= 1:
        for c in cells:
            rows.append(_eval_one_cell(**c, **kw))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_eval_one_cell, **c, **kw): c["cell_id"] for c in cells}
            for f in as_completed(futs):
                rows.append(f.result())

    rows.sort(key=lambda r: r["cell_id"])
    _print_table(rows)

    summary_dir = (
        Path(args.summary_dir).resolve() if args.summary_dir
        else run_paths.get_runs_root() / "_summary"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = summary_dir / f"evaluator_{ts}.json"
    summary_path.write_text(json.dumps({
        "stage": "evaluator",
        "timestamp": ts,
        "mode": mode,
        "args": {k: v for k, v in vars(args).items()},
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    print(f"\nevaluator summary → {summary_path}")

    # Exit-code policy:
    #   1  any cell crashed (ok=False — exception during eval)
    #   2  --strict and any cell incomplete (missing TS or LS trajectories)
    #   0  otherwise
    # If both conditions hit, prefer 2 (incomplete data is the more visible
    # signal — caller should investigate whether the runner finished cleanly).
    n_fail = sum(1 for r in rows if not r["ok"])
    n_incomplete = sum(1 for r in rows if r.get("incomplete"))
    if args.strict and n_incomplete:
        sys.exit(2)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
