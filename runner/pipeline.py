"""Runner pipeline — execute trajectories for one or more cells.

Layer boundary:
    released data provides episode JSON  (data/personas/<p>/episodes/<eid>.json)
    runner reads episodes and writes trajectories + cell_manifest       ← THIS
    evaluator reads episodes + trajectories and writes eval.json + summary

Two modes:

  Sweep mode (default) — full Cartesian product:
    --personas P [P ...]   --seeds S [S ...]
    --variants V [V ...]   --models M [M ...]
    plus run-params (--memory-layer / --max-steps / --timeout)

  Single-cell mode — explicit episode path, no compose convention:
    --episode <path>  --variant V  [--model M]  [--seed S] ...
    Useful for ad-hoc episodes outside the canonical data/personas/ layout.

Each persona has one composed episode per seed, named
`<persona>_seed<NNN>.json`.

Examples:

  uv run python -m runner.pipeline \\
      --personas alice bob \\
      --variants atr always_ask oracle_target --models gpt-5.4 \\
      --memory-layer context

  uv run python -m runner.pipeline \\
      --episode /tmp/custom_ep.json --variant atr --model gpt-5.4
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "runner"))

from lib.benchmark_data import PERSONAS_DIR, read_json
from runner.run_episode import run_episode, load_episode
from runner import paths as run_paths
from runner._constants import (
    CLEAN_TERMINATIONS as _CLEAN_TERMINATIONS,
    LAYER_CHOICES,
    ORACLE_VARIANTS as _ORACLE_VARIANTS,
    TS_ONLY_VARIANTS as _TS_ONLY_VARIANTS,
    VARIANTS,
)
from evaluator.pipeline import evaluate_test_session
# Speak architecture reads `interaction_events` directly.

logger = logging.getLogger("runner.pipeline")

# Silence chatty third-party INFO loggers that bury the actual run signal.
for _noisy in ("httpx", "openai", "openai._base_client"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# Local list alias for argparse choices (argparse needs a list, not a tuple).
_VARIANTS = list(VARIANTS)
_LAYER_CHOICES = list(LAYER_CHOICES)


# ── Per-cell + sweep-wide live counters ──────────────────────────────────────
# All counters are integer-only; derived metrics (avg turns, hit rate, etc.)
# are computed in the formatters below. Rich renders task.fields[X] for every
# task on every column, so each cell row + the overall row both keep the same
# field set; "meaningless for overall" entries (e.g. ATR) just stay empty.

# Serializes per-callback state mutation + rich progress.update across:
# `evaluate_test_session` runs OUTSIDE the lock — it has no shared state and
# is the only heavy step in the callback, so the lock's critical section is
# microseconds and never gates LLM calls.
#
# Per-sweep-invocation: the lock that serializes the small callback critical
# section is now instantiated inside `_run_with_progress`,
# rather than at module level. A module-level lock would be silently shared
# across independent sweep invocations in the same process — fine for the
# canonical "one sweep per process" usage but a quiet hazard for any future
# embedded / library usage.


def _new_state() -> dict:
    """Per-cell live counters under the send-to-user architecture. Reads
    InteractionEvents so the source of truth matches what evaluator/metrics.py
    aggregates.
    """
    return {
        # Headline counts. send_routed_rule = send_to_user calls Router routed
        # to rule; cls_hit/cls_miss are classifier-classified outcomes.
        "send_routed_rule": 0, "cls_hit": 0, "cls_miss": 0,
        # cls scaffolding fault — excluded from routed_rule/cls_hit so the
        # hit rate stays interpretable.
        "cls_err": 0,
        # send_to_user calls Router handled as task user.
        "send_routed_task": 0,
        # off_protocol_ask events: text-turn leaks (no tool_calls, non-empty
        # content in LS) that hook did NOT rescue (or hook was off). Per
        # this is persisted only on rescue failure.
        "off_protocol": 0,
        # scaffolding-hook counters (turn-level). All zero on
        # hook=off cells. hook_appended = total retry attempts across
        # all problem turns (one event per retry); hook_rescued = number
        # of send_to_user calls marked was_hook_rescued (1 per rescued
        # problem turn).
        "hook_appended": 0,
        "hook_rescued": 0,
        # STU same-turn anomaly counters (LS-only). Both count distinct
        # offending assistant turns, not events:
        #   stu_mix_turns — turns where send_to_user co-occurred with
        #                   other tool calls (sibling tools recorded in
        #                   the event, not aggregated here).
        #   stu_dup_turns — turns where the agent emitted multiple
        #                   send_to_user calls; orchestrator kept the
        #                   first and dropped the rest.
        "stu_mix_turns": 0,
        "stu_dup_turns": 0,
        "rules_covered": set(),  # distinct rule_ids hit across LS
        "n_rules": 0,             # episode rule pool size, populated lazily
        # Session counts
        "n_done": 0, "turns_total": 0,
        "ls_total": 0, "ts_total": 0,
        "ls_with_routed_rule": 0,
        # Termination outcomes split by kind
        "ls_clean": 0, "ls_bad": 0,
        "ts_clean": 0, "ts_bad": 0,
        # TS task_success (real eval via evaluator.evaluate_test_session)
        "ts_pass": 0,
        # Per-role cumulative tokens + call counts.
        "tok_agent_in": 0, "tok_agent_out": 0, "tok_agent_calls": 0,
        "tok_router_in": 0, "tok_router_out": 0, "tok_router_calls": 0,
        "tok_user_in": 0, "tok_user_out": 0, "tok_user_calls": 0,
        "tok_cls_in": 0, "tok_cls_out": 0, "tok_cls_calls": 0,
    }


def _update_state(s: dict, kind: str, traj, ts_eval: dict | None = None) -> None:
    """Fold one finished session's trajectory into the running counters.

    Counters are additive (+=) and idempotent across cells, so the same
    function works for per-cell state and for the sweep-wide aggregate —
    caller picks which dict to fold into. `rules_covered` is a set so it's
    only meaningful per-cell (sweep keeps the union but doesn't display it).
    """
    s["n_done"] += 1
    s["turns_total"] += int(getattr(traj, "step_count", 0) or 0)

    is_clean = traj.termination_reason in _CLEAN_TERMINATIONS
    if kind == "learning":
        s["ls_total"] += 1
        if is_clean: s["ls_clean"] += 1
        else: s["ls_bad"] += 1
    else:  # test
        s["ts_total"] += 1
        if is_clean: s["ts_clean"] += 1
        else: s["ts_bad"] += 1
        if ts_eval and ts_eval.get("task_success"):
            s["ts_pass"] += 1

    # Per-role token bucketing — keys come from runner/orchestrator's
    # token_scope() calls (agent / router / user_sim / classifier).
    for role, role_usage in (getattr(traj, "token_usage", None) or {}).items():
        in_tok = int(role_usage.get("prompt_tokens") or 0)
        out_tok = int(role_usage.get("completion_tokens") or 0)
        n_calls = int(role_usage.get("calls") or 0)
        if role == "agent":
            s["tok_agent_in"] += in_tok
            s["tok_agent_out"] += out_tok
            s["tok_agent_calls"] += n_calls
        elif role == "router":
            s["tok_router_in"] += in_tok
            s["tok_router_out"] += out_tok
            s["tok_router_calls"] += n_calls
        elif role == "user_sim":
            s["tok_user_in"] += in_tok
            s["tok_user_out"] += out_tok
            s["tok_user_calls"] += n_calls
        elif role == "classifier":
            s["tok_cls_in"] += in_tok
            s["tok_cls_out"] += out_tok
            s["tok_cls_calls"] += n_calls

    if kind == "learning":
        traj_has_routed_rule = False
        for ev in (getattr(traj, "interaction_events", None) or []):
            ek = getattr(ev, "kind", None)
            if ek == "send_event":
                if getattr(ev, "was_hook_rescued", False):
                    s["hook_rescued"] += 1
            elif ek == "route_decision":
                route = getattr(ev, "route", None)
                if route == "rule":
                    s["send_routed_rule"] += 1
                    traj_has_routed_rule = True
                elif route == "task":
                    s["send_routed_task"] += 1
            elif ek == "cls_verdict":
                if getattr(ev, "cls_error", False):
                    s["cls_err"] += 1
                else:
                    rid = getattr(ev, "rule_id", None)
                    if rid:
                        s["cls_hit"] += 1
                        s["rules_covered"].add(rid)
                    else:
                        s["cls_miss"] += 1
            elif ek == "off_protocol_ask":
                s["off_protocol"] += 1
            elif ek == "hook_appended":
                s["hook_appended"] += 1
            elif ek == "stu_mixed_with_tools":
                s["stu_mix_turns"] += 1
            elif ek == "stu_duplicate_same_turn":
                s["stu_dup_turns"] += 1
        if traj_has_routed_rule:
            s["ls_with_routed_rule"] += 1


def _fmt_hook(s: dict) -> str:
    """ hook column. Displays `<problem>/<rescued> ×<avg>`. `off_protocol_ask` events are persisted only on
    rescue failure (rescue is invisible in trajectory.messages). To
    derive the total problem-turn count we must therefore add rescued
    + unrescued:

      problem  = off_protocol + hook_rescued
                 (failed rescues + successful rescues = total leaks)
      rescued  = hook_rescued
                 (text-turn leaks that ended in a was_hook_rescued
                 send_to_user)
      avg      = hook_appended / problem
                 (avg retry attempts per problem turn; 1.0 = single
                 retry succeeded, up to 3.0 = exhausted budget)

    `—` when the hook never fired (hook=off cells, or hook=on cells
    whose agent stayed on-protocol).
    """
    fire = s.get("hook_appended", 0)
    if fire == 0:
        return "—"
    rescued = s.get("hook_rescued", 0)
    unrescued = s.get("off_protocol", 0)
    problem = rescued + unrescued
    avg = fire / problem if problem else 0.0
    return f"{problem}/{rescued} ×{avg:.1f}"


def _fmt_atr(s: dict) -> str:
    """Rule-routed-send display. Headline is `cls_hit / classified`.
    `classified` is hits + misses. cls_err appears as `+φK` when non-zero:
    those are rule-routed send_to_user calls where the classifier crashed,
    so they are kept visible but excluded from the hit-rate denominator.

    Layout examples:
      routed_rule=0 cls_err=0  → "—"
      routed_rule=3 hit=1 miss=2      → "1/3"
      routed_rule=3 hit=1 miss=1 err=1 → "1/2+φ1"
      routed_rule=1 err=1             → "φ1"
    """
    classified = s.get("cls_hit", 0) + s.get("cls_miss", 0)
    cls_err = s.get("cls_err", 0)
    if classified == 0 and cls_err == 0:
        return "—"
    parts: list[str] = []
    if classified:
        parts.append(f"{s['cls_hit']}/{classified}")
    if cls_err:
        parts.append(f"φ{cls_err}")
    return "+".join(parts) if parts else "—"


def _fmt_ls_term(s: dict) -> str:
    if s["ls_total"] == 0:
        return "—"
    return f"✓{s['ls_clean']}/✗{s['ls_bad']}"


def _fmt_ts_term(s: dict) -> str:
    if s["ts_total"] == 0:
        return "—"
    return f"✓{s['ts_clean']}/✗{s['ts_bad']}"


def _fmt_cov(s: dict) -> str:
    if s["n_rules"] == 0:
        return "—"
    return f"{len(s['rules_covered'])}/{s['n_rules']}"


def _fmt_acc(s: dict) -> str:
    """TS task_success rate (programmatic check vs gold). Distinct from
    ts_term, which only reports protocol-clean TS terminations."""
    if s["ts_total"] == 0:
        return "—"
    pct = s["ts_pass"] * 100 / s["ts_total"]
    return f"{s['ts_pass']}/{s['ts_total']} {pct:.0f}%"


def _fmt_turns(s: dict) -> str:
    if s["n_done"] == 0:
        return "—"
    return f"Tn{s['turns_total'] / s['n_done']:.1f}"


def _fmt_tok_pair(in_tok: int, out_tok: int) -> str:
    """Compact token-pair render: '↑Xk↓Yk' (rounded to nearest k)."""
    if in_tok == 0 and out_tok == 0:
        return "—"
    return f"↑{in_tok / 1000:.1f}k↓{out_tok / 1000:.1f}k"


# Thread-local cell id used by `_CellThreadFilter` to gate per-cell file
# handlers. With `--workers > 1` multiple cells run concurrently and each
# attaches its own FileHandler to the root logger; without per-thread
# gating, every active handler receives every record and cell A's run.log
# ends up with cell B's lines (and vice versa). The filter consults this
# TLS slot to drop records that don't originate in the owning thread.
_CELL_LOG_TLS = threading.local()


class _CellThreadFilter(logging.Filter):
    """Pass records only when the emitting thread matches the handler's owner.

    Each per-cell FileHandler has one of these attached with its own
    `cell_thread_id`. At emit time the filter compares against the current
    thread id; mismatch → drop. The console StreamHandler has no filter so
    aggregated stdout still shows every cell's output.
    """

    def __init__(self, cell_thread_id: int):
        super().__init__()
        self._owner = cell_thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return threading.get_ident() == self._owner


def _setup_cell_log(out_log: Path):
    """Attach a per-cell file handler scoped to the calling thread.

    Returns the handler; caller removes it from the root logger when done.
    The console StreamHandler is added once globally (idempotent). The file
    handler is gated by `_CellThreadFilter` so concurrent cells do not
    cross-contaminate each other's run.log.
    """
    out_log.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)
    fh = logging.FileHandler(out_log, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.addFilter(_CellThreadFilter(threading.get_ident()))
    root.addHandler(fh)
    return fh


_episode_path = run_paths.episode_path


def _run_one_cell(
    *,
    ep_path: Path,
    variant: str,
    model: str | None,
    memory_layer: str,
    seed: int | None,
    max_steps: int,
    timeout: float | None,
    traces: bool,
    fresh: bool,
    allow_stale_resume: bool,
    cell_id: str,
    agent_reasoning_effort: str | None = None,
    on_session_done: Callable[..., None] | None = None,
    ts_workers: int = 1,
    hook_enabled: bool = False,
    smoke: bool = False,
) -> dict:
    """Execute one (episode × variant × model × layer × seed) cell.

    Returns a row dict suitable for the sweep summary. `ok=False` on
    exception or non-fatal anomaly; caller decides whether to continue.
    """
    t0 = time.time()
    row: dict = {
        "cell_id": cell_id,
        "episode_path": str(ep_path),
        "variant": variant,
        "model": model,
        "memory_layer": memory_layer,
        "seed": seed,
        "ok": False,
        "duration_sec": None,
        "error": None,
    }
    file_handler = None
    try:
        if not ep_path.exists():
            raise FileNotFoundError(
                f"Episode not found: {ep_path}. Check the released data path."
            )
        episode = load_episode(ep_path)
        runs_dir = run_paths.runs_dir(
            episode.raw_persona.persona_id, episode.episode_id, variant,
            model=model, memory_layer=memory_layer, seed=seed,
            hook_enabled=hook_enabled, smoke=smoke,
        )
        if fresh and runs_dir.exists():
            logger.info("[--fresh] wiping %s", runs_dir)
            shutil.rmtree(runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Cell-level resume short-circuit: a prior successful run persists
        # the full row dict to `_cell_done.json`. If that file exists with
        # ok=True, skip the entire cell — no episode reload, no run_episode,
        # no per-session resume scan. `--fresh` wipes runs_dir above so the
        # marker is gone in fresh mode. Aborted/failed cells don't write
        # the marker, so they re-run on the next sweep.
        done_marker = runs_dir / "_cell_done.json"
        if done_marker.exists():
            try:
                cached = json.loads(done_marker.read_text())
                if cached.get("ok") is True:
                    logger.info(
                        "=== SKIP  %s  (already done; remove _cell_done.json to force rerun)",
                        cell_id,
                    )
                    cached["cell_id"] = cell_id
                    cached["episode_path"] = str(ep_path)
                    cached["duration_sec"] = 0.0
                    cached["skipped"] = True
                    return cached
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[skip-check] %s: _cell_done.json unreadable (%s); will rerun",
                    cell_id, e,
                )

        traces_dir = runs_dir / "traces" if traces else None
        file_handler = _setup_cell_log(runs_dir / "run.log")
        logger.info("=== START %s ===", cell_id)
        logger.info("episode: %s", ep_path)
        logger.info("runs_dir: %s", runs_dir)
        logger.info(
            "mode: %s",
            "fresh (cleared cache)" if fresh else "resume (reuse trajectories)",
        )
        if traces_dir is not None:
            logger.info("traces (live tail): %s/<sid>.txt", traces_dir)

        trajs = run_episode(
            str(ep_path), variant,
            memory_layer=memory_layer,
            model=model,
            max_steps=max_steps,
            timeout=timeout,
            seed=seed,
            out_dir=runs_dir / "trajectories",
            traces_dir=traces_dir,
            allow_stale_resume=allow_stale_resume,
            on_session_done=on_session_done,
            agent_reasoning_effort=agent_reasoning_effort,
            ts_workers=ts_workers,
            hook_enabled=hook_enabled,
            smoke=smoke,
        )

        # Reuse the same folding helper used by live progress so the final
        # row + the live counters can never disagree (one source of truth
        # for what counts as a "clean termination", a "TS pass", etc.).
        cell_summary = _new_state()
        cell_summary["n_rules"] = len(episode.rules)
        session_map = {s.session_id: s for s in episode.test_sessions}
        rule_map = {r.rule_id: r for r in episode.rules}
        for t in trajs:
            ts_eval = None
            if t.session_type == "test":
                sess = session_map.get(t.session_id)
                rule = rule_map.get(sess.rule_id) if sess else None
                if sess and rule:
                    try:
                        ts_eval = evaluate_test_session(t, sess, rule)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "evaluate_test_session failed for %s: %s",
                            t.session_id, e,
                        )
            _update_state(cell_summary, t.session_type, t, ts_eval=ts_eval)

        ls_n = cell_summary["ls_total"]
        ts_n = cell_summary["ts_total"]
        avg_turns = (
            cell_summary["turns_total"] / cell_summary["n_done"]
            if cell_summary["n_done"] else 0.0
        )
        row.update({
            "n_learning": ls_n,
            "n_test": ts_n,
            # Speak-architecture per-cell counters
            "send_routed_rule": cell_summary["send_routed_rule"],
            "cls_hit": cell_summary["cls_hit"],
            "cls_miss": cell_summary["cls_miss"],
            "cls_classified": cell_summary["cls_hit"] + cell_summary["cls_miss"],
            "cls_err": cell_summary["cls_err"],
            "send_routed_task": cell_summary["send_routed_task"],
            "off_protocol": cell_summary["off_protocol"],
            # hook counters (turn-level).
            "hook_appended": cell_summary["hook_appended"],
            "hook_rescued": cell_summary["hook_rescued"],
            # STU misuse diagnostics surfaced
            # in the per-cell sweep row (already in eval.json; this
            # mirrors them so the sweep summary shows the same
            # information without re-reading eval.json).
            "stu_mix_turns": cell_summary["stu_mix_turns"],
            "stu_dup_turns": cell_summary["stu_dup_turns"],
            "rules_covered": len(cell_summary["rules_covered"]),
            "n_rules": cell_summary["n_rules"],
            "ls_with_routed_rule": cell_summary["ls_with_routed_rule"],
            "ls_clean": cell_summary["ls_clean"],
            "ls_bad": cell_summary["ls_bad"],
            "ts_clean": cell_summary["ts_clean"],
            "ts_bad": cell_summary["ts_bad"],
            "ts_pass": cell_summary["ts_pass"],
            "ts_total": cell_summary["ts_total"],
            # `non_clean_terminations` retained for callers that read
            # it (status flag in the table, summary JSON consumers).
            "non_clean_terminations": cell_summary["ls_bad"] + cell_summary["ts_bad"],
            "avg_turns": round(avg_turns, 2),
            "tok_agent_in": cell_summary["tok_agent_in"],
            "tok_agent_out": cell_summary["tok_agent_out"],
            "tok_agent_calls": cell_summary["tok_agent_calls"],
            "tok_router_in": cell_summary["tok_router_in"],
            "tok_router_out": cell_summary["tok_router_out"],
            "tok_router_calls": cell_summary["tok_router_calls"],
            "tok_user_in": cell_summary["tok_user_in"],
            "tok_user_out": cell_summary["tok_user_out"],
            "tok_user_calls": cell_summary["tok_user_calls"],
            "tok_cls_in": cell_summary["tok_cls_in"],
            "tok_cls_out": cell_summary["tok_cls_out"],
            "tok_cls_calls": cell_summary["tok_cls_calls"],
            "runs_dir": str(runs_dir),
        })
        hook_avg = (
            cell_summary["hook_appended"] / cell_summary["off_protocol"]
            if cell_summary["off_protocol"] else 0.0
        )
        logger.info(
            "=== DONE  %s   ls=%d ts=%d rule=%d/%d classified (%d routed, φ%d) "
            "task=%d hook=%d/%d ×%.1f cov=%d/%d "
            "LS=✓%d/✗%d TS=✓%d/✗%d acc=%d/%d Tn=%.1f",
            cell_id, ls_n, ts_n,
            cell_summary["cls_hit"],
            cell_summary["cls_hit"] + cell_summary["cls_miss"],
            cell_summary["send_routed_rule"], cell_summary["cls_err"],
            cell_summary["send_routed_task"],
            cell_summary["off_protocol"], cell_summary["hook_rescued"], hook_avg,
            len(cell_summary["rules_covered"]), cell_summary["n_rules"],
            cell_summary["ls_clean"], cell_summary["ls_bad"],
            cell_summary["ts_clean"], cell_summary["ts_bad"],
            cell_summary["ts_pass"], cell_summary["ts_total"],
            avg_turns,
        )
        row["ok"] = True
        # Persist a cell-done marker so a subsequent sweep can skip this
        # cell entirely (see the short-circuit at the top of the try block).
        try:
            (runs_dir / "_cell_done.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False, default=str)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[done-marker] %s failed to write: %s", cell_id, e)
    except Exception as exc:  # noqa: BLE001 — top-level driver
        # SweepAbort signals the entire sweep should stop.
        # Re-raise so the sweep loop sets ABORT_EVENT and refuses to
        # launch more cells. The cell's partial trajectory (if any) is
        # NOT persisted (P31).
        from _abort import SweepAbort
        if isinstance(exc, SweepAbort):
            logger.error(
                "=== ABORT %s  SweepAbort from module=%s session=%s: %s",
                cell_id, exc.module, exc.session_id, exc.original,
            )
            row["error"] = f"SweepAbort: {exc}"
            row["aborted"] = True
            if file_handler is not None:
                logging.getLogger().removeHandler(file_handler)
                file_handler.close()
            raise
        row["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("=== FAIL  %s   %s\n%s", cell_id, row["error"], traceback.format_exc())
    finally:
        row["duration_sec"] = round(time.time() - t0, 1)
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
    return row


def _build_cells_sweep(args) -> list[dict]:
    """Cartesian product of sweep dimensions (persona x seed x variant x model).

    Each persona has one composed episode per seed.
    """
    cells = []
    hook_enabled = bool(getattr(args, "hook", False))
    for p in args.personas:
        for s in args.seeds:
            for v in args.variants:
                for m in args.models:
                    ep = _episode_path(p, s)
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
                        "ep_path": ep,
                        "variant": v,
                        "model": m,
                        "seed": s,
                        "cell_id": cell_id,
                    })
    return cells


@contextlib.contextmanager
def _suppress_root_console_handler():
    """Detach the root logger's StreamHandler for the duration of the block.

    Rich's Progress live-renders to stderr by repaint; if other handlers also
    write to stderr/stdout the bars get torn. We keep per-cell FileHandlers
    attached (run.log capture intact) but pull stderr handlers out so the
    Progress display owns the terminal during the run.
    """
    root = logging.getLogger()
    detached = [
        h for h in list(root.handlers)
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    for h in detached:
        root.removeHandler(h)
    try:
        yield
    finally:
        for h in detached:
            root.addHandler(h)


def _run_with_progress(
    cells: list[dict],
    cell_kwargs: dict,
    *,
    workers: int,
    no_progress: bool,
) -> list[dict]:
    """Drive cell execution with a rich progress bar.

    Renders one overall task ("Sweep N/M cells") and one task per cell
    (showing `<completed>/<total>` sessions plus the latest session id /
    kind in the task description). When `no_progress` is set or no TTY
    is detected, falls back to plain sequential / ThreadPool execution
    so logs flow normally — useful for CI / piped runs.
    """
    rows: list[dict] = []
    use_progress = not no_progress and sys.stderr.isatty()

    from _abort import ABORT_EVENT, SweepAbort

    if not use_progress:
        # No bars: behave exactly like the pre-progress code path.
        if workers <= 1:
            for c in cells:
                if ABORT_EVENT.is_set():
                    logger.warning(
                        "[sweep-abort] skipping cell %s (ABORT_EVENT set)",
                        c["cell_id"],
                    )
                    continue
                try:
                    rows.append(_run_one_cell(**c, **cell_kwargs))
                except SweepAbort as exc:
                    # Per-cell isolation: an abort in one cell does not
                    # cascade to others. Record the failure, treat the cell
                    # as finished (failed), keep iterating the sweep.
                    logger.error(
                        "[cell-abort] cell %s raised SweepAbort; recording as "
                        "failed and continuing sweep. (%s)",
                        c["cell_id"], exc,
                    )
                    rows.append({
                        "cell_id": c["cell_id"],
                        "ok": False,
                        "aborted": True,
                        "error": f"SweepAbort: {exc}",
                    })
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_run_one_cell, **c, **cell_kwargs): c["cell_id"] for c in cells}
                for f in as_completed(futs):
                    cid = futs[f]
                    try:
                        rows.append(f.result())
                    except SweepAbort as exc:
                        # Per-cell isolation: this future's abort does not
                        # affect pending or running futures.
                        logger.error(
                            "[cell-abort] cell %s raised SweepAbort; "
                            "recording as failed and continuing sweep. (%s)",
                            cid, exc,
                        )
                        rows.append({
                            "cell_id": cid,
                            "ok": False,
                            "aborted": True,
                            "error": f"SweepAbort: {exc}",
                        })
        return rows

    console = Console(stderr=True)
    # Column legend (per cell, all derived from _fmt_*):
    #   ls    "✓<clean>/✗<bad>"                  LS termination outcomes
    #   ts    "✓<clean>/✗<bad>"                  TS termination outcomes
    #   atr   "<hits>/<asks>"                    formal ATR action labels (LS)
    #   cov   "<hit_rules>/<n_rules>"            distinct rule coverage (LS)
    #   acc   "<pass>/<total> <pct>%"            TS task_success vs gold
    #   turns "Tn<avg>"                          mean step_count per session
    # Sweep row reuses these — atr / cov / acc are still meaningful as
    # totals since they aggregate across variants/cells.
    # Speak-architecture rich columns: n_done/total, ts_pass, rule/task,
    # hit, off. Per-cell row + sweep row both render the same fields.
    # LS/TS termination + cov are dropped from live rich (still surfaced
    # in eval.json + end-of-run table).
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}", justify="left"),
        BarColumn(bar_width=24),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("[green]term {task.fields[term]}"),
        TextColumn("[bold yellow]ts {task.fields[ts_pass]}"),
        TextColumn("[cyan]rule {task.fields[rule]}"),
        TextColumn("[magenta]hit {task.fields[hit]}"),
        TextColumn("[yellow]cov {task.fields[cov]}"),
        TextColumn("[red]stu {task.fields[stu]}"),
        TextColumn("[blue]hook {task.fields[hook]}"),
        TextColumn("{task.fields[note]}"),
        console=console,
        transient=False,
        refresh_per_second=4,
    )

    # Pre-compute expected session counts per cell (so each bar shows
    # a proper denominator from the start, not "?" until first update).
    expected_per_cell = {
        c["cell_id"]: _expected_session_count(c["ep_path"], c["variant"])
        for c in cells
    }

    cell_state: dict[str, dict] = {
        c["cell_id"]: _new_state() for c in cells
    }
    sweep_state = _new_state()
    # Lazy-loaded eval contexts per cell — populated on first session
    # callback (so we don't pay the JSON read up-front for cells that
    # might fail before producing any TS).
    _eval_ctx: dict[str, dict] = {c["cell_id"]: {"loaded": False} for c in cells}

    def _ensure_eval_ctx(cid: str, ep_path: Path) -> dict:
        ctx = _eval_ctx[cid]
        if ctx["loaded"]:
            return ctx
        try:
            ep = load_episode(ep_path)
            ctx["session_map"] = {s.session_id: s for s in ep.test_sessions}
            ctx["rule_map"] = {r.rule_id: r for r in ep.rules}
            ctx["n_rules"] = len(ep.rules)
        except Exception as e:  # noqa: BLE001
            logger.warning("eval ctx load failed for %s: %s", cid, e)
            ctx["session_map"] = {}
            ctx["rule_map"] = {}
            ctx["n_rules"] = 0
        ctx["loaded"] = True
        return ctx

    def _cell_fields(s: dict) -> dict[str, str]:
        rule = s.get("send_routed_rule", 0)
        clean = s.get("ls_clean", 0) + s.get("ts_clean", 0)
        bad = s.get("ls_bad", 0) + s.get("ts_bad", 0)
        term = f"✓{clean}/✗{bad}" if (clean or bad) else "—"
        hit = s.get("cls_hit", 0)
        n_distinct = len(s.get("rules_covered", ()) or ())
        mix = s.get("stu_mix_turns", 0)
        dup = s.get("stu_dup_turns", 0)
        stu = f"m{mix}/d{dup}" if (mix or dup) else "—"
        return {
            "term": term,
            "ts_pass": _fmt_acc(s),
            "rule": str(rule) if rule else "—",
            "hit": str(hit) if hit else "—",
            "cov": str(n_distinct) if n_distinct else "—",
            "stu": stu,
            "hook": _fmt_hook(s),  # <problem>/<rescued> ×<avg>
        }

    # Concurrency: two layers can fire `_cb` from worker threads —
    # (a) per-cell TS concurrency inside `run_episode` (ts_workers, default 1),
    # (b) cell-level `--workers` > 1.
    # Per-sweep-invocation `_cb_lock` serializes the small critical
    # section that mutates `cell_state` / `sweep_state` and calls
    # `progress.update`. `evaluate_test_session` runs *outside* the lock to
    # keep the critical section short.
    _cb_lock = threading.Lock()
    with _suppress_root_console_handler(), progress:
        overall = progress.add_task(
            "Sweep", total=len(cells), note="",
            **_cell_fields(_new_state()),
        )
        # Per-cell tasks created up front so the user sees the full plan.
        cell_tasks: dict[str, TaskID] = {
            c["cell_id"]: progress.add_task(
                f"  {c['cell_id']}",
                total=max(expected_per_cell[c["cell_id"]], 1),
                note="queued", **_cell_fields(_new_state()),
            )
            for c in cells
        }
        cell_ep_paths = {c["cell_id"]: c["ep_path"] for c in cells}

        def make_callback(cid: str):
            tid = cell_tasks[cid]
            cs = cell_state[cid]
            ep_path = cell_ep_paths[cid]

            def _cb(sid: str, kind: str, traj) -> None:
                # evaluate_test_session has no shared mutable state; run it
                # outside the lock so the critical section stays microseconds.
                ts_eval = None
                if kind == "test":
                    ctx = _ensure_eval_ctx(cid, ep_path)
                    sess = ctx["session_map"].get(sid)
                    rule = ctx["rule_map"].get(sess.rule_id) if sess else None
                    if sess and rule:
                        try:
                            ts_eval = evaluate_test_session(traj, sess, rule)
                        except Exception as e:  # noqa: BLE001
                            logger.warning(
                                "evaluate_test_session failed for %s: %s",
                                sid, e,
                            )
                with _cb_lock:
                    if kind == "test":
                        # Refresh n_rules denom inside the lock; idempotent.
                        cs["n_rules"] = _ensure_eval_ctx(cid, ep_path)["n_rules"]
                    elif cs["n_rules"] == 0:
                        # First LS update — populate denom for cov ratio.
                        cs["n_rules"] = _ensure_eval_ctx(cid, ep_path)["n_rules"]
                    _update_state(cs, kind, traj, ts_eval=ts_eval)
                    _update_state(sweep_state, kind, traj, ts_eval=ts_eval)
                    progress.update(
                        tid, advance=1,
                        note=f"{sid} ({kind})",
                        **_cell_fields(cs),
                    )
                    # Sweep row reuses the same field set as per-cell.
                    progress.update(overall, **_cell_fields(sweep_state))
            return _cb

        if workers <= 1:
            for c in cells:
                if ABORT_EVENT.is_set():
                    progress.update(cell_tasks[c["cell_id"]], note="⊘ aborted")
                    progress.advance(overall)
                    continue
                cb = make_callback(c["cell_id"])
                try:
                    rows.append(_run_one_cell(**c, **cell_kwargs, on_session_done=cb))
                    progress.update(cell_tasks[c["cell_id"]], note="✓ done")
                except SweepAbort as exc:
                    # Per-cell isolation: abort this cell only; continue sweep.
                    logger.error(
                        "[cell-abort] cell %s raised SweepAbort: %s; "
                        "continuing sweep.",
                        c["cell_id"], exc,
                    )
                    rows.append({
                        "cell_id": c["cell_id"],
                        "ok": False,
                        "aborted": True,
                        "error": f"SweepAbort: {exc}",
                    })
                    progress.update(cell_tasks[c["cell_id"]], note="✗ ABORT")
                progress.advance(overall)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {}
                for c in cells:
                    cb = make_callback(c["cell_id"])
                    fut = ex.submit(
                        _run_one_cell, **c, **cell_kwargs, on_session_done=cb,
                    )
                    futs[fut] = c["cell_id"]
                for f in as_completed(futs):
                    cid = futs[f]
                    try:
                        row = f.result()
                        rows.append(row)
                        note = "✓ done" if row.get("ok") else f"✗ {row.get('error', 'fail')}"
                    except SweepAbort as exc:
                        # Per-cell isolation: this future failed but other
                        # pending and running futures continue uninterrupted.
                        logger.error(
                            "[cell-abort] cell %s raised SweepAbort: %s; "
                            "continuing sweep.",
                            cid, exc,
                        )
                        rows.append({
                            "cell_id": cid,
                            "ok": False,
                            "aborted": True,
                            "error": f"SweepAbort: {exc}",
                        })
                        note = "✗ ABORT"
                    progress.update(cell_tasks[cid], note=note)
                    progress.advance(overall)
    return rows


def _expected_session_count(ep_path: Path, variant: str) -> int:
    """How many sessions (LS + TS) this cell will run.

    Cheap: parse only the two count-bearing arrays from the episode JSON.
    Oracle variants skip Phase 1 entirely, so LS count is 0.
    """
    if not ep_path.exists():
        return 0
    try:
        data = read_json(ep_path)
    except (OSError, json.JSONDecodeError):
        return 0
    ts_count = len(data.get("test_sessions") or [])
    if variant in _TS_ONLY_VARIANTS:
        return ts_count
    ls_count = len(data.get("learning_sessions") or [])
    return ls_count + ts_count


def _build_cells_single(args) -> list[dict]:
    """Single-cell mode — explicit episode path."""
    ep = Path(args.episode).resolve()
    seed = args.seed
    if seed is None:
        seed = run_paths.seed_from_episode_id(ep.stem)
    hook_enabled = bool(getattr(args, "hook", False))
    cell_id = (
        f"{ep.stem}/"
        f"{run_paths.cell_segment(args.variant, args.model, args.memory_layer, seed, hook_enabled=hook_enabled)}"
    )
    return [{
        "ep_path": ep,
        "variant": args.variant,
        "model": args.model,
        "seed": seed,
        "cell_id": cell_id,
    }]


def _print_table(rows: list[dict]) -> None:
    """One line per cell — every metric the rich progress shows, expanded
    into reader-friendly columns, plus per-role token columns the
    rich bar omits.

    Status flags:
      ✓  cell ran cleanly
      !  ran but had non-clean terminations
      X  cell failed entirely (exception)

    Token columns: ↑in/↓out is per-call average for the *_avg group and
    cumulative total for the *_tot group, summed only over sessions that
    successfully exercised the role (agent runs every session; user_sim
    runs LS only; router runs on user-visible LS turns; classifier fires only
    when Router routes to a rule check).
    """
    if not rows:
        return
    print()
    # Per-cell columns: cell_id | ok | LS | TS | rule | hit | task | hook |
    # cov | acc | Tn | agent_avg | agent_tot | router_avg | router_tot |
    # user_avg | user_tot | cls_avg | cls_tot | sec. `rule` is
    # rule-routed send_to_user volume; `hit`
    # displays classifier hits over classified rule asks (+φ errors); `hook`
    # is `<problem>/<rescued> ×<avg>`.
    cols = [
        ("cell_id", 60),
        ("ok", 2),
        ("LS", 8),
        ("TS", 8),
        ("rule", 5),
        ("hit", 7),
        ("task", 5),
        ("hook", 14),
        ("cov", 7),
        ("acc", 12),
        ("Tn", 5),
        ("agent_avg", 14),
        ("agent_tot", 14),
        ("router_avg", 14),
        ("router_tot", 14),
        ("user_avg", 14),
        ("user_tot", 14),
        ("cls_avg", 14),
        ("cls_tot", 14),
        ("sec", 6),
    ]
    header = "  ".join(f"{name:>{w}}" if name not in ("cell_id",) else f"{name:<{w}}"
                       for name, w in cols)
    print(header)
    print("-" * len(header))

    sweep = _new_state()
    for r in rows:
        if not r["ok"]:
            status = "X"
        elif r.get("non_clean_terminations"):
            status = "!"
        else:
            status = "✓"

        if r["ok"]:
            n_done = (r.get("n_learning", 0) or 0) + (r.get("n_test", 0) or 0)
            ls_term = f"✓{r.get('ls_clean', 0)}/✗{r.get('ls_bad', 0)}"
            ts_term = f"✓{r.get('ts_clean', 0)}/✗{r.get('ts_bad', 0)}"
            # Reuse _fmt_atr so table + rich agree on the empty-state shape.
            rule_count = r.get("send_routed_rule", 0) or 0
            rule_str = str(rule_count) if rule_count else "—"
            hit_str = _fmt_atr({
                "send_routed_rule": r.get("send_routed_rule", 0),
                "cls_hit": r.get("cls_hit", 0),
                "cls_miss": r.get("cls_miss", 0),
                "cls_err": r.get("cls_err", 0),
            })
            task_count = r.get("send_routed_task", 0) or 0
            task_str = str(task_count) if task_count else "—"
            hook_str = _fmt_hook({
                "hook_appended": r.get("hook_appended", 0),
                "hook_rescued": r.get("hook_rescued", 0),
                "off_protocol": r.get("off_protocol", 0),
            })
            cov = f"{r.get('rules_covered', 0)}/{r.get('n_rules', 0)}"
            ts_tot = r.get("ts_total", 0) or 0
            if ts_tot:
                pct = r.get("ts_pass", 0) * 100 / ts_tot
                acc = f"{r.get('ts_pass', 0)}/{ts_tot} {pct:.0f}%"
            else:
                acc = "—"
            tn = f"{r.get('avg_turns', 0):.1f}"

            ls_n = r.get("n_learning", 0) or 0
            ts_n = r.get("n_test", 0) or 0
            # Per-CALL denominators (not per-session). Each role's call count
            # comes from the trajectory's `token_usage[role][calls]` field
            # accumulated by `_update_state` — these are the only correct
            # denominators for per-call averages, since one session can
            # produce many LLM calls per role.
            agent_n = r.get("tok_agent_calls", 0) or 0
            router_n = r.get("tok_router_calls", 0) or 0
            user_n = r.get("tok_user_calls", 0) or 0
            cls_n = r.get("tok_cls_calls", 0) or 0

            def _avg_pair(in_tok: int, out_tok: int, n: int) -> str:
                if n == 0:
                    return "—"
                return _fmt_tok_pair(in_tok // n, out_tok // n)

            agent_avg = _avg_pair(r.get("tok_agent_in", 0), r.get("tok_agent_out", 0), agent_n)
            agent_tot = _fmt_tok_pair(r.get("tok_agent_in", 0), r.get("tok_agent_out", 0))
            router_avg = _avg_pair(r.get("tok_router_in", 0), r.get("tok_router_out", 0), router_n)
            router_tot = _fmt_tok_pair(r.get("tok_router_in", 0), r.get("tok_router_out", 0))
            user_avg = _avg_pair(r.get("tok_user_in", 0), r.get("tok_user_out", 0), user_n)
            user_tot = _fmt_tok_pair(r.get("tok_user_in", 0), r.get("tok_user_out", 0))
            cls_avg = _avg_pair(r.get("tok_cls_in", 0), r.get("tok_cls_out", 0), cls_n)
            cls_tot = _fmt_tok_pair(r.get("tok_cls_in", 0), r.get("tok_cls_out", 0))

            # Roll into sweep totals — only successful cells contribute.
            sweep["send_routed_rule"] += r.get("send_routed_rule", 0)
            sweep["cls_hit"] += r.get("cls_hit", 0)
            sweep["cls_miss"] += r.get("cls_miss", 0)
            sweep["cls_err"] += r.get("cls_err", 0)
            sweep["send_routed_task"] += r.get("send_routed_task", 0)
            sweep["off_protocol"] += r.get("off_protocol", 0)
            sweep["hook_appended"] += r.get("hook_appended", 0)
            sweep["hook_rescued"] += r.get("hook_rescued", 0)
            sweep["ls_total"] += ls_n
            sweep["ts_total"] += ts_n
            sweep["ls_with_routed_rule"] += r.get("ls_with_routed_rule", 0)
            sweep["ls_clean"] += r.get("ls_clean", 0)
            sweep["ls_bad"] += r.get("ls_bad", 0)
            sweep["ts_clean"] += r.get("ts_clean", 0)
            sweep["ts_bad"] += r.get("ts_bad", 0)
            sweep["ts_pass"] += r.get("ts_pass", 0)
            sweep["tok_agent_in"] += r.get("tok_agent_in", 0)
            sweep["tok_agent_out"] += r.get("tok_agent_out", 0)
            sweep["tok_agent_calls"] += r.get("tok_agent_calls", 0)
            sweep["tok_router_in"] += r.get("tok_router_in", 0)
            sweep["tok_router_out"] += r.get("tok_router_out", 0)
            sweep["tok_router_calls"] += r.get("tok_router_calls", 0)
            sweep["tok_user_in"] += r.get("tok_user_in", 0)
            sweep["tok_user_out"] += r.get("tok_user_out", 0)
            sweep["tok_user_calls"] += r.get("tok_user_calls", 0)
            sweep["tok_cls_in"] += r.get("tok_cls_in", 0)
            sweep["tok_cls_out"] += r.get("tok_cls_out", 0)
            sweep["tok_cls_calls"] += r.get("tok_cls_calls", 0)
            sweep["n_done"] += n_done
            sweep["turns_total"] += int(round(
                (r.get("avg_turns", 0) or 0) * n_done
            ))
        else:
            ls_term = ts_term = rule_str = hit_str = task_str = hook_str = cov = acc = tn = "-"
            agent_avg = agent_tot = router_avg = router_tot = "-"
            user_avg = user_tot = cls_avg = cls_tot = "-"

        # Aborted/failed cells skip the `_run_one_cell` finally block that
        # stamps duration_sec, so fall back to "-" when the key is absent.
        duration_str = (
            f"{r['duration_sec']}" if "duration_sec" in r else "-"
        )
        vals = [
            r["cell_id"], status, ls_term, ts_term, rule_str, hit_str, task_str,
            hook_str,
            cov, acc, tn,
            agent_avg, agent_tot, router_avg, router_tot,
            user_avg, user_tot, cls_avg, cls_tot,
            duration_str,
        ]
        line_parts = []
        for (name, w), v in zip(cols, vals):
            if name == "cell_id":
                line_parts.append(f"{str(v):<{w}}")
            else:
                line_parts.append(f"{str(v):>{w}}")
        print("  ".join(line_parts))

    if sweep["n_done"]:
        print("-" * len(header))
        agent_n = sweep["tok_agent_calls"]
        router_n = sweep["tok_router_calls"]
        user_n = sweep["tok_user_calls"]
        cls_n = sweep["tok_cls_calls"]

        def _avg(in_tok: int, out_tok: int, n: int) -> str:
            if n == 0:
                return "—"
            return _fmt_tok_pair(in_tok // n, out_tok // n)

        ts_pct = sweep["ts_pass"] * 100 / sweep["ts_total"] if sweep["ts_total"] else 0
        print(
            f"sweep totals  "
            f"sessions={sweep['n_done']}  "
            f"LS=✓{sweep['ls_clean']}/✗{sweep['ls_bad']}  "
            f"TS=✓{sweep['ts_clean']}/✗{sweep['ts_bad']}  "
            f"hit={_fmt_atr(sweep)}  "
            f"task={sweep['send_routed_task']}  "
            f"hook={_fmt_hook(sweep)}  "
            f"acc={sweep['ts_pass']}/{sweep['ts_total']} ({ts_pct:.0f}%)  "
            f"Tn={sweep['turns_total'] / sweep['n_done']:.1f}"
        )
        print(
            f"              "
            f"agent_avg={_avg(sweep['tok_agent_in'], sweep['tok_agent_out'], agent_n)}  "
            f"agent_tot={_fmt_tok_pair(sweep['tok_agent_in'], sweep['tok_agent_out'])}  "
            f"router_avg={_avg(sweep['tok_router_in'], sweep['tok_router_out'], router_n)}  "
            f"router_tot={_fmt_tok_pair(sweep['tok_router_in'], sweep['tok_router_out'])}  "
            f"user_avg={_avg(sweep['tok_user_in'], sweep['tok_user_out'], user_n)}  "
            f"user_tot={_fmt_tok_pair(sweep['tok_user_in'], sweep['tok_user_out'])}  "
            f"cls_avg={_avg(sweep['tok_cls_in'], sweep['tok_cls_out'], cls_n)}  "
            f"cls_tot={_fmt_tok_pair(sweep['tok_cls_in'], sweep['tok_cls_out'])}"
        )


def _print_variant_aggregate(rows: list[dict]) -> None:
    """Per-variant aggregate table (research-signal table).

    Groups successful runner cells by variant and reports only metrics that
    runner rows actually own: TS pass rate, rule-routed ask volume,
    classifier hit rate over classified asks, task-routed send_to_user volume,
    off-protocol asks, classifier errors, and always_ask compliance.

    Coverage×hit×pass cells live in evaluator metrics; runner does not load
    the full evaluator aggregate here. Keeping those out of this table avoids
    printing empty result columns before evaluator has run.

    Cells with errors / explicit failures are excluded from the aggregate
    (keeps mean/std interpretable on the surviving subset).

    Columns:
      variant | n_cells | acc | rule | hit_rate | task | hook | cls_err |
      forced_compl

    `hook` is the mean±std of `<problem>/<rescued> ×<avg>` per
    cell (rendered as `mean_problem/mean_rescued ×mean_avg`).
    """
    if not rows:
        return
    from statistics import mean, stdev

    # Group by variant; only successful cells contribute.
    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        if not r.get("ok"):
            continue
        v = r.get("variant") or "unknown"
        by_variant.setdefault(v, []).append(r)

    if not by_variant:
        return

    print()
    print("─── per-variant aggregate (single seed=0; std across persona×tag×model×layer) ───")
    cols = [
        ("variant", 12),
        ("n_cells", 7),
        ("acc", 12),
        ("rule", 11),
        ("hit_rate", 12),
        ("task", 11),
        ("hook", 18),
        ("cls_err", 9),
        ("forced_c%", 10),
    ]
    header = "  ".join(f"{n:<{w}}" if i == 0 else f"{n:>{w}}"
                       for i, (n, w) in enumerate(cols))
    print(header)
    print("-" * len(header))

    def _fmt_mean_std(xs: list[float], pct: bool = False, places: int = 1) -> str:
        if not xs:
            return "—"
        xs = [float(x) for x in xs]
        m = mean(xs)
        s = stdev(xs) if len(xs) >= 2 else 0.0
        if pct:
            return f"{m*100:.{places}f}±{s*100:.{places}f}"
        return f"{m:.{places}f}±{s:.{places}f}"

    # Variant-stable display order matching the send-to-user architecture set.
    order = list(VARIANTS)
    seen = list(by_variant.keys())
    sorted_variants = [v for v in order if v in by_variant] + [v for v in seen if v not in order]

    for v in sorted_variants:
        cells_for_v = by_variant[v]
        n_cells = len(cells_for_v)

        # Per-cell vectors
        accs = [
            (r.get("ts_pass", 0) / r["ts_total"]) for r in cells_for_v
            if r.get("ts_total")
        ]
        rule_routes = [r.get("send_routed_rule", 0) for r in cells_for_v]
        rule_rates = [
            r.get("cls_hit", 0) / (
                r.get("cls_classified", r.get("cls_hit", 0) + r.get("cls_miss", 0))
            )
            for r in cells_for_v
            if r.get("cls_classified", r.get("cls_hit", 0) + r.get("cls_miss", 0))
        ]
        task_routes = [r.get("send_routed_task", 0) for r in cells_for_v]
        cls_errs = [r.get("cls_err", 0) for r in cells_for_v]
        # hook stats per cell: <problem>/<rescued> ×<avg>.
        # mean over cells (std would compound oddly across three numbers
        # so we collapse to a single rendered string from the per-cell
        # mean of each component).
        hook_problems = [r.get("off_protocol", 0) for r in cells_for_v]
        hook_rescues = [r.get("hook_rescued", 0) for r in cells_for_v]
        hook_fires = [r.get("hook_appended", 0) for r in cells_for_v]
        if any(hook_fires):
            mp = mean(hook_problems) if hook_problems else 0
            mr = mean(hook_rescues) if hook_rescues else 0
            mf = mean(hook_fires) if hook_fires else 0
            mavg = mf / mp if mp else 0.0
            hook_str_v = f"{mp:.1f}/{mr:.1f} ×{mavg:.1f}"
        else:
            hook_str_v = "—"

        # forced_rule_ask_compliance is meaningful only for always_ask:
        # percent of LS sessions with at least one rule-routed send_to_user.
        # This must not degrade into "percent of cells with any rule ask",
        # which hides isolated compliance failures inside a large cell.
        if v == "always_ask":
            ls_total = sum(r.get("n_learning", 0) or 0 for r in cells_for_v)
            ls_with_rule = sum(
                r.get("ls_with_routed_rule", 0) or 0
                for r in cells_for_v
            )
            forced_str = f"{ls_with_rule*100/ls_total:.1f}" if ls_total else "—"
        else:
            forced_str = "—"

        vals = [
            v,
            str(n_cells),
            _fmt_mean_std(accs, pct=True),
            _fmt_mean_std(rule_routes, places=2),
            _fmt_mean_std(rule_rates, pct=True) if rule_rates else "—",
            _fmt_mean_std(task_routes, places=2),
            hook_str_v,
            _fmt_mean_std(cls_errs, places=2),
            forced_str,
        ]

        line_parts = []
        for i, ((name, w), v_str) in enumerate(zip(cols, vals)):
            if i == 0:
                line_parts.append(f"{str(v_str):<{w}}")
            else:
                line_parts.append(f"{str(v_str):>{w}}")
        print("  ".join(line_parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runner pipeline — execute trajectories. "
                    "Sweep mode: all of --personas/--seeds/--variants. "
                    "Single-cell mode: --episode + --variant.",
    )
    # ── Sweep dims (all required together for sweep mode) ───────────────
    parser.add_argument("--personas", nargs="+",
                        help="Persona IDs (sweep mode)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Episode-composition seed(s) selecting which built "
                             "episodes to run; the released benchmark is constructed "
                             "at seed 0 (baked into episode filenames). Not a "
                             "result-variance axis.")
    parser.add_argument(
        "--variants", nargs="+", choices=_VARIANTS,
        default=list(_VARIANTS),
    )
    parser.add_argument("--models", nargs="+", default=[None],
                        help=(
                            "Agent model(s). Router is fixed to Gemini; "
                            "user_sim and classifier are always GPT."
                        ))
    # ── Single-cell mode ────────────────────────────────────────────────
    parser.add_argument("--episode", default=None,
                        help="Single-cell mode: path to episode JSON. "
                             "When set, --personas are "
                             "ignored; --variant + (optional) --model/--seed "
                             "apply.")
    parser.add_argument("--variant", default=None, choices=_VARIANTS,
                        help="Single-cell mode: agent variant.")
    parser.add_argument("--model", default=None,
                        help="Single-cell mode: agent model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single-cell mode: LLM seed. If omitted and "
                             "--episode stem contains _seedNNN, that seed is "
                             "inferred so canonical single-cell reruns land in "
                             "the same cell as sweep mode.")
    # ── Run params ──────────────────────────────────────────────────────
    parser.add_argument("--memory-layer", default="context",
                        choices=_LAYER_CHOICES,
                        help="Cross-session layer kind: raw/native/context all map to "
                             "the raw-transcript ContextLayer.")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-session wall-clock timeout (seconds).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel cells (>1 uses ThreadPool; LLM concurrency "
                             "still bounded by lib/llm.py semaphores).")
    parser.add_argument("--ts-workers", type=int, default=1,
                        help="Per-cell concurrency for test sessions (Phase 2). "
                             "Each TS reads a frozen snapshot with no shared "
                             "mutable state, so parallel execution is safe. Defaults "
                             "to 1 (serial); raise to run test sessions in parallel.")
    parser.add_argument("--reasoning", default="on",
                        choices=["on", "off"],
                        help="Binary reasoning switch for the agent-under-test "
                             "LLM: 'on' enables the model's reasoning/thinking mode "
                             "where the provider exposes one, 'off' disables it. "
                             "Router, user_sim, and classifier always use the "
                             "provider default. Stamped into the cell manifest, so "
                             "flipping this requires --fresh or a new cell directory.")
    parser.add_argument("--hook", action="store_true",
                        help="Enable scaffolding hook (LS only). "
                             "On every off-protocol assistant turn, inject a "
                             "<scaffolding_note> user message asking the agent "
                             "to re-emit via send_to_user. No per-session cap; "
                             "runaway loops are bounded by the existing "
                             "non-ask-text budget and max_steps. Stamped into "
                             "manifest as hook_enabled and into the cell path "
                             "(__hook segment), so hook-on / hook-off are "
                             "distinct cells.")
    parser.add_argument("--smoke", action="store_true",
                        help="Mark this run as smoke / debug / validation "
                             ". Stamped into cell_manifest and the "
                             "cell directory path (__smoke segment) so smoke "
                             "runs never collide with main-table cells, even "
                             "when every other dimension matches.")
    # ── Trace + resume controls ─────────────────────────────────────────
    parser.add_argument("--traces", action=argparse.BooleanOptionalAction, default=True,
                        help="Write pretty-trace .txt files (default: on).")
    parser.add_argument("--fresh", action="store_true",
                        help="Wipe each cell's runs_dir before launching "
                             "(per-cell scoped). Default: idempotent resume.")
    parser.add_argument("--allow-stale-resume", action="store_true",
                        help="Bypass cell manifest mismatch (debugging only).")
    # ── Output ──────────────────────────────────────────────────────────
    parser.add_argument("--outputs-root", default=None,
                        help="Override the output root for ALL artifacts "
                             "(cell data + summary). Cell layout becomes "
                             "<root>/<persona>/<short_ep>/<cell>/... and the "
                             "default summary lands in <root>/_summary/. Use "
                             "this for reported-results runs, prompt "
                             "ablations, or any sweep that must not collide "
                             "with the canonical outputs/ tree. Overrides "
                             "the ATR_OUTPUTS_ROOT env var when both are set.")
    parser.add_argument("--summary-dir", default=None,
                        help="Sweep summary output dir (default: "
                             "<outputs-root>/_summary/runner_, i.e. follows "
                             "--outputs-root if set, else outputs/_summary/).")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable rich progress bars; emit plain log lines "
                             "to stderr (useful in CI / piped runs).")

    args = parser.parse_args()

    # Apply --outputs-root before any cell paths are resolved. This must
    # come before _build_cells_* so runs_dir() / episode discovery /
    # manifest paths all see the redirected root from the very first call.
    if args.outputs_root:
        run_paths.set_runs_root(args.outputs_root)

    # Mode selection.
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

    logger.info(
        "Runner pipeline (%s): %d cell(s)",
        mode, len(cells),
    )

    rows: list[dict] = []
    kw = dict(
        memory_layer=args.memory_layer,
        max_steps=args.max_steps, timeout=args.timeout,
        traces=args.traces, fresh=args.fresh,
        allow_stale_resume=args.allow_stale_resume,
        agent_reasoning_effort=args.reasoning,
        ts_workers=args.ts_workers,
        hook_enabled=args.hook,
        smoke=args.smoke,
    )
    rows = _run_with_progress(cells, kw, workers=args.workers, no_progress=args.no_progress)

    rows.sort(key=lambda r: r["cell_id"])
    _print_table(rows)
    _print_variant_aggregate(rows)

    summary_dir = (
        Path(args.summary_dir).resolve() if args.summary_dir
        else run_paths.get_runs_root() / "_summary"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = summary_dir / f"runner_{ts}.json"
    # surface incomplete / aborted cells in the runner sweep
    # summary so they are visible at a glance, not buried in evaluator
    # output. A cell is incomplete if it didn't run cleanly (ok=False)
    # OR was aborted by SweepAbort (aborted=True). Counted both ways so
    # the user immediately sees how many cells need attention.
    incomplete_cells = [
        {
            "cell_id": r.get("cell_id"),
            "ok": r.get("ok", False),
            "aborted": r.get("aborted", False),
            "error": r.get("error"),
        }
        for r in rows
        if (not r.get("ok")) or r.get("aborted")
    ]
    aborted_count = sum(1 for r in rows if r.get("aborted"))
    summary_payload = {
        "stage": "runner",
        "timestamp": ts,
        "mode": mode,
        "args": {k: v for k, v in vars(args).items()},
        "rows": rows,
        # top-level incomplete summary.
        "n_cells_total": len(rows),
        "n_cells_ok": sum(1 for r in rows if r.get("ok")),
        "n_cells_aborted": aborted_count,
        "n_cells_incomplete": len(incomplete_cells),
        "incomplete_cells": incomplete_cells,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    print(f"\nrunner summary → {summary_path}")
    if incomplete_cells:
        print(
            f"\n[summary] {len(incomplete_cells)}/{len(rows)} cells are "
            f"incomplete (aborted: {aborted_count}). "
            f"They will not enter main-table aggregation."
        )
    print("Next: uv run python -m evaluator.pipeline ... (same sweep dims)")

    n_fail = sum(1 for r in rows if not r["ok"])
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
