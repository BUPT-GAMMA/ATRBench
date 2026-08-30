"""Run a full episode: load episode JSON → execute sessions → save trajectories."""
from __future__ import annotations

import argparse
import functools
import json
import sys
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluator"))
from schemas import Episode, SessionTrajectory
from orchestrator import run_session
from memory import CrossSessionLayer
from runner.environment.base import PersonaProfile
import paths as run_paths  # runner/paths.py — runner dir is on sys.path
import manifest as manifest_module  # runner/manifest.py — same path setup
from _constants import (
    LAYER_CHOICES,
    ORACLE_VARIANTS,
    TS_ONLY_VARIANTS,
    VARIANTS,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import lib.llm as _llm


def _build_persona(episode: Episode) -> PersonaProfile:
    """Derive a minimal PersonaProfile from raw_persona + structured_persona.

    Nemotron track no longer persists persona_profile.json; the handful of
    tool-internal identity fields (shipping_address / payment_method /
    contact) are synthesized deterministically from raw_persona.persona_id so
    tool returns stay reproducible. Evaluator doesn't compare these fields,
    so their literal values don't matter for benchmark correctness.
    """
    pid = episode.raw_persona.persona_id
    demo = (episode.structured_persona or {}).get("demographics") or {}
    city = (demo.get("city") or "").strip()
    state = (demo.get("state") or "").strip()
    zipcode = (demo.get("zipcode") or "").strip()

    home_city = f"{city}, {state}".strip(", ") if city or state else ""
    shipping = f"{city}, {state} {zipcode}".strip(", ").strip() if city else ""

    return PersonaProfile(
        persona_id=pid,
        narrative=episode.raw_persona.narrative,
        home_city=home_city,
        home_zone="suburban",
        default_shipping_address=shipping,
        default_payment_method=f"CARD_{pid[:6].upper()}",
        default_contact=f"{pid}@mail.example",
    )

logger = logging.getLogger(__name__)
# Logging is configured per-run in `main()` so we can attach a file handler
# pointing at the run's `run.log`. Importers that call `run_episode()` directly
# should configure logging themselves (or pass `out_dir` and add a handler).


@functools.lru_cache(maxsize=128)
def _load_episode_cached(resolved_path_str: str) -> Episode:
    with open(resolved_path_str) as f:
        ep_data = json.load(f)
    return Episode(**ep_data)


def load_episode(episode_path: Path) -> Episode:
    """Load episode JSON into an Episode object.

    Cached by resolved path for the process lifetime — runner+evaluator
    pipelines parse the same episode JSON multiple times per cell. Callers
    must not mutate the returned Episode; tests that rewrite an episode
    file in place must call `_load_episode_cached.cache_clear()`.
    """
    return _load_episode_cached(str(Path(episode_path).resolve()))


def run_episode(
    episode_path: str,
    agent_variant: str,
    memory_layer: str = "context",
    model: str | None = None,
    max_steps: int = 20,
    timeout: float | None = None,
    out_dir: Path | None = None,
    traces_dir: Path | None = None,
    seed: int | None = None,
    allow_stale_resume: bool = False,
    on_session_done: Callable[[str, str, SessionTrajectory], None] | None = None,
    agent_reasoning_effort: str | None = None,
    ts_workers: int = 1,
    hook_enabled: bool = False,
    smoke: bool = False,
) -> list[SessionTrajectory]:
    """Execute all sessions in strict two-phase order, with per-session checkpointing.

    Phase 1 (learning): sessions run serially; `layer.after_session(traj)` is
        invoked after each one so the layer updates its store.
    Phase 2 (test): the layer is frozen into a snapshot; every test
        session reads the same snapshot (parallel semantics) and never
        writes back.

    Resume protocol:
      - Each session's trajectory is written immediately on completion to
        `<out_dir>/<session_id>.json` (variant is implicit in the parent
        directory name, so file names don't carry it).
      - On startup, if a trajectory file already exists for a session, it is
        loaded and replayed into the cross-session layer (Phase 1) or simply
        re-collected (Phase 2). This makes the runner idempotent — re-running
        a partially completed episode skips finished sessions and continues
        from the first un-saved one.
      - Replay re-feeds the saved trajectory through `layer.after_session()`,
        which for `ContextLayer` is a cheap, deterministic transcript dump
        (no LLM call).
      - To force a clean re-run, delete the run's `trajectories/` directory.

    Args:
        episode_path: Path to episode JSON.
        agent_variant: default / atr / always_ask / oracle_target / oracle_full
            (send-to-user architecture).
        memory_layer: Cross-session layer kind — `raw` / `context` / `native`.
        out_dir: Directory for per-session trajectory checkpoints. Defaults
            to the canonical runs/ layout
            (`outputs/<persona>/<short_episode_id>/<cell_segment>/trajectories/`).
    """
    _model = model or _llm.GPT

    # Agent reasoning is a binary experimental condition ("on" / "off").
    # It is stamped into the manifest so a later run with the other condition
    # fails the gating check instead of silently mixing trajectories.
    _agent_reasoning_effort = agent_reasoning_effort

    ep_path = Path(episode_path).resolve()
    episode = load_episode(ep_path)

    persona = _build_persona(episode)
    logger.info("Loaded persona %s", persona.persona_id)

    # Hard fail combinations that produce silently-broken oracle runs. Oracle
    # variants need to inject rule canonical_answers into the cross-session
    # layer, and only ContextLayer implements `inject_prior_user_statement`.
    # Running oracle on memory layers used to silently drop the rules and
    # produce a non-oracle "oracle" — uppermost-bound numbers that are wrong.
    if agent_variant in ORACLE_VARIANTS and memory_layer not in (
        "context", "raw", "native",
    ):
        raise ValueError(
            f"agent_variant={agent_variant!r} requires memory_layer in "
            f"{{context, raw, native}} (only ContextLayer implements "
            f"inject_prior_user_statement). Got memory_layer={memory_layer!r}. "
            f"Either change the layer or use a non-oracle variant."
        )

    if out_dir is None:
        out_dir = run_paths.trajectories_dir(
            persona.persona_id, episode.episode_id, agent_variant,
            model=_model, memory_layer=memory_layer, seed=seed,
            hook_enabled=hook_enabled, smoke=smoke,
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cell manifest gate: refuse to resume if any sweep-relevant dimension
    # (variant / model / memory_layer / seed / hook_enabled / prompt_hashes
    # / code_hash) has changed since this cell was first written. Without
    # this, switching any of those between runs would silently fold new
    # results into stale trajectories.
    cell_dir = out_dir.parent  # trajectories_dir is <cell>/trajectories
    manifest_file = cell_dir / "cell_manifest.json"
    proposed_manifest = manifest_module.build_manifest(
        persona_id=persona.persona_id,
        episode_id=episode.episode_id,
        variant=agent_variant,
        model=_model,
        memory_layer=memory_layer,
        seed=seed,
        max_steps=max_steps,
        timeout=timeout,
        agent_reasoning_effort=_agent_reasoning_effort,
        hook_enabled=hook_enabled,
        smoke=smoke,
    )
    manifest_module.gate_or_init(
        manifest_file,
        proposed=proposed_manifest,
        allow_stale=allow_stale_resume,
    )

    # Pretty-trace streaming: when traces_dir is set, every fresh session
    # opens a SessionStreamWriter so its conversation lands in
    # <traces_dir>/<sid>.txt as the dialogue unfolds (tail -f friendly).
    # Resumed sessions skip the writer entirely — re-rendering them from
    # the saved trajectory at cell-end is cheaper.
    _trace_writer_factory = None
    if traces_dir is not None:
        traces_dir.mkdir(parents=True, exist_ok=True)
        from tools.pretty_trace import SessionStreamWriter as _SSW

        def _trace_writer_factory(session, agent_variant):  # noqa: ANN001
            meta = {
                "session_id": session.session_id,
                "session_type": session.session_type,
                "agent_variant": agent_variant,
                # LS uses reason_for_call (vague intent for user_sim);
                # TestSession uses instruction (self-contained, user offline).
                "instruction": (
                    getattr(session, "instruction", None)
                    or getattr(session, "reason_for_call", None)
                ),
                "rule_id": getattr(session, "rule_id", None),
            }
            return _SSW(traces_dir / f"{session.session_id}.txt", meta)

    # Episode-level rule pool for the ATR classifier (learning phase).
    episode_rules = episode.rules

    # Oracle pre-flight validation. Both variants need canonical_answers;
    # oracle_target validates per-test rule lookup, oracle_full validates
    # every rule in the episode (since all are injected).
    if agent_variant == "oracle_target":
        rules_by_id = {r.rule_id: r for r in episode_rules}
        for session in episode.test_sessions:
            target_rule = rules_by_id.get(session.rule_id)
            if target_rule is None:
                raise RuntimeError(
                    f"Oracle (Targeted): test {session.session_id} references "
                    f"rule_id={getattr(session, 'rule_id', '<missing>')!r} "
                    f"which is not in episode_rules. Fix the episode JSON."
                )
            if not target_rule.canonical_answer:
                raise RuntimeError(
                    f"Oracle (Targeted): rule {target_rule.rule_id} "
                    f"(referenced by test {session.session_id}) has empty "
                    f"canonical_answer. Fix the episode JSON."
                )
    elif agent_variant == "oracle_full":
        # Inject ALL rules' canonical_answers (information upper bound).
        # Every rule must have a canonical_answer.
        for rule in episode_rules:
            if not rule.canonical_answer:
                raise RuntimeError(
                    f"Oracle (Full): rule {rule.rule_id} has empty "
                    f"canonical_answer; cannot seed full-information "
                    f"oracle. Fix the episode JSON."
                )

    # TS-only variants (oracle_*) skip the learning phase entirely and
    # inject rule canonical_answers at TS time (the application upper bound).
    learning_sessions = sorted(episode.learning_sessions, key=lambda s: s.day_offset)
    if agent_variant in TS_ONLY_VARIANTS and learning_sessions:
        logger.info(
            "TS-only variant (%s): skipping %d learning sessions.",
            agent_variant, len(learning_sessions),
        )
        learning_sessions = []

    layer = CrossSessionLayer.create(memory_layer, model=_model)

    # Oracle (Targeted) defers per-test injection to the Phase 2 loop below.

    trajectories: list[SessionTrajectory] = []

    def _path_for(sid: str) -> Path:
        # variant is implicit in the parent dir; file name carries only sid
        return out_dir / f"{sid}.json"

    # Termination states whose trajectories should NOT count as "resumed" —
    # `_try_load` returns None for these so the next run re-attempts the
    # session. The on-disk trajectory is preserved (overwritten on rerun)
    # so the user can still inspect what happened.
    #
    # under the unified sweep-abort regime, new code paths no
    # longer WRITE termination_reason ∈ {error, sim_error,
    # error_after_retry} — LLM retry exhaustion raises SweepAbort and no
    # partial trajectory is persisted. These reasons remain here ONLY
    # for compatibility with on-disk trajectories that contain them;
    # new sweeps will not produce them.
    #
    # max_steps / task_no_progress / timeout are NOT here: they reflect
    # the agent's own struggle, retrying won't change the outcome.
    _RESUME_RERUN_TERMINATIONS = frozenset({
        "error", "error_after_retry", "sim_error",
    })

    def _try_load(sid: str) -> SessionTrajectory | None:
        p = _path_for(sid)
        if not p.exists():
            return None
        try:
            with open(p) as f:
                traj = SessionTrajectory(**json.load(f))
        except Exception as e:
            logger.warning("Existing %s failed to parse (%s); will re-run.", p, e)
            return None
        if traj.termination_reason in _RESUME_RERUN_TERMINATIONS:
            logger.info(
                "[%s] on-disk trajectory has term=%s; treating as not-resumable, "
                "will re-run (existing file preserved until overwrite).",
                sid, traj.termination_reason,
            )
            return None
        return traj

    def _save(traj: SessionTrajectory) -> None:
        # Self-heal if parent dir got removed externally between session saves
        # (e.g. user cleaning runs/ while a run is still active).
        out_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to .tmp sibling then os.replace.
        # SIGINT mid-write leaves a .tmp orphan but never a truncated target.
        target = _path_for(traj.session_id)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.write(traj.model_dump_json(indent=2))
        tmp.replace(target)

    # ---- Phase 1: learning sessions (serial, writes back to layer) ----
    # Resume note: replaying via layer.after_session is byte-deterministic
    # for ContextLayer (transcript dump -> string), so a partial resume
    # reproduces the same snapshot as an uninterrupted run.
    from _abort import ABORT_EVENT
    n_resumed = 0
    for session in learning_sessions:
        # at LS session boundary, honor sweep abort signal.
        # Already-in-flight LS runs to completion before this check fires
        # (this check is at the *start* of each new session).
        if ABORT_EVENT.is_set():
            logger.warning(
                "[sweep-abort] skipping remaining LS sessions in cell "
                "(ABORT_EVENT set).",
            )
            break
        sid = session.session_id
        traj = _try_load(sid)
        if traj is not None:
            # Replay into layer so cross-session state matches a fresh run
            # (a deterministic transcript dump for ContextLayer).
            layer.before_session(sid)
            if traj.termination_reason != "error":
                layer.after_session(traj)
            trajectories.append(traj)
            n_resumed += 1
            logger.info(
                "[learning %s] resumed from disk (steps=%d, term=%s)",
                sid, traj.step_count, traj.termination_reason,
            )
            if on_session_done:
                on_session_done(sid, "learning", traj)
            continue
        logger.info(
            "Learning session %s (variant=%s, layer=%s)",
            sid, agent_variant, memory_layer,
        )
        layer.before_session(sid)
        _writer = _trace_writer_factory(session, agent_variant) if _trace_writer_factory else None
        traj = run_session(
            session, agent_variant, layer, persona,
            model=_model,
            max_steps=max_steps,
            timeout=timeout,
            seed=seed,
            episode_rules=episode_rules,
            on_turn=(_writer.on_event if _writer else None),
            agent_reasoning_effort=_agent_reasoning_effort,
            hook_enabled=hook_enabled,
        )
        if _writer:
            _writer.on_finish(traj.model_dump())
        trajectories.append(traj)
        if traj.termination_reason != "error":
            layer.after_session(traj)
        traj.memory_snapshot = layer.get_snapshot()
        _save(traj)  # checkpoint immediately (preserved for analysis)
        logger.info(
            "  → done: steps=%d, termination=%s, duration=%.1fs (saved)",
            traj.step_count, traj.termination_reason, traj.duration_seconds,
        )
        if on_session_done:
            on_session_done(sid, "learning", traj)
        # LS error abort: TS phase depends on a clean memory layer
        # accumulation, so we cannot proceed past an LS that died mid-
        # turn. Raise to abort run_episode; the partial trajectory is
        # already on disk for inspection. Resume re-runs this LS because
        # `_try_load` filters out `_RESUME_RERUN_TERMINATIONS`.
        if traj.termination_reason in _RESUME_RERUN_TERMINATIONS:
            raise RuntimeError(
                f"LS {sid} ended with termination_reason={traj.termination_reason!r}; "
                f"aborting episode. Partial trajectory saved to {_path_for(sid)}. "
                f"Resume to re-attempt this LS once the underlying issue is fixed."
            )
    if n_resumed:
        logger.info(
            "Phase 1: %d / %d learning sessions resumed from disk",
            n_resumed, len(learning_sessions),
        )

    # ---- Phase 1 → Phase 2 boundary: freeze snapshot ----
    snapshot = layer.freeze()
    probe = snapshot.get_context_string()
    logger.info(
        "Frozen memory snapshot: %d entries, fixed_context=%s",
        len(snapshot.get_snapshot()),
        "present" if probe else "absent (query-driven or empty)",
    )

    # ---- Oracle (Full) snapshot: inject ALL rules' canonical_answers ----
    # Built once and shared across every TS in the cell. Pre-flight
    # validation already ensured every rule.canonical_answer is non-empty.
    oracle_full_snapshot = None
    if agent_variant == "oracle_full":
        _full_layer = CrossSessionLayer.create(memory_layer, model=_model)
        if not hasattr(_full_layer, "inject_prior_user_statement"):
            raise RuntimeError(
                f"Oracle (Full): layer {type(_full_layer).__name__} does not "
                f"implement inject_prior_user_statement. Use --memory-layer "
                f"context."
            )
        for rule in episode_rules:
            _full_layer.inject_prior_user_statement(rule.canonical_answer)
        oracle_full_snapshot = _full_layer.freeze()
        logger.info(
            "Oracle (Full): injected %d rules' canonical_answers into "
            "shared snapshot.",
            len(episode_rules),
        )

    # ---- Phase 2: test sessions (user offline, all read same snapshot) ----
    # TS sessions are independent (read-only snapshot, no shared mutable
    # state, per-session checkpoint files). Run them concurrently within
    # the cell to cut wall-clock; concurrency capped by `ts_workers`.
    n_resumed_test = 0

    def _run_one_test_attempt(session) -> tuple[SessionTrajectory, bool]:
        """Single attempt at one TS. Raises on pre-run errors (e.g. oracle
        config) or unforeseen exceptions in run_session. The wrapping
        `_run_one_test` adds retry + synthesized fallback so a failure here
        never aborts the whole TS phase."""
        sid = session.session_id
        cached = _try_load(sid)
        if cached is not None:
            logger.info(
                "[test %s] resumed from disk (steps=%d, term=%s)",
                sid, cached.step_count, cached.termination_reason,
            )
            return cached, True
        logger.info("Test session %s (variant=%s)", sid, agent_variant)
        _writer = _trace_writer_factory(session, agent_variant) if _trace_writer_factory else None

        # Oracle (Targeted): build a per-test snapshot containing ONLY
        # the canonical_answer of this test's corresponding rule. Strips
        # retrieval cost from upper bound (peer convention: τ2/PrefIX/
        # AMemGym all use targeted oracles). Other variants share the
        # snapshot frozen from learning phase.
        # Oracle (Full) uses the shared `oracle_full_snapshot` containing
        # all rules' canonical_answers, built once before the TS loop.
        if agent_variant == "oracle_target":
            per_test_layer = CrossSessionLayer.create(memory_layer, model=_model)
            target_rule = next(
                (r for r in episode_rules if r.rule_id == session.rule_id),
                None,
            )
            # Bad-input policy: silent degradation turns an upper-bound
            # baseline into a noisy lower bound. Fail fast and surface the
            # distinct cause.
            if target_rule is None:
                raise RuntimeError(
                    f"Oracle (Targeted): test {sid} references rule_id="
                    f"{getattr(session, 'rule_id', '<missing>')!r} which is "
                    f"not in episode_rules. Fix the episode JSON."
                )
            if not target_rule.canonical_answer:
                raise RuntimeError(
                    f"Oracle (Targeted): rule {target_rule.rule_id} (referenced "
                    f"by test {sid}) has empty canonical_answer. Fix the "
                    f"episode JSON."
                )
            if not hasattr(per_test_layer, "inject_prior_user_statement"):
                raise RuntimeError(
                    f"Oracle (Targeted): layer {type(per_test_layer).__name__} "
                    f"does not implement inject_prior_user_statement. "
                    f"Use --memory-layer context."
                )
            per_test_layer.inject_prior_user_statement(target_rule.canonical_answer)
            logger.info(
                "Oracle (Targeted): injected rule %s canonical_answer for test %s",
                target_rule.rule_id, sid,
            )
            active_snapshot = per_test_layer.freeze()
        elif agent_variant == "oracle_full":
            # Built once before the loop (see Oracle-Full snapshot block).
            active_snapshot = oracle_full_snapshot
        else:
            active_snapshot = snapshot

        traj = run_session(
            session, agent_variant, active_snapshot, persona,
            model=_model,
            max_steps=max_steps,
            timeout=timeout,
            seed=seed,
            episode_rules=episode_rules,
            on_turn=(_writer.on_event if _writer else None),
            agent_reasoning_effort=_agent_reasoning_effort,
            hook_enabled=hook_enabled,
        )
        if _writer:
            _writer.on_finish(traj.model_dump())
        traj.memory_snapshot = active_snapshot.get_snapshot()
        _save(traj)
        logger.info(
            "  → done: steps=%d, termination=%s, duration=%.1fs (saved)",
            traj.step_count, traj.termination_reason, traj.duration_seconds,
        )
        return traj, False

    def _run_one_test(session) -> tuple[SessionTrajectory, bool]:
        """Run one TS attempt — no wrapper retry, no synthesized fallback.

        the `error_after_retry` wrapper
        was retired. Any exception propagates up to the pipeline-level
        sweep-abort handler. LLM retry exhaustion inside `run_session`
        raises `SweepAbort` directly; oracle config errors (rule_id
        mismatch, empty canonical_answer, layer without
        inject_prior_user_statement) propagate as normal exceptions.
        """
        return _run_one_test_attempt(session)

    if ts_workers <= 1 or len(episode.test_sessions) <= 1:
        for session in episode.test_sessions:
            # TS session-boundary abort check.
            if ABORT_EVENT.is_set():
                logger.warning(
                    "[sweep-abort] skipping remaining TS sessions in cell "
                    "(ABORT_EVENT set).",
                )
                break
            traj, was_resumed = _run_one_test(session)
            trajectories.append(traj)
            if was_resumed:
                n_resumed_test += 1
            if on_session_done:
                on_session_done(session.session_id, "test", traj)
    else:
        with ThreadPoolExecutor(max_workers=ts_workers) as ex:
            future_to_session = {}
            for s in episode.test_sessions:
                # Per-submission abort check. Already-running TS workers
                # will exit at the next session boundary (or by raising
                # SweepAbort themselves).
                if ABORT_EVENT.is_set():
                    logger.warning(
                        "[sweep-abort] not submitting TS %s (ABORT_EVENT set).",
                        s.session_id,
                    )
                    break
                future_to_session[ex.submit(_run_one_test, s)] = s
            for fut in as_completed(future_to_session):
                session = future_to_session[fut]
                traj, was_resumed = fut.result()
                trajectories.append(traj)
                if was_resumed:
                    n_resumed_test += 1
                if on_session_done:
                    on_session_done(session.session_id, "test", traj)
    if n_resumed_test:
        logger.info(
            "Phase 2: %d / %d test sessions resumed from disk",
            n_resumed_test, len(episode.test_sessions),
        )

    # Stamp the manifest as complete so downstream tooling can tell a
    # cleanly-finished cell from one that was killed mid-run.
    manifest_module.mark_complete(manifest_file)
    return trajectories


def _configure_logging(log_file: Path) -> None:
    """Attach console + file handlers to the root logger.

    Called from main() once we know the run's `runs_dir` so the file
    handler points at the right `run.log`. Keeps console output identical
    to before.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Avoid duplicate handlers if main() is re-entered (rare).
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers)
    if not has_console:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
    file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_h.setFormatter(fmt)
    root.addHandler(file_h)


def main():
    parser = argparse.ArgumentParser(description="ATR Runner")
    parser.add_argument("--episode", required=True, help="Path to episode JSON")
    parser.add_argument("--agent-variant", required=True,
                        choices=list(VARIANTS))
    parser.add_argument("--memory-layer", default="context",
                        choices=list(LAYER_CHOICES),
                        help="Cross-session layer kind: raw/native/context all map to "
                             "the raw-transcript ContextLayer. runner.pipeline is "
                             "the main entrypoint; this CLI is for debug/library use.")
    parser.add_argument("--model", default=None, help="LLM model override")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps per session")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Wall-clock timeout per session (seconds)")
    parser.add_argument("--seed", type=int, default=None,
                        help="LLM determinism hint forwarded to all agent, router, "
                             "classifier, and user_sim calls. Independent from "
                             "the episode-composition seed (which is baked into "
                             "the episode JSON).")
    parser.add_argument("--reasoning", default="on",
                        choices=["on", "off"],
                        help="Binary reasoning switch for the agent-under-test LLM: "
                             "'on' enables the model's reasoning/thinking mode where "
                             "the provider exposes one, 'off' disables it. Router, "
                             "user_sim, and classifier always use the provider default.")
    parser.add_argument("--traces", action="store_true",
                        help="Write pretty-trace .txt files alongside trajectories")
    parser.add_argument("--out-dir", default=None,
                        help="Override output directory (overrides canonical path)")
    parser.add_argument("--ts-workers", type=int, default=1,
                        help="Per-cell concurrency for test sessions (Phase 2). "
                             "TS reads a frozen snapshot and has no shared mutable "
                             "state, so parallel execution is safe. Defaults to 1 "
                             "(serial); raise to run test sessions in parallel.")
    parser.add_argument("--hook", action="store_true",
                        help="Enable scaffolding hook: on every "
                             "off-protocol assistant turn (LS only), inject a "
                             "<scaffolding_note> user message asking the agent "
                             "to re-emit via send_to_user. No per-session cap; "
                             "runaway loops are bounded by the existing "
                             "non-ask-text budget and max_steps. Recorded in "
                             "cell manifest as hook_enabled and stamped into "
                             "the cell path (__hook segment), so hook-on / "
                             "hook-off are distinct cells.")
    args = parser.parse_args()

    # Resolve canonical paths from the episode + variant, then set up
    # logging to write to <runs_dir>/run.log alongside trajectories/ and eval.json.
    ep_path = Path(args.episode).resolve()
    episode = load_episode(ep_path)
    persona_id = episode.raw_persona.persona_id
    runs_dir = run_paths.runs_dir(
        persona_id, episode.episode_id, args.agent_variant,
        model=args.model, memory_layer=args.memory_layer, seed=args.seed,
        hook_enabled=args.hook,
    )
    runs_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(runs_dir / "run.log")

    logger.info("Run dir: %s", runs_dir)

    trajectories = run_episode(
        args.episode, args.agent_variant,
        memory_layer=args.memory_layer,
        model=args.model,
        max_steps=args.max_steps,
        timeout=args.timeout,
        seed=args.seed,
        out_dir=runs_dir / "trajectories",
        traces_dir=runs_dir / "traces" if args.traces else None,
        agent_reasoning_effort=args.reasoning,
        ts_workers=args.ts_workers,
        hook_enabled=args.hook,
    )
    logger.info("Episode complete: %d trajectories under %s",
                len(trajectories), runs_dir / "trajectories")


if __name__ == "__main__":
    main()
