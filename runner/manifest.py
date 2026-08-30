"""Cell manifest — config snapshot that gates resume.

A `cell_manifest.json` lives next to a cell's `trajectories/` directory and
captures the configuration the cell was run under. On resume, the runner
compares the proposed config against the manifest; any mismatch on a
sweep-relevant dimension (variant / model / memory_layer / seed /
prompt_hash / code_hash) refuses the resume and aborts the cell with a
clear error.

Why this exists
---------------
Path-based cell keying alone can't catch every drift dimension — prompt
edits, code changes, or schema bumps don't naturally translate into a
new directory name. The manifest captures the full signature and treats
any deviation as "this cell needs to be re-run", instead of silently
overwriting partial trajectories with a different protocol.

Fields (version 1)
------------------
    version           int — bump on schema change
    persona_id        str
    episode_id        str
    variant           str
    model             str — resolved to DEFAULT_MODEL when caller passes None
    memory_layer      str
    seed              int | null
    agent_reasoning_effort  str | null — binary reasoning condition ("on" / "off")
    hook_enabled      bool — scaffolding hook gating; gated, so
                              flipping requires --fresh or a new cell dir
    max_steps         int
    timeout           float | null
    prompt_hashes     {filename: sha256-hex} — every .md under runner/prompts/
    code_hash         str | null — git rev-parse HEAD; null when not in a git tree
    created_at        ISO 8601
    updated_at        ISO 8601
    complete          bool — true after run_episode finishes all sessions
    schema_version    str — trajectory/eval schema version.
                            "v7_user_sim_free_output" = free-text
                            user_sim + mark_task_complete() tool;
                            user_sim_reply returns (reply, intent) where
                            intent is "end" iff mark_task_complete was
                            called. Router/cls/user_sim module split from and the <RULE_ANSWER> token mechanism
                            are preserved.
                            "v6_router_rule_answer" = Router +
                            <RULE_ANSWER> user_sim token contract:
                            route_decision carries is_strict_rule_question +
                            rule_question_span; user_sim emits intent + reply.
                            "v5_split_user_gate" = split user_sim
                            gate: reason_has_cross_session_rule_intent +
                            output_asks_cross_session_rule_question + intent +
                            reply, with is_cross_session_ask derived by the
                            runner. "v4_bool_gate" = boolean user_sim
                            gate: is_cross_session_ask + intent + reply.
                            "v3_send" =
                            send-to-user tool architecture:
                            send_to_user(output, reason) + interaction_events
                            with send_event / route_decision / cls_verdict.
                            Older cells either lack the field or have an
                            earlier value. Not gated; informational only.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fields that must match exactly for a resume to be accepted. Other fields
# (timestamps, max_steps, timeout) are recorded for audit but are NOT gated —
# changing max_steps mid-resume isn't a correctness issue, just a runtime
# parameter.
_GATING_FIELDS = (
    "persona_id",
    "episode_id",
    "variant",
    "model",
    "memory_layer",
    "seed",
    "agent_reasoning_effort",
    "hook_enabled",
    "smoke",  # distinguishes smoke runs from main-table cells.
    "prompt_hashes",
    "code_hash",
)


@functools.lru_cache(maxsize=1)
def compute_prompt_hashes() -> dict[str, str]:
    """Hash every .md under runner/prompts/ + every .json under
    runner/cls_calibration/shots/ (the active shot banks for cls).

    Sorted by stable name so the dict is reproducible across processes.
    Calibration shot files are tracked alongside prompts so a manifest
    pins both prompt wording AND classifier shot version — swapping
    `_DEFAULT_FEWSHOT_PATH` (or a downstream `ATR_CLS_FEWSHOT_PATH`)
    surfaces as a hash diff.

    Cached for the process lifetime (prompts/shots don't change mid-sweep);
    tests that monkey-patch prompts must call
    `compute_prompt_hashes.cache_clear()` first.
    """
    out: dict[str, str] = {}
    if _PROMPTS_DIR.is_dir():
        for p in sorted(_PROMPTS_DIR.iterdir()):
            if p.is_dir() or p.suffix != ".md":
                continue
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    shots_dir = _PROMPTS_DIR.parent / "cls_calibration" / "shots"
    if shots_dir.is_dir():
        for p in sorted(shots_dir.iterdir()):
            if p.is_dir() or p.suffix != ".json":
                continue
            out[f"cls_calibration/shots/{p.name}"] = (
                hashlib.sha256(p.read_bytes()).hexdigest()
            )
    return out


@functools.lru_cache(maxsize=1)
def compute_code_hash() -> str | None:
    """Return a hash that changes whenever core runtime code changes.

    Two independent inputs:
      1. git HEAD sha — moves when commits land
      2. SHA-256 of all .py under runner/, evaluator/, lib/ — moves on
         uncommitted edits (the "dirty worktree" case git rev-parse can't
         see)

    Returning HEAD alone (the previous behavior) silently let dirty-tree
    resumes mix new code with old trajectories — exactly the failure mode
    this manifest is meant to prevent. Hashing the source tree closes that
    gap regardless of git state and works in non-git checkouts too.

    Cached for the process lifetime (source code doesn't change mid-sweep);
    tests that monkey-patch `_hash_source_tree` must call
    `compute_code_hash.cache_clear()` first to force a recompute.
    """
    head: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            head = result.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        head = None

    src_sha = _hash_source_tree()
    if head and src_sha:
        return f"{head[:12]}+src{src_sha[:12]}"
    if src_sha:
        return f"src{src_sha[:12]}"
    return head


def _hash_source_tree() -> str | None:
    """SHA-256 over the on-disk content of runner/, evaluator/, lib/ .py files.

    Stable: files visited in sorted order; each contributes its relative
    path + bytes. Excludes __pycache__ and other non-source artifacts.
    """
    hasher = hashlib.sha256()
    seen_any = False
    for sub in ("runner", "evaluator", "lib"):
        root = _PROJECT_ROOT / sub
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            hasher.update(str(path.relative_to(_PROJECT_ROOT)).encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(data)
            hasher.update(b"\n")
            seen_any = True
    return hasher.hexdigest() if seen_any else None


SCHEMA_VERSION = "v7_user_sim_free_output"


def build_manifest(
    *,
    persona_id: str,
    episode_id: str,
    variant: str,
    model: str,
    memory_layer: str,
    seed: int | None,
    max_steps: int,
    timeout: float | None,
    agent_reasoning_effort: str | None = None,
    hook_enabled: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Construct a fresh manifest dict for the current cell.

    `smoke` flag distinguishes smoke / debug / validation runs
    from main-table cells so they cannot share a cell directory even
    when every other gated dimension matches.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "version": MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "persona_id": persona_id,
        "episode_id": episode_id,
        "variant": variant,
        "model": model,
        "memory_layer": memory_layer,
        "seed": seed,
        "agent_reasoning_effort": agent_reasoning_effort,
        "hook_enabled": bool(hook_enabled),
        "smoke": bool(smoke),
        "max_steps": max_steps,
        "timeout": timeout,
        "prompt_hashes": compute_prompt_hashes(),
        "code_hash": compute_code_hash(),
        "created_at": now,
        "updated_at": now,
        "complete": False,
    }


def read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        from lib.benchmark_data import read_json  # noqa: PLC0415
        return read_json(path)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read cell manifest %s: %s", path, e)
        return None


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    from lib.benchmark_data import write_json  # noqa: PLC0415
    write_json(path, manifest)


def diff_manifest(
    existing: dict[str, Any],
    proposed: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return {field: (existing, proposed)} for every gating field that differs.

    Empty dict ⇒ safe to resume.
    """
    out: dict[str, tuple[Any, Any]] = {}
    for k in _GATING_FIELDS:
        if existing.get(k) != proposed.get(k):
            out[k] = (existing.get(k), proposed.get(k))
    return out


class ManifestMismatch(RuntimeError):
    """Raised when a resume is attempted with a config that doesn't match
    the cell's existing manifest. Message is pre-formatted for direct
    surfacing in run.log / stderr.
    """


def gate_or_init(
    manifest_path: Path,
    *,
    proposed: dict[str, Any],
    allow_stale: bool = False,
) -> dict[str, Any]:
    """Either accept the existing manifest (matching) or initialise a fresh one.

    Returns the manifest dict that should be considered "current" for this
    run. Raises `ManifestMismatch` when an existing manifest disagrees on
    any gating field, unless `allow_stale=True` (debug escape hatch).

    Side effect: writes the manifest to disk (either fresh or with bumped
    `updated_at`). Caller is responsible for calling `mark_complete`
    after run_episode finishes.
    """
    existing = read_manifest(manifest_path)
    if existing is None:
        write_manifest(manifest_path, proposed)
        return proposed

    diff = diff_manifest(existing, proposed)
    if diff and not allow_stale:
        lines = ["Cell manifest mismatch — refusing resume."]
        lines.append(f"  manifest: {manifest_path}")
        for field, (was, now) in diff.items():
            lines.append(f"  {field}: {was!r} → {now!r}")
        lines.append(
            "  Resolution: delete the cell directory to start fresh, or "
            "pass --allow-stale-resume to override (debugging only)."
        )
        raise ManifestMismatch("\n".join(lines))

    if diff and allow_stale:
        logger.warning(
            "Cell manifest mismatch on %d field(s); --allow-stale-resume set, "
            "proceeding anyway. Differences: %s",
            len(diff),
            ", ".join(f"{k}({was!r}→{now!r})" for k, (was, now) in diff.items()),
        )

    # Match: bump updated_at, keep created_at + complete unchanged.
    existing["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Refresh non-gated runtime params (max_steps / timeout) so the manifest
    # reflects the latest invocation's runtime knobs.
    existing["max_steps"] = proposed.get("max_steps", existing.get("max_steps"))
    existing["timeout"] = proposed.get("timeout", existing.get("timeout"))
    write_manifest(manifest_path, existing)
    return existing


def mark_complete(manifest_path: Path) -> None:
    """Stamp the manifest as complete after run_episode finishes."""
    m = read_manifest(manifest_path)
    if m is None:
        logger.warning(
            "mark_complete called but manifest does not exist at %s",
            manifest_path,
        )
        return
    m["complete"] = True
    m["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_manifest(manifest_path, m)
