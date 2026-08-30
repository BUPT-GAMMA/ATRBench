"""Canonical paths for runner / evaluator outputs.

Layout:

  outputs/<persona_id>/<short_episode_id>/<cell_segment>/
      ├── cell_manifest.json        (config snapshot — gates resume; see runner/manifest.py)
      ├── trajectories/             (per-session JSON checkpoints)
      ├── traces/                   (pretty-trace .txt, regenerated on demand)
      ├── eval.json                 (evaluator report)
      └── run.log                   (runner stdout/stderr)

`cell_segment` encodes ALL sweep-relevant dimensions so two cells with
different memory_layer / seed / model never collide:

    <variant>__model-<safe_model>__layer-<memory_layer>__seed<seed>[__hook]

`model` is normalised — the absence of `--model` (i.e. `model=None`) is
resolved to `lib.llm.DEFAULT_MODEL` here so a default-model run produces
the SAME path as `--model gpt-5.4`. Without that, resume would silently
miss whenever a sweep flipped between explicit and implicit model.

`memory_layer` and `seed` are likewise stamped — earlier versions encoded
only `variant + model`, which let a sweep that flipped `--memory-layer`
or `--seed` reuse stale trajectories from the previous run.

`__hook` is appended only when scaffolding-hook is enabled, so
existing hook-off paths are unchanged. Hook-on and hook-off runs land in
distinct cells.

`short_episode_id` is `episode.episode_id` with the persona_id prefix
stripped (e.g. "elizabeth_pesacreta_seed000" → "seed000").

Special case — oracle variant:
  oracle is bound to test_sessions and does not consume the episode's
  learning trajectory, so its output is independent of which episode it
  is paired with. oracle cells are stored at:

    outputs/<persona_id>/_oracle/<cell_segment>/

  `cell_segment` for oracle still encodes seed + memory_layer + model
  (all of which DO affect oracle output).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Output root resolution order (first hit wins, with CLI override available
# via set_runs_root() — see runner.pipeline --outputs-root):
#   1. ATR_OUTPUTS_ROOT env var (used by stability launcher + nested subprocs)
#   2. <repo>/outputs/ (canonical)
# Either can be overridden at runtime by calling set_runs_root(...) early
# in the entrypoint (before any path-dependent code runs).
_OUTPUTS_ROOT_ENV = os.environ.get("ATR_OUTPUTS_ROOT")
_RUNS_ROOT = (
    Path(_OUTPUTS_ROOT_ENV).expanduser().resolve()
    if _OUTPUTS_ROOT_ENV else _PROJECT_ROOT / "outputs"
)


def _safe_path_component(value: str, field: str) -> str:
    """Validate an identifier before using it as a directory component.

    Episode metadata can come from a user-supplied custom episode, so it is
    not safe to trust the strings even though the released ids are curated.
    Keep the accepted format broad (spaces and punctuation are allowed), but
    reject anything that can introduce another path component or a control
    character.  The resolved-path check in ``_within_runs_root`` is a second
    defence for all callers.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field} must be a single path component")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{field} contains a control character")
    if Path(value).is_absolute():
        raise ValueError(f"{field} must be a relative path component")
    return value


def _within_runs_root(path: Path) -> Path:
    """Resolve *path* and reject output paths outside the configured root."""
    root = _RUNS_ROOT.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"output path escapes outputs root: {resolved} (root: {root})"
        ) from exc
    return resolved


def set_runs_root(path) -> Path:
    """Override the output root at runtime. All subsequent calls to
    `runs_dir()` / `trajectories_dir()` / `eval_path()` / `log_path()` /
    `manifest_path()` will resolve under this root. Returns the resolved
    new root for convenience.

    Used by `runner.pipeline --outputs-root` and
    `evaluator.pipeline --outputs-root` so a single CLI flag redirects
    BOTH cell data and summary files into the same parallel tree.
    """
    global _RUNS_ROOT
    _RUNS_ROOT = Path(path).expanduser().resolve()
    return _RUNS_ROOT


def get_runs_root() -> Path:
    """Read the currently active output root. Prefer this in callers that
    need the live value (e.g. summary-dir default) over importing
    `_RUNS_ROOT` directly, since the global may be mutated by
    `set_runs_root()` after module import."""
    return _RUNS_ROOT

# Resolve DEFAULT_MODEL lazily to avoid a hard import cycle when paths.py is
# imported from contexts that haven't finished setting up sys.path. lib/llm.py
# is the canonical owner of the default; we mirror its value here.
sys.path.insert(0, str(_PROJECT_ROOT))
from lib.llm import DEFAULT_MODEL  # noqa: E402

sys.path.insert(0, str(_PROJECT_ROOT / "runner"))
from _constants import (  # noqa: E402
    ORACLE_VARIANTS as _ORACLE_VARIANTS,
    TS_ONLY_VARIANTS as _TS_ONLY_VARIANTS,
)


def episode_path(persona_id: str, seed: int) -> Path:
    """Canonical filesystem path for a composed episode JSON.

    Encoded as `<personas>/<persona>/episodes/<persona>_seed<NNN>.json`.
    Each persona has one episode per seed in the released data. The runner and
    evaluator both read here, so this is the single source of truth for episode
    discovery.
    """
    from lib.benchmark_data import PERSONAS_DIR  # noqa: PLC0415
    persona_id = _safe_path_component(persona_id, "persona_id")
    return (
        PERSONAS_DIR / persona_id / "episodes" /
        f"{persona_id}_seed{seed:03d}.json"
    )


def short_episode_id(episode_id: str, persona_id: str) -> str:
    """Strip persona_id prefix from episode_id for cleaner paths."""
    persona_id = _safe_path_component(persona_id, "persona_id")
    episode_id = _safe_path_component(episode_id, "episode_id")
    prefix = f"{persona_id}_"
    if episode_id.startswith(prefix):
        return episode_id[len(prefix):]
    return episode_id


def seed_from_episode_id(episode_id: str) -> int | None:
    """Infer the composition seed encoded in canonical episode ids.

    Canonical released episodes end with `_seedNNN`, e.g.
    `alice_seed000`. Single-cell reruns should land in the
    same `seedNNN` cell as sweep mode even when the operator points directly
    at the episode JSON. Return None for ad-hoc/custom episode ids.
    """
    m = re.search(r"(?:^|_)seed(?P<seed>\d+)(?:$|_)", episode_id)
    if not m:
        return None
    return int(m.group("seed"))


def _safe_model(model: str | None) -> str:
    """Normalise model string for filesystem use.

    None is resolved to DEFAULT_MODEL — a missing `--model` and an explicit
    `--model gpt-5.4` MUST produce the same path, otherwise resume silently
    breaks whenever a sweep mixes the two forms.
    """
    return (model or DEFAULT_MODEL).replace("/", "-")


def cell_segment(
    variant: str,
    model: str | None = None,
    memory_layer: str | None = None,
    seed: int | None = None,
    hook_enabled: bool = False,
    smoke: bool = False,
) -> str:
    """Build the per-cell directory name from sweep dimensions.

    Six dimensions participate: variant + model + memory_layer + seed +
    hook_enabled + smoke. Missing memory_layer / seed degrade to
    "default" / "noseed" so the cell name is always well-formed (callers
    that don't thread the full key won't crash, but they will accumulate
    everything under the catch-all bucket — visible at a glance in the
    directory listing).

    `hook_enabled` appears as a `__hook` segment when True;
    when False the segment is omitted so existing hook-off cell paths
    are unchanged. This keeps hook-on and hook-off runs in distinct
    cells — they produce different metric distributions and must not
    overwrite each other.

    `smoke` appears as a `__smoke` segment when True so smoke
    / debug runs never collide with main-table cells, even when every
    other dimension matches.
    """
    parts = [variant, f"model-{_safe_model(model)}"]
    parts.append(f"layer-{memory_layer or 'default'}")
    parts.append(f"seed{int(seed):03d}" if seed is not None else "noseed")
    if hook_enabled:
        parts.append("hook")
    if smoke:
        parts.append("smoke")
    return "__".join(parts)


def runs_dir(
    persona_id: str,
    episode_id: str,
    variant: str,
    model: str | None = None,
    memory_layer: str | None = None,
    seed: int | None = None,
    hook_enabled: bool = False,
    smoke: bool = False,
) -> Path:
    """Canonical output dir for one cell.

    Non-TS-only: `outputs/<persona>/<short_ep>/<cell_segment>/`.
    Oracle:      `outputs/<persona>/_oracle/<cell_segment>/`
                 (independent of the learning trajectory).
    """
    persona_id = _safe_path_component(persona_id, "persona_id")
    episode_id = _safe_path_component(episode_id, "episode_id")
    seg = cell_segment(variant, model, memory_layer, seed, hook_enabled, smoke)
    if variant in _ORACLE_VARIANTS:
        path = _RUNS_ROOT / persona_id / "_oracle" / seg
        return _within_runs_root(path)
    short = short_episode_id(episode_id, persona_id)
    path = _RUNS_ROOT / persona_id / short / seg
    return _within_runs_root(path)


def trajectories_dir(
    persona_id: str,
    episode_id: str,
    variant: str,
    model: str | None = None,
    memory_layer: str | None = None,
    seed: int | None = None,
    hook_enabled: bool = False,
    smoke: bool = False,
) -> Path:
    return runs_dir(
        persona_id, episode_id, variant, model, memory_layer, seed,
        hook_enabled, smoke,
    ) / "trajectories"


def eval_path(
    persona_id: str,
    episode_id: str,
    variant: str,
    model: str | None = None,
    memory_layer: str | None = None,
    seed: int | None = None,
    hook_enabled: bool = False,
) -> Path:
    return runs_dir(
        persona_id, episode_id, variant, model, memory_layer, seed,
        hook_enabled,
    ) / "eval.json"


def log_path(
    persona_id: str,
    episode_id: str,
    variant: str,
    model: str | None = None,
    memory_layer: str | None = None,
    seed: int | None = None,
    hook_enabled: bool = False,
) -> Path:
    return runs_dir(
        persona_id, episode_id, variant, model, memory_layer, seed,
        hook_enabled,
    ) / "run.log"


def manifest_path(
    persona_id: str,
    episode_id: str,
    variant: str,
    model: str | None = None,
    memory_layer: str | None = None,
    seed: int | None = None,
    hook_enabled: bool = False,
) -> Path:
    return runs_dir(
        persona_id, episode_id, variant, model, memory_layer, seed,
        hook_enabled,
    ) / "cell_manifest.json"
