"""Build Table 1 (TS Payoff) and Table 2 (LS Acquisition + Application) for the
main results tables.

Aggregation:
  - Persona-Macro primary (each persona contributes equally).
  - TS-Micro secondary (sensitivity), reported alongside.
  - 95% CI via persona-level cluster bootstrap (B=1000).

Reads outputs/<persona>/<short_episode>/<cell>/eval.json (and oracle_target
under outputs/<persona>/_oracle/<cell>/eval.json) produced by
`evaluator.pipeline`.

Usage:
  uv run python scripts/build_main_tables.py
  uv run python scripts/build_main_tables.py --bootstrap 1000 --out tables.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS = Path(
    os.environ.get("ATR_OUTPUTS_ROOT", REPO_ROOT / "outputs")
).expanduser().resolve()

CELL_RE = re.compile(
    r"^(?P<variant>default|atr|always_ask|oracle_target)__"
    r"model-(?P<model>[^_]+(?:-[^_]+)*)__"
    r"layer-(?P<layer>[a-z]+)__"
    r"seed(?P<seed>\d+)__hook$"
)

# Main panel: the 8 agent models under test, in the paper's Table 1/2
# column order (GPT, Opus, GF, GP, Qw, MM, DP, DF).
MODELS = [
    "gpt-5.4",
    "claude-opus-4-7",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "qwen3.6-plus",
    "MiniMax-M2.7",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
]
VARIANTS_TABLE1 = ["default", "atr", "always_ask", "oracle_target"]
VARIANTS_TABLE2 = ["atr", "always_ask"]


def parse_cell_dir(cell_dir: Path) -> dict | None:
    """Return {persona, panel, variant, model} or None if not a cell dir.

    Each persona has one episode cell tree under `outputs/<persona>/seedNNN/`
    (panel="main"); oracle cells live under `_oracle/` (panel="oracle").
    Any other parent directory is not part of the main table.
    """
    m = CELL_RE.match(cell_dir.name)
    if not m:
        return None
    parent = cell_dir.parent.name  # e.g. seed000 or _oracle
    persona = cell_dir.parent.parent.name
    if parent == "_oracle":
        panel = "oracle"
    elif re.match(r"^seed\d+$", parent):
        panel = "main"
    else:
        return None
    return {
        "persona": persona,
        "panel": panel,
        "variant": m.group("variant"),
        "model": m.group("model"),
        "cell_dir": cell_dir,
    }


def _cell_is_complete(cell_dir: Path) -> bool:
    """Aborted / interrupted cells must NOT enter main-table
    aggregation. `runner.manifest.mark_complete` stamps `complete=True`
    on `cell_manifest.json` only after a clean cell finish; aborted cells
    never reach that step (the orchestrator raises SweepAbort and
    `pipeline.py` re-raises without calling `mark_complete`). Treat
    absence-of-stamp as "not complete" and exclude the cell.
    """
    manifest_path = cell_dir / "cell_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(manifest.get("complete"))


def load_cells(outputs_root: Path = DEFAULT_OUTPUTS) -> list[dict]:
    """Walk outputs/, parse cell metadata + load eval.json metrics.

    Cells whose `cell_manifest.json` does not carry `complete=True` are
    skipped (aborted / interrupted cells stay out of the
    main table). A warning is logged listing each skipped cell so the
    operator can see what's missing.
    """
    cells = []
    skipped_incomplete: list[str] = []
    if not outputs_root.exists():
        return cells
    for persona_dir in sorted(outputs_root.iterdir()):
        if not persona_dir.is_dir() or persona_dir.name.startswith(("_", ".")):
            continue
        for panel_dir in sorted(persona_dir.iterdir()):
            if not panel_dir.is_dir():
                continue
            for cell_dir in sorted(panel_dir.iterdir()):
                meta = parse_cell_dir(cell_dir)
                if meta is None:
                    continue
                ej_path = cell_dir / "eval.json"
                if not ej_path.exists():
                    continue
                if not _cell_is_complete(cell_dir):
                    skipped_incomplete.append(str(cell_dir.relative_to(outputs_root)))
                    continue
                ej = json.loads(ej_path.read_text())
                m = ej["metrics"]
                if (
                    m.get("missing_test_sessions")
                    or m.get("missing_learning_sessions")
                ):
                    skipped_incomplete.append(str(cell_dir.relative_to(outputs_root)))
                    continue
                cells.append({
                    **meta,
                    "total_test": m.get("total_test", 0),
                    "success_count": m.get("success_count", 0),
                    "payoff_accuracy": m.get("payoff_accuracy"),
                    "session_results": ej.get("session_results", []),
                    # LS metrics
                    "ls_total": m.get("total_learning", 0),
                    "ls_routed_rule": m.get("ls_send_calls_routed_rule", 0),
                    "rule_coverage_ratio": (m.get("rule_coverage") or {}).get("ratio", 0.0),
                    # 8-cell: AppliedRate uses covered_hit; AcqPrec numerator
                    # uses all hit cells (covered + uncovered), per-rule deduped.
                    "covered_hit_pass": m.get("covered_hit_pass", 0),
                    "covered_hit_fail": m.get("covered_hit_fail", 0),
                    "uncovered_hit_pass": m.get("uncovered_hit_pass", 0),
                    "uncovered_hit_fail": m.get("uncovered_hit_fail", 0),
                })
    if skipped_incomplete:
        print(
            f"[build_main_tables] skipped {len(skipped_incomplete)} "
            f"incomplete cell(s) (cell_manifest.complete != True):"
        )
        for name in skipped_incomplete:
            print(f"  - {name}")
    return cells


def cluster_bootstrap_ci(
    per_persona_values: list[float], B: int = 1000, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile cluster bootstrap CI: resample personas with replacement."""
    n = len(per_persona_values)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(42)
    means = []
    for _ in range(B):
        sample = [per_persona_values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    hi = means[int(B * (1 - alpha / 2)) - 1]
    return (lo, hi)


def build_table1(cells: list[dict], bootstrap: int) -> dict:
    """Table 1: (variant, model, panel) → Persona-Macro task_success + CI.

    Returns {(variant, model, panel): {macro, ci_lo, ci_hi, micro, n_personas, n_ts}}
    """
    # Group cells by (variant, model, panel)
    groups = defaultdict(list)
    for c in cells:
        if c["variant"] not in VARIANTS_TABLE1:
            continue
        if c["model"] not in MODELS:
            continue
        groups[(c["variant"], c["model"], c["panel"])].append(c)

    out = {}
    for key, group_cells in groups.items():
        variant, model, panel = key
        # Per-persona Micro: accuracy within each persona
        per_persona_acc = []
        total_success = 0
        total_n = 0
        for c in group_cells:
            n = c["total_test"]
            s = c["success_count"]
            if n > 0:
                per_persona_acc.append(s / n)
                total_success += s
                total_n += n
        if not per_persona_acc:
            continue
        macro = sum(per_persona_acc) / len(per_persona_acc)
        micro = total_success / total_n if total_n else 0
        ci_lo, ci_hi = cluster_bootstrap_ci(per_persona_acc, B=bootstrap)
        out[key] = {
            "macro": macro,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "micro": micro,
            "n_personas": len(per_persona_acc),
            "n_ts_total": total_n,
            "per_persona": per_persona_acc,
        }
    return out


def build_table2(cells: list[dict], bootstrap: int) -> dict:
    """Table 2: (variant, model) → Persona-Macro of {ask_rate, acqprec,
    rule_coverage, AppliedRate}.

    Only atr / always_ask (default has no asks, oracle has no LS).
    """
    # Each persona contributes one episode cell. Aggregate within
    # (persona, variant, model), then average across personas Macro.
    by_persona = defaultdict(lambda: defaultdict(lambda: {
        "ls_total": 0, "routed": 0, "acquired": 0,
        "covered_hit_pass": 0, "covered_hit_fail": 0,
        "rule_cov_ratios": [],  # one per cell
    }))
    for c in cells:
        if c["variant"] not in VARIANTS_TABLE2:
            continue
        if c["model"] not in MODELS:
            continue
        if c["panel"] != "main":
            continue
        key = (c["variant"], c["model"])
        bucket = by_persona[c["persona"]][key]
        bucket["ls_total"] += c["ls_total"]
        bucket["routed"] += c["ls_routed_rule"]
        # AcqPrec numerator: unique rules newly acquired. The 8-cell hit
        # tallies are per-rule (deduped), so all hit cells (covered +
        # uncovered) sum to the count of distinct rules acquired.
        bucket["acquired"] += (
            c["covered_hit_pass"] + c["covered_hit_fail"]
            + c["uncovered_hit_pass"] + c["uncovered_hit_fail"]
        )
        bucket["covered_hit_pass"] += c["covered_hit_pass"]
        bucket["covered_hit_fail"] += c["covered_hit_fail"]
        bucket["rule_cov_ratios"].append(c["rule_coverage_ratio"])

    # Compute per-persona metrics (Micro within persona)
    rows = {}
    for variant in VARIANTS_TABLE2:
        for model in MODELS:
            key = (variant, model)
            per_persona_ask, per_persona_acqprec, per_persona_cov, per_persona_app = [], [], [], []
            for persona, persona_buckets in by_persona.items():
                b = persona_buckets.get(key)
                if not b or b["ls_total"] == 0:
                    continue
                ar = b["routed"] / b["ls_total"]
                # AcqPrec: share of strict rule questions that newly acquire a
                # hidden rule (unique rules acquired / strict rule questions).
                acqprec = b["acquired"] / b["routed"] if b["routed"] > 0 else None
                cov = sum(b["rule_cov_ratios"]) / len(b["rule_cov_ratios"]) if b["rule_cov_ratios"] else 0
                applied_d = b["covered_hit_pass"] + b["covered_hit_fail"]
                applied = b["covered_hit_pass"] / applied_d if applied_d > 0 else None
                per_persona_ask.append(ar)
                if acqprec is not None:
                    per_persona_acqprec.append(acqprec)
                per_persona_cov.append(cov)
                if applied is not None:
                    per_persona_app.append(applied)
            rows[key] = {
                "ask_rate_macro": sum(per_persona_ask) / len(per_persona_ask) if per_persona_ask else None,
                "ask_rate_ci": cluster_bootstrap_ci(per_persona_ask, B=bootstrap) if per_persona_ask else None,
                "acqprec_macro": sum(per_persona_acqprec) / len(per_persona_acqprec) if per_persona_acqprec else None,
                "acqprec_ci": cluster_bootstrap_ci(per_persona_acqprec, B=bootstrap) if per_persona_acqprec else None,
                "coverage_macro": sum(per_persona_cov) / len(per_persona_cov) if per_persona_cov else None,
                "coverage_ci": cluster_bootstrap_ci(per_persona_cov, B=bootstrap) if per_persona_cov else None,
                "applied_macro": sum(per_persona_app) / len(per_persona_app) if per_persona_app else None,
                "applied_ci": cluster_bootstrap_ci(per_persona_app, B=bootstrap) if per_persona_app else None,
                "n_personas": len(per_persona_ask),
                "n_applied_personas": len(per_persona_app),
            }
    return rows


def fmt_cell(macro: float | None, ci: tuple[float, float] | None = None) -> str:
    if macro is None:
        return "—"
    return f"{macro*100:.1f}"


def fmt_pct(v: float | None) -> str:
    return f"{v*100:5.1f}" if v is not None else "  —  "


def print_main_table(t1: dict, t2: dict) -> str:
    """Single mega-table: (variants × models) × 3 cols
    (RuleAsk, RuleCov, TSAcc). RuleAsk/RuleCov are n/a for default and
    oracle (default has no rule-ask guidance; oracle skips learning)."""
    panel_key = "main"
    lines = []
    lines.append("## Main Table: ATRBench Results (Persona-Macro)\n")
    lines.append("RuleAsk: asks per LS session (lower=cheaper). "
                 "RuleCov: \\% of rule pool acquired (Classifier-confirmed). "
                 "TSAcc: \\% TS pass rate.\n")
    lines.append("| Variant | Model | RuleAsk ↓ | RuleCov ↑ | TSAcc ↑ |")
    lines.append("|---|---|---|---|---|")
    for variant in VARIANTS_TABLE1:
        for model in MODELS:
            # TSAcc from t1
            if variant == "oracle_target":
                ts_val = t1.get((variant, model, "oracle"))
            else:
                ts_val = t1.get((variant, model, panel_key))
            tsacc = f"{ts_val['macro']*100:.1f}" if ts_val else "—"
            # RuleAsk + RuleCov from t2 (only atr/always_ask)
            if variant in VARIANTS_TABLE2:
                r = t2.get((variant, model))
                if r and r["ask_rate_macro"] is not None:
                    rule_ask = f"{r['ask_rate_macro']:.2f}"
                    rule_cov = (f"{r['coverage_macro']*100:.1f}"
                                if r["coverage_macro"] is not None else "—")
                else:
                    rule_ask = rule_cov = "—"
            else:
                # default / oracle: no rule-ask channel use
                rule_ask = rule_cov = "n/a"
            lines.append(f"| {variant} | {model} | {rule_ask} | {rule_cov} | {tsacc} |")
    return "\n".join(lines)


def print_gap_table(t1: dict, t2: dict) -> str:
    """Acquisition diagnostic: AcqPrec + AppliedRate + GapRec (cf. paper
    Tables 1-2). GapRec is the share of each model's default->oracle TSAcc
    gap recovered by the variant."""
    lines = []
    lines.append("\n## Acquisition Diagnostic (Persona-Macro)\n")
    lines.append("AcqPrec: \\% of strict rule questions that newly acquire a hidden rule. "
                 "AppliedRate: \\% of acquired rules whose bound TS passes. "
                 "GapRec: (var-default)/(oracle-default) in TSAcc, \\%.\n")
    lines.append("| Variant | Model | AcqPrec ↑ | AppliedRate ↑ | GapRec ↑ |")
    lines.append("|---|---|---|---|---|")
    for variant in VARIANTS_TABLE2:
        for model in MODELS:
            r = t2.get((variant, model))
            if not r:
                continue
            ap = (f"{r['acqprec_macro']*100:.1f}"
                  if r.get("acqprec_macro") is not None else "—")
            app = (f"{r['applied_macro']*100:.1f}"
                   if r["applied_macro"] is not None else "—")
            # GapRec from TSAcc: (variant - default) / (oracle - default)
            dflt = t1.get(("default", model, "main"))
            orc = t1.get(("oracle_target", model, "oracle"))
            var = t1.get((variant, model, "main"))
            gaprec = "—"
            if dflt and orc and var:
                denom = orc["macro"] - dflt["macro"]
                if abs(denom) > 1e-9:
                    gaprec = f"{(var['macro'] - dflt['macro']) / denom * 100:.1f}"
            lines.append(f"| {variant} | {model} | {ap} | {app} | {gaprec} |")
    return "\n".join(lines)


def print_ts_micro_sensitivity(t1: dict) -> str:
    """Optional sensitivity panel for the appendix."""
    panel_key = "main"
    lines = ["\n## Sensitivity: TSAcc TS-Micro (pooled)\n"]
    lines.append("| Variant | Model | TSAcc (Micro) |")
    lines.append("|---|---|---|")
    for variant in VARIANTS_TABLE1:
        for model in MODELS:
            if variant == "oracle_target":
                val = t1.get((variant, model, "oracle"))
            else:
                val = t1.get((variant, model, panel_key))
            cell = f"{val['micro']*100:.1f}" if val else "—"
            lines.append(f"| {variant} | {model} | {cell} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--out", default=None, help="Write output to file (markdown)")
    p.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS,
        help="Runner/evaluator output root (default: ATR_OUTPUTS_ROOT or outputs/)",
    )
    args = p.parse_args()

    cells = load_cells(args.outputs_root.expanduser().resolve())
    print(f"# Main tables ({len(cells)} cells loaded)\n")

    t1 = build_table1(cells, args.bootstrap)
    t2 = build_table2(cells, args.bootstrap)

    out_main = print_main_table(t1, t2)
    out_gap = print_gap_table(t1, t2)
    out_sens = print_ts_micro_sensitivity(t1)
    text = out_main + "\n" + out_gap + "\n" + out_sens
    print(text)

    if args.out:
        Path(args.out).write_text(f"# Main tables ({len(cells)} cells)\n\n{text}\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
