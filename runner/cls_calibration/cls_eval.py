"""Run cls against a labeled test set; report per-class counts + precision /
recall / accuracy.

Usage:
    uv run python -m runner.cls_calibration.cls_eval \\
        --test-set runner/cls_calibration/test_set.jsonl \\
        [--shots runner/cls_calibration/shots/v12.json] \\
        [--model gpt-5.4] [--seed 0] [--concurrency 15]

Test-set entries (one per line, JSONL):
    {"rule_pool": [{"rule_id": str, "statement": str, "bound_tool": str}, ...],
     "agent_text": str,
     "rule_question_span": str,  # optional; preferred when present
     "gold_rule_id": str | null,
     "source": str}

Confusion buckets:
    TP                — predicted == gold (both non-null)
    FP-wrong-match    — predicted != gold, predicted non-null, gold non-null
    FP-should-miss    — predicted non-null, gold null  (over-claim)
    FN                — predicted null, gold non-null  (under-claim)
    TN                — predicted null, gold null      (correct miss)
    cls-error         — scaffolding fault (excluded from precision/recall)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from typing import Iterator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "runner"))

from runner.route_agent_text import route_agent_text  # noqa: E402
from runner.schemas import Rule, ActionStep  # noqa: E402


def _iter_examples(path: Path) -> Iterator[dict]:
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield json.loads(line)


def _to_rules(pool: list[dict]) -> list[Rule]:
    """Wrap minimal pool dicts into Rule objects so route_agent_text can
    format them. Test-set entries carry only what the matcher needs."""
    out: list[Rule] = []
    for r in pool:
        out.append(Rule(
            rule_id=r["rule_id"],
            rule_text=r.get("rule_text") or r["statement"],
            canonical_answer=r["statement"],
            check_type=r.get("check_type", "tool_identity"),
            action_step=ActionStep(
                tool=r.get("bound_tool") or "",
                param=r.get("bound_param"),
            ),
        ))
    return out


def _query_text(ex: dict) -> str:
    """Return the production-shaped cls input for a test example."""
    span = ex.get("rule_question_span")
    if isinstance(span, str) and span.strip():
        return span
    return ex["agent_text"]


def _bucket_result(i: int, total: int, ex: dict, result: dict) -> dict:
    gold = ex.get("gold_rule_id")
    if result.get("error"):
        bucket = "cls_error"
        pred = None
    else:
        pred = result.get("rule_id")
        if pred is None and gold is None:
            bucket = "TN"
        elif pred is None and gold is not None:
            bucket = "FN"
        elif pred is not None and gold is None:
            bucket = "FP-should-miss"
        elif pred == gold:
            bucket = "TP"
        else:
            bucket = "FP-wrong-match"
    return {
        "i": i, "total": total, "source": ex.get("source", "-"),
        "query": _query_text(ex),
        "gold": gold, "pred": pred, "bucket": bucket,
    }


def _eval_one(
    i: int,
    total: int,
    ex: dict,
    *,
    model: str,
    seed: int,
    shots: Path | None,
) -> dict:
    rules = _to_rules(ex["rule_pool"])
    result = route_agent_text(
        query=_query_text(ex),
        rules=rules,
        model=model,
        seed=seed,
        fewshot_path=shots,
    )
    return _bucket_result(i, total, ex, result)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test-set", required=True, type=Path,
                   help="JSONL test-set path")
    p.add_argument("--shots", default=None, type=Path,
                   help="Shot bank to use (overrides ATR_CLS_FEWSHOT_PATH "
                        "and the compiled-in default)")
    p.add_argument("--model", default="gpt-5.4",
                   help="LLM model for cls (default: gpt-5.4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1,
                   help="Number of cls calls to run concurrently")
    p.add_argument("--report", type=Path, default=None,
                   help="Optional JSON output path for the metric summary")
    args = p.parse_args()

    examples = list(_iter_examples(args.test_set))
    if not examples:
        print(f"No examples found in {args.test_set}", file=sys.stderr)
        return 2

    if args.concurrency < 1:
        print("--concurrency must be >= 1", file=sys.stderr)
        return 2

    total = len(examples)
    rows: list[dict] = []
    if args.concurrency == 1:
        for i, ex in enumerate(examples, 1):
            rows.append(_eval_one(
                i, total, ex,
                model=args.model, seed=args.seed, shots=args.shots,
            ))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    _eval_one, i, total, ex,
                    model=args.model, seed=args.seed, shots=args.shots,
                )
                for i, ex in enumerate(examples, 1)
            ]
            for future in as_completed(futures):
                rows.append(future.result())

    rows.sort(key=lambda row: row["i"])

    tp = fp_wrong = fp_should_miss = fn = tn = err = 0
    for row in rows:
        bucket = row["bucket"]
        if bucket == "cls_error":
            err += 1
        elif bucket == "TN":
            tn += 1
        elif bucket == "FN":
            fn += 1
        elif bucket == "FP-should-miss":
            fp_should_miss += 1
        elif bucket == "TP":
            tp += 1
        elif bucket == "FP-wrong-match":
            fp_wrong += 1
        print(f"[{row['i']:3d}/{row['total']}] "
              f"{bucket:<15s} "
              f"gold={row['gold']!r:>20s}  "
              f"pred={row['pred']!r:>20s}  "
              f"({row['source']})")

    n_judged = len(examples) - err
    fp = fp_wrong + fp_should_miss
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    accuracy = (tp + tn) / n_judged if n_judged else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (tp + fp) and (tp + fn) else float("nan")
    )
    print()
    print(f"=== cls eval: {len(examples)} examples ({err} cls_errors excluded) ===")
    print(f"  TP                : {tp}")
    print(f"  FP-wrong-match    : {fp_wrong}")
    print(f"  FP-should-miss    : {fp_should_miss}")
    print(f"  FN                : {fn}")
    print(f"  TN                : {tn}")
    print(f"  precision  (TP / (TP+FP))  = {precision:.3f}")
    print(f"  recall     (TP / (TP+FN))  = {recall:.3f}")
    print(f"  F1                          = {f1:.3f}")
    print(f"  accuracy   ((TP+TN) / N)    = {accuracy:.3f}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "n_examples": len(examples),
            "n_judged": n_judged,
            "shots": str(args.shots) if args.shots else None,
            "model": args.model,
            "concurrency": args.concurrency,
            "tp": tp, "fp_wrong_match": fp_wrong,
            "fp_should_miss": fp_should_miss,
            "fn": fn, "tn": tn, "cls_error": err,
            "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy,
            "rows": [
                {k: v for k, v in row.items() if k != "total"}
                for row in rows
            ],
        }, indent=2))
        print(f"\n  → report written to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
