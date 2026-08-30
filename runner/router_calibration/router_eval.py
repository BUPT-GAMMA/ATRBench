"""Run Router against a labeled test set.

Usage:
    uv run python -m runner.router_calibration.router_eval \
        --test-set runner/router_calibration/test_set.jsonl \
        [--shots runner/router_calibration/shots/v1.json] \
        [--model gemini-3-flash-preview] [--workers 15]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "runner"))

from lib.llm import GEMINI  # noqa: E402
from runner.router import route_agent_turn  # noqa: E402


def _iter_examples(path: Path) -> Iterator[dict]:
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield json.loads(line)


def _bucket(pred: bool, gold: bool, error: str | None) -> str:
    if error:
        return "router_error"
    if pred and gold:
        return "TP"
    if pred and not gold:
        return "FP"
    if not pred and gold:
        return "FN"
    return "TN"


def _score(rows: list[dict]) -> dict:
    tp = fp = fn = tn = err = 0
    span_exact = span_missing = span_not_verbatim = span_mismatch = 0
    span_eval_total = 0
    for row in rows:
        bucket = row["bucket"]
        if bucket == "TP":
            tp += 1
            span_eval_total += 1
            if row["pred_span"] is None:
                span_missing += 1
            elif row["pred_span"] not in row["output"]:
                span_not_verbatim += 1
            elif row["pred_span"] == row["gold_span"]:
                span_exact += 1
            else:
                span_mismatch += 1
        elif bucket == "FP":
            fp += 1
        elif bucket == "FN":
            fn += 1
        elif bucket == "TN":
            tn += 1
        elif bucket == "router_error":
            err += 1
    judged = len(rows) - err
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    accuracy = (tp + tn) / judged if judged else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    span_accuracy = span_exact / span_eval_total if span_eval_total else None
    return {
        "n_examples": len(rows),
        "n_judged": judged,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "router_error": err,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "span_eval_total": span_eval_total,
        "span_exact": span_exact,
        "span_missing": span_missing,
        "span_not_verbatim": span_not_verbatim,
        "span_mismatch": span_mismatch,
        "span_accuracy": span_accuracy,
    }


def _run_one(i: int, ex: dict, model: str, shots: Path | None, seed: int | None) -> dict:
    result = route_agent_turn(
        reason=str(ex.get("reason") or ""),
        output=str(ex.get("output") or ""),
        model=model,
        seed=seed,
        fewshot_path=shots,
    )
    pred = bool(result.get("is_strict_rule_question"))
    gold = bool(ex.get("gold_is_strict_rule_question"))
    error = result.get("error")
    return {
        "i": i,
        "source": ex.get("source", "-"),
        "output": str(ex.get("output") or ""),
        "gold": gold,
        "pred": pred,
        "gold_span": ex.get("gold_rule_question_span"),
        "pred_span": result.get("rule_question_span"),
        "router_error": error,
        "bucket": _bucket(pred, gold, error),
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("runner/router_calibration/test_set.jsonl"),
    )
    parser.add_argument(
        "--shots",
        type=Path,
        default=None,
        help="Shot bank to use. Omit to use Router default/env resolution.",
    )
    parser.add_argument("--model", default=GEMINI)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    examples = list(_iter_examples(args.test_set))
    if not examples:
        print(f"No examples found in {args.test_set}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    if args.workers <= 1:
        for i, ex in enumerate(examples, 1):
            rows.append(_run_one(i, ex, args.model, args.shots, args.seed))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(_run_one, i, ex, args.model, args.shots, args.seed)
                for i, ex in enumerate(examples, 1)
            ]
            for fut in as_completed(futures):
                rows.append(fut.result())
        rows.sort(key=lambda r: r["i"])

    for row in rows:
        print(
            f"[{row['i']:3d}/{len(examples)}] {row['bucket']:<12s} "
            f"gold={row['gold']!s:<5s} pred={row['pred']!s:<5s} "
            f"span={'ok' if row['bucket'] != 'TP' or row['gold_span'] == row['pred_span'] else 'bad'} "
            f"({row['source']})"
        )

    summary = _score(rows)
    print()
    print(
        f"=== router eval: {summary['n_examples']} examples "
        f"({summary['router_error']} router_errors excluded) ==="
    )
    print(f"  TP        : {summary['tp']}")
    print(f"  FP        : {summary['fp']}")
    print(f"  FN        : {summary['fn']}")
    print(f"  TN        : {summary['tn']}")
    print(f"  precision : {_fmt(summary['precision'])}")
    print(f"  recall    : {_fmt(summary['recall'])}")
    print(f"  F1        : {_fmt(summary['f1'])}")
    print(f"  accuracy  : {_fmt(summary['accuracy'])}")
    print(f"  span exact: {summary['span_exact']}/{summary['span_eval_total']} "
          f"({_fmt(summary['span_accuracy'])})")
    print(f"    missing      : {summary['span_missing']}")
    print(f"    not verbatim : {summary['span_not_verbatim']}")
    print(f"    mismatch     : {summary['span_mismatch']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            **summary,
            "test_set": str(args.test_set),
            "shots": str(args.shots) if args.shots else None,
            "model": args.model,
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"\n  -> report written to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
