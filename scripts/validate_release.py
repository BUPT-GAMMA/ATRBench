"""Validate the self-contained released benchmark without API calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.schemas import Episode
from lib.benchmark_data import tool_map


PERSONAS = ROOT / "data" / "personas"
EXPECTED_PERSONAS = 20
EXPECTED_RULES = 284
EXPECTED_LEARNING = 568
EXPECTED_TEST = 284
EXPECTED_TOOLS = 74


def main() -> None:
    episode_paths = sorted(PERSONAS.glob("*/episodes/*_seed000.json"))
    if len(episode_paths) != EXPECTED_PERSONAS:
        raise SystemExit(
            f"expected {EXPECTED_PERSONAS} released episodes, found {len(episode_paths)}"
        )

    total_rules = total_learning = total_test = 0
    for path in episode_paths:
        episode = Episode.model_validate(json.loads(path.read_text(encoding="utf-8")))
        rule_ids = {rule.rule_id for rule in episode.rules}
        test_rule_ids = {session.rule_id for session in episode.test_sessions}
        if rule_ids != test_rule_ids:
            raise SystemExit(f"rule/test mismatch: {path}")
        if episode.metadata.get("trajectory_length") != len(episode.learning_sessions):
            raise SystemExit(f"learning metadata mismatch: {path}")
        if episode.metadata.get("K") != len(episode.test_sessions) * 2:
            raise SystemExit(f"K metadata mismatch: {path}")
        total_rules += len(episode.rules)
        total_learning += len(episode.learning_sessions)
        total_test += len(episode.test_sessions)

    actual = (total_rules, total_learning, total_test, len(tool_map()))
    expected = (EXPECTED_RULES, EXPECTED_LEARNING, EXPECTED_TEST, EXPECTED_TOOLS)
    if actual != expected:
        raise SystemExit(f"release counts mismatch: expected {expected}, found {actual}")
    print(
        "ATRBench release OK: "
        f"{EXPECTED_PERSONAS} personas, {EXPECTED_RULES} rules, "
        f"{EXPECTED_LEARNING} learning sessions, {EXPECTED_TEST} test sessions, "
        f"{EXPECTED_TOOLS} tools"
    )


if __name__ == "__main__":
    main()
