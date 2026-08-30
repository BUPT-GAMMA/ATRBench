"""Shared utilities for comparing rule-bound tool-call arguments against
agent-generated trajectories.

Lives outside `evaluator/` so runtime components can share the same safe-type
normalization rules.
"""
from __future__ import annotations

import json
from typing import Any


def safe_for_compare(name: str, type_name: str) -> bool:
    """A compare_args slot is safe iff its type+name combination supports
    exact-equality matching against an LLM-generated trajectory.

    Allowed:
      - boolean
      - enum[...] (closed value space)
      - string named *_id (id pool with closed membership) or `target_tool`
      - array[string] named *_ids (set of ids)
      - object named `target_params` (dict whose safe-typed sub-fields
        match against the target_tool's signature at compare time —
        the dict itself isn't exact-matched, so accepting it is safe)

    Rejected:
      - date / datetime (paraphrase-brittle)
      - integer / number (JSON-vs-string-coerced LLM output)
      - free-text string (open content)
      - array[string] not _ids (open content list)
    """
    if type_name == "boolean":
        return True
    if type_name.startswith("enum["):
        return True
    if type_name == "string" and (name.endswith("_id") or name == "target_tool"):
        return True
    if type_name == "array[string]" and name.endswith("_ids"):
        return True
    if type_name == "object" and name == "target_params":
        return True
    return False


def coerce_target_params(value: Any) -> Any:
    """Normalize an LLM-produced `target_params` value to a dict (or None).

    `target_params` is declared as `type: object` in the ontology, but
    different LLMs serialize nested objects differently in tool calls:

      - GPT-family / Claude: emit a native dict — {"reservation_id": "RK72M"}
      - Gemini-family: tend to emit a JSON-encoded string —
        '{"reservation_id": "RK72M"}'

    Both are legitimate representations of the same nested-object value
    under OpenAI-style function-calling specs (`"type": "object"` is
    permissive about JSON-string vs native dict). For benchmark
    comparability across model families, the evaluator coerces both
    shapes to a dict before comparison.

    Returns:
      - dict (input dict): unchanged
      - str (input JSON-encoded dict): json.loads result if it parses to
        a dict; else the original string (caller's equality check will
        then fail naturally)
      - None / anything else: unchanged

    Caller decides what to do with non-dict outputs; this function only
    *normalizes* what's normalizable.
    """
    if isinstance(value, dict) or value is None:
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
        return parsed if isinstance(parsed, dict) else value
    return value
