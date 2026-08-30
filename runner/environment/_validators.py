"""Shared input validators for domain tools.

Centralizes the typing checks that the yaml ontology declares but the
Python signatures don't enforce on their own (datetime / date strings,
modify_*.new_value's per-field type). Each helper raises ValueError on
failure — toolkit method bodies let it propagate; ATREnv.get_response
wraps it as a structured tool error.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any


def parse_iso_datetime(value: Any, field_name: str) -> str:
    """Validate that `value` is an ISO 8601 datetime string. Returns the
    value verbatim on success; raises ValueError otherwise. Accepts the
    forms Python's `datetime.fromisoformat` understands (3.11+ — full
    ISO 8601 including timezone offsets like '-05:00' and 'Z').
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name}={value!r} is not a valid ISO 8601 datetime "
            f"(expected e.g. '2025-02-06T20:15:00-05:00')"
        )
    s = value.strip()
    # fromisoformat accepts 'Z' suffix only in 3.11+; normalize defensively.
    candidate = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        raise ValueError(
            f"{field_name}={value!r} is not a valid ISO 8601 datetime "
            f"(expected e.g. '2025-02-06T20:15:00-05:00')"
        )
    return s


def parse_iso_date(value: Any, field_name: str) -> str:
    """Validate ISO 8601 date (YYYY-MM-DD). Datetimes also accepted —
    `pause_until` style fields take either."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name}={value!r} is not a valid ISO 8601 date "
            f"(expected e.g. '2025-03-15' or full datetime)"
        )
    s = value.strip()
    candidate = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        date.fromisoformat(candidate[:10])
    except ValueError:
        raise ValueError(
            f"{field_name}={value!r} is not a valid ISO 8601 date "
            f"(expected e.g. '2025-03-15')"
        )
    # If a longer string was passed, also validate the full datetime parses
    if len(candidate) > 10:
        try:
            datetime.fromisoformat(candidate)
        except ValueError:
            raise ValueError(
                f"{field_name}={value!r} is not a valid ISO 8601 date/datetime"
            )
    return s


def parse_int(value: Any, field_name: str) -> int:
    """Coerce `value` to int. Accepts bare int and integer-shaped str."""
    if isinstance(value, bool):
        # bool is a subclass of int in Python; reject explicitly to avoid
        # treating True/False as 1/0 silently.
        raise ValueError(f"{field_name}={value!r} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError(f"{field_name}={value!r} is not an integer")


def parse_enum(value: Any, allowed: set[str], field_name: str) -> str:
    """Validate `value` is in the allowed enum set."""
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"{field_name}={value!r} not in allowed values: {sorted(allowed)}"
        )
    return value
