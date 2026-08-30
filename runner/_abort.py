"""Sweep-level abort signaling.

When any LLM module (agent / user_sim / Router / cls) exhausts its
internal retry budget, it raises `SweepAbort` instead of degrading or
writing `termination_reason="error" | "sim_error"`. The sweep runner
catches the exception in `runner/pipeline.py`, sets `ABORT_EVENT`, and
arranges remaining workers to:

  - finish their current session,
  - skip launching the next session (LS or TS) within the cell, and
  - never start cells that haven't begun yet.

, no `termination_reason` is written and no partial
trajectory is persisted for the offending session. On `--resume`, that
session is treated as unfinalized (same as never started) and re-runs.
"""
from __future__ import annotations

import threading


# Global abort event. Workers consult this at session boundaries —
# "finish current session, don't start next" — to honor the
# graceful-stop semantics in .
ABORT_EVENT = threading.Event()


class SweepAbort(RuntimeError):
    """LLM retry budget exhausted; sweep should stop gracefully.

    Caught at the sweep-runner level (`runner/pipeline.py`). The handler
    sets `ABORT_EVENT`, lets in-flight workers complete the current
    session, and prevents new cells / sessions from launching.

    Attributes:
        module: Which LLM-using component raised this (e.g.
                "agent", "user_sim", "router", "cls", "hook_retry").
        session_id: The session that was running when the abort fired
                    (None if pre-session phase).
        original: The underlying exception that caused retry exhaustion.
    """

    def __init__(
        self,
        module: str,
        session_id: str | None,
        original: BaseException,
    ) -> None:
        self.module = module
        self.session_id = session_id
        self.original = original
        super().__init__(
            f"SweepAbort: {module} LLM retry exhausted "
            f"(session={session_id!r}): "
            f"{type(original).__name__}: {original}"
        )
