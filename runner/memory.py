"""Cross-session information layer (raw transcript).

Every session boundary invokes two hooks: `before_session` (read the
snapshot into the agent system prompt) and `after_session` (update stored
state from this session's transcript). The payoff / test phase runs in
parallel semantics: all test sessions consume a frozen snapshot taken at
the end of the learning phase, and none of them write back.

The benchmark uses a single layer, `ContextLayer` (selected as "raw"):
it renders the cleaned raw dialogue plus business API-call history as
text, injected into the system prompt `<context>` block. Runner
communication artifacts such as `send_to_user.reason` and delivery acks
are not exposed. No extraction, no retrieval.

Read interface: `get_context_string(query=None)` returns a text blob for
the system prompt `<context>`.
"""
from __future__ import annotations

import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import Message, MemoryEntry, SessionTrajectory
from _constants import (
    CONTEXT_HIDDEN_TOOL_NAMES as _CONTEXT_HIDDEN_TOOL_NAMES,
    FINISH_SESSION_TOOL_NAME as _FINISH_SESSION_TOOL_NAME,
    SEND_TO_USER_TOOL_NAME as _SEND_TO_USER_TOOL_NAME,
    USER_END_MARKER as _USER_END_MARKER,
)

logger = logging.getLogger(__name__)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _render_transcript(messages: list[Message]) -> str:
    """Render a session transcript as text for the ContextLayer.

    Preserves user-visible dialogue and business tool calls/results, while
    hiding runner control-plane artifacts:
      - send_to_user.reason and its delivered ack are hidden
      - send_to_user.output is rendered as assistant-visible dialogue
      - assistant content on tool-call turns is hidden as internal protocol
      - finish_session and its synthetic ack are hidden
    """
    lines: list[str] = []
    hidden_tool_call_ids: set[str] = set()
    for m in messages:
        if m.role == "system":
            continue
        content = (m.content or "").strip()
        if m.role == "user":
            if content == _USER_END_MARKER:
                continue
            lines.append(f"[user]: {_truncate(content, 500)}")
        elif m.role == "assistant":
            if m.tool_calls:
                seen_send_to_user = False
                for tc in m.tool_calls:
                    if tc.name in _CONTEXT_HIDDEN_TOOL_NAMES:
                        hidden_tool_call_ids.add(tc.id)

                    if tc.name == _SEND_TO_USER_TOOL_NAME:
                        if seen_send_to_user:
                            continue
                        seen_send_to_user = True
                        output = str((tc.arguments or {}).get("output") or "").strip()
                        if output:
                            lines.append(
                                f"[assistant to user]: {_truncate(output, 500)}"
                            )
                        continue

                    if tc.name == _FINISH_SESSION_TOOL_NAME:
                        continue

                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    lines.append(f"[tool call]: {tc.name}({_truncate(args_str, 300)})")
            elif content:
                lines.append(f"[assistant to user]: {_truncate(content, 500)}")
        elif m.role == "tool":
            if m.tool_call_id and m.tool_call_id in hidden_tool_call_ids:
                continue
            lines.append(f"[tool result]: {_truncate(content, 500)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frozen snapshot (consumed by test phase in parallel semantics)
# ---------------------------------------------------------------------------

class FrozenSnapshot:
    """Immutable view of the cross-session layer at a point in time.

    Carries the fixed_context string for system prompt injection.
    """

    def __init__(
        self,
        entries: list[MemoryEntry],
        fixed_context: str | None = None,
    ):
        self._entries = list(entries)
        self._fixed_context = fixed_context

    def get_context_string(self, query: str | None = None) -> str | None:
        return self._fixed_context

    def get_snapshot(self) -> list[MemoryEntry]:
        return list(self._entries)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CrossSessionLayer(ABC):
    """Abstract base for cross-session information layers.

    Implementations maintain their own internal store but share a uniform
    interface:
      - `get_context_string(query)`    — text blob for system prompt <context>
      - `after_session(trajectory)`    — ingest a finished session's transcript
      - `freeze()`                     — produce an immutable FrozenSnapshot
    All layers inject via `get_context_string()` into the system prompt.
    """

    @abstractmethod
    def get_context_string(self, query: str | None = None) -> str | None:
        """Return text blob for system prompt <context>."""

    @abstractmethod
    def get_snapshot(self) -> list[MemoryEntry]:
        """Return structured MemoryEntry list — may be empty for layers
        that don't decompose state into entries (e.g. ContextLayer)."""

    def before_session(self, session_id: str) -> None:
        """Default no-op. Layers can override to prepare per-session state."""
        return None

    @abstractmethod
    def after_session(self, trajectory: SessionTrajectory) -> None:
        """Ingest a completed session's transcript into the store.

        Only called at the end of learning sessions (test phase uses a
        frozen snapshot and never writes back).
        """

    @abstractmethod
    def freeze(self) -> FrozenSnapshot:
        """Produce an immutable snapshot of current state."""

    @classmethod
    def create(
        cls,
        kind: Literal["raw", "context", "native"],
        initial_entries: list[MemoryEntry] | None = None,
        model: str | None = None,
        seed: int | None = None,
    ) -> "CrossSessionLayer":
        """Factory. `raw` / `context` / `native` all map to ContextLayer
        (the raw-transcript layer used by the benchmark)."""
        if kind in ("raw", "context", "native"):
            return ContextLayer()
        raise ValueError(f"Unknown layer kind: {kind!r} (only raw is supported)")


# ---------------------------------------------------------------------------
# ContextLayer — raw transcript rendered as text
# ---------------------------------------------------------------------------

class ContextLayer(CrossSessionLayer):
    """Stores cleaned raw transcript from each learning session as text.

    On read, returns a single text blob for system prompt injection.
    Each session is rendered with `_render_transcript`, preserving
    user-visible dialogue and business tool calls/results in chronological
    order. Runner control-plane details are intentionally removed:
    send_to_user.reason, delivery acks, finish_session acks, and assistant
    content attached to tool-call turns do not enter <context>.

    Two entry kinds are tracked separately so the rendered context can label
    them honestly:
      - "session" : a real LS transcript (added via after_session)
      - "prior"   : a synthetic prior user statement (added via
                    inject_prior_user_statement; used by oracle variants)
    Each kind is independently numbered in the rendered output so the agent
    can tell "[Session 2]" (real interaction history) apart from
    "[Prior statement 2]" (oracle-injected canonical answer).
    """

    # Entry kinds — kept as class-level constants to avoid magic strings.
    _KIND_SESSION = "session"
    _KIND_PRIOR = "prior"

    def __init__(self):
        # Each entry: (kind, text) where kind in {_KIND_SESSION, _KIND_PRIOR}.
        # Append-only; ordering is preserved end-to-end (session N is always
        # rendered before session N+1, prior K before prior K+1).
        self._entries: list[tuple[str, str]] = []

    def get_context_string(self, query: str | None = None) -> str | None:
        if not self._entries:
            return None
        parts: list[str] = []
        session_idx = 0
        prior_idx = 0
        for kind, text in self._entries:
            if kind == self._KIND_SESSION:
                session_idx += 1
                parts.append(f"[Session {session_idx}]\n{text}")
            else:  # _KIND_PRIOR
                prior_idx += 1
                parts.append(f"[Prior statement {prior_idx}]\n[user]: {text}")
        return "\n\n".join(parts)

    def get_snapshot(self) -> list[MemoryEntry]:
        return []

    def after_session(self, trajectory: SessionTrajectory) -> None:
        text = _render_transcript(trajectory.messages)
        if text.strip():
            self._entries.append((self._KIND_SESSION, text))

    def inject_prior_user_statement(self, statement: str) -> None:
        """Inject a synthetic prior user statement.

        Used by oracle variants to seed the layer with rule canonical_answers
        without running learning sessions. Statement is rendered as a
        first-person user line under a "[Prior statement N]" header so the
        agent can distinguish it from real session transcripts.
        """
        self._entries.append((self._KIND_PRIOR, statement))

    def freeze(self) -> FrozenSnapshot:
        return FrozenSnapshot(
            entries=[],
            fixed_context=self.get_context_string(),
        )
