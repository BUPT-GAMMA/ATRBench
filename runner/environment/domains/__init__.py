"""Domain registry for ATR environments.

Each domain provides:
  - A DB class (ATRDB subclass)
  - A ToolKit class (ATRToolKitBase subclass)
  - build_db(session_task, persona) -> DB

The registry maps domain_name → builder callable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from runner.environment.base import ATRDB, ATRToolKitBase


class DomainBuilder(Protocol):
    def build_env(self, session_task, persona, allowed_tools): ...


from runner.environment.domains import (  # noqa: E402
    commerce, reservation, travel, communication, scheduling, workspace,
)

_REGISTRY: dict[str, Callable] = {
    "commerce": commerce.build_env,
    "reservation": reservation.build_env,
    "travel": travel.build_env,
    "communication": communication.build_env,
    "scheduling": scheduling.build_env,
    "workspace": workspace.build_env,
}


def build_env_for_session(
    session_task,
    persona,
    seed: int | None = None,
):
    """Dispatch to the domain-specific env builder.

    seed (optional) is stamped on the DB and used to initialize the stdlib
    RNG — threaded through for τ²-parity reproducibility. Pass through from
    the episode / evaluator config so runs are replay-stable.
    """
    domain = session_task.domain
    if domain not in _REGISTRY:
        raise ValueError(
            f"Unknown domain: {domain}. Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[domain](
        session_task=session_task,
        persona=persona,
        allowed_tools=session_task.local_env.tools,
        seed=seed,
    )


def register_domain(name: str, builder: Callable) -> None:
    _REGISTRY[name] = builder


def registered_domains() -> list[str]:
    return sorted(_REGISTRY.keys())
