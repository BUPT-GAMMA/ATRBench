"""Declarative table specs for ATR domain databases.

Single source of truth for table metadata consumed by:
  - ATRDB.hydrate_all()           — env auto-builds tables from refs
  - episode validation            — secondary-entity coverage check
  - learning-session environments — search tool → ref type mapping
  - agent.md prompt renderer      — ID discovery paths section
  - binding validator             — reject rules targeting unconstructable tools

Three table kinds:
  primary   — id == ref.id, ref carries the full attribute payload
  derived   — id from a primary ref's attribute (e.g. orders <- product.order_id)
  runtime   — created by write tools, never hydrated from refs
"""
from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TableSpec(BaseModel):
    """Declarative spec for one DB table in a domain."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    model: Optional[type[BaseModel]] = None
    kind: Literal["primary", "derived", "runtime"]

    # primary: ref_type that lifts to this table (e.g. "product")
    # derived: source primary ref_type (e.g. "product" for orders)
    # runtime: None
    source_ref_type: Optional[str] = None

    # primary only: alternate ref_type strings accepted as aliases
    # (e.g. scheduling.events accepts both "calendar_event" and "event").
    aliases: list[str] = Field(default_factory=list)

    # derived only: attribute name on the source ref that carries this
    # table's id (e.g. "order_id" for commerce.orders)
    source_attr: Optional[str] = None

    # primary only: attributes to promote to first-class model fields.
    promoted_attrs: list[str] = Field(default_factory=list)

    # Tools that operate on rows of this table by id.
    operating_tools: list[str] = Field(default_factory=list)

    # Tools through which an agent can discover this table's row ids.
    discovery_tools: list[str] = Field(default_factory=list)

    # primary only: (ref_id, attrs, persona, spec) -> kwargs for model(**)
    build_row: Optional[Callable] = None

    # derived only: (source_ref_id, source_attrs, persona, spec) -> kwargs
    derive_row: Optional[Callable] = None

    # derived only: (existing_row, new_source_ref_id, new_attrs, persona)
    # -> updated_row. Default: first-wins dedup.
    merge_row: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, list[TableSpec]] = {}
_LOADED: bool = False


def register_domain_tables(domain: str, specs: list[TableSpec]) -> None:
    """Called by each domain module at import time."""
    primary_ref_types: set[str] = set()
    for s in specs:
        if s.kind == "primary" and s.source_ref_type:
            primary_ref_types.add(s.source_ref_type)
            primary_ref_types.update(s.aliases or [])
    for s in specs:
        if s.kind == "derived" and s.source_ref_type is not None:
            if s.source_ref_type not in primary_ref_types:
                raise ValueError(
                    f"{domain}.{s.name}: derived table source_ref_type="
                    f"{s.source_ref_type!r} but no primary with that "
                    f"ref_type in this domain"
                )
            if not s.source_attr:
                raise ValueError(
                    f"{domain}.{s.name}: derived needs source_attr"
                )
    _REGISTRY[domain] = specs


def _ensure_loaded() -> None:
    """Lazy import all domain modules so register_domain_tables fires."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from runner.environment import domains  # noqa: F401


def all_table_specs() -> list[tuple[str, TableSpec]]:
    """All registered (domain, spec) pairs."""
    _ensure_loaded()
    out = []
    for domain in sorted(_REGISTRY):
        for s in _REGISTRY[domain]:
            out.append((domain, s))
    return out


def specs_for_domain(domain: str) -> list[TableSpec]:
    _ensure_loaded()
    return list(_REGISTRY.get(domain, []))


def spec_for_tool(tool_name: str) -> Optional[tuple[str, TableSpec]]:
    """Find the (domain, table) that operates on this tool."""
    _ensure_loaded()
    for domain, specs in _REGISTRY.items():
        for s in specs:
            if tool_name in s.operating_tools:
                return domain, s
    return None
