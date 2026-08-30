"""Session-scoped executable environment layer for ATR.

Each session builds a fresh database from its local references and routes
validated tool calls through typed domain toolkits.

Public API:
    ATRDB                       Pydantic base for domain databases
    ATRToolKitBase              Subclass this + decorate methods with @is_tool
    @is_tool(ToolType.READ)     Register a method as a callable tool
    ATRTool                     Wraps a bound method, exposes openai_schema
    ATREnv                      Holds a toolkit, routes tool calls via get_response
    ToolResponse                (content: str, is_error: bool)
"""
from __future__ import annotations

import inspect
import json
import logging
import random
import re
import typing
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
from typing import Any, Callable, ClassVar, Literal, Optional

from docstring_parser import parse as parse_docstring
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persona (episode-level, shared across all sessions in an episode)
# ---------------------------------------------------------------------------

class PersonaProfile(BaseModel):
    """Episode-level user identity, shared by all sessions in an episode.

    `default_shipping_address` / `default_payment_method` / `default_contact`
    are consumed inside tool implementations (commerce.place_order,
    travel / reservation booking tools) as auto-fills when the agent omits
    the corresponding identity arg. The agent cannot read them directly.

    `narrative` is the raw persona text consumed by user_sim and
    route_agent_text for tone/style grounding — not surfaced to the agent.

    `home_city` / `home_zone` are identity projections kept on the model
    for future projection-based rules; no tool currently consumes them.
    """
    persona_id: str
    narrative: str = ""
    home_city: str = ""
    home_zone: str = ""
    default_shipping_address: str = ""
    default_payment_method: str = ""
    default_contact: str = ""


# ---------------------------------------------------------------------------
# DB base
# ---------------------------------------------------------------------------

class ATRDB(BaseModel):
    """Base class for domain databases.

    Subclasses define typed fields (e.g. products, orders). `persona` is
    inherited from the base and shared via the episode. Built fresh from
    session_task.local_env.references + the episode-level PersonaProfile.
    Session-scoped: discarded at session end.

    Subclasses MUST declare `_KNOWN_REF_TYPES` — the set of reference types
    their `from_references` classmethod knows how to lift. The base provides
    `_warn_unknown_ref_types` which scans the incoming references and logs a
    warning for each unrecognized type. Unknown types are the single most
    common authoring footgun (`type: "item"` vs `"product"` quietly drops
    the reference and the session runs against an empty DB).
    """

    persona: PersonaProfile

    model_config = ConfigDict(extra="forbid")

    # Subclasses override with the ref_type values their from_references consumes.
    _KNOWN_REF_TYPES: ClassVar[set[str]] = set()

    # Declarative table specs — subclasses populate for hydrate_all.
    _TABLES: ClassVar[list] = []

    # Seed stamped on the DB at env construction. None means "not seeded"
    # — consumers (data generators, future stochastic tools) should treat
    # that as non-reproducible. Set by ATREnv.__init__(seed=...).
    _seed: Optional[int] = PrivateAttr(default=None)

    @classmethod
    def _warn_unknown_ref_types(cls, references: list) -> None:
        """Log once per unknown ref_type. References are accepted as either
        ReferenceObject instances or plain dictionaries.
        """
        seen: set[str] = set()
        for ref in references:
            _, rt, _ = cls._unpack_ref(ref)
            if rt and rt not in cls._KNOWN_REF_TYPES and rt not in seen:
                seen.add(rt)
                logger.warning(
                    "Unknown reference type %r for %s — dropping. "
                    "Known types for this domain: %s. "
                    "Check REF_TYPE_SCHEMA.md for the supported set.",
                    rt, cls.__name__, sorted(cls._KNOWN_REF_TYPES) or ["(none declared)"],
                )

    @staticmethod
    def _unpack_ref(ref: Any) -> tuple[str | None, str | None, dict[str, Any]]:
        """Normalize a reference to (id, type, attributes).

        Accepts ReferenceObject instances (Pydantic) and raw dictionaries.
        Handles empty attributes safely — previously `getattr(...) or ref.get(...)`
        short-circuited on {} and crashed with AttributeError.
        """
        if isinstance(ref, dict):
            return ref.get("id"), ref.get("type"), ref.get("attributes") or {}
        return (
            getattr(ref, "id", None),
            getattr(ref, "type", None),
            getattr(ref, "attributes", None) or {},
        )

    def get_statistics(self) -> dict[str, Any]:
        return {}

    @classmethod
    def hydrate_all(
        cls,
        persona: "PersonaProfile",
        references: list,
    ) -> "ATRDB":
        """Build a session DB from refs by walking the _TABLES declaration.

        Phase 1: hydrate primary tables (each ref of declared ref_type).
        Phase 2: hydrate derived tables (each primary ref carrying the
                  declared source_attr).
        Runtime tables stay empty — write tools populate them later.
        """
        from runner.environment.tables import TableSpec

        specs: list[TableSpec] = cls._TABLES
        if not specs:
            raise ValueError(
                f"{cls.__name__} has no _TABLES — use from_references or "
                f"populate _TABLES for hydrate_all"
            )

        cls._warn_unknown_ref_types(references)

        primary_tables: dict[str, dict] = {}
        derived_tables: dict[str, dict] = {}
        runtime_tables: dict[str, dict] = {}

        primary_by_ref: dict[str, TableSpec] = {}
        derived_by_source_ref: dict[str, list[TableSpec]] = {}
        ref_alias_map: dict[str, str] = {}

        for t in specs:
            if t.kind == "primary":
                primary_tables[t.name] = {}
                if t.source_ref_type:
                    primary_by_ref[t.source_ref_type] = t
                    for alias in (t.aliases or []):
                        ref_alias_map[alias] = t.source_ref_type
            elif t.kind == "derived":
                derived_tables[t.name] = {}
                if t.source_ref_type:
                    derived_by_source_ref.setdefault(t.source_ref_type, []).append(t)
            elif t.kind == "runtime":
                runtime_tables[t.name] = {}

        for ref in references:
            ref_id, ref_type, attrs = cls._unpack_ref(ref)
            if not ref_id or not ref_type:
                continue
            ref_type = ref_alias_map.get(ref_type, ref_type)

            primary_spec = primary_by_ref.get(ref_type)
            if primary_spec and primary_spec.build_row:
                row_kwargs = primary_spec.build_row(
                    ref_id, attrs, persona, primary_spec
                )
                primary_tables[primary_spec.name][ref_id] = (
                    primary_spec.model(**row_kwargs)
                )

            for derived_spec in derived_by_source_ref.get(ref_type, []):
                derived_id = attrs.get(derived_spec.source_attr)
                if not derived_id:
                    continue
                existing = derived_tables[derived_spec.name].get(derived_id)
                if existing is not None:
                    if derived_spec.merge_row:
                        derived_tables[derived_spec.name][derived_id] = (
                            derived_spec.merge_row(
                                existing, ref_id, attrs, persona
                            )
                        )
                    continue
                if derived_spec.derive_row:
                    row_kwargs = derived_spec.derive_row(
                        ref_id, attrs, persona, derived_spec
                    )
                    derived_tables[derived_spec.name][derived_id] = (
                        derived_spec.model(**row_kwargs)
                    )

        all_tables = {**primary_tables, **derived_tables, **runtime_tables}
        return cls(persona=persona, **all_tables)


# ---------------------------------------------------------------------------
# Tool type taxonomy
# ---------------------------------------------------------------------------

class ToolType(str, Enum):
    READ = "read"
    WRITE = "write"
    THINK = "think"
    GENERIC = "generic"


# τ²-bench attribute conventions — matched verbatim so any τ² tutorial /
# reference that inspects tool metadata works on our ToolKits too.
_TOOL_ATTR = "__tool__"
_TOOL_TYPE_ATTR = "__tool_type__"
_MUTATES_STATE_ATTR = "__mutates_state__"


def is_tool(
    tool_type: ToolType = ToolType.READ,
    mutates_state: Optional[bool] = None,
) -> Callable:
    """Decorator: mark a ToolKit method as a callable tool.

    Args:
        tool_type: READ / WRITE / THINK / GENERIC. Drives prompt taxonomy
            and (via mutates_state) replay semantics.
        mutates_state: Whether this tool modifies DB state. If None,
            defaults to True for WRITE and False for everything else.
            WRITE tools are re-executed during trajectory replay; READ /
            THINK tools reuse the recorded response.
    """
    if mutates_state is None:
        mutates_state = tool_type == ToolType.WRITE

    def decorator(func: Callable) -> Callable:
        setattr(func, _TOOL_ATTR, True)
        setattr(func, _TOOL_TYPE_ATTR, tool_type)
        setattr(func, _MUTATES_STATE_ATTR, mutates_state)
        return func

    return decorator


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------

class ATRTool:
    """Wraps a bound ToolKit method and exposes an OpenAI-compatible schema.

    Follows the τ²-bench Tool contract: the Python function signature + its
    docstring are the single source of truth. docstring-parser extracts
    short_desc / long_desc / Args descriptions / Returns / Raises / Examples;
    `params` and `returns` are dynamic Pydantic models generated via
    `create_model`. Parameters with no type hint are typed as Any.
    """

    def __init__(self, func: Callable):
        self._func = func
        self.name: str = func.__name__
        doc = parse_docstring(func.__doc__ or "")
        self.short_desc: str = (doc.short_description or "").strip()
        self.long_desc: str = (doc.long_description or "").strip()
        self.params_model: type[BaseModel] = self._build_params_model(func, doc)
        self.returns_model: type[BaseModel] = self._build_returns_model(func, doc)
        self.raises: list[dict[str, str]] = [
            {"type": e.type_name or "", "desc": e.description or ""}
            for e in doc.raises
        ]
        self.examples: list[str] = [
            (ex.description or "").strip() for ex in doc.examples
        ]
        self.tool_type: ToolType = getattr(func, _TOOL_TYPE_ATTR, ToolType.GENERIC)
        self.mutates_state: bool = getattr(
            func, _MUTATES_STATE_ATTR, self.tool_type == ToolType.WRITE
        )

    @staticmethod
    def _build_params_model(func: Callable, doc: Any) -> type[BaseModel]:
        sig = inspect.signature(func)
        # Resolve annotations in the function's defining module scope so type
        # aliases (e.g. `SelectionBasis = Literal[...]`) resolve correctly.
        try:
            resolved = typing.get_type_hints(func)
        except Exception:
            resolved = {}
        doc_params = {p.arg_name: p for p in doc.params}
        fields: dict[str, tuple[Any, Any]] = {}
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            anno = resolved.get(pname, param.annotation)
            if anno is inspect.Parameter.empty:
                anno = Any
            default: Any = ... if param.default is inspect.Parameter.empty else param.default
            description = ""
            if pname in doc_params and doc_params[pname].description:
                description = doc_params[pname].description.strip()
            if description:
                default = Field(default, description=description)
            fields[pname] = (anno, default)
        model = create_model(f"{func.__name__}_params", **fields)  # type: ignore
        try:
            model.model_rebuild()
        except Exception:
            pass
        return model

    @staticmethod
    def _build_returns_model(func: Callable, doc: Any) -> type[BaseModel]:
        sig = inspect.signature(func)
        try:
            resolved = typing.get_type_hints(func)
        except Exception:
            resolved = {}
        anno = resolved.get("return", sig.return_annotation)
        if anno is inspect.Signature.empty:
            anno = Any
        description = ""
        if doc.returns and doc.returns.description:
            description = doc.returns.description.strip()
        return create_model(
            f"{func.__name__}_returns",
            returns=(anno, Field(..., description=description)),
        )

    # ── Description accessors ────────────────────────────────────────────
    # `description` is the short form used in tool schemas (OpenAI
    # function_def.description); `_get_description` is the τ² helper that
    # composes short + long. Kept as a property for back-compat with
    # earlier ATR code that read `tool.description`.

    @property
    def description(self) -> str:
        return self.short_desc

    def _get_description(self) -> str:
        if self.long_desc:
            return f"{self.short_desc}\n\n{self.long_desc}"
        return self.short_desc

    @staticmethod
    def _flatten_anyof(schema: dict) -> dict:
        """Recursively flatten anyOf [{type: X}, {type: null}] → {type: X, nullable: true}.

        Pydantic v2 emits ``anyOf`` for ``Optional[T]`` fields. Gemini rejects
        ``anyOf`` — it requires a direct ``type`` key on every property. GPT and
        DeepSeek both accept ``nullable`` so this conversion is safe across all
        providers we use.
        """
        if "anyOf" in schema:
            alts = schema["anyOf"]
            nulls = [a for a in alts if a.get("type") == "null"]
            others = [a for a in alts if a.get("type") != "null"]
            if len(nulls) == 1 and len(others) == 1:
                merged = {**schema, **others[0]}
                merged.pop("anyOf", None)
                merged["nullable"] = True
                schema = merged
        for key, val in list(schema.items()):
            if isinstance(val, dict):
                schema[key] = ATRTool._flatten_anyof(val)
            elif isinstance(val, list):
                schema[key] = [
                    ATRTool._flatten_anyof(v) if isinstance(v, dict) else v
                    for v in val
                ]
        return schema

    # Keys whose values are dicts of {name: sub-schema}. The dict KEYS are
    # user-defined names (e.g. real param names like "title"), not metadata,
    # so we must only recurse into the values — never pop "title" off them.
    _PROPERTY_BAG_KEYS = ("properties", "$defs", "definitions", "patternProperties")

    @staticmethod
    def _strip_pydantic_noise(schema: dict) -> dict:
        """Recursively drop `default`, `title`, `nullable` METADATA from a JSON Schema.

        `default` / `title` are pydantic-generated metadata noise; the OpenAI
        tool spec doesn't require them.

        `nullable: true` is OpenAPI 3.0, not JSON Schema. Some providers
        (observed: deepseek-v4-flash) reject it with "null is not of type X"
        when applied to typed array/object fields. Optional params don't
        need `nullable` — omitting the field from a tool call (when not in
        `required`) is sufficient. Stripping it keeps the schema portable.

        IMPORTANT: only strip these as metadata at the schema-node level.
        A `properties` (or `$defs`) dict uses user-chosen names as keys; if
        a tool happens to declare a field literally named `title` (see
        `scheduling.CalendarEventArgs.title`), pop()-ing "title" from
        `properties` would delete the field while leaving `required: ["title"]`
        — Gemini's strict tool-schema validator rejects that with
        "required fields ['title'] are not defined in the schema properties".
        GPT/DeepSeek tolerate the inconsistency, which is why this only
        surfaced once we tried gemini-3.1-* as the agent model.
        """
        if not isinstance(schema, dict):
            return schema
        schema.pop("default", None)
        schema.pop("title", None)
        schema.pop("nullable", None)
        for k, v in list(schema.items()):
            if k in ATRTool._PROPERTY_BAG_KEYS and isinstance(v, dict):
                # Recurse into each named sub-schema, but don't strip metadata
                # off the bag itself — its keys are property names.
                schema[k] = {
                    name: ATRTool._strip_pydantic_noise(sub) if isinstance(sub, dict) else sub
                    for name, sub in v.items()
                }
            elif isinstance(v, dict):
                schema[k] = ATRTool._strip_pydantic_noise(v)
            elif isinstance(v, list):
                schema[k] = [
                    ATRTool._strip_pydantic_noise(x) if isinstance(x, dict) else x
                    for x in v
                ]
        return schema

    @property
    def openai_schema(self) -> dict[str, Any]:
        raw = self.params_model.model_json_schema()
        params = self._strip_pydantic_noise(self._flatten_anyof(raw))
        # Pydantic omits `required` entirely when no field is required; some
        # providers (observed: deepseek-v4-flash) treat the absent key as
        # `null` and fail strict-schema validation with
        # "null is not of type array". Ensuring an empty list is present
        # keeps the spec well-formed across providers.
        if isinstance(params, dict) and "required" not in params:
            params["required"] = []
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self._get_description(),
                "parameters": params,
            },
        }

    def to_text_description(self) -> str:
        """Render as a compact text description for text-mode prompts."""
        schema = self.params_model.model_json_schema()
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        param_lines = []
        for pname, pschema in props.items():
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            enum = pschema.get("enum")
            req_mark = "required" if pname in required else "optional"
            type_str = ptype
            if enum:
                type_str = f"{ptype}, enum: {enum}"
            param_lines.append(f"  - {pname} ({type_str}, {req_mark}): {pdesc}")
        params_block = "\n".join(param_lines) if param_lines else "  (no params)"
        return f"### {self.name}\n{self.description}\nParams:\n{params_block}"

    def __call__(self, **kwargs: Any) -> Any:
        return self._func(**kwargs)


def as_tool(func: Callable) -> ATRTool:
    return ATRTool(func)


# ---------------------------------------------------------------------------
# ToolKit base
# ---------------------------------------------------------------------------

class _ToolKitMeta(type):
    """Metaclass that collects @is_tool-decorated methods into _func_tools."""

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        tools: dict[str, Callable] = {}
        # Inherit from base
        for base in bases:
            inherited = getattr(base, "_func_tools", {})
            tools.update(inherited)
        for attr_name, method in attrs.items():
            fn = method.fget if isinstance(method, property) else method
            if callable(fn) and getattr(fn, _TOOL_ATTR, False):
                tools[attr_name] = fn
        cls._func_tools = tools


class ATRToolKitBase(metaclass=_ToolKitMeta):
    """Base class for domain toolkits.

    Subclasses bind a typed ATRDB and decorate methods with @is_tool.
    Tools are auto-collected via the metaclass.

    Base tool available in every domain: get_user_confirmation.
    """

    _func_tools: dict[str, Callable] = {}
    # Subclass hook consumed by ATREnv:
    #   _shape_guards: {tool_name -> {exactly_one_of | at_least_one_of: [...]}}
    #     Declarative structural guards the env enforces before dispatch.
    # Id params are not schema-narrowed: enum narrowing leaked the full id
    # set and let agents skip search/list calls. Unknown ids now raise
    # "Not found" from method bodies (see tools.yaml id_discovery_convention).
    _shape_guards: dict[str, dict] = {}

    def __init__(self, db: ATRDB):
        self.db = db

    @is_tool(ToolType.READ)
    def get_user_confirmation(
        self,
        target_tool: str,
        target_params: dict | None = None,
    ) -> dict:
        """Obtain user confirmation before executing a destructive or
        hard-to-reverse action (cancel_*, delete_*, archive_*, send_*,
        modify_* on already-placed items, etc.).

        Confirmation is a first-class tool call rather than a self-reported
        boolean flag so the trajectory records an auditable turn of
        agent→user→agent before the destructive call. Evaluators perform
        paired matching: a call X(params) is accepted only if the same
        session's trajectory contains a preceding
        get_user_confirmation(target_tool='X', target_params=<safe-typed
        subset matches>). The match is selective — only safe-typed fields
        (*_id / *_ids / enum / boolean) participate; free-text content
        and datetime fields are ignored.

        Args:
            target_tool: The name of the tool the agent intends to call
                next (e.g. "cancel_event", "delete_files", "place_order").
                Must be a tool available in this session's allowed_tools.
            target_params: Optional full arguments dict the agent plans to
                pass to target_tool, quoted verbatim. May be omitted for
                confirm-only rules where only the target tool identity is
                being confirmed.

        Returns:
            A dict with keys "confirmed" (bool), "target_tool" (echo), and
            "target_params" (echo). The base stub returns confirmed=True;
            a user-simulator-backed runtime overrides this via
            toolkit._confirmation_policy (future hook — v1 leaves it
            permissive).
        """
        target_params = target_params or {}
        return {
            "confirmed": True,
            "target_tool": target_tool,
            "target_params": target_params,
        }

    def get_tools(self, include: Optional[list[str]] = None) -> dict[str, ATRTool]:
        """Return {name: ATRTool} for all registered tools.

        Args:
            include: If provided, only return tools whose names are in this list.
                Raises ValueError if an included name is not registered.
        """
        all_tools = {
            name: as_tool(getattr(self, name))
            for name in self._func_tools.keys()
        }
        if include is None:
            return all_tools
        allowed = set(include)
        unknown = allowed - set(all_tools.keys())
        if unknown:
            raise ValueError(
                f"Tool(s) not registered: {sorted(unknown)}. "
                f"Available: {sorted(all_tools.keys())}"
            )
        return {n: t for n, t in all_tools.items() if n in allowed}

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._func_tools

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self._func_tools:
            raise ValueError(f"Tool '{tool_name}' not registered.")
        method = getattr(self, tool_name)
        return method(**kwargs)

    def tool_type(self, tool_name: str) -> ToolType:
        fn = self._func_tools[tool_name]
        return getattr(fn, _TOOL_TYPE_ATTR, ToolType.GENERIC)

    def tool_mutates_state(self, tool_name: str) -> bool:
        """Whether a given tool modifies DB state (τ² replay hook).

        Defaults: WRITE → True, everything else → False. Tools can override
        via `@is_tool(mutates_state=...)`. Used by trajectory-replay logic
        to decide whether to re-execute a tool or reuse its recorded output.
        """
        fn = self._func_tools[tool_name]
        default = getattr(fn, _TOOL_TYPE_ATTR, ToolType.GENERIC) == ToolType.WRITE
        return getattr(fn, _MUTATES_STATE_ATTR, default)

    def get_db_hash(self) -> str:
        """Hash of the current DB state — used to detect mutation drift in
        replay / determinism checks. Returns a hex SHA-256 of the DB's
        canonical JSON dump.
        """
        if self.db is None:
            return ""
        try:
            payload = self.db.model_dump_json(exclude_none=True)
        except Exception:
            payload = repr(self.db)
        return sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class ToolResponse:
    """Env → caller reply to a single tool call.

    Mirrors τ²'s `ToolMessage` shape (id / role=tool / content / requestor /
    error). We keep the class name `ToolResponse` because orchestrator
    wraps it into a trajectory `Message(role="tool")` before recording —
    the on-disk schema stays unchanged.
    """
    id: str                              # matches the originating ToolCall.id
    content: str                         # JSON-serialized tool output (or error message)
    is_error: bool = False
    requestor: Literal["assistant", "user"] = "assistant"


class ATREnv:
    """Session-scoped environment holding a toolkit and a tool-name whitelist.

    Contract:
        get_response(tool_call) catches exceptions from the toolkit method
        and wraps both success and error as ToolResponse with JSON content.
        Tool outputs are JSON-encoded; BaseModel outputs go through
        model_dump.
    """

    # Always-available base tools. Any session allowed_tools list is
    # auto-unioned with these so agents can always reach confirmation
    # primitives without each task listing them explicitly.
    _BASE_TOOLS: ClassVar[frozenset[str]] = frozenset({
        "get_user_confirmation",
    })

    def __init__(
        self,
        domain: str,
        toolkit: ATRToolKitBase,
        allowed_tools: list[str] | None = None,
        seed: Optional[int] = None,
    ):
        self.domain = domain
        self.toolkit = toolkit
        if allowed_tools is not None:
            self.allowed_tools = set(allowed_tools) | set(self._BASE_TOOLS)
        else:
            self.allowed_tools = None

        # Seed / replay infra (τ²-bench parity). Stamps the seed on the DB
        # (via PrivateAttr) and initializes the stdlib RNG. Tools today are
        # deterministic so this is a no-op in practice, but full
        # reproducibility guarantees require the seed to be threaded end
        # to end (data-gen → runner → evaluator). seed=None preserves
        # previous behavior.
        self.seed: Optional[int] = seed
        if seed is not None:
            random.seed(seed)
            if toolkit.db is not None:
                toolkit.db._seed = seed

    def _check_shape_guards(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str | None:
        """Enforce toolkit._shape_guards (exactly_one_of / at_least_one_of)
        before dispatching the call. Returns an error message on violation,
        else None.
        """
        guards = getattr(self.toolkit, "_shape_guards", {}).get(tool_name, {})
        if not guards:
            return None

        def _present(f: str) -> bool:
            v = arguments.get(f)
            return v not in (None, "", [])

        if "exactly_one_of" in guards:
            fields = guards["exactly_one_of"]
            present = [f for f in fields if _present(f)]
            if len(present) != 1:
                return (
                    f"{tool_name} requires exactly one of {fields}; "
                    f"got {len(present)}: {present}"
                )
        if "at_least_one_of" in guards:
            fields = guards["at_least_one_of"]
            if not any(_present(f) for f in fields):
                return f"{tool_name} requires at least one of {fields}"
        return None

    def get_tool_descriptions(self) -> str:
        """Render all allowed tools as a text block for prompt injection."""
        tools = self.toolkit.get_tools(
            include=sorted(self.allowed_tools) if self.allowed_tools else None
        )
        return "\n\n".join(t.to_text_description() for t in tools.values())

    def get_openai_schemas(self) -> list[dict[str, Any]]:
        tools = self.toolkit.get_tools(
            include=sorted(self.allowed_tools) if self.allowed_tools else None
        )
        return [tool.openai_schema for tool in tools.values()]

    def get_response(self, message: Any) -> ToolResponse:
        """Execute a single tool call (τ² signature) and return a JSON-wrapped response.

        Args:
            message: A ToolCall-like object exposing `id`, `name`, `arguments`,
                and optionally `requestor`. Duck-typed — any object with those
                attrs works (matches τ²-bench Environment.get_response(ToolCall)).

        Policy:
          - Unknown tool name → error
          - Tool exists but not in allowed_tools → error
          - Shape guard violation → error
          - Raised exception → error (caught and serialized)
          - Successful return → JSON-encoded content, is_error=False
        """
        name = getattr(message, "name", None)
        arguments = getattr(message, "arguments", {}) or {}
        call_id = getattr(message, "id", "") or ""
        requestor: Literal["assistant", "user"] = getattr(
            message, "requestor", "assistant"
        ) or "assistant"

        def _err(msg: str) -> ToolResponse:
            return ToolResponse(
                id=call_id,
                content=json.dumps({"error": msg}, ensure_ascii=False),
                is_error=True,
                requestor=requestor,
            )

        if not name or not self.toolkit.has_tool(name):
            return _err(f"Unknown tool: {name}")
        if self.allowed_tools is not None and name not in self.allowed_tools:
            return _err(f"Tool not available in this session: {name}")
        shape_violation = self._check_shape_guards(name, arguments)
        if shape_violation is not None:
            return _err(shape_violation)
        try:
            result = self.toolkit.use_tool(name, **arguments)
            return ToolResponse(
                id=call_id,
                content=_to_json_str(result),
                is_error=False,
                requestor=requestor,
            )
        except Exception as e:
            return _err(str(e))


def _loose_string_match(query: str | None, target: str | None) -> bool:
    """Loose string matching used by domain search_* filters.

    Returns True if:
      - query is empty/None, OR
      - query is substring of target (common case), OR
      - target is substring of query (agent sends more specific term), OR
      - query and target share any whitespace-separated token
        (handles "Portland, OR" vs "Downtown Portland" — both contain "portland").

    This matches the expected user intent of "agent specifies a location/
    type and env returns items that semantically live in that region/
    category", rather than requiring exact string containment.
    """
    if not query:
        return True
    if not target:
        return False
    q = query.lower().strip()
    t = target.lower().strip()
    if q in t or t in q:
        return True
    # token intersection (after stripping punctuation)
    q_tokens = set(re.split(r"[\s,./\-_]+", q)) - {""}
    t_tokens = set(re.split(r"[\s,./\-_]+", t)) - {""}
    return bool(q_tokens & t_tokens)


def _empty_with_hint(filter_summary: str, suggestion: str) -> dict:
    """Empty search result with helpful hint. Used by domain search/list
    tools when filters eliminate every candidate from a non-empty pool.

    Why: ATR's session-scoped DB is small (3-6 task-relevant refs) — a
    0-result rarely means "nothing exists" and almost always means
    "agent's free-text query token doesn't align with how data-gen wrote
    the ref's geographic / temporal field" (e.g. agent searches
    'Vancouver' but gold ref has location='Delta, BC'). Surfacing a hint
    encourages the agent to (a) drop a filter, (b) try a broader term,
    or (c) re-read instruction tokens — instead of fabricating ids or
    abandoning the task.

    Mirrors real-world search-API UX (Google "did you mean..." / Yelp
    "no results, try fewer keywords"); not an ATR-specific behavior.
    """
    return {
        "count": 0,
        "results": [],
        "hint": f"filter matched 0 items: {filter_summary}. Hint: {suggestion}",
    }


def _to_json_str(resp: Any) -> str:
    """Serialize tool output. Handles BaseModel, datetime, nested structures."""
    def _proc(v: Any) -> Any:
        if isinstance(v, BaseModel):
            return v.model_dump()
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        if isinstance(v, list):
            return [_proc(x) for x in v]
        if isinstance(v, tuple):
            return [_proc(x) for x in v]
        if isinstance(v, dict):
            return {k: _proc(x) for k, x in v.items()}
        return v
    if isinstance(resp, str):
        return resp
    return json.dumps(_proc(resp), ensure_ascii=False, default=str)
