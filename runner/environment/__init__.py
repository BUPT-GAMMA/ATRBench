"""ATR environment layer (session-scoped).

Each LearningSession / TestSession constructs an ATREnv with:
  - a domain-specific ATRDB (persona + references lifted into typed Pydantic)
  - a ToolKit bound to that DB
  - an allowed-tools whitelist from local_env.tools

The env exposes get_response(tool_call) → (content_json, is_error).
Orchestrator routes parsed tool_calls through the env, collects results as
tool messages.
"""
from runner.environment.base import (
    ATRDB,
    ATREnv,
    ATRToolKitBase,
    ATRTool,
    PersonaProfile,
    ToolResponse,
    ToolType,
    is_tool,
    as_tool,
)

__all__ = [
    "ATRDB",
    "ATREnv",
    "ATRToolKitBase",
    "ATRTool",
    "PersonaProfile",
    "ToolResponse",
    "ToolType",
    "is_tool",
    "as_tool",
]
