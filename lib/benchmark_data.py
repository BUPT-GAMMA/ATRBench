"""Runtime access to released episodes, ontology, and JSON artifacts."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = PROJECT_ROOT / "data" / "personas"
ONTOLOGY_PATH = PROJECT_ROOT / "ontology" / "tools.yaml"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> Path:
    """Write JSON atomically through a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(path)
    return path


@lru_cache(maxsize=1)
def load_ontology() -> dict[str, Any]:
    return yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def tool_map() -> dict[str, dict[str, Any]]:
    """Return ontology tools keyed by name, with their domain attached."""
    tools: dict[str, dict[str, Any]] = {}
    ontology = load_ontology()
    for domain_name, domain in ontology["domains"].items():
        for tool in domain["tools"]:
            tools[tool["name"]] = {**tool, "domain": domain_name}
    for tool in (ontology.get("base_tools", {}) or {}).get("tools", []) or []:
        tools[tool["name"]] = {**tool, "domain": "base"}
    return tools
