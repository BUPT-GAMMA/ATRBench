"""Prompt file loader for runner.

Centralizes runner prompts in `runner/prompts/*.md`. Each file may contain
multiple named sections delimited by lines of the form `--- <name> ---`
(by themselves on a line). A file with no delimiters is loaded whole.

Example file (`runner/prompts/user_sim.md`):
    --- open ---
    <opening prompt body>
    --- reply ---
    <reply prompt body>

Usage:
    from _prompts import load_prompt
    text = load_prompt("user_sim", "reply")  # section
    text = load_prompt("classifier")         # whole file
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_SECTION_RE = re.compile(r"^---\s+(\S+)\s+---\s*$", re.M)


@lru_cache(maxsize=None)
def _read(file: str) -> dict[str, str]:
    """Parse a prompt file into {section_name: body}.

    Files without any `--- name ---` markers are returned as
    `{"": <whole file>}`.
    """
    text = (_PROMPTS_DIR / f"{file}.md").read_text()
    parts = _SECTION_RE.split(text)
    if len(parts) == 1:
        return {"": text.strip()}
    # parts = [pre, name1, body1, name2, body2, ...]; pre is whatever came
    # before the first marker — must be empty / whitespace.
    pre = parts[0].strip()
    if pre:
        raise ValueError(
            f"prompts/{file}.md: content before first `--- name ---` marker "
            f"is not allowed (got {pre!r:.40s}...)"
        )
    sections: dict[str, str] = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        sections[name] = body.strip()
    return sections


def load_prompt(file: str, section: str | None = None) -> str:
    """Load a prompt body. `section=None` loads the whole file (only valid
    for single-section files)."""
    sections = _read(file)
    if section is None:
        if list(sections.keys()) != [""]:
            raise ValueError(
                f"prompts/{file}.md has multiple sections "
                f"({list(sections.keys())}); pass section=..."
            )
        return sections[""]
    if section not in sections:
        raise KeyError(
            f"prompts/{file}.md has no section {section!r}; "
            f"available: {list(sections.keys())}"
        )
    return sections[section]
