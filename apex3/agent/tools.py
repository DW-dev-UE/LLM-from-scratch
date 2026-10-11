from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[[dict], str] = field(repr=False)
    confirm: bool = False
    read_only: bool = False

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class ToolRegistry:
    def __init__(self, tools: list[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        self.extend(tools)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def extend(self, tools) -> None:
        for t in tools:
            self.add(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list, "null": type(None)}


def _type_ok(value, kind) -> bool:
    kinds = kind if isinstance(kind, list) else [kind]
    for k in kinds:
        py = _TYPES.get(k)
        if py is None:
            return True
        if isinstance(value, bool) and k in ("integer", "number"):
            continue
        if isinstance(value, py):
            return True
    return False


def validate(schema: dict, args: dict) -> str | None:
    props = schema.get("properties") or {}
    missing = [k for k in schema.get("required") or [] if k not in args]
    if missing:
        return f"missing required arguments: {', '.join(missing)}"
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            if schema.get("additionalProperties") is False:
                return f"unknown argument: {key}"
            continue
        if "type" in spec and not _type_ok(value, spec["type"]):
            return f"argument {key} must be of type {spec['type']}"
        if "enum" in spec and value not in spec["enum"]:
            return f"argument {key} must be one of {spec['enum']}"
    return None


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - 2 * half} characters cut] ...\n{text[-half:]}"
