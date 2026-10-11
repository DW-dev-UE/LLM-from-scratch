from __future__ import annotations

import os
from dataclasses import dataclass

from .sandbox import resolve
from .tools import Tool

SANDBOX_PATH = "/skills"


@dataclass
class Skill:
    name: str
    description: str
    path: str


def front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    meta = {}
    for line in text[3:end].splitlines():
        if ":" in line and not line[:1].isspace():
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, text[end + 4 :].lstrip("\n")


def load_skills(root: str) -> list[Skill]:
    skills = []
    for entry in sorted(os.listdir(root)):
        md = os.path.join(root, entry, "SKILL.md")
        if not os.path.isfile(md):
            continue
        with open(md, encoding="utf-8") as f:
            meta, _ = front_matter(f.read())
        skills.append(Skill(meta.get("name", entry), meta.get("description", ""), os.path.join(root, entry)))
    return skills


def catalog(skills: list[Skill], in_sandbox: bool = True) -> str:
    if not skills:
        return ""
    head = "# Skills\nBefore a task that matches a skill, read it with read_skill and follow it."
    if in_sandbox:
        head += f" Skill files are also available at {SANDBOX_PATH}/<folder>/ for bash."
    return head + "\n" + "\n".join(f"- {s.name}: {s.description}" for s in skills)


def skill_tool(skills: list[Skill]) -> Tool:
    by_name = {s.name: s for s in skills}

    def read_skill(a: dict) -> str:
        s = by_name.get(a["name"])
        if s is None:
            return f"error: no skill named {a['name']}. Available: {', '.join(by_name)}"
        path = a.get("file", "SKILL.md")
        with open(resolve(s.path, path), encoding="utf-8", errors="replace") as f:
            text = f.read()
        if path == "SKILL.md":
            _, text = front_matter(text)
            files = sorted(os.path.relpath(os.path.join(d, n), s.path).replace(os.sep, "/") for d, _, names in os.walk(s.path) for n in names if n != "SKILL.md")
            if files:
                text += f"\n\n[files in this skill: {', '.join(files)}; folder {os.path.basename(s.path)}]"
        return text

    return Tool("read_skill", "Read a skill's instructions (SKILL.md) or one of its files.",
                {"type": "object", "properties": {"name": {"type": "string", "enum": list(by_name)}, "file": {"type": "string", "description": "default SKILL.md"}}, "required": ["name"]}, read_skill, read_only=True)
