from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import uuid

from .tools import Tool

WORK = "/work"


class DockerSandbox:
    def __init__(self, workdir: str, image: str = "python:3.12-slim", cpus: float = 2, memory: str = "4g", pids: int = 256, timeout: int = 120, network: bool = False, runtime: str | None = None, mounts: dict[str, str] | None = None):
        self.workdir = os.path.realpath(workdir)
        self.image, self.cpus, self.memory, self.pids = image, cpus, memory, pids
        self.timeout, self.network, self.runtime = timeout, network, runtime
        self.mounts = mounts or {}
        self.name = f"apex3-sbx-{uuid.uuid4().hex[:12]}"
        self.started = False

    def start(self) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("docker is not installed; run without the bash tool or install docker")
        cmd = ["docker", "run", "-d", "--rm", "--name", self.name, "--network", "bridge" if self.network else "none", "--cpus", str(self.cpus), "--memory", self.memory, "--memory-swap", self.memory, "--pids-limit", str(self.pids), "--read-only", "--tmpfs", "/tmp:rw,exec,size=1g", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "-e", "HOME=/tmp", "-v", f"{self.workdir}:{WORK}:rw", "-w", WORK]
        if hasattr(os, "getuid"):
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if self.runtime:
            cmd += ["--runtime", self.runtime]
        for host, dst in self.mounts.items():
            cmd += ["-v", f"{os.path.realpath(host)}:{dst}:ro"]
        subprocess.run(cmd + [self.image, "sleep", "infinity"], check=True, capture_output=True)
        self.started = True

    def exec(self, command: str, timeout: int | None = None) -> tuple[int, str]:
        if not self.started:
            self.start()
        t = min(timeout or self.timeout, self.timeout)
        cmd = ["docker", "exec", self.name, "timeout", "-s", "KILL", str(t), "bash", "-c", command]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=t + 30)
        except subprocess.TimeoutExpired:
            return 137, f"timed out after {t} s"
        out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
        if r.returncode == 137:
            out += f"\n[killed: time limit {t} s or memory limit {self.memory}]"
        return r.returncode, out

    def stop(self) -> None:
        if self.started:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            self.started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def resolve(workdir: str, path: str) -> str:
    root = os.path.realpath(workdir)
    if path == WORK or path.startswith(WORK + "/"):
        path = path[len(WORK) :].lstrip("/")
    if os.path.isabs(path):
        raise ValueError(f"path must be inside the workspace ({WORK}): {path}")
    full = os.path.realpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError(f"path escapes the workspace: {path}")
    return full


def _rel(workdir: str, full: str) -> str:
    return os.path.relpath(full, os.path.realpath(workdir)).replace(os.sep, "/")


def coding_tools(workdir: str, sandbox: DockerSandbox | None = None, max_read_bytes: int = 256_000, max_list: int = 500) -> list[Tool]:
    def read_file(a: dict) -> str:
        full = resolve(workdir, a["path"])
        with open(full, "rb") as f:
            data = f.read(max_read_bytes + 1)
        if b"\0" in data[:8192]:
            return f"{a['path']} is a binary file"
        text = data[:max_read_bytes].decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        start = max(int(a.get("offset", 1)), 1) - 1
        limit = int(a.get("limit", len(lines)))
        out = "".join(lines[start : start + limit])
        if len(data) > max_read_bytes:
            out += f"\n[file is larger than {max_read_bytes} bytes; the rest was not read]"
        elif start > 0 or start + limit < len(lines):
            out = f"[lines {start + 1}-{min(start + limit, len(lines))} of {len(lines)}]\n" + out
        return out

    def write_file(a: dict) -> str:
        full = resolve(workdir, a["path"])
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="") as f:
            f.write(a["content"])
        return f"wrote {len(a['content'])} characters to {_rel(workdir, full)}"

    def edit_file(a: dict) -> str:
        full = resolve(workdir, a["path"])
        with open(full, encoding="utf-8", newline="") as f:
            text = f.read()
        old, new = a["old"], a["new"]
        n = text.count(old) if old else 0
        if n == 0:
            return "error: old string not found (it must match the file exactly, including whitespace)"
        if n > 1 and not a.get("replace_all", False):
            return f"error: old string found {n} times; add surrounding lines to make it unique or set replace_all"
        with open(full, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace(old, new) if a.get("replace_all") else text.replace(old, new, 1))
        return f"replaced {n if a.get('replace_all') else 1} occurrence(s) in {_rel(workdir, full)}"

    def list_files(a: dict) -> str:
        base = resolve(workdir, a.get("path", "."))
        pattern = a.get("pattern", "*")
        out = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in (".git", "__pycache__", "node_modules", ".venv"))
            for name in sorted(filenames):
                rel = _rel(workdir, os.path.join(dirpath, name))
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                    out.append(rel)
                    if len(out) >= max_list:
                        return "\n".join(out) + f"\n[stopped at {max_list} files]"
        return "\n".join(out) if out else "no files"

    path = {"type": "string", "description": f"path relative to the workspace ({WORK})"}
    tools = [Tool("read_file", "Read a text file from the workspace.", {"type": "object", "properties": {"path": path, "offset": {"type": "integer", "description": "first line, 1-based"}, "limit": {"type": "integer", "description": "number of lines"}}, "required": ["path"]}, read_file, read_only=True), Tool("write_file", "Create or overwrite a file in the workspace.", {"type": "object", "properties": {"path": path, "content": {"type": "string"}}, "required": ["path", "content"]}, write_file),
             Tool("edit_file", "Replace an exact string in a file. The old string must be unique unless replace_all is true.", {"type": "object", "properties": {"path": path, "old": {"type": "string"}, "new": {"type": "string"}, "replace_all": {"type": "boolean"}}, "required": ["path", "old", "new"]}, edit_file), Tool("list_files", "List files under a workspace directory, optionally filtered by a glob pattern.", {"type": "object", "properties": {"path": path, "pattern": {"type": "string"}}}, list_files, read_only=True)]
    if sandbox is not None:
        def bash(a: dict) -> str:
            code, out = sandbox.exec(a["command"], a.get("timeout"))
            return f"exit code {code}\n{out}"

        tools.append(Tool("bash", f"Run a bash command in an isolated container (no network) with the workspace at {WORK}.", {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "description": "seconds"}}, "required": ["command"]}, bash))
    return tools
