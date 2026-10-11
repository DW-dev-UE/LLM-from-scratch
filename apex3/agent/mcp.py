from __future__ import annotations

import fnmatch
import itertools
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout

from .tools import Tool

PROTOCOL_VERSION = "2025-06-18"


class MCPError(RuntimeError):
    pass


class MCPClient:
    def __init__(self, name: str, command: str, args: list[str] = (), env: dict | None = None, timeout: float = 60, stderr_path: str | None = None):
        self.name, self.timeout = name, timeout
        self._ids = itertools.count(1)
        self._pending: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._exited = False
        self._stderr = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen([command, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, env={**os.environ, **(env or {})})
        threading.Thread(target=self._read_loop, daemon=True, name=f"mcp-{name}").start()
        try:
            init = self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "apex3-agent", "version": "0.1.0"}})
            self.notify("notifications/initialized")
        except Exception:
            self.close()
            raise
        self.server_info = init.get("serverInfo", {})
        self.protocol_version = init.get("protocolVersion")

    def _send(self, msg: dict) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode()
        with self._lock:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})})

    def request(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        rid = next(self._ids)
        fut: Future = Future()
        self._pending[rid] = fut
        try:
            if self._exited:
                raise OSError
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params is not None else {})})
        except OSError:
            self._pending.pop(rid, None)
            raise MCPError(f"{self.name}: server exited") from None
        try:
            return fut.result(timeout or self.timeout)
        except FutureTimeout:
            self._pending.pop(rid, None)
            self.notify("notifications/cancelled", {"requestId": rid, "reason": "timeout"})
            raise MCPError(f"{self.name}: {method} timed out") from None

    def _read_loop(self) -> None:
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            if "method" not in msg:
                fut = self._pending.pop(msg.get("id"), None)
                if fut is not None:
                    err = msg.get("error")
                    if err is not None:
                        fut.set_exception(MCPError(f"{self.name}: {err.get('message', err) if isinstance(err, dict) else err}"))
                    else:
                        fut.set_result(msg.get("result") or {})
            elif "id" in msg:
                reply = {"result": {}} if msg["method"] == "ping" else {"error": {"code": -32601, "message": f"method not supported: {msg['method']}"}}
                try:
                    self._send({"jsonrpc": "2.0", "id": msg["id"], **reply})
                except OSError:
                    break
        self._exited = True
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(MCPError(f"{self.name}: server exited"))
        self._pending.clear()

    def list_tools(self) -> list[dict]:
        tools, cursor = [], None
        while True:
            r = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += r.get("tools", [])
            cursor = r.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict) -> str:
        r = self.request("tools/call", {"name": name, "arguments": arguments})
        text = content_text(r)
        return f"error: {text}" if r.get("isError") else text

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(5)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
        if self._stderr is not subprocess.DEVNULL:
            self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def content_text(result: dict) -> str:
    parts = []
    for c in result.get("content") or []:
        kind = c.get("type")
        if kind == "text":
            parts.append(c.get("text", ""))
        elif kind == "resource":
            res = c.get("resource") or {}
            parts.append(res["text"] if "text" in res else f"[binary resource {res.get('uri', '')}]")
        elif kind == "resource_link":
            parts.append(f"[resource {c.get('uri', '')}]")
        else:
            parts.append(f"[{kind} omitted: {c.get('mimeType', 'unknown type')}]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(parts)


def _match(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def mcp_tools(client: MCPClient, allow=("*",), deny=(), confirm="auto") -> list[Tool]:
    out = []
    for t in client.list_tools():
        name = t["name"]
        if not _match(name, allow) or _match(name, deny):
            continue
        read_only = (t.get("annotations") or {}).get("readOnlyHint") is True
        needs = not read_only if confirm == "auto" else _match(name, confirm)
        out.append(Tool(name=f"{client.name}__{name}", description=t.get("description") or t.get("title") or name, parameters=t.get("inputSchema") or {"type": "object"}, fn=lambda args, n=name: client.call_tool(n, args), confirm=needs, read_only=read_only and not needs))
    return out


def connect(name: str, spec: dict) -> MCPClient:
    env = {k: os.path.expandvars(v) for k, v in (spec.get("env") or {}).items()}
    command = sys.executable if spec["command"] == "python" else spec["command"]
    return MCPClient(name, command, spec.get("args", []), env, spec.get("timeout", 60), spec.get("stderr"))
