from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .template import VERSION, ChatTemplate
from .tools import ToolRegistry, truncate, validate

DECLINED = "error: the user declined this action. Do not retry it unless the user asks."


def _deny(name: str, args: dict) -> bool:
    return False


class Agent:
    def __init__(self, backend, template: ChatTemplate, tools: ToolRegistry, system: str = "", skills: str = "", confirm: Callable[[str, dict], bool] = _deny, on_event: Callable[[str, dict], None] | None = None,
                 max_steps: int = 32, max_new_tokens: int = 4096, max_context: int = 131_072, result_chars: int = 20_000, workers: int = 8, confidence: Callable[[object], float | None] | None = None, confirm_below: float | None = None):
        self.backend, self.t, self.tools = backend, template, tools
        self.confirm, self.on_event = confirm, on_event or (lambda kind, data: None)
        self.confidence, self.confirm_below = confidence, confirm_below
        self.max_steps, self.max_new_tokens, self.max_context = max_steps, max_new_tokens, max_context
        self.result_chars, self.workers = result_chars, workers
        self.messages = [{"role": "system", "content": template.system_text(system, tools.schemas(), skills)}]
        self.ids, _ = template.render(self.messages)

    def ask(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        self.ids += self.t.turn("user", text)
        for _ in range(self.max_steps):
            self.ids.append(self.t.sp["<|assistant|>"])
            budget = min(self.max_new_tokens, self.max_context - len(self.ids) - 1)
            if budget <= 0:
                return self._stop("the context is full")
            out = self.backend.generate(self.ids, self.t.stop, budget)
            body = out[:-1] if out and out[-1] in self.t.stop else out
            self.ids += body + [self.t.end]
            msg = self.t.parse(body)
            score = self.confidence(self.backend) if self.confidence else None
            low = score is not None and self.confirm_below is not None and score < self.confirm_below
            if score is not None:
                msg["confidence"] = round(score, 4)
            self.messages.append(msg)
            self.on_event("assistant", msg)
            if not msg["tool_calls"]:
                return msg["content"]
            for call, result in zip(msg["tool_calls"], self.run_tools(msg["tool_calls"], low)):
                self.messages.append({"role": "tool", "name": call["name"], "content": result})
                self.ids += self.t.tool_result(result)
                self.on_event("tool_result", {"name": call["name"], "content": result})
        return self._stop(f"the step limit ({self.max_steps}) was reached")

    def _stop(self, reason: str) -> str:
        if self.ids[-1] == self.t.sp["<|assistant|>"]:
            self.ids.append(self.t.end)
            self.messages.append({"role": "assistant", "content": ""})
        self.on_event("stopped", {"reason": reason})
        return f"[stopped: {reason}]"

    def run_tools(self, calls: list[dict], low_confidence: bool = False) -> list[str]:
        results: list[str | None] = [None] * len(calls)
        jobs = []
        for k, c in enumerate(calls):
            tool = self.tools.get(c["name"]) if "error" not in c else None
            if "error" in c:
                results[k] = f"error: {c['error']}"
            elif tool is None:
                results[k] = f"error: unknown tool {c['name']}"
            elif (err := validate(tool.parameters, c["arguments"])) is not None:
                results[k] = f"error: {err}"
            else:
                ask = tool.confirm or (low_confidence and not tool.read_only)
                self.on_event("tool_call", {"name": tool.name, "arguments": c["arguments"], "confirm": ask, "low_confidence": low_confidence})
                if ask and not self.confirm(tool.name, c["arguments"]):
                    results[k] = DECLINED
                else:
                    jobs.append((k, tool, c["arguments"]))
        with ThreadPoolExecutor(self.workers) as ex:
            i = 0
            while i < len(jobs):
                j = i
                while j < len(jobs) and jobs[j][1].read_only:
                    j += 1
                group = jobs[i:j] if j > i else jobs[i : i + 1]
                for (k, _, _), r in zip(group, ex.map(lambda job: self._call(job[1], job[2]), group)):
                    results[k] = r
                i += len(group)
        return [truncate(r, self.result_chars) for r in results]

    @staticmethod
    def _call(tool, args: dict) -> str:
        try:
            out = tool.fn(args)
            return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"

    def transcript(self) -> dict:
        return {"template": VERSION, "messages": self.messages}
