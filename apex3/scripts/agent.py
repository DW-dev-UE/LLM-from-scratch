from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apex3.agent import Agent, ChatTemplate, ToolRegistry
from apex3.agent.mcp import connect, mcp_tools
from apex3.agent.sandbox import DockerSandbox, coding_tools
from apex3.agent.skills import SANDBOX_PATH, catalog, load_skills, skill_tool


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-config", default="configs/agent/default.json")
    p.add_argument("--config", default="C", help="model preset, used only if the checkpoint has no model_config")
    p.add_argument("--set", nargs="*", default=[], help="ModelConfig overrides, e.g. rope_scale_global=8")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--workdir", default=None, help="workspace for the coding tools (mounted at /work)")
    p.add_argument("--no-sandbox", action="store_true", help="file tools only, no bash")
    p.add_argument("--transcript", default=None, help="write the conversation (template form) here after each turn")
    p.add_argument("--list-tools", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--experts", default="grouped_mm", choices=["grouped_mm", "loop"])
    p.add_argument("--max-context", type=int, default=131_072)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--confidence-head", default=None, help="head from scripts/confidence.py train (this checkpoint)")
    p.add_argument("--confirm-below", type=float, default=None, help="ask before side-effect tool calls of turns scored below this (default: agent config)")
    return p.parse_args()


def confirm(name: str, args: dict) -> bool:
    print(f"\n[confirm] {name}\n{json.dumps(args, ensure_ascii=False, indent=2)}")
    try:
        return input("Allow? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def on_event(kind: str, data: dict) -> None:
    if kind == "assistant" and "confidence" in data:
        print(f"[confidence {data['confidence']:.2f}]")
    if kind == "assistant" and data["tool_calls"]:
        if data["content"].strip():
            print(data["content"].strip())
    elif kind == "tool_call":
        print(f"  -> {data['name']}({json.dumps(data['arguments'], ensure_ascii=False)[:300]})")
    elif kind == "tool_result":
        text = data["content"].replace("\n", " ")
        print(f"  <- {text[:300]}{'...' if len(text) > 300 else ''}")
    elif kind == "stopped":
        print(f"[stopped: {data['reason']}]")


def build_tools(cfg: dict, args, stack: contextlib.ExitStack) -> tuple[ToolRegistry, str]:
    tools = ToolRegistry()
    skills = load_skills(cfg["skills_dir"]) if cfg.get("skills_dir") and os.path.isdir(cfg["skills_dir"]) else []
    use_sandbox = args.workdir is not None and not args.no_sandbox and cfg.get("sandbox") is not None
    if args.workdir:
        sandbox = None
        if use_sandbox:
            mounts = {cfg["skills_dir"]: SANDBOX_PATH} if skills else {}
            sandbox = stack.enter_context(DockerSandbox(args.workdir, mounts=mounts, **cfg["sandbox"]))
        tools.extend(coding_tools(args.workdir, sandbox))
    if skills:
        tools.add(skill_tool(skills))
    for name, spec in (cfg.get("mcp_servers") or {}).items():
        if not spec.get("enabled", True):
            continue
        client = stack.enter_context(connect(name, spec))
        tools.extend(mcp_tools(client, spec.get("allow", ["*"]), spec.get("deny", []), spec.get("confirm", "auto")))
    return tools, catalog(skills, in_sandbox=use_sandbox)


def main():
    args = parse_args()
    with open(args.agent_config) as f:
        cfg = json.load(f)
    with contextlib.ExitStack() as stack:
        tools, skills = build_tools(cfg, args, stack)
        if args.list_tools:
            for t in tools:
                flags = [n for n, on in (("confirm", t.confirm), ("read-only", t.read_only)) if on]
                print(f"{t.name:<40} {' '.join(flags):<18} {t.description[:80]}")
            return
        if not (args.ckpt and args.tokenizer):
            sys.exit("--ckpt and --tokenizer are required to run the model")

        from apex3.agent.backend import TorchBackend, load_model
        from apex3.checkpoint import read_trainer_state
        from apex3.config import parse_overrides, resolve_config
        from apex3.tokens import load_tokenizer

        gen = cfg.get("generation", {})
        template = ChatTemplate(load_tokenizer(args.tokenizer, raw_text=True))
        model_cfg, _ = resolve_config(read_trainer_state(args.ckpt), args.config, None, parse_overrides(args.set))
        model = load_model(model_cfg, args.ckpt, args.device, experts=args.experts)
        backend = TorchBackend(model, args.max_context, gen.get("temperature", 0.7), gen.get("top_p", 0.95), args.seed)
        conf_cfg = cfg.get("confidence") or {}
        head_path = args.confidence_head or conf_cfg.get("head")
        score, below = None, None
        if head_path:
            from apex3.confidence import ConfidenceHead

            head = ConfidenceHead.load(head_path)
            if head.meta.get("ckpt") and os.path.normpath(head.meta["ckpt"]) != os.path.normpath(args.ckpt):
                print(f"[warning] the confidence head was trained on {head.meta['ckpt']}, not this checkpoint")
            backend.capture_layers = tuple(i - 1 for i in head.layers)
            score = head.score_turn
            below = args.confirm_below if args.confirm_below is not None else conf_cfg.get("confirm_below")
        agent = Agent(backend, template, tools, cfg.get("system", ""), skills, confirm, on_event, cfg.get("max_steps", 32),
                      gen.get("max_new_tokens", 4096), args.max_context, cfg.get("result_chars", 20_000), confidence=score, confirm_below=below)
        print(f"{len(tools)} tools. Empty line or Ctrl-D to quit.")
        while True:
            try:
                text = input("\n> ").strip()
            except EOFError:
                break
            if not text:
                break
            print(agent.ask(text))
            if args.transcript:
                with open(args.transcript, "w", encoding="utf-8") as f:
                    json.dump(agent.transcript(), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
