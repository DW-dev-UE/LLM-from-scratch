from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict

import torch

from .agent.template import VERSION
from .config import ModelConfig

HF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf")
CODE_FILES = ["configuration_apex3.py", "modeling_apex3.py", "apex3_tool_parser.py"]
CONFIG_KEYS = ["vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "global_every", "window", "rope_theta_local", "rope_theta_global", "rope_scale_global", "n_dense_layers", "dense_ffn", "n_experts", "expert_ffn", "top_k", "shared_ffn", "route_scale", "norm_eps", "init_std"]
STOP_TOKENS = ["<|end|>", "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|tool_result|>", "<|/tool_result|>"]
MAX_CONTEXT = 131_072


def hf_state_dict(sd: dict[str, torch.Tensor], dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        name = "lm_head.weight" if k == "head.weight" else f"model.{k}"
        out[name] = v.detach().to(torch.float32 if ".router." in k else dtype).contiguous()
    if "lm_head.weight" not in out:
        out["lm_head.weight"] = out["model.embed.weight"].clone()
    return out


def hf_config(cfg: ModelConfig, ids: dict[str, int], dtype=torch.bfloat16) -> dict:
    c = {k: v for k, v in asdict(cfg).items() if k in CONFIG_KEYS}
    c.update(architectures=["Apex3ForCausalLM"], model_type="apex3", auto_map={"AutoConfig": "configuration_apex3.Apex3Config", "AutoModelForCausalLM": "modeling_apex3.Apex3ForCausalLM"}, torch_dtype=str(dtype).removeprefix("torch."),
             max_position_embeddings=MAX_CONTEXT, tie_word_embeddings=False, use_cache=True, bos_token_id=None, eos_token_id=[ids[t] for t in STOP_TOKENS], pad_token_id=ids["<|pad|>"], apex3_chat_template=VERSION)
    return c


def write_hf_folder(out: str, cfg: ModelConfig, sd: dict[str, torch.Tensor], tokenizer_path: str, dtype=torch.bfloat16, max_shard_size: str = "5GB") -> None:
    from huggingface_hub import save_torch_state_dict

    from .tokens import load_tokenizer, special_ids

    ids = special_ids(load_tokenizer(tokenizer_path))
    os.makedirs(out, exist_ok=True)
    save_torch_state_dict(hf_state_dict(sd, dtype), out, max_shard_size=max_shard_size)

    def dump(name, obj):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    stop = [ids[t] for t in STOP_TOKENS]
    dump("config.json", hf_config(cfg, ids, dtype))
    dump("generation_config.json", {"eos_token_id": stop, "pad_token_id": ids["<|pad|>"], "do_sample": True, "temperature": 0.7, "top_p": 0.95})
    shutil.copy(tokenizer_path, os.path.join(out, "tokenizer.json"))
    with open(os.path.join(HF_DIR, "chat_template.jinja"), encoding="utf-8") as f:
        template = f.read()
    dump("tokenizer_config.json", {"tokenizer_class": "PreTrainedTokenizerFast", "chat_template": template, "eos_token": "<|end|>", "pad_token": "<|pad|>", "bos_token": None, "unk_token": None, "model_max_length": MAX_CONTEXT, "clean_up_tokenization_spaces": False})
    dump("special_tokens_map.json", {"eos_token": "<|end|>", "pad_token": "<|pad|>"})
    for name in CODE_FILES:
        shutil.copy(os.path.join(HF_DIR, name), os.path.join(out, name))
    card = os.path.join(out, "README.md")
    if not os.path.exists(card):
        shutil.copy(os.path.join(HF_DIR, "MODEL_CARD.md"), card)
