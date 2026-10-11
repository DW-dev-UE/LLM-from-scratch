from __future__ import annotations

from .config import ModelConfig

PEAK_BF16 = [("H100 PCIe", 756e12), ("H100 NVL", 835e12), ("H100", 989e12), ("GH200", 989e12), ("H200", 989e12), ("GB200", 2.5e15), ("B200", 2.25e15)]


def peak_bf16_flops(device_name: str) -> float | None:
    for key, flops in PEAK_BF16:
        if key in device_name:
            return flops
    return None


def param_counts(cfg: ModelConfig) -> dict[str, int]:
    d, L = cfg.d_model, cfg.n_layers
    q_dim, kv_dim = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    attn = L * (2 * d * q_dim + 2 * d * kv_dim)
    dense = 3 * d * cfg.dense_ffn
    expert = 3 * d * cfg.expert_ffn
    shared = 3 * d * cfg.shared_ffn
    router = d * cfg.n_experts
    n_moe = cfg.n_moe_layers
    ffn_total = (L - n_moe) * dense + n_moe * (cfg.n_experts * expert + shared + router)
    ffn_active = (L - n_moe) * dense + n_moe * (cfg.top_k * expert + shared + router)
    embed = cfg.vocab_size * d
    norms = L * (2 * d + 2 * cfg.head_dim) + d
    active = attn + ffn_active + embed
    return {"attn": attn, "ffn_total": ffn_total, "ffn_active": ffn_active, "embed": embed, "norms": norms,
            "total": attn + ffn_total + embed * (1 if cfg.tie_embeddings else 2) + norms, "active": active, "active_with_embed": active + embed}


def attn_context_sum(cfg: ModelConfig, seq_len: int) -> int:
    return sum(seq_len if cfg.is_global(i) else min(2 * cfg.window, seq_len) for i in range(cfg.n_layers))


def flops_per_token(cfg: ModelConfig, seq_len: int, training: bool = True) -> float:
    f = 6 * param_counts(cfg)["active"] + 6 * cfg.n_heads * cfg.head_dim * attn_context_sum(cfg, seq_len)
    return f if training else f / 3


def mfu(tokens_per_sec_per_gpu: float, flops_per_tok: float, peak_flops: float) -> float:
    return tokens_per_sec_per_gpu * flops_per_tok / peak_flops


def gpu_hours(tokens: float, tokens_per_sec_per_gpu: float) -> float:
    return tokens / tokens_per_sec_per_gpu / 3600
