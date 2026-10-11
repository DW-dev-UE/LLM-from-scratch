from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apex3.config import ATTN_RATIOS, PRESETS, get_config
from apex3.flops import flops_per_token, param_counts


def main():
    print(f"{'preset':<7}{'total':>9}{'active':>9}{'act+emb':>9}{'ffn act':>9}   GFLOP/token at 8K / 32K / 128K")
    for name, cfg in PRESETS.items():
        c = param_counts(cfg)
        ffn = c["ffn_active"] / c["ffn_total"]
        g = [flops_per_token(cfg, s) / 1e9 for s in (8192, 32768, 131072)]
        print(f"{name:<7}{c['total'] / 1e9:>8.2f}B{c['active'] / 1e9:>8.2f}B{c['active_with_embed'] / 1e9:>8.2f}B" f"{ffn:>9.1%}   {g[0]:.1f} / {g[1]:.1f} / {g[2]:.1f}")
    print("\nattention ratio (preset C): GFLOP/token 8K / 32K / 128K")
    for ratio in ATTN_RATIOS:
        cfg = get_config("C", attn=ratio)
        g = [flops_per_token(cfg, s) / 1e9 for s in (8192, 32768, 131072)]
        print(f"  {ratio:<5} global layers {cfg.n_global_layers:>2}: {g[0]:.1f} / {g[1]:.1f} / {g[2]:.1f}")


if __name__ == "__main__":
    main()
