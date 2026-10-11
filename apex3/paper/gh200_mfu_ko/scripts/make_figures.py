from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle

FIGURES = {"fig_system": "figs_system", "fig_ladder": "figs_progress", "fig_train_mfu": "figs_progress", "fig_loss": "figs_progress",
           "fig_power": "figs_progress", "fig_time_budget": "figs_breakdown", "fig_fp8": "figs_breakdown", "fig_cpu_accum": "figs_breakdown", "fig_diloco": "figs_breakdown"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="build only these figures")
    args = ap.parse_args()
    figstyle.setup()
    failed = []
    for name in args.only or list(FIGURES):
        try:
            module = importlib.import_module(FIGURES[name])
            getattr(module, name)()
            print(f"ok    {name}: figures/{name}.pdf, figures/{name}.png")
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
