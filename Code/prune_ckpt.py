"""Keep the last N committed checkpoints; delete the rest.

composelm has no retention logic — every save_steps checkpoint accumulates
forever. A 3.02B checkpoint is ~36 GB (fp32 params 12 + AdamW states 24), so at
save_steps=250 over a 214K-step run an unpruned directory reaches ~31 TB. The
instance has 5.5 TiB.

Run from cron every 30 minutes:

    */30 * * * * cd /home/ubuntu/llm && python apex2/prune_ckpt.py runs/apex2-3b --keep 5

Safety: the checkpoint named by the `latest` pointer is never removed, and only
directories carrying a COMMITTED marker are considered (an in-flight
`.checkpoint-N.tmp-*` is left alone).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

STEP_RE = re.compile(r"^checkpoint-(\d+)$")


def committed_checkpoints(root: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = STEP_RE.match(path.name)
        if match and (path / "COMMITTED").is_file():
            found.append((int(match.group(1)), path))
    return sorted(found)


def latest_target(root: Path) -> str | None:
    pointer = root / "latest"
    if not pointer.is_file():
        return None
    try:
        return pointer.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.output_dir
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1
    if args.keep < 1:
        print("--keep must be at least 1", file=sys.stderr)
        return 1

    checkpoints = committed_checkpoints(root)
    if len(checkpoints) <= args.keep:
        return 0

    protected = latest_target(root)
    doomed = checkpoints[: -args.keep]
    freed = 0
    for step, path in doomed:
        if path.name == protected:
            # Should not happen (latest is always the newest), but deleting the
            # resume target would end the run.
            print(f"skip {path.name}: latest pointer")
            continue
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if args.dry_run:
            print(f"would remove {path.name}  ({size / 1e9:.1f} GB)")
        else:
            shutil.rmtree(path)
            print(f"removed {path.name}  ({size / 1e9:.1f} GB)")
        freed += size

    print(f"{'would free' if args.dry_run else 'freed'} {freed / 1e9:.1f} GB; "
          f"kept {min(args.keep, len(checkpoints))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
