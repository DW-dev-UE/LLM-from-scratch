"""Pin every corpus source to an immutable revision before tokenizing.

HF dataset repos are mutable — shards get added, filters get retuned, opt-out
removals land. A model released with "trained on FineWeb-Edu" names something
that no longer exists. This records the exact commit sha of every source so the
run is reproducible and the model card can cite it.

    python apex2/lock_datasets.py                 # write DATASET_LOCK.json
    python apex2/lock_datasets.py --verify        # fail if anything drifted

Run once before prepare_data.py, then --verify before publishing weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://huggingface.co/api/datasets/{repo}"


def fetch(repo: str) -> dict:
    request = urllib.request.Request(API.format(repo=repo),
        headers={"User-Agent": "apex2-lock/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{repo}: HTTP {error.code} — gated repo needs "
                         f"`huggingface-cli login` and terms acceptance") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"{repo}: {error.reason}") from error


def describe(repo: str) -> dict:
    info = fetch(repo)
    card = info.get("cardData") or {}
    license_value = card.get("license") or info.get("license")
    if isinstance(license_value, list):
        license_value = ", ".join(str(v) for v in license_value)
    return {
        "repo": repo,
        "sha": info.get("sha"),
        "last_modified": info.get("lastModified"),
        "license": license_value,
        "gated": bool(info.get("gated")),
        "downloads": info.get("downloads"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, default=Path("apex2/mixture.json"))
    parser.add_argument("--out", type=Path, default=Path("DATASET_LOCK.json"))
    parser.add_argument("--verify", action="store_true",
        help="compare live revisions against the lock file instead of writing it")
    args = parser.parse_args()

    mixture = json.loads(args.mixture.read_text(encoding="utf-8"))
    repos = sorted({source["hf_id"] for source in mixture["sources"]})

    entries: dict[str, dict] = {}
    for repo in repos:
        entry = describe(repo)
        entries[repo] = entry
        flag = " [GATED]" if entry["gated"] else ""
        print(f"{repo:44s} {str(entry['sha'])[:12]}  {entry['license']}{flag}")

    if args.verify:
        if not args.out.is_file():
            print(f"\nno lock file at {args.out} — run without --verify first",
                file=sys.stderr)
            return 1
        locked = json.loads(args.out.read_text(encoding="utf-8"))["datasets"]
        drift = [repo for repo, entry in entries.items()
            if repo not in locked or locked[repo]["sha"] != entry["sha"]]
        if drift:
            print("\nREVISION DRIFT — these no longer match the lock:", file=sys.stderr)
            for repo in drift:
                was = locked.get(repo, {}).get("sha", "<absent>")
                print(f"  {repo}\n    locked {was}\n    live   {entries[repo]['sha']}",
                    file=sys.stderr)
            return 1
        print("\nall sources match DATASET_LOCK.json")
        return 0

    weights = {source["name"]: source["weight"] for source in mixture["sources"]}
    payload = {
        "schema_version": 1,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "budget_tokens": mixture["budget_tokens"],
        "mixture_weights": weights,
        "datasets": entries,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\nwrote {args.out} ({len(entries)} repos)")

    unknown = [r for r, e in entries.items() if not e["license"]]
    if unknown:
        # A source whose card declares no license is the one you cannot defend
        # in a model card. Decide before spending 106 days on it.
        print("\nWARNING — no license declared on the dataset card:", file=sys.stderr)
        for repo in unknown:
            print(f"  {repo}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
