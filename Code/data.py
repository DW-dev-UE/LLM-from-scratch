"""PackedCorpus — a StatefulDataSource over pre-tokenized uint16 shards.

Implements composelm's streaming resume contract:
    set_shard(shard_id, num_shards) / state_dict() / load_state_dict(state)

Exact resume works because the only mutable state is (a) a per-source file
index + token offset and (b) the PCG64 state of the mixture sampler. Both
serialize to plain JSON-able values, so the cursor survives a kill -9 with no
fast-forward replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import IterableDataset


class PackedCorpus(IterableDataset):
    """Weighted mixture over per-source token shards, yielding packed blocks.

    Parameters
    ----------
    mixture_path :
        mixture.json — source names and sampling weights.
    token_root :
        directory holding ``<source>/index.json`` written by prepare_data.py.
    seq_len :
        tokens per emitted block. Blocks are dense: no padding, no attention
        mask, documents run together separated by the EOS id baked in at
        tokenization time.
    seed :
        base seed for the mixture sampler. The per-shard stream is derived from
        ``seed`` and ``shard_id`` so two ranks never draw the same order.
    """

    def __init__(
        self,
        mixture_path: str | Path,
        token_root: str | Path,
        *,
        seq_len: int = 4096,
        seed: int = 42,
    ) -> None:
        mixture = json.loads(Path(mixture_path).read_text(encoding="utf-8"))
        self.token_root = Path(token_root)
        self.seq_len = seq_len
        self.seed = seed

        self.names: list[str] = []
        self.weights: list[float] = []
        self.all_files: dict[str, list[dict[str, Any]]] = {}
        for source in mixture["sources"]:
            name = source["name"]
            index_path = self.token_root / name / "index.json"
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"{index_path} missing — run prepare_data.py --source {name}"
                )
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if not index["shards"]:
                raise ValueError(f"source {name} has no shards")
            self.names.append(name)
            self.weights.append(float(source["weight"]))
            self.all_files[name] = index["shards"]

        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        self.shard_id = 0
        self.num_shards = 1
        self._apply_shard()
        self._reset_cursors()

    # -- StatefulDataSource ------------------------------------------------

    def set_shard(self, shard_id: int, num_shards: int) -> None:
        if not 0 <= shard_id < num_shards:
            raise ValueError(f"invalid shard {shard_id}/{num_shards}")
        self.shard_id = shard_id
        self.num_shards = num_shards
        self._apply_shard()
        self._reset_cursors()

    def state_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "num_shards": self.num_shards,
            "seq_len": self.seq_len,
            "rng": self._rng.bit_generator.state,
            "cursors": {name: dict(cursor) for name, cursor in self._cursors.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("num_shards") != self.num_shards or state.get("shard_id") != self.shard_id:
            raise ValueError(
                f"data state is for shard {state.get('shard_id')}/"
                f"{state.get('num_shards')}, current is {self.shard_id}/{self.num_shards}"
            )
        if state.get("seq_len") != self.seq_len:
            raise ValueError("data state seq_len mismatch")
        self._rng.bit_generator.state = state["rng"]
        for name, cursor in state["cursors"].items():
            if name not in self._cursors:
                raise ValueError(f"data state references unknown source {name!r}")
            self._cursors[name] = dict(cursor)

    def close(self) -> None:
        """Release every open shard mapping.

        Iteration alone never needs this — ``_take`` copies out of the map and
        the cache self-evicts. It matters for anything that has to delete or
        replace a shard file afterwards: on Windows an open mapping makes the
        file undeletable, and on Linux it keeps the inode alive after unlink.
        """
        for memmap in self._memmaps.values():
            handle = getattr(memmap, "_mmap", None)
            if handle is not None:
                handle.close()
        self._memmaps.clear()

    # -- iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            index = int(self._rng.choice(len(self.names), p=self.weights))
            tokens = self._take(self.names[index])
            block = torch.from_numpy(tokens.astype(np.int64))
            yield {"input_ids": block, "labels": block.clone()}

    # -- internals ---------------------------------------------------------

    def _apply_shard(self) -> None:
        """Deterministically split each source's shard files across ranks."""
        self._files: dict[str, list[dict[str, Any]]] = {}
        for name, files in self.all_files.items():
            mine = files[self.shard_id :: self.num_shards]
            if not mine:
                raise ValueError(
                    f"source {name} has {len(files)} shard files but "
                    f"{self.num_shards} ranks — re-shard with more/smaller files"
                )
            self._files[name] = mine
        # Independent stream per rank; same seed would replay identical data.
        self._rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([self.seed, self.shard_id]))
        )
        self._memmaps: dict[tuple[str, int], np.memmap] = {}

    def _reset_cursors(self) -> None:
        self._cursors = {
            name: {"file": 0, "offset": 0, "epoch": 0} for name in self.names
        }

    def _memmap(self, name: str, file_index: int) -> np.memmap:
        key = (name, file_index)
        cached = self._memmaps.get(key)
        if cached is None:
            entry = self._files[name][file_index]
            path = self.token_root / name / entry["path"]
            cached = np.memmap(path, dtype=np.uint16, mode="r")
            # Keep at most a couple of maps open per source; shards are read
            # front-to-back so older ones are never revisited within an epoch.
            if len(self._memmaps) > 2 * len(self.names):
                self._memmaps.clear()
            self._memmaps[key] = cached
        return cached

    def _take(self, name: str) -> np.ndarray:
        cursor = self._cursors[name]
        files = self._files[name]
        for _ in range(len(files) + 1):
            data = self._memmap(name, cursor["file"])
            start = cursor["offset"]
            if start + self.seq_len <= data.size:
                cursor["offset"] = start + self.seq_len
                return np.asarray(data[start : start + self.seq_len])
            # Tail shorter than one block: drop it and move on. At 256M tokens
            # per shard the discarded remainder is under 0.002% of the source.
            cursor["file"] += 1
            cursor["offset"] = 0
            if cursor["file"] >= len(files):
                cursor["file"] = 0
                cursor["epoch"] += 1
                self._memmaps.clear()
        raise RuntimeError(f"source {name}: no shard holds a full {self.seq_len}-token block")


def corpus_fingerprint(mixture_path: str | Path, token_root: str | Path) -> str:
    """Stable id for the tokenized corpus — pass as TrainingConfig.data_fingerprint.

    Covers source names, weights and per-source token totals, so swapping in a
    re-tokenized shard set makes resume fail loudly instead of silently
    training on different data.
    """
    import hashlib

    mixture = json.loads(Path(mixture_path).read_text(encoding="utf-8"))
    root = Path(token_root)
    parts = []
    for source in mixture["sources"]:
        name = source["name"]
        index = json.loads((root / name / "index.json").read_text(encoding="utf-8"))
        parts.append(f"{name}:{source['weight']}:{index['total_tokens']}:{len(index['shards'])}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]
