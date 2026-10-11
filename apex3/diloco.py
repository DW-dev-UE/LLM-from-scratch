from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import torch


def _ranges(n: int, parts: int) -> list[tuple[int, int]]:
    return [(n * i // parts, n * (i + 1) // parts) for i in range(parts)]


def _read(path: str, lo: int, hi: int) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.float32, count=hi - lo, offset=lo * 4))


def _write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class LocalExchange:
    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)

    def publish(self, step: int, rank: int, files: list[str]) -> None:
        open(os.path.join(self.dir, f"s{step:06d}_r{rank}.done"), "w").close()

    def wait(self, step: int, rank: int, log: Callable) -> None:
        marker, t0 = os.path.join(self.dir, f"s{step:06d}_r{rank}.done"), time.time()
        while not os.path.exists(marker):
            time.sleep(1)
            if time.time() - t0 > 600:
                log(f"[diloco] waiting for worker {rank} (sync {step})")
                t0 = time.time()


class SshExchange(LocalExchange):
    def __init__(self, directory: str, peers: list[str], key: str | None = None, streams: int = 8, recv: str = "numactl --membind=0"):
        super().__init__(directory)
        os.chmod(directory, 0o1777)
        self.peers, self.streams, self.recv = peers, streams, recv
        self.ssh = ["-q", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=60", "-c", "aes128-gcm@openssh.com"] + (["-i", key] if key else [])

    def _send(self, src: str, peer: str) -> None:
        dst = f"{self.dir}/{os.path.basename(src)}"
        remote = f"{self.recv} sh -c 'cat > {dst}.part && mv {dst}.part {dst}'"
        for attempt in range(10):
            with open(src, "rb") as f:
                if subprocess.run(["ssh", *self.ssh, peer, remote], stdin=f).returncode == 0:
                    return
            time.sleep(min(600, 30 * 2**attempt))
        raise RuntimeError(f"sending {src} to {peer} failed")

    def publish(self, step: int, rank: int, files: list[str]) -> None:
        with ThreadPoolExecutor(self.streams) as pool:
            list(pool.map(lambda job: self._send(*job), [(f, p) for p in self.peers for f in files]))
        marker = os.path.join(self.dir, f"s{step:06d}_r{rank}.done")
        open(marker, "w").close()
        for p in self.peers:
            self._send(marker, p)


class DiLoCo:
    def __init__(self, master: torch.Tensor, extras: list[torch.Tensor], refresh: Callable[[], None], *, rank: int, workers: int,
                 state_dir: str, exchange: LocalExchange, lr: float = 0.7, momentum: float = 0.9, parts: int = 8, chunk: int = 1 << 26, log: Callable = print):
        if not (master.dtype == torch.float32 and master.is_contiguous() and master.device.type == "cpu"):
            raise ValueError("master must be a contiguous FP32 CPU tensor")
        self.master, self.extras, self.refresh = master.view(-1), extras, refresh
        self.rank, self.workers, self.dir, self.xchg = rank, workers, state_dir, exchange
        self.lr, self.mu, self.parts, self.chunk, self.log = lr, momentum, parts, chunk, log
        self.n = self.master.numel()
        os.makedirs(state_dir, exist_ok=True)
        self.meta_path = os.path.join(state_dir, "meta.json")

    def _gen(self, name: str, gen: int) -> str:
        return os.path.join(self.dir, f"{name}_{gen}.bin" if name != "extras" else f"extras_{gen}.npy")

    def _x(self, step: int, rank: int, part: int | None) -> str:
        tail = "x.npy" if part is None else f"p{part}.bin"
        return os.path.join(self.xchg.dir, f"s{step:06d}_r{rank}_{tail}")

    def meta(self) -> dict | None:
        if not os.path.exists(self.meta_path):
            return None
        with open(self.meta_path) as f:
            return json.load(f)

    @property
    def completed(self) -> int:
        return self.meta()["completed"]

    def _extras_flat(self) -> np.ndarray:
        if not self.extras:
            return np.zeros(0, dtype=np.float32)
        return torch.cat([e.detach().float().reshape(-1).cpu() for e in self.extras]).numpy()

    def _set_extras(self, flat: np.ndarray) -> None:
        o = 0
        for e in self.extras:
            k = e.numel()
            e.copy_(torch.from_numpy(flat[o:o + k]).view_as(e).to(e.device, e.dtype))
            o += k

    def fingerprint(self, t: torch.Tensor | None = None) -> float:
        t = self.master if t is None else t
        return float(t[:: max(1, self.n // 4096)].double().sum())

    def start(self, resumed: dict | None) -> bool:
        meta = self.meta()
        done, at_sync = (resumed or {}).get("completed", 0), (resumed or {}).get("at_sync", False)
        if meta is None:
            if done:
                raise RuntimeError(f"checkpoint has {done} DiLoCo syncs but {self.dir} has no meta.json")
            g = np.memmap(self._gen("global", 0), dtype=np.float32, mode="w+", shape=(self.n,))
            for lo, hi in _ranges(self.n, max(1, self.n // self.chunk)):
                g[lo:hi] = self.master[lo:hi].numpy()
            g.flush()
            del g
            np.memmap(self._gen("momentum", 0), dtype=np.float32, mode="w+", shape=(self.n,)).flush()
            np.save(self._gen("extras", 0), self._extras_flat())
            _write_json(self.meta_path, {"gen": 0, "completed": 0, "workers": self.workers, "n": self.n})
            self.log(f"[diloco] worker {self.rank}/{self.workers}: new outer state in {self.dir}, " f"fingerprint {self.fingerprint():.6e}")
            return False
        if meta["n"] != self.n or meta["workers"] != self.workers:
            raise RuntimeError(f"{self.meta_path}: n {meta['n']} workers {meta['workers']}, " f"this run n {self.n} workers {self.workers}")
        if meta["completed"] == done:
            self.log(f"[diloco] worker {self.rank}: {done} syncs" + (", redoing the interrupted sync" if at_sync else ""))
            return at_sync
        if meta["completed"] == done + 1 and at_sync:
            self._load_global(meta["gen"])
            self.log(f"[diloco] worker {self.rank}: sync {done} finished after the checkpoint, continuing from the " f"global weights, fingerprint {self.fingerprint():.6e}")
            return False
        raise RuntimeError(f"checkpoint at {done} syncs (at_sync {at_sync}) does not match {self.meta_path} " f"({meta['completed']} syncs)")

    def _load_global(self, gen: int) -> None:
        path = self._gen("global", gen)
        for lo, hi in _ranges(self.n, max(1, self.n // self.chunk)):
            self.master[lo:hi].copy_(_read(path, lo, hi))
        self._set_extras(np.load(self._gen("extras", gen)))
        self.refresh()

    def sync(self) -> dict:
        t0 = time.time()
        meta = self.meta()
        gen, step = meta["gen"], meta["completed"]
        g_path, m_path = self._gen("global", gen), self._gen("momentum", gen)

        own = []
        for i, (lo, hi) in enumerate(_ranges(self.n, self.parts)):
            path = self._x(step, self.rank, i)
            out = np.memmap(path, dtype=np.float32, mode="w+", shape=(hi - lo,))
            for a in range(lo, hi, self.chunk):
                b = min(hi, a + self.chunk)
                torch.sub(_read(g_path, a, b), self.master[a:b], out=torch.from_numpy(out[a - lo:b - lo]))
            out.flush()
            del out
            own.append(path)
        np.save(self._x(step, self.rank, None), self._extras_flat())
        own.append(self._x(step, self.rank, None))
        t_write = time.time()
        self.xchg.publish(step, self.rank, own)
        t_push = time.time()
        for r in range(self.workers):
            if r != self.rank:
                self.xchg.wait(step, r, self.log)
        t_wait = time.time()

        new_g = np.memmap(self._gen("global", gen + 1), dtype=np.float32, mode="w+", shape=(self.n,))
        new_m = np.memmap(self._gen("momentum", gen + 1), dtype=np.float32, mode="w+", shape=(self.n,))
        for i, (lo, hi) in enumerate(_ranges(self.n, self.parts)):
            for a in range(lo, hi, self.chunk):
                b = min(hi, a + self.chunk)
                g = _read(self._x(step, 0, i), a - lo, b - lo)
                for r in range(1, self.workers):
                    g.add_(_read(self._x(step, r, i), a - lo, b - lo))
                g.div_(self.workers)
                buf = _read(m_path, a, b).mul_(self.mu).add_(g)
                w = _read(g_path, a, b).sub_(g.add_(buf, alpha=self.mu), alpha=self.lr)
                new_m[a:b] = buf.numpy()
                new_g[a:b] = w.numpy()
                self.master[a:b].copy_(w)
        new_g.flush()
        new_m.flush()
        del new_g, new_m
        ex = np.mean([np.load(self._x(step, r, None)) for r in range(self.workers)], axis=0).astype(np.float32)
        np.save(self._gen("extras", gen + 1), ex)
        _write_json(self.meta_path, {**meta, "gen": gen + 1, "completed": step + 1})
        self._set_extras(ex)
        self.refresh()

        for name in ("global", "momentum", "extras"):
            if os.path.exists(self._gen(name, gen)):
                os.remove(self._gen(name, gen))
        prefix = f"s{step - 1:06d}_"
        for f in os.listdir(self.xchg.dir):
            if f.startswith(prefix):
                os.remove(os.path.join(self.xchg.dir, f))
        stats = {"sync": step + 1, "write_s": t_write - t0, "push_s": t_push - t_write, "wait_s": t_wait - t_push, "update_s": time.time() - t_wait, "fingerprint": self.fingerprint()}
        self.log(f"[diloco] sync {step + 1} done: write {stats['write_s']:.0f}s push {stats['push_s']:.0f}s " f"wait {stats['wait_s']:.0f}s update {stats['update_s']:.0f}s " f"fingerprint {stats['fingerprint']:.6e}")
        return stats
