from __future__ import annotations

import statistics
from collections import deque

import torch


class G2Monitor:
    def __init__(self, load_ratio: float = 1.5, sustain_steps: int = 1000, grad_factor: float = 5.0, window: int = 200, spike_sigma: float = 4.0, recover_steps: int = 200, gap_growth: float = 0.01):
        self.load_ratio, self.sustain = load_ratio, sustain_steps
        self.grad_factor, self.spike_sigma, self.recover = grad_factor, spike_sigma, recover_steps
        self.gap_growth = gap_growth
        self.grads: deque[float] = deque(maxlen=window)
        self.losses: deque[float] = deque(maxlen=window)
        self.high_load_steps = 0
        self.zero_steps: torch.Tensor | None = None
        self.spike_start: int | None = None
        self.reported_dead = 0
        self.gap_epoch: dict[str, int] = {}
        self.gap_now: dict[str, list[float]] = {}
        self.gap_done: dict[str, list[tuple[int, float]]] = {}

    def update_gap(self, gaps: dict[str, float], epochs: dict[str, float]) -> list[str]:
        alarms = []
        for d, gap in gaps.items():
            e = int(epochs.get(d, 0.0))
            if d in self.gap_epoch and e > self.gap_epoch[d] and self.gap_now.get(d):
                done = self.gap_done.setdefault(d, [])
                done.append((self.gap_epoch[d], statistics.fmean(self.gap_now[d])))
                self.gap_now[d] = []
                if len(done) >= 2 and done[-1][1] - done[-2][1] > self.gap_growth:
                    (e0, g0), (e1, g1) = done[-2], done[-1]
                    alarms.append(f"train-val gap of {d} grew from {g0:.4f} (epoch {e0}) to {g1:.4f} nats " f"(epoch {e1}): memorization signal, lower its weight")
            self.gap_epoch[d] = e
            self.gap_now.setdefault(d, []).append(gap)
        return alarms

    def update(self, step: int, loss: float, grad_norm: float, moe_stats: dict, counts: torch.Tensor | None) -> list[str]:
        alarms = []
        if len(self.grads) >= 20:
            med = statistics.median(self.grads)
            if grad_norm > self.grad_factor * med:
                alarms.append(f"grad_norm {grad_norm:.3g} > {self.grad_factor}x median {med:.3g}")
        self.grads.append(grad_norm)

        if moe_stats:
            ratio = moe_stats["moe/load_max_over_mean"]
            self.high_load_steps = self.high_load_steps + 1 if ratio > self.load_ratio else 0
            if self.high_load_steps == self.sustain:
                alarms.append(f"expert load max/mean > {self.load_ratio} for {self.sustain} steps (check gamma, alpha)")
        if counts is not None:
            zero = (counts == 0).long()
            self.zero_steps = zero if self.zero_steps is None else (self.zero_steps + 1) * zero
            dead = int((self.zero_steps >= self.sustain).sum())
            if dead > self.reported_dead:
                alarms.append(f"{dead} experts had zero load for {self.sustain} steps")
            self.reported_dead = dead

        if len(self.losses) >= 50:
            mean, std = statistics.fmean(self.losses), statistics.pstdev(self.losses)
            if loss > mean + self.spike_sigma * max(std, 1e-3 * mean):
                if self.spike_start is None:
                    self.spike_start = step
                elif step - self.spike_start == self.recover:
                    alarms.append(f"loss spike since step {self.spike_start} not recovered: roll back and skip data")
                return alarms
            self.spike_start = None
        self.losses.append(loss)
        return alarms
