from __future__ import annotations

import warnings

import torch
import triton
import triton.language as tl


@triton.jit
def _ce_grad_kernel(logits_ptr, labels_ptr, lse_ptr, target_ptr, n_valid_ptr, z_coef, n_cols, row_stride, IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * row_stride
    cols = tl.arange(0, BLOCK)
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        c = start + cols
        x = tl.load(base + c, mask=c < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        s = tl.where(m_new > float("-inf"), s * tl.exp(m - m_new) + tl.exp(x - m_new), 0.0)
        m = m_new
    m_row = tl.max(m, axis=0)
    lse = m_row + tl.log(tl.sum(tl.where(m > float("-inf"), s * tl.exp(m - m_row), 0.0), axis=0))

    label = tl.load(labels_ptr + row)
    valid = label != IGNORE
    y = tl.where(valid, label, 0)
    target = tl.load(base + y).to(tl.float32)
    scale = tl.where(valid, 1.0 / tl.load(n_valid_ptr), 0.0)
    zfac = 1.0 + 2.0 * z_coef * lse
    for start in range(0, n_cols, BLOCK):
        c = start + cols
        mask = c < n_cols
        x = tl.load(base + c, mask=mask, other=0.0).to(tl.float32)
        g = tl.exp(x - lse) * zfac
        g = tl.where(c == y, g - 1.0, g) * scale
        tl.store(base + c, g.to(logits_ptr.dtype.element_ty), mask=mask)
    tl.store(lse_ptr + row, lse)
    tl.store(target_ptr + row, target)


def ce_grad_(logits: torch.Tensor, labels: torch.Tensor, n_valid: torch.Tensor, z_coef: float, ignore_index: int):
    n, V = logits.shape
    assert logits.stride(1) == 1 and n_valid.dtype == torch.float32
    labels = labels.contiguous()
    lse = torch.empty(n, dtype=torch.float32, device=logits.device)
    target = torch.empty_like(lse)
    if n:
        _ce_grad_kernel[(n,)](logits, labels, lse, target, n_valid, float(z_coef), V, logits.stride(0), IGNORE=ignore_index, BLOCK=min(4096, triton.next_power_of_2(V)), num_warps=8)
    return lse, target


_ADDMM_FP32: dict = {}


def _probe_addmm_fp32(device: torch.device) -> bool:
    try:
        g = (torch.arange(16 * 24, device=device) % 9 - 4).view(16, 24).to(torch.bfloat16)
        hc = (torch.arange(16 * 8, device=device) % 7 - 3).view(16, 8).to(torch.bfloat16)
        acc = torch.full((24, 8), 1024.125, dtype=torch.float32, device=device)
        torch.addmm(acc, g.t(), hc, out_dtype=torch.float32, out=acc)
        ref = 1024.125 + g.double().t() @ hc.double()
        return bool(torch.equal(acc.double(), ref))
    except (TypeError, RuntimeError):
        return False


def accumulate_wgrad(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dtype == torch.float32:
        acc.addmm_(a, b)
        return
    ok = _ADDMM_FP32.get(acc.device)
    if ok is None:
        ok = _ADDMM_FP32[acc.device] = _probe_addmm_fp32(acc.device)
        if not ok:
            warnings.warn("torch.addmm(out_dtype=float32) is not usable: grad_w adds BF16 products in FP32")
    if ok:
        torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
    else:
        acc.add_(a @ b)
