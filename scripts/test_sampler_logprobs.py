"""CPU unit test of train_rank1_pg.sampler_logprobs (no GPU, no model).

Checks (a) the value against an independent numpy implementation of the vLLM sampler order
(presence penalty -> temperature -> top-k -> top-p), including chunk boundaries, and (b) review item A-1: the gradient is finite and
equals an independent float64 autograd reference even when the target token is inside the top-k but cut off by top-p
(the case where the old logsumexp version produced NaN).

    python -I scripts/test_sampler_logprobs.py
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rank1_pg as T  # noqa: E402


def ref_value(z_row, y_prev, y_t):
    """numpy reference for one position. z_row: logits (V,), y_prev: tokens generated before this position, y_t: target."""
    z = z_row.astype(np.float64).copy()
    for tok in set(int(t) for t in y_prev):
        z[tok] -= T.PRES
    z = z / T.TEMP
    top = np.argsort(-z, kind="stable")[:T.TOPK]
    vals = z[top]
    p = np.exp(vals - vals.max())
    p /= p.sum()
    keep = (np.cumsum(p) - p) < T.TOPP
    kv = vals[keep]
    lse = kv.max() + math.log(np.exp(kv - kv.max()).sum())
    lp = vals - lse
    where = np.nonzero((top == y_t) & keep)[0]
    return float(lp[where[0]]) if len(where) else -math.inf


def ref_torch64(z_row, y_prev, y_t):
    """float64 autograd reference of the same quantity (differentiable w.r.t. z_row); None if y_t is outside the support."""
    z = z_row.double()
    pen = torch.zeros_like(z)
    for tok in set(int(t) for t in y_prev):
        pen[tok] = T.PRES
    z = (z - pen) / T.TEMP
    vals, idx = z.topk(T.TOPK)
    p = torch.softmax(vals.detach(), -1)
    keep = (p.cumsum(-1) - p) < T.TOPP
    sel = ((idx == y_t) & keep).nonzero()
    if len(sel) == 0:
        return None
    lse = torch.logsumexp(vals[keep], -1)
    return vals[sel[0, 0]] - lse


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    V, dm, L = 400, 24, 700                      # L > 2 chunks of 256
    head = torch.nn.Linear(dm, V, bias=False)
    h = torch.randn(L, dm) * 3.0

    # make a sequence y: mostly sampled from the reference distribution, some forced to be rank 19 (inside top-k, cut by top-p)
    z_all = head(h).detach().numpy().astype(np.float64)
    y = np.zeros(L, dtype=np.int64)
    forced = set(rng.choice(L, size=60, replace=False).tolist())
    for t in range(L):
        z = z_all[t].copy()
        for tok in set(y[:t].tolist()):
            z[tok] -= T.PRES
        z = z / T.TEMP
        top = np.argsort(-z, kind="stable")[:T.TOPK]
        if t in forced:
            y[t] = top[T.TOPK - 1]                # the 20th token: almost always outside the top-p set
        else:
            vals = z[top]
            p = np.exp(vals - vals.max())
            p /= p.sum()
            y[t] = top[rng.choice(T.TOPK, p=p)]
    yt = torch.tensor(y)

    # (a) values
    lp = T.sampler_logprobs(h.clone().requires_grad_(True), head, yt, chunk=256).detach().numpy()
    ref = np.array([ref_value(z_all[t], y[:t], y[t]) for t in range(L)])
    fin_ref, fin_new = np.isfinite(ref), np.isfinite(lp)
    n_out = int((~fin_ref).sum())
    print(f"positions {L}; targets outside the support (reference): {n_out}; finite pattern identical: {bool((fin_ref == fin_new).all())}")
    d = np.abs(lp[fin_ref & fin_new] - ref[fin_ref & fin_new])
    print(f"max |value diff| on finite positions: {d.max():.2e}")
    ok_val = bool((fin_ref == fin_new).all()) and d.max() < 1e-3

    # (b) gradient: finite, and equal to the float64 autograd reference
    hh = h.clone().requires_grad_(True)
    out = T.sampler_logprobs(hh, head, yt, chunk=256)
    out[torch.isfinite(out)].sum().backward()
    g = hh.grad
    finite_grad = bool(torch.isfinite(g).all())
    print("gradient finite:", finite_grad, "| n_out > 0 (the A-1 case is exercised):", n_out > 0)

    hr = h.clone().double().requires_grad_(True)
    zr = head.double()(hr)
    tot = 0.0
    for t in range(L):
        r = ref_torch64(zr[t], y[:t], int(y[t]))
        if r is not None:
            tot = tot + r
    tot.backward()
    gref = hr.grad.float()
    rel = float((g - gref).norm() / gref.norm())
    print(f"relative gradient difference vs float64 autograd reference: {rel:.2e}")
    ok_grad = finite_grad and rel < 1e-3

    print("PASS" if (ok_val and ok_grad and n_out > 0) else "FAIL")
    sys.exit(0 if (ok_val and ok_grad and n_out > 0) else 1)


if __name__ == "__main__":
    main()
