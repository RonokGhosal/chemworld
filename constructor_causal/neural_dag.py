"""
Topological-ORDER recovery by batched neural RESIT (Phase 1 only) -- large-scale neural causal discovery.

AUDIT CAVEATS (pass 3 -- claims corrected):
  * This recovers a topological ORDER over the variables, NOT a connected DAG: resit_order returns only a
    permutation (no adjacency / edge-pruning). order_accuracy iterates over the GROUND-TRUTH edges only, so
    FALSE edges are never scored -- the metric structurally cannot see false positives. Full RESIT needs a
    Phase-2 parent-selection step that is not implemented here.
  * "Beats classical additive-noise methods on multiplicative noise" is NOT measured: the only baseline is
    a random permutation; no ANM estimator is ever run, so the comparative claim is unfalsifiable from this
    code. Also graded against random chance, not the gameability baselines (varsortability / sortnregress,
    Reisach 2021 "Beware the Simulated DAG!").
  * All synthetic noise is Gaussian, i.e. the LSNM location-scale model is EXACTLY specified (best case);
    the multiplicative-regime claim is not tested where Gaussian-LSNM is known to fail (misspecified noise).

neural_scale.py oriented INDEPENDENT pairs. This recovers a topological ORDER over a connected DAG's
variables (variables share parents, confounding is possible) by RESIT Phase-1 (Regression with Subsequent
Independence Test, Peters et al. 2014) made neural + batched:

  while variables remain:
    for each candidate variable i, regress x_i on ALL other remaining variables with a MULTIVARIATE
    location-scale net (mean mu(x_-i) + scale exp(s(x_-i))*eps), Gaussian NLL;  measure how dependent
    its standardized residual still is on the others (sum of HSIC).  The variable whose residual is
    MOST INDEPENDENT is a SINK (it causes none of the rest) -> remove it, record it, repeat.
  Reverse the removal order -> a topological order (causes first).

The per-round candidate regressions are BATCHED (k multivariate nets in one bmm pass); the rounds are
sequential (each depends on the previous sink). Total ~ d(d+1)/2 multivariate net fits -- O(d^2).
Score = ORDER accuracy: fraction of true edges (i->j) whose cause i precedes effect j in the recovered
order (1.0 perfect, ~0.5 chance) -- an ORDER metric, not a graph-recovery (SHD/F1) metric. The scale net
lets the location-scale regression fit heteroscedastic noise; whether that beats additive ANM is NOT
established here (see AUDIT CAVEATS -- no ANM baseline is run).

CLI: python -m constructor_causal.neural_dag [--d 40] [--n 1000] [--regime multiplicative|additive]
                                             [--indeg 2] [--h 64] [--epochs 200] [--seeds 3]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from .neural_scale import DEVICE, batched_hsic, _w, _b


class BatchedMVLSNM(nn.Module):
    """T parallel MULTIVARIATE location-scale nets: in_dim -> h -> h -> {mu, log-scale}."""

    def __init__(self, T, in_dim, h=64):
        super().__init__()
        self.W1, self.b1 = _w(T, in_dim, h), _b(T, 1, h)
        self.W2, self.b2 = _w(T, h, h), _b(T, 1, h)
        self.Wm, self.bm = _w(T, h, 1), _b(T, 1, 1)
        self.Ws, self.bs = _w(T, h, 1), _b(T, 1, 1)

    def forward(self, x):                               # x: (T,n,in_dim)
        z = torch.tanh(torch.baddbmm(self.b1, x, self.W1))
        z = torch.tanh(torch.baddbmm(self.b2, z, self.W2))
        mu = torch.baddbmm(self.bm, z, self.Wm)
        ls = torch.baddbmm(self.bs, z, self.Ws).clamp(-4.0, 4.0)
        return mu, ls


def mv_residuals(INP, TGT, h=64, epochs=200, lr=1e-2):
    """INP: (T,n,in_dim), TGT: (T,n) tensors on device. Returns standardized residuals (T,n)."""
    T, n, in_dim = INP.shape
    m = BatchedMVLSNM(T, in_dim, h).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Y = TGT.unsqueeze(-1)
    for _ in range(epochs):
        opt.zero_grad()
        mu, ls = m(INP)
        nll = (ls + 0.5 * ((Y - mu) * torch.exp(-ls)) ** 2).mean()
        nll.backward()
        opt.step()
    with torch.no_grad():
        mu, ls = m(INP)
        return ((Y - mu) * torch.exp(-ls)).squeeze(-1)


def resit_order(X, h=64, epochs=200, hsic_m=256):
    """Recover a topological order (causes first) by batched neural RESIT. X: (n,d) numpy."""
    n, d = X.shape
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xt = torch.tensor(Xs, dtype=torch.float32, device=DEVICE)        # (n,d)
    active = list(range(d))
    removed = []                                                     # sinks, in removal order
    while len(active) > 1:
        A = active
        k = len(A)
        Xa = Xt[:, A]                                                # (n,k)
        INP = Xa.t().unsqueeze(0).repeat(k, 1, 1).transpose(1, 2).contiguous()  # (k,n,k): task t sees Xa
        idx = torch.arange(k, device=DEVICE)
        INP[idx, :, idx] = 0.0                                       # mask the candidate's own column
        TGT = Xa.t()                                                 # (k,n): task t target = Xa[:,t]
        R = mv_residuals(INP, TGT, h=h, epochs=epochs)               # (k,n)
        dep = torch.zeros(k, device=DEVICE)                          # sum_s HSIC(residual_t, x_s), s!=t
        Xa_t = Xa.t()                                                # (k,n)
        for s in range(k):
            xs = Xa_t[s].unsqueeze(0).repeat(k, 1)                   # (k,n)
            hs = batched_hsic(xs, R, m=hsic_m)
            hs[s] = 0.0
            dep += hs
        sink = int(torch.argmin(dep).item())
        removed.append(A[sink])
        active.pop(sink)
    removed.append(active[0])
    return removed[::-1]                                             # reverse removal -> topo (causes first)


def gen_connected_dag(d, n, regime, rng, indeg=2):
    """Random connected DAG (topo order 0..d-1, each node up to `indeg` parents from earlier), then
    LABELS permuted. Returns (X (n,d) in label space, true_edges set of (cause,effect) in label space)."""
    parents = {0: []}
    for j in range(1, d):
        parents[j] = sorted(int(p) for p in rng.choice(j, size=min(indeg, j), replace=False))
    X = np.zeros((n, d))
    X[:, 0] = rng.normal(0, 1, n)
    for j in range(1, d):
        pa = parents[j]
        e = rng.normal(0, 1, n)
        if not pa:
            X[:, j] = rng.normal(0, 1, n)
            continue
        mean = sum(np.sin(2.0 * X[:, p]) for p in pa) / np.sqrt(len(pa))
        if regime == "additive":
            X[:, j] = mean + 0.5 * e
        elif regime == "multiplicative":
            scale = 0.4 + 0.6 * np.abs(sum(np.sin(1.5 * X[:, p]) for p in pa)) / np.sqrt(len(pa))
            X[:, j] = mean + scale * e                              # heteroscedastic in the parents
        else:
            raise ValueError(regime)
    perm = rng.permutation(d)
    lab = {i: int(perm[i]) for i in range(d)}
    Xp = np.empty_like(X)
    for i in range(d):
        Xp[:, lab[i]] = X[:, i]
    true_edges = {(lab[p], lab[j]) for j in range(d) for p in parents[j]}
    return Xp, true_edges


def order_accuracy(order, true_edges):
    """Fraction of true edges (cause->effect) whose cause precedes its effect in the recovered order."""
    pos = {v: i for i, v in enumerate(order)}
    if not true_edges:
        return 1.0
    return sum(1 for (a, b) in true_edges if pos[a] < pos[b]) / len(true_edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=40)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--regime", default="multiplicative", choices=["additive", "multiplicative"])
    ap.add_argument("--indeg", type=int, default=2)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print("=" * 92)
    print(f"NEURAL RESIT -- connected-DAG topological recovery   device={DEVICE}")
    print(f"  d={args.d} n={args.n} regime={args.regime} indeg={args.indeg} h={args.h} "
          f"epochs={args.epochs} seeds={args.seeds}")
    print("=" * 92)
    print(f"  {'seed':>5} {'#edges':>7} {'order-acc':>10} {'rand-acc':>9} {'wall_s':>8}")
    accs, rng_accs = [], []
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        X, edges = gen_connected_dag(args.d, args.n, args.regime, rng, indeg=args.indeg)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        order = resit_order(X, h=args.h, epochs=args.epochs)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        acc = order_accuracy(order, edges)
        rnd = order_accuracy(list(rng.permutation(args.d)), edges)   # chance baseline ~0.5
        accs.append(acc); rng_accs.append(rnd)
        print(f"  {s:>5} {len(edges):>7} {acc:>10.2f} {rnd:>9.2f} {dt:>8.1f}")
    print("=" * 92)
    print(f"  ORDER accuracy {np.mean(accs):.2f} (vs random {np.mean(rng_accs):.2f}): fraction of true")
    print(f"  cause->effect edges the recovered topological order gets right. RESIT recovers the order")
    print(f"  from OBSERVATION alone in the {args.regime} regime by learning the conditional scale.")
    print(f"  ~{args.d*(args.d+1)//2} multivariate net fits / DAG -- O(d^2), batched per round on the GPU.")
    print("=" * 92)


if __name__ == "__main__":
    main()
