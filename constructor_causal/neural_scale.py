"""
GPU SCALE HARNESS for neural causal discovery -- what actually justifies an A100.

Two engines, both fully BATCHED so T independent 1-D nets train in ONE forward/backward pass (3-D
weight tensors via bmm). This is the GPU win: orienting a DAG over d variables needs ~d(d-1) directional
fits; sequential CPU fits are the bottleneck, batched GPU fits are not.

  1. BatchedLSNM  -- T parallel location-scale nets  y = mu(x) + exp(s(x))*eps, Gaussian-NLL.
     Orient i->j if the standardized residual (y-mu)/scale is more independent of the cause (batched
     HSIC). This is the engine from neural_discovery.py, vectorized over tasks. Fits multiplicative /
     heteroscedastic noise a fixed-variance additive ANM cannot -- but see AUDIT CAVEATS below.

  2. BatchedPNLFlow -- T parallel conditional normalizing flows for the POST-NONLINEAR frontier
     (y = g(f(x)+n), g invertible). Per task: an MLP f(x) and a MONOTONIC h(y) ~= g^-1
     (h(y) = a*y + sum_k b_k*sigmoid(c_k*y+d_k), a,b,c >= 0 => invertible), trained by exact
     change-of-variables MLE: loglik = -0.5*(h(y)-f(x))^2 + log h'(y). Orient by likelihood ratio
     (the causal direction admits the higher-likelihood flow). This is the tool a location-scale model
     could not be (neural_discovery.py left post_nonlinear at ~0.19) -- the open piece, now attacked.

AUDIT CAVEATS (pass 3):
  * All synthetic noise is Gaussian (gen_pairs draws eps ~ N(0,1)), so the multiplicative regime
    y = scale(x)*eps is EXACTLY a conditionally-Gaussian location-scale model -- the best case the NLL
    assumes. Gaussian-LSNM orientation fails under MISSPECIFIED (non-Gaussian) noise; that regime, and any
    ANM baseline, are untested here. The "additive ANM can't touch" line is not measured.
  * BatchedPNLFlow orients by LIKELIHOOD RATIO with a fixed standard-normal base -- exactly the ML rule
    Immer et al. 2023 ("On the Identifiability and Estimation of Causal Location-Scale Noise Models" /
    "Maximum Likelihood vs. Independence Testing") show is FRAGILE under misspecification; the LSNM engine's
    residual-independence (HSIC) test is the more robust one. Do not read LR-orientation accuracy as a
    general guarantee.
  * The benchmark's ground truth is always x->y, so orientation accuracy counts agreement with a CONSTANT
    label -- a fixed directional bias scores as well as genuine skill.

Device-aware (CUDA on the A100, MPS on a Mac, else CPU). `main` runs a correctness+throughput sweep:
orientation accuracy AND pairs/sec at growing batch sizes, so the GPU scaling is measured, not asserted.

CLI:  python -m constructor_causal.neural_scale [--P 512] [--n 1500] [--h 64] [--epochs 400]
                                                [--method lsnm|flow|both] [--regimes additive,...]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from .neural_discovery import get_device

DEVICE = get_device()


# ============================== shared: batched MLP trunk ==============================
def _w(*shape):
    fan_in = shape[-2]
    return nn.Parameter(torch.randn(*shape) * (2.0 / fan_in) ** 0.5)


def _b(*shape):
    return nn.Parameter(torch.zeros(*shape))


def _trunk_params(T, h):
    """A 1->h->h batched MLP trunk as (W1,b1,W2,b2)."""
    return nn.ParameterList([_w(T, 1, h), _b(T, 1, h), _w(T, h, h), _b(T, 1, h)])


def _trunk_fwd(p, x):                                   # x: (T,n,1) -> (T,n,h)
    z = torch.tanh(torch.baddbmm(p[1], x, p[0]))
    return torch.tanh(torch.baddbmm(p[3], z, p[2]))


# ============================== engine 1: batched location-scale ==============================
class BatchedLSNM(nn.Module):
    def __init__(self, T, h=64):
        super().__init__()
        self.trunk = _trunk_params(T, h)
        self.Wm, self.bm = _w(T, h, 1), _b(T, 1, 1)
        self.Ws, self.bs = _w(T, h, 1), _b(T, 1, 1)

    def forward(self, x):                               # x: (T,n,1)
        z = _trunk_fwd(self.trunk, x)
        mu = torch.baddbmm(self.bm, z, self.Wm)
        ls = torch.baddbmm(self.bs, z, self.Ws).clamp(-4.0, 4.0)
        return mu, ls


def lsnm_residuals(causes, effects, h=64, epochs=400, lr=1e-2):
    """causes/effects: (T,n) numpy (already standardized). Returns standardized residuals (T,n) tensor."""
    T = causes.shape[0]
    C = torch.tensor(causes, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    Y = torch.tensor(effects, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    m = BatchedLSNM(T, h).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        mu, ls = m(C)
        nll = (ls + 0.5 * ((Y - mu) * torch.exp(-ls)) ** 2).mean()
        nll.backward()
        opt.step()
    with torch.no_grad():
        mu, ls = m(C)
        return ((Y - mu) * torch.exp(-ls)).squeeze(-1)


def batched_hsic(X, R, m=256):
    """HSIC statistic per task. X,R: (T,n) tensors on device. Gaussian kernels, median bandwidth."""
    T, n = X.shape
    idx = torch.randperm(n, device=X.device)[:min(m, n)]
    x = X[:, idx].unsqueeze(-1)                         # (T,mm,1)
    r = R[:, idx].unsqueeze(-1)
    mm = x.shape[1]

    def K(a):
        d2 = (a - a.transpose(1, 2)) ** 2               # (T,mm,mm)
        med = d2.flatten(1).median(1).values.clamp_min(1e-6)
        sig = (0.5 * med).sqrt().view(T, 1, 1)
        return torch.exp(-d2 / (2 * sig * sig))
    H = torch.eye(mm, device=X.device) - 1.0 / mm
    KxH = K(x) @ H
    KrH = K(r) @ H
    return (KxH * KrH.transpose(1, 2)).sum((1, 2)) / (mm - 1) ** 2   # trace(KxH KrH)/(mm-1)^2


def _stdz(A):
    return (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)


def orient_lsnm(Xp, Yp, h=64, epochs=400):
    """Xp,Yp: (P,n) numpy, true direction x->y. Returns bool (P,) True => predicted x->y."""
    Xs, Ys = _stdz(Xp), _stdz(Yp)
    causes = np.vstack([Xs, Ys])                        # forward then backward, one batch of 2P
    effects = np.vstack([Ys, Xs])
    r = lsnm_residuals(causes, effects, h=h, epochs=epochs)
    cz = torch.tensor(causes, dtype=torch.float32, device=DEVICE)
    hs = batched_hsic(cz, r).cpu().numpy()
    P = Xp.shape[0]
    return hs[:P] < hs[P:]                              # forward residual more independent => x->y


# ============================== engine 2: batched PNL flow ==============================
class BatchedPNLFlow(nn.Module):
    """Per task: MLP f(x) and a MONOTONIC h(y)=a*y + sum_k b_k*sigmoid(c_k*y+d_k) (a,b,c>=0 via
    softplus => strictly increasing, invertible). Exact change-of-variables MLE."""

    def __init__(self, T, h=64, K=8):
        super().__init__()
        self.trunk = _trunk_params(T, h)
        self.Wf, self.bf = _w(T, h, 1), _b(T, 1, 1)
        self.a = _b(T, 1, 1)                            # softplus-> slope (>0)
        self.bk = _b(T, 1, K)                           # softplus-> bump heights (>=0)
        self.ck = _b(T, 1, K)                           # softplus-> bump slopes (>=0)
        self.dk = nn.Parameter(torch.randn(T, 1, K) * 0.5)

    def f(self, x):                                     # (T,n,1)
        return torch.baddbmm(self.bf, _trunk_fwd(self.trunk, x), self.Wf)

    def h_and_logderiv(self, y):                        # y: (T,n,1)
        a = torch.nn.functional.softplus(self.a)        # (T,1,1)
        bk = torch.nn.functional.softplus(self.bk)      # (T,1,K)
        ck = torch.nn.functional.softplus(self.ck)      # (T,1,K)
        z = ck * y + self.dk                            # (T,n,K)
        s = torch.sigmoid(z)
        h = a * y + (bk * s).sum(-1, keepdim=True)      # (T,n,1)
        hp = a + (bk * ck * s * (1 - s)).sum(-1, keepdim=True)   # h'(y) > 0
        return h, torch.log(hp.clamp_min(1e-6))

    def loglik(self, x, y):
        h, logdh = self.h_and_logderiv(y)
        n = h - self.f(x)                               # base noise (standard normal target)
        return (-0.5 * n ** 2 - 0.5 * np.log(2 * np.pi) + logdh).squeeze(-1)   # (T,n)


def flow_loglik(causes, effects, h=64, K=8, epochs=600, lr=5e-3):
    """Fit one conditional flow per task; return mean per-sample loglik (T,) on device."""
    C = torch.tensor(causes, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    Y = torch.tensor(effects, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    T = causes.shape[0]
    m = BatchedPNLFlow(T, h, K).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        nll = -m.loglik(C, Y).mean()
        nll.backward()
        opt.step()
    with torch.no_grad():
        return m.loglik(C, Y).mean(1)                   # (T,) per-task mean loglik


def orient_flow(Xp, Yp, h=64, K=8, epochs=600):
    """Orient by likelihood ratio: the causal direction admits the higher-likelihood PNL flow."""
    Xs, Ys = _stdz(Xp), _stdz(Yp)
    causes = np.vstack([Xs, Ys]); effects = np.vstack([Ys, Xs])
    ll = flow_loglik(causes, effects, h=h, K=K, epochs=epochs).cpu().numpy()
    P = Xp.shape[0]
    return ll[:P] > ll[P:]                              # forward likelihood higher => x->y


# ============================== worlds: P independent bivariate pairs ==============================
def gen_pairs(regime, P, n, rng):
    """P independent bivariate pairs, TRUE direction x->y (x exogenous N(0,1))."""
    x = rng.normal(0, 1, (P, n))
    e = rng.normal(0, 1, (P, n))
    if regime == "additive":
        y = np.sin(2 * x) + 0.5 * e
    elif regime == "multiplicative":
        y = (0.4 + 0.8 * np.abs(np.sin(1.5 * x))) * e
    elif regime == "post_nonlinear":
        y = np.tanh(2.5 * (np.sin(2 * x) + 0.5 * e))
    else:
        raise ValueError(regime)
    return x, y


# ============================== benchmark ==============================
def benchmark(regime, method, P, n, h, epochs, seed=0):
    rng = np.random.default_rng(seed)
    Xp, Yp = gen_pairs(regime, P, n, rng)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    if method == "flow":
        pred = orient_flow(Xp, Yp, h=h, epochs=epochs)
    else:
        pred = orient_lsnm(Xp, Yp, h=h, epochs=epochs)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    acc = float(np.mean(pred))                          # fraction predicted x->y (=accuracy, truth is x->y)
    return acc, dt, 2 * P / dt                          # 2P directional fits done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--method", default="both", choices=["lsnm", "flow", "both"])
    ap.add_argument("--regimes", default="additive,multiplicative,post_nonlinear")
    ap.add_argument("--scaling", action="store_true", help="also sweep P to measure GPU throughput scaling")
    args = ap.parse_args()
    regimes = args.regimes.split(",")

    print("=" * 96)
    print(f"NEURAL SCALE HARNESS  device={DEVICE}  P={args.P} n={args.n} h={args.h} epochs={args.epochs}")
    print("=" * 96)
    print(f"  {'regime':>15} {'method':>7} {'orient-acc':>11} {'wall_s':>8} {'fits/sec':>10}")
    methods = ["lsnm", "flow"] if args.method == "both" else [args.method]
    for regime in regimes:
        for meth in methods:
            # the flow is only the right tool for post_nonlinear; lsnm is the general engine
            acc, dt, fps = benchmark(regime, meth, args.P, args.n, args.h, args.epochs)
            print(f"  {regime:>15} {meth:>7} {acc:>11.2f} {dt:>8.2f} {fps:>10.0f}")
    print("=" * 96)
    print("  orient-acc: fraction of pairs oriented correctly (1.00=perfect, ~0.50=chance).")
    print("  THE WIN is multiplicative/lsnm (additive ANM is ~chance there); the FRONTIER is")
    print("  post_nonlinear/flow (a location-scale lsnm cannot, a flow should). do() not run here --")
    print("  this harness is the OBSERVATIONAL neural engines at scale (the GPU-bound part).")

    if args.scaling:
        # activation memory ~ 2P * n * h * 4 bytes PER tensor; keep n modest so big P fits 40GB.
        ns = min(args.n, 1000)
        print(f"\n  GPU THROUGHPUT SCALING (lsnm, multiplicative, n={ns}): does fits/sec hold as P grows?")
        print(f"  {'P':>7} {'dir-fits':>9} {'orient-acc':>11} {'wall_s':>8} {'fits/sec':>10} {'~act_GB':>8}")
        for P in (128, 512, 1024, 2048):
            acc, dt, fps = benchmark("multiplicative", "lsnm", P, ns, args.h, args.epochs)
            act_gb = 2 * P * ns * args.h * 4 / 1e9
            print(f"  {P:>7} {2*P:>9} {acc:>11.2f} {dt:>8.2f} {fps:>10.0f} {act_gb:>8.2f}")
        print("  (On an A100 fits/sec should RISE then plateau as P grows -- fixed launch overhead is")
        print("   amortized over more parallel nets. That plateau is parallelism a CPU cannot match.)")
        print("  (If you OOM at large P: lower --n, or we add task-chunking -- ping me with the error.)")


if __name__ == "__main__":
    main()
