"""
RUNG 6A -- the THESIS-BREAKER: where does intervention STOP being necessary?

Rung 5 found intervention NECESSARY to orient an instantaneous chain a->b->c, but only in the
LINEAR-GAUSSIAN regime (LiNGAM rescued the non-Gaussian case). Nonlinearity cuts the other way: a
nonlinear additive-noise model is observationally IDENTIFIABLE (Hoyer 2009) -- even under Gaussian
noise -- so a proper observational method (ANM/RESIT: regress child on parent, test residual
independence) orients the chain with NO intervention. This rung is built to find that boundary and to
report intervention LOSING its necessity, honestly.

We sweep the SAME chain a->b->c under four links (linear, tanh, quad, sin), Gaussian noise throughout,
and orient with three methods:
  * ANM   (observation, nonlinear-aware): RBF-ridge regression + HSIC residual-independence.
  * LiNGAM(observation, linear-aware)  : the Rung-5 tool -- shown to be the WRONG tool here.
  * do()  (intervention)               : orients regardless.
Honest expectation: LINEAR-Gaussian -> ANM & LiNGAM at chance, do necessary (Rung 5). NONLINEAR-
Gaussian -> ANM ~1.0, do redundant -> intervention NOT necessary. The boundary is NONLINEARITY: it,
like non-Gaussianity, breaks the linear-Gaussian symmetry that made acting the only option. So
intervention is strictly necessary for orientation only in the doubly-degenerate LINEAR-AND-GAUSSIAN
case. (Next, 6B adds a hidden confounder, which breaks ANM's causal-sufficiency assumption -- and the
necessity of intervention should RETURN.)
"""
from __future__ import annotations

import numpy as np


class NLChainWorld:
    """Instantaneous additive-noise chain 0->1->...->depth, labels permuted. x_{i+1}=f(x_i)+e,
    x_0 exogenous; Gaussian noise. link in {linear,tanh,quad,sin}. do() clamps a label."""

    def __init__(self, depth, rng, link="tanh", noise=0.5, w=1.0, ew=2.0):
        self.depth = depth; self.rng = rng; self.link = link
        self.noise = noise; self.w = w; self.ew = ew; self.d = depth + 1
        perm = rng.permutation(self.d)
        self.lab = {i: int(perm[i]) for i in range(self.d)}
        self.true_dir = {(self.lab[i], self.lab[i + 1]) for i in range(depth)}

    def _f(self, x):
        if self.link == "linear":
            return self.w * x
        if self.link == "tanh":
            return np.tanh(self.ew * x)
        if self.link == "quad":
            return self.w * x + 0.5 * x ** 2
        if self.link == "sin":
            return np.sin(self.ew * x)
        raise ValueError(self.link)

    def sample(self, n, do=None):
        do_i = {}
        if do:
            inv = {v: k for k, v in self.lab.items()}
            do_i = {inv[k]: v for k, v in do.items()}
        X = np.zeros((n, self.d))
        for i in range(self.d):
            if i in do_i:
                X[:, i] = do_i[i]
            elif i == 0:
                X[:, i] = self.rng.normal(0, 1, n)
            else:
                X[:, i] = self._f(X[:, i - 1]) + self.rng.normal(0, self.noise, n)
        out = np.empty((n, self.d))
        for i in range(self.d):
            out[:, self.lab[i]] = X[:, i]
        return out


# ----------------------------- observational orientation -----------------------------
def _rbf_ridge(x, y, n_c=25, lam=1e-2):
    """Flexible nonlinear 1-D regression: RBF features + ridge. Returns fitted y_hat."""
    xs = (x - x.mean()) / (x.std() + 1e-9)
    c = np.linspace(xs.min(), xs.max(), n_c)
    F = np.exp(-(xs[:, None] - c[None, :]) ** 2)
    F = np.column_stack([F, np.ones(len(xs))])
    w = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ y)
    return F @ w


def _hsic(x, y, m=400, seed=0):
    """Biased HSIC statistic (Gaussian kernels, median bandwidth). Larger => more dependent."""
    idx = np.random.default_rng(seed).choice(len(x), min(m, len(x)), replace=False)
    x = x[idx].reshape(-1, 1).astype(float); y = y[idx].reshape(-1, 1).astype(float); n = len(x)

    def K(a):
        sq = np.sum((a[:, None] - a[None, :]) ** 2, 2)
        s = np.sqrt(np.median(sq[sq > 0]) / 2) + 1e-9
        return np.exp(-sq / (2 * s * s))
    H = np.eye(n) - 1.0 / n
    return float(np.trace(K(x) @ H @ K(y) @ H) / (n - 1) ** 2)


def anm_dir(x, y):
    """ANM/RESIT: +1 => x->y if residual of y~f(x) is more independent of x than the reverse."""
    r_yx = y - _rbf_ridge(x, y)
    r_xy = x - _rbf_ridge(y, x)
    return 1 if _hsic(x, r_yx) < _hsic(y, r_xy) else -1


def lingam_dir(x, y):
    """Hyvarinen-Smith pairwise linear measure (the Rung-5 tool; the WRONG tool under nonlinearity)."""
    x = (x - x.mean()) / (x.std() + 1e-9); y = (y - y.mean()) / (y.std() + 1e-9)
    rho = float(np.mean(x * y))
    M = rho * (np.mean(x * np.tanh(y)) - np.mean(np.tanh(x) * y))
    return 1 if M > 0 else -1


def do_dir(world, la, lb, delta=1.0, n=300, thresh=0.1):
    """+1 => la->lb if do(la) moves lb and do(lb) does not move la."""
    eab = (world.sample(n, {la: delta}).mean(0)[lb] - world.sample(n, {la: -delta}).mean(0)[lb]) / (2 * delta)
    eba = (world.sample(n, {lb: delta}).mean(0)[la] - world.sample(n, {lb: -delta}).mean(0)[la]) / (2 * delta)
    if abs(eab) > thresh and abs(eba) <= thresh:
        return 1
    if abs(eba) > thresh and abs(eab) <= thresh:
        return -1
    return 0


def _chain_edges(world, X):
    """Per chain edge: (anm_correct, lingam_correct, do_correct) as 0/1 (method +1 => first->second)."""
    out = []
    for i in range(world.depth):
        la, lb = world.lab[i], world.lab[i + 1]                # true la->lb
        out.append((1.0 if anm_dir(X[:, la], X[:, lb]) == 1 else 0.0,
                    1.0 if lingam_dir(X[:, la], X[:, lb]) == 1 else 0.0,
                    1.0 if do_dir(world, la, lb) == 1 else 0.0))
    return out


def _verdict(a, do_minus_anm):
    if a < 0.7:
        return "do NECESSARY (obs ~chance)"
    if a < 0.98 or do_minus_anm > 0.02:
        return "do BETTER (obs mostly orients; do strictly more reliable)"
    return "do REDUNDANT (obs fully orients)"


def main(seeds=range(15), n_obs=4000, depth=2):
    print("=" * 100)
    print(f"RUNG 6A -- where intervention STOPS being necessary (nonlinear additive, Gaussian noise) "
          f"({len(list(seeds))} seeds)")
    print("=" * 100)
    print(f"  chain a->b->c orientation fraction (1.00 = correct, ~0.50 = chance); "
          f"'do>ANM' = edges where do() right & ANM wrong")
    print(f"  {'link':>8} {'ANM (obs)':>11} {'LiNGAM':>8} {'do (interv)':>12} {'do>ANM':>8}   verdict")
    rows = {}
    for link in ("linear", "tanh", "quad", "sin"):
        edges = []
        for s in seeds:
            w = NLChainWorld(depth, np.random.default_rng(s), link=link)
            edges += _chain_edges(w, w.sample(n_obs))
        E = np.array(edges)                                     # (n_edges, 3): anm, lingam, do
        a, l, d = E[:, 0].mean(), E[:, 1].mean(), E[:, 2].mean()
        do_gt_anm = float(np.mean((E[:, 2] == 1) & (E[:, 0] == 0)))   # do right, ANM wrong
        anm_gt_do = float(np.mean((E[:, 0] == 1) & (E[:, 2] == 0)))   # ANM right, do wrong
        rows[link] = (a, l, d, do_gt_anm, anm_gt_do)
        print(f"  {link:>8} {a:>11.2f} {l:>8.2f} {d:>12.2f} {do_gt_anm:>8.2f}   {_verdict(a, do_gt_anm)}")
    print("=" * 100)
    lin = rows["linear"]
    clean = [k for k in ("tanh", "quad", "sin") if rows[k][0] >= 0.98 and rows[k][3] <= 0.02]
    print(f"  BOUNDARY: LINEAR-Gaussian -> observation at chance (ANM {lin[0]:.2f}, LiNGAM {lin[1]:.2f}),")
    print(f"  do() NECESSARY ({lin[2]:.2f}) -- Rung 5. Add NONLINEARITY and a nonlinear-aware")
    print(f"  observational method (ANM) orients the chain WITHOUT acting:")
    print(f"   * {', '.join(clean)}: ANM fully orients (>=0.98, 0 do>ANM edges) -> do() REDUNDANT, so")
    print(f"     intervention is NOT necessary in these additive-Gaussian nonlinear chains.")
    print(f"   * tanh: ANM {rows['tanh'][0]:.2f} (saturation flattens the child, costing the HSIC test")
    print(f"     power; asymptotes ~0.975<1.0) and do>ANM={rows['tanh'][3]:.2f} -- here do() is strictly")
    print(f"     MORE reliable, so we do NOT claim 'not necessary', only 'observation mostly suffices'.")
    print(f"  Honest read: intervention is strictly necessary for orientation only in the doubly-")
    print(f"  degenerate LINEAR-AND-GAUSSIAN case; nonlinearity (like non-Gaussianity) hands orientation")
    print(f"  back to observation -- cleanly for quad/sin, partially for the saturating tanh. (LiNGAM,")
    print(f"  the linear tool, gets sin {rows['sin'][1]:.2f} -- the WRONG baseline would fake a do win.)")
    print(f"  SCOPE: additive-noise, acyclic. Off it (post-nonlinear / multiplicative noise) ANM")
    print(f"  collapses and this boundary does NOT hold -- not claimed here.")
    print("=" * 100)
    return rows


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(seeds=range(ns))
