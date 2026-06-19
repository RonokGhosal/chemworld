"""
Observation-gating: where the noisy-TV trap finally bites.

In the fully-observed worlds, naive surprise ties curiosity, because the noise
source can't be acted on -- so chasing surprise costs nothing. The trap needs the
action to gate *what you observe*. Here it does: every step the agent forces the
knob (to excite the informative variable) but RECORDS only ONE channel, and only
that channel updates its belief. The world offers two channels:

    chain1  -- informative: chain1 := 0.3·chain1 + 0.8·a0 + small noise (a LAW)
    static  -- a parent-free channel of pure large noise               (NO law)

Two objectives, differing only in what they call "interesting":

    EIG (curiosity)  -- watch the channel with the most expected information gain
                        about its PARAMETERS. Static has a single parameter (its
                        mean); two looks pin it, after which its EIG ≈ 0, so the
                        agent watches chain1 and learns its law.
    surprise (naive) -- watch the least predictable channel. Static is endlessly
                        surprising (large irreducible noise), so the agent stares
                        at it forever and never learns chain1. The noisy-TV trap.

Each channel is modelled as exactly what it is -- chain1 by a Bayesian linear
regression on its true regressors, static by a Bayesian mean -- so the contrast is
about the OBJECTIVE, not the model class. This is the dynamical, constructor-
framework restatement of the minimal point in the repo-root ``discovery.py``.
"""
from __future__ import annotations

import numpy as np

from .world import DynamicalCausalWorld

CHAIN1, STATIC, A0 = 2, 5, 0


class BayesLR:
    """Bayesian linear regression with online noise estimate; gives EIG & surprise."""

    def __init__(self, p, alpha=1.0, sigma0=1.0):
        self.p = p
        self.S = np.zeros((p, p))
        self.r = np.zeros(p)
        self.alpha = alpha
        self.sigma2 = sigma0 ** 2
        self.n = 0
        self.sse = 0.0

    def _post(self):
        Cov = np.linalg.inv(self.alpha * np.eye(self.p) + self.S / self.sigma2)
        return Cov @ (self.r / self.sigma2), Cov

    def eig(self, phi):
        _, Cov = self._post()
        return 0.5 * np.log1p(max(phi @ Cov @ phi, 0.0) / self.sigma2)

    def surprise(self, phi):
        _, Cov = self._post()
        return 0.5 * np.log(2 * np.pi * np.e * max(self.sigma2 + phi @ Cov @ phi, 1e-12))

    def update(self, phi, y):
        if self.n >= self.p + 2:
            m, _ = self._post()
            self.sse += (y - m @ phi) ** 2
            self.sigma2 = max(self.sse / max(self.n, 1), 1e-6)
        self.S += np.outer(phi, phi)
        self.r += phi * y
        self.n += 1

    def mean(self):
        return self._post()[0]


class BayesMean:
    """A parent-free channel: one parameter (the mean) over a noisy signal."""

    def __init__(self, prior_var=100.0, sigma0=1.0):
        self.prior_prec = 1.0 / prior_var
        self.sigma2 = sigma0 ** 2
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def _mean_prec(self):
        return self.prior_prec + self.n / self.sigma2

    def eig(self):
        # info about the mean from one more observation -> 0 as n grows
        return 0.5 * np.log1p((1.0 / self.sigma2) / self._mean_prec())

    def surprise(self):
        return 0.5 * np.log(2 * np.pi * np.e * max(self.sigma2 + 1.0 / self._mean_prec(), 1e-12))

    def update(self, y):
        self.n += 1
        self.sum += y
        self.sumsq += y * y
        if self.n >= 3:
            mean = self.sum / self.n
            self.sigma2 = max(self.sumsq / self.n - mean ** 2, 1e-6)


def run_gated(objective: str, budget: int = 30, seed: int = 0,
              static_noise: float = 4.0, chain1_noise: float = 0.4):
    rng = np.random.default_rng(seed)
    world = DynamicalCausalWorld.default(rng)
    world.noise_std = world.noise_std.copy()
    world.noise_std[CHAIN1] = chain1_noise
    world.noise_std[STATIC] = static_noise

    chain1 = BayesLR(p=3)             # regress chain1_next on [a0, chain1, 1]
    static = BayesMean()
    looks = {CHAIN1: 0, STATIC: 0}

    x = world.reset()
    for _ in range(budget):
        a0 = float(rng.choice((-2.0, 0.0, 2.0)))
        phi = np.array([a0, x[CHAIN1], 1.0])
        x_next = world.step({A0: a0})
        if objective == "eig":
            watch = CHAIN1 if chain1.eig(phi) >= static.eig() else STATIC
        elif objective == "surprise":
            watch = CHAIN1 if chain1.surprise(phi) >= static.surprise() else STATIC
        else:
            raise ValueError(objective)
        looks[watch] += 1
        if watch == CHAIN1:
            chain1.update(phi, x_next[CHAIN1])
        else:
            static.update(x_next[STATIC])
        x = x_next

    w_a0 = float(chain1.mean()[0]) if chain1.n else 0.0    # true a0 weight is 0.80
    err = abs(w_a0 - 0.80)
    return {"looks_static": looks[STATIC], "looks_chain1": looks[CHAIN1],
            "chain1_a0_weight": w_a0, "weight_err": err,
            "learned_chain1": err < 0.15 and chain1.n > 0}


def compare(budget: int = 30, seeds=range(12)):
    out = {}
    for obj in ("eig", "surprise"):
        rs = [run_gated(obj, budget, s) for s in seeds]
        out[obj] = {
            "frac_static": float(np.mean([r["looks_static"] / budget for r in rs])),
            "weight_err": float(np.mean([r["weight_err"] for r in rs])),
            "learned": float(np.mean([r["learned_chain1"] for r in rs])),
        }
    return out


if __name__ == "__main__":
    print("observation-gating (noisy-TV): EIG vs naive surprise\n" + "-" * 64)
    for obj, m in compare().items():
        print(f"  {obj:9s}  watched static {100*m['frac_static']:4.0f}% of looks   "
              f"chain1 law err={m['weight_err']:.2f}   learned chain1: {100*m['learned']:3.0f}%")
