"""
Stronger curiosity baselines (commander Order 2) -- the real trial for the EIG claim.

  Ensemble (Pathak disagreement)  -- K bootstrapped mean-models; act where they DISAGREE.
  Heteroscedastic disagreement    -- disagreement DOWN-WEIGHTED by aleatoric sigma^2 (the
                                     'fixed' version that should ignore action-noise).
  LPTracker (learning progress)   -- act where recent prediction ERROR is DROPPING; a
                                     noise-controlling knob yields no progress, so it should
                                     be ignored (Oudeyer).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .model import BayesianDynamicsModel


class Ensemble:
    """K bootstrapped homoscedastic mean-models; disagreement = per-sensor variance of their
    next-state mean predictions (Pathak). Lured by irreducible noise at finite data."""

    def __init__(self, d, actuators, hidden=(), interaction_pairs=(), K=4, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.models = [BayesianDynamicsModel(d, actuators, hidden=hidden,
                                             interaction_pairs=interaction_pairs,
                                             rng=np.random.default_rng(int(rng.integers(1 << 31))))
                       for _ in range(K)]
        self.K = K
        self.rng = rng
        self.sensors = self.models[0].sensors

    def update(self, xc, xn):
        for m in self.models:
            if self.rng.random() < 0.6:                  # bootstrap mask -> real disagreement
                m.update(xc, xn)

    def disagreement(self, xc):
        """Per-sensor dict of Var_k(mu_k,i(xc))."""
        preds = np.array([m.predict_next(xc)[0] for m in self.models])   # (K, d)
        return {i: float(np.var(preds[:, i])) for i in self.sensors}


class LPTracker:
    """Per-actuator empirical learning progress: how much the model's prediction error has
    DROPPED recently when that actuator was driven. Irreducible-noise actions show no drop."""

    def __init__(self, actuators, window=12):
        self.acts = list(actuators)
        self.err = {a: deque(maxlen=2 * window) for a in self.acts}
        self.window = window

    def observe(self, xc, step_err):
        for a in self.acts:
            if abs(float(xc[a])) > 1e-6:                  # this actuator was driven this step
                self.err[a].append(step_err)

    def lp(self, a):
        e = self.err[a]
        if len(e) < self.window + 2:
            return 1.0                                    # unexplored -> optimistic
        half = len(e) // 2
        old = np.mean(list(e)[:half]); recent = np.mean(list(e)[half:])
        return max(old - recent, 0.0)                     # positive = error dropping = progress

    def score(self, macro):
        return sum(self.lp(a) * k for cmd, k in macro for a in cmd if abs(cmd[a]) > 1e-6)


__all__ = ["Ensemble", "LPTracker"]
