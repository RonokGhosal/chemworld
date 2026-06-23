"""
HETEROSCEDASTIC noise head + noise-aware action scorers.

Orders 4 & 5: only a model of sigma^2(x,a) makes the action noisy-TV visible. Then the
objectives split into two families on the noise knob:

  AVOID it (target REDUCIBLE uncertainty):
    eig_hetero   -- sigma^2-WEIGHTED parameter info gain: a sample under high predicted
                    noise carries little info, so driving a_noise scores low.
  LURED by it (target OBSERVATION unpredictability / irreducible noise):
    surprise_hetero -- heteroscedastic predictive ENTROPY (max-entropy / surprise)
    pred_error      -- expected predictive ERROR magnitude (ICM / prediction-first)

The VarianceHead is a per-sensor sigma^2_i(x) = floor + sum_a w_{i,a}*relu(x_a), fit by
least squares on the squared residuals of the mean model (refit from the buffer).
"""
from __future__ import annotations

import numpy as np


class VarianceHead:
    def __init__(self, model, actuators, floor=0.05):
        self.model = model
        self.acts = list(actuators)
        self.floor = float(floor)
        self.coef = {i: np.zeros(len(self.acts) + 1) for i in model.sensors}
        for i in model.sensors:
            self.coef[i][0] = floor

    def _feat(self, xc):
        return np.concatenate([[1.0], [max(float(xc[a]), 0.0) for a in self.acts]])

    def fit(self, buf):
        if len(buf) < 20:
            return
        F = np.array([self._feat(xc) for xc, _ in buf])
        for i in self.model.sensors:
            mean = self.model._mean(i)
            r2 = np.array([(xn[i] - float(mean @ self.model._phi(xc))) ** 2 for xc, xn in buf])
            self.coef[i], *_ = np.linalg.lstsq(F, r2, rcond=None)

    def sigma2(self, xc, i):
        return max(float(self._feat(xc) @ self.coef[i]), self.floor)


def eig_hetero(model, head, states):
    """sigma^2-weighted info gain (REDUCIBLE): down-weights noise-swamped samples."""
    Pinv = model._ensure_pinv()
    g = 0.0
    for xc, _ in states:
        phi = model._phi(xc)
        q = float(phi @ Pinv @ phi)
        for i in model.sensors:
            g += 0.5 * np.log1p(max(q / head.sigma2(xc, i), 0.0))
    return g


def surprise_hetero(model, head, states):
    """heteroscedastic predictive ENTROPY (LURED by noise-generating actions)."""
    g = 0.0
    for xc, _ in states:
        for i in model.sensors:
            g += 0.5 * np.log(2 * np.pi * np.e * head.sigma2(xc, i))
    return g


def pred_error(model, head, states):
    """expected predictive ERROR magnitude (ICM / prediction-first; LURED)."""
    g = 0.0
    for xc, _ in states:
        for i in model.sensors:
            g += np.sqrt(head.sigma2(xc, i))
    return float(g)


HETERO_SCORERS = {"eig_hetero": eig_hetero, "surprise_hetero": surprise_hetero,
                  "pred_error": pred_error}

__all__ = ["VarianceHead", "eig_hetero", "surprise_hetero", "pred_error", "HETERO_SCORERS"]
