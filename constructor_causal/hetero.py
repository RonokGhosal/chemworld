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

from .model import BayesianDynamicsModel, _t_crit


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


class BayesianVarianceHead:
    """Per-sensor BAYESIAN regression of squared residuals on psi(x)=[1, relu(a)...], tracking
    the posterior precision-inverse Q_i online (rank-1 Sherman-Morrison). Unlike the batch
    VarianceHead it also scores INFORMATION GAIN about the variance law (var_info), which is
    what lets EIG value learning a_noise->Var(n) ENOUGH, then stop -- resolving the tension
    where mean-only EIG avoids the noise knob and never learns the (real) variance edge."""

    def __init__(self, model, actuators, alpha_v=1e-2, floor=0.05):
        self.model = model
        self.acts = list(actuators)
        self.floor = float(floor)
        self.pv = len(self.acts) + 1
        self.Q = {i: np.eye(self.pv) / alpha_v for i in model.sensors}     # precision inverse
        self.rr = {i: np.zeros(self.pv) for i in model.sensors}

    def psi(self, xc):
        return np.concatenate([[1.0], [max(float(xc[a]), 0.0) for a in self.acts]])

    def coef(self, i):
        return self.Q[i] @ self.rr[i]

    def sigma2(self, xc, i):
        return max(float(self.psi(xc) @ self.coef(i)), self.floor)

    @property
    def coef_dict(self):
        return {i: self.coef(i) for i in self.model.sensors}

    def update(self, xc, xn):
        psi = self.psi(xc)
        for i in self.model.sensors:
            mean = self.model._mean(i)
            r2 = (float(xn[i]) - float(mean @ self.model._phi(xc))) ** 2
            Qp = self.Q[i] @ psi
            self.Q[i] = self.Q[i] - np.outer(Qp, Qp) / (1.0 + float(psi @ Qp))
            self.rr[i] = self.rr[i] + psi * r2

    def var_info(self, xc, i):
        """Expected info gain about sensor i's VARIANCE coefficients from a sample at x."""
        psi = self.psi(xc)
        return 0.5 * np.log1p(max(float(psi @ self.Q[i] @ psi), 0.0))


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


class WeightedHeteroModel:
    """The RIGOROUS heteroscedastic model (Order 3): per-sensor WEIGHTED precision

        Lambda_i = alpha*I + sum_t [1/sigma_i^2(x_t)] phi_t phi_t^T,   m_i = Lambda_i^{-1} r_i,
        r_i = sum_t [1/sigma_i^2(x_t)] phi_t y_{t,i}

    maintained online by a weighted Sherman-Morrison rank-1 update (the homoscedastic
    shared-Gram optimization is given up on purpose -- each sensor now has its own weighted
    precision because the noise weight differs per sensor and per sample). The variance head
    sigma_i^2(x) supplies the weights and co-adapts. EIG per sensor is
    ΔI_i = 0.5 log(1 + phi^T Lambda_i^{-1} phi / sigma_i^2(x)) -- a sample taken under high
    predicted noise carries little information, so EIG avoids noise-generating actions."""

    def __init__(self, d, actuators, hidden=(), interaction_pairs=(), alpha=1e-2,
                 sigma0=1.0, rng=None, bayes_head=False):
        self.base = BayesianDynamicsModel(d, actuators, hidden=hidden,
                                          interaction_pairs=interaction_pairs, alpha=alpha,
                                          sigma0=sigma0, rng=rng)
        self.sensors = self.base.sensors
        self.cols = self.base.cols
        self.actuators = self.base.actuators
        self.p = self.base.p
        self.alpha = float(alpha)
        self.P = {i: np.eye(self.p) / self.alpha for i in self.sensors}   # Lambda_i^{-1}
        self.r = {i: np.zeros(self.p) for i in self.sensors}
        self.n = {i: 0.0 for i in self.sensors}
        self.bayes_head = bool(bayes_head)
        self.head = (BayesianVarianceHead(self, actuators) if bayes_head
                     else VarianceHead(self, actuators, floor=0.05))
        self.buf = []

    def _phi(self, x):
        return self.base._phi(x)

    def _mean(self, i):
        return self.P[i] @ self.r[i]

    def sigma2(self, xc, i):
        return self.head.sigma2(xc, i)

    def update(self, xc, xn):
        phi = self._phi(xc)
        for i in self.sensors:
            w = 1.0 / self.sigma2(xc, i)
            Pphi = self.P[i] @ phi
            self.P[i] = self.P[i] - w * np.outer(Pphi, Pphi) / (1.0 + w * float(phi @ Pphi))
            self.r[i] = self.r[i] + w * phi * float(xn[i])
            self.n[i] += 1.0
        if self.bayes_head:
            self.head.update(xc, xn)            # online Bayesian variance update
        else:
            self.buf.append((xc.copy(), xn.copy()))

    def refit_head(self):
        if not self.bayes_head:
            self.head.fit(self.buf)

    def predict_next(self, xc, command=None):
        x = np.asarray(xc, float).copy()
        if command:
            for j, v in command.items():
                x[j] = v
        phi = self._phi(x)
        mu = x.copy()
        for i in self.sensors:
            mu[i] = float(self._mean(i) @ phi)
        return mu, np.zeros(self.base.d)

    def seq_eig(self, states):
        """Chain-rule sigma^2-WEIGHTED info gain over a macro rollout, per-sensor."""
        Ps = {i: self.P[i].copy() for i in self.sensors}
        g = 0.0
        for xc, _ in states:
            phi = self._phi(xc)
            for i in self.sensors:
                w = 1.0 / self.sigma2(xc, i)
                Pphi = Ps[i] @ phi
                q = w * float(phi @ Pphi)
                if q <= 0:
                    continue
                g += 0.5 * np.log1p(q)
                Ps[i] = Ps[i] - w * np.outer(Pphi, Pphi) / (1.0 + q)
        return g

    def seq_var_eig(self, states):
        """Chain-rule info gain about the VARIANCE law over a macro (needs a Bayesian head).
        HIGH for driving a noise-controlling knob the agent hasn't characterised yet, and
        falls to ~0 once it has -- so the agent probes the noise mechanism enough, then stops."""
        if not self.bayes_head:
            return 0.0
        Qs = {i: self.head.Q[i].copy() for i in self.sensors}
        g = 0.0
        for xc, _ in states:
            psi = self.head.psi(xc)
            for i in self.sensors:
                Qp = Qs[i] @ psi
                q = float(psi @ Qp)
                if q <= 0:
                    continue
                g += 0.5 * np.log1p(q)
                Qs[i] = Qs[i] - np.outer(Qp, Qp) / (1.0 + q)
        return g

    def seq_eig_mv(self, states, var_weight=1.0):
        """Total info gain over BOTH mean and variance parameters: the agent avoids wasting
        budget on irreducible mean-noise (seq_eig) yet still values learning the variance
        mechanism (seq_var_eig) until it is characterised."""
        return self.seq_eig(states) + var_weight * self.seq_var_eig(states)

    def recovered_edges(self, z=3.0, eps=0.05):
        """Weighted per-coefficient t-test on the linear cross-edges (Cov_i = Lambda_i^{-1},
        since the noise is already in the weights)."""
        E = set()
        for i in self.sensors:
            mean = self._mean(i)
            crit = _t_crit(z, max(self.n[i], 2.0))
            for j in self.cols:
                if j == i:
                    continue
                k = self.base._lin_k(j)
                std = float(np.sqrt(max(self.P[i][k, k], 1e-12)))
                if abs(mean[k]) > eps and abs(mean[k]) / std > crit:
                    E.add((j, i))
        return E


__all__ = ["VarianceHead", "eig_hetero", "surprise_hetero", "pred_error", "HETERO_SCORERS",
           "WeightedHeteroModel"]
