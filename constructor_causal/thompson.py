"""Thompson-sampling experimenter -- active_core's acquisition, brought into constructor_causal.

Drop-in for EpistemicExperimenter (same ``.choose(state) -> command`` interface), but instead of
an epsilon-greedy argmax over the MEAN model's info-gain rollout, it draws ONE structure from the
posterior each step and scores info gain along a rollout under that draw. Exploration is then
intrinsic to the sampling -- there is NO epsilon and NO exploration constant -- and under-probed
edges (which carry posterior uncertainty) periodically light up, so the agent does not tunnel-vision
onto a few knobs. This is the fix verified in pi_ct_nsdm/active_core.py, expressed against this
package's Bayesian dynamics model.
"""
from __future__ import annotations

import itertools

import numpy as np


class ThompsonExperimenter:
    def __init__(self, model, actuators, setpoints=(-2.0, 0.0, 2.0), horizon: int = 2,
                 max_candidates: int = 64, rng=None):
        self.model = model
        self.actuators = tuple(actuators)
        self.setpoints = tuple(setpoints)
        self.horizon = int(horizon)
        self.max_candidates = int(max_candidates)
        self.rng = rng if rng is not None else np.random.default_rng()
        full = len(self.setpoints) ** len(self.actuators)
        self.enumerate = full <= self.max_candidates
        self.candidates = ([dict(zip(self.actuators, combo))
                            for combo in itertools.product(self.setpoints,
                                                           repeat=len(self.actuators))]
                           if self.enumerate else None)

    def _sample_candidates(self):
        return [{j: float(self.rng.choice(self.setpoints)) for j in self.actuators}
                for _ in range(self.max_candidates)]

    def _sample_weights(self):
        """One posterior draw of every sensor's weight vector (the structure sample).

        The NIG weight posterior is multivariate STUDENT-t (dof nu = 2 a_N, scale = Cov),
        not Gaussian. Draw it exactly:  t = mean + sqrt(nu / chi2_nu) * N(0, Cov).
        Sampling a Gaussian with the t-scale would UNDER-disperse exploration (by
        ~sqrt((nu-2)/nu)) precisely at the small n where Thompson exploration matters."""
        W = {}
        for i in self.model.sensors:
            mean, Cov = self.model._posterior(i)
            # floor the sampling dof: early on nu = 2 a0 + n is ~0 (a t with ~0 dof is
            # a meaningless point mass and chisquare(~0) underflows to 0). nu>=2 gives a
            # sane heavy-tailed draw; chi guarded against residual underflow.
            nu = max(float(self.model._dof[i]), 2.0)
            g = self.rng.multivariate_normal(np.zeros_like(mean), Cov)
            chi = float(self.rng.chisquare(nu))
            s = float(np.sqrt(nu / chi)) if chi > 1e-9 else 1.0
            W[i] = mean + s * g
        return W

    def _rollout_sampled(self, state, command, W):
        """Roll the SAMPLED model forward holding command; return the clamped states."""
        x = np.asarray(state, float).copy()
        states = []
        for _ in range(self.horizon):
            xc = x.copy()
            for j, v in command.items():
                xc[j] = v
            states.append(xc)
            phi = self.model._phi(xc)
            xn = xc.copy()
            for i in self.model.sensors:
                xn[i] = float(W[i] @ phi)
            x = xn
        return states

    def score(self, state, command, W) -> float:
        # chain-rule sequential EIG over the sampled rollout (no double-counting of
        # overlapping information across correlated held-command steps).
        phis = [self.model._phi(xc) for xc in self._rollout_sampled(state, command, W)]
        return self.model.seq_info_gain(phis)

    def choose(self, state) -> dict:
        W = self._sample_weights()
        cands = self.candidates if self.enumerate else self._sample_candidates()
        scores = np.array([self.score(state, c, W) for c in cands])
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        return dict(cands[int(self.rng.choice(best))])


__all__ = ["ThompsonExperimenter"]
