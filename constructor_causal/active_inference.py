"""
Active inference, with the reward term deleted.

In active inference an agent picks actions to minimise EXPECTED FREE ENERGY,

        G(pi)  =  -  (epistemic value)  -  (pragmatic value).
                       \\__ info gain __/      \\__ reward / preferences __/

This project's thesis is that the *pragmatic* term is optional. Strike it and the
agent is left with pure epistemic drive: act so as to learn the most about the
world's causal mechanism. No reward, no goals, no preferred observations -- only
the imperative to resolve uncertainty about the parameters of the causal model.

``EpistemicExperimenter`` picks, at each step, the actuator command whose
short model-rollout maximises summed expected information gain (model.py). Because
EIG measures shrinkage of uncertainty about the causal *weights* -- not raw
surprise -- it walks straight past the noisy ``static`` sensor and instead drives
actuators through the regions that most sharply disambiguate cause from decoy.

Two foils are provided to show the choice of objective is what matters:
  * ``RandomExperimenter``        -- the sanity floor (PLAN.md: "beat random").
  * ``NaiveSurpriseExperimenter`` -- maximises predictive entropy; the noisy-TV
                                     trap, kept for contrast (cf. discovery.py).
"""
from __future__ import annotations

import itertools

import numpy as np


def rollout_model(model, x0, command, horizon):
    """Deterministically roll the *learned* model forward holding ``command``.
    Returns the list of clamped states fed into each predicted transition."""
    x = np.asarray(x0, float).copy()
    clamped_states = []
    for _ in range(horizon):
        xc = x.copy()
        for j, v in command.items():
            xc[j] = v
        clamped_states.append(xc)
        mu, _ = model.predict_next(xc, command)
        x = mu
    return clamped_states


class EpistemicExperimenter:
    """Reward-free action selection by expected information gain."""

    def __init__(self, model, actuators, setpoints=(-2.0, 0.0, 2.0),
                 horizon: int = 2, epsilon: float = 0.1, max_candidates: int = 64,
                 rng=None):
        self.model = model
        self.actuators = tuple(actuators)
        self.setpoints = tuple(setpoints)
        self.horizon = int(horizon)
        self.epsilon = float(epsilon)
        self.max_candidates = int(max_candidates)
        self.rng = rng if rng is not None else np.random.default_rng()
        # candidate commands: every joint setting of the actuators -- but with many
        # knobs that grid is exponential, so above max_candidates we SAMPLE joint
        # settings each step instead of enumerating (keeps EIG usable on wide worlds).
        full = len(self.setpoints) ** len(self.actuators)
        self.enumerate = full <= self.max_candidates
        self.candidates = ([dict(zip(self.actuators, combo))
                            for combo in itertools.product(self.setpoints,
                                                           repeat=len(self.actuators))]
                           if self.enumerate else None)

    def _sample_candidates(self):
        return [{j: float(self.rng.choice(self.setpoints)) for j in self.actuators}
                for _ in range(self.max_candidates)]

    def score(self, state, command) -> float:
        """EIG over an H-step learned-model rollout under this command, summed by the
        CHAIN RULE (seq_info_gain) so overlapping information along the held-command
        rollout is not double-counted."""
        phis = [self.model._phi(xc)
                for xc in rollout_model(self.model, state, command, self.horizon)]
        return self.model.seq_info_gain(phis)

    def choose(self, state) -> dict:
        cands = self.candidates if self.enumerate else self._sample_candidates()
        if self.rng.random() < self.epsilon:                 # keep coverage honest
            return dict(cands[int(self.rng.integers(len(cands)))])
        scores = np.array([self.score(state, c) for c in cands])
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        return dict(cands[int(self.rng.choice(best))])


class CertifyingExperimenter:
    """Reward-free action selection with an EXTENDED epistemic objective: expected
    information gain about the MODEL (exactly EpistemicExperimenter) PLUS information gain
    about POSSIBILITY -- act so the agent's UNDECIDED skills become certifiable.

        score(c) = coverage_gain(c)  +  eig_weight * EIG(c)

    ``targets`` is a list of (command, deficit): the holds whose reset-free certificate
    (certify.certify_library) came back UNDECIDED *for lack of on-policy coverage*, each
    weighted by how far below the coverage threshold it is. ``coverage_gain(c)`` sums the
    deficit of every target ``c`` satisfies (within ``action_tol``), so a candidate that
    practises the most-under-covered skill scores highest. Because the integer deficit
    dwarfs the EIG bits, OPEN CERTIFICATES are resolved first and the EIG term only breaks
    ties (e.g. settles the OTHER knobs) -- and with ``targets`` empty the whole thing
    reduces EXACTLY to pure-EIG EpistemicExperimenter, so reward-free initial learning is
    untouched. A target's deficit persists until its skill is covered, so SUSTAINED holds
    EMERGE on their own (no hard-coded commit length) -- which is precisely what slow
    downstream variables need to enter their effect region. This operationalises Constructor
    Theory's drive: not 'reduce my parameter uncertainty' but 'resolve what I can DO'."""

    def __init__(self, model, actuators, setpoints=(-2.0, 0.0, 2.0), horizon: int = 2,
                 eig_weight: float = 1.0, action_tol: float = 0.4, epsilon: float = 0.05,
                 max_candidates: int = 64, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.eig = EpistemicExperimenter(model, actuators, setpoints=setpoints,
                                         horizon=horizon, epsilon=0.0,
                                         max_candidates=max_candidates, rng=self.rng)
        self.actuators = tuple(actuators)
        self.action_tol = float(action_tol)
        self.eig_weight = float(eig_weight)
        self.epsilon = float(epsilon)
        self.targets: list = []                  # [(command_dict, deficit), ...]

    @property
    def model(self):
        return self.eig.model

    @model.setter
    def model(self, m):
        self.eig.model = m

    def coverage_gain(self, command) -> float:
        g = 0.0
        for (tcmd, deficit) in self.targets:
            if all(j in command and abs(command[j] - v) <= self.action_tol
                   for j, v in tcmd.items()):
                g += float(deficit)
        return g

    def choose(self, state) -> dict:
        cands = (self.eig.candidates if self.eig.enumerate
                 else self.eig._sample_candidates())
        if self.rng.random() < self.epsilon:                 # keep coverage honest
            return dict(cands[int(self.rng.integers(len(cands)))])
        scores = np.array([self.coverage_gain(c) + self.eig_weight * self.eig.score(state, c)
                           for c in cands])
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        return dict(cands[int(self.rng.choice(best))])


class RandomExperimenter:
    """Uniform random commands -- the sanity floor."""

    def __init__(self, model, actuators, setpoints=(-2.0, 0.0, 2.0), rng=None):
        self.actuators = tuple(actuators)
        self.setpoints = tuple(setpoints)
        self.rng = rng if rng is not None else np.random.default_rng()

    def choose(self, state) -> dict:
        return {j: float(self.rng.choice(self.setpoints)) for j in self.actuators}


class PassiveExperimenter:
    """Never intervenes -- just watches the world evolve. The foil for the hidden-
    confounder test: passive observation cannot tell a shared hidden cause from a
    direct edge, so it infers spurious links that intervention would reject."""

    def __init__(self, *args, **kwargs):
        pass

    def choose(self, state) -> dict:
        return {}


class NaiveSurpriseExperimenter:
    """Maximise predictive entropy of the next observation -- the noisy-TV trap.
    Kept to demonstrate WHY the objective, not the machinery, is what matters."""

    def __init__(self, model, actuators, setpoints=(-2.0, 0.0, 2.0),
                 epsilon: float = 0.1, rng=None):
        self.model = model
        self.actuators = tuple(actuators)
        self.setpoints = tuple(setpoints)
        self.epsilon = float(epsilon)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.candidates = [dict(zip(self.actuators, combo))
                           for combo in itertools.product(self.setpoints,
                                                          repeat=len(self.actuators))]

    def choose(self, state) -> dict:
        if self.rng.random() < self.epsilon:
            return dict(self.candidates[int(self.rng.integers(len(self.candidates)))])
        scores = np.array([self.model.raw_surprise(state, c) for c in self.candidates])
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        return dict(self.candidates[int(self.rng.choice(best))])


__all__ = ["EpistemicExperimenter", "CertifyingExperimenter", "RandomExperimenter",
           "NaiveSurpriseExperimenter", "PassiveExperimenter", "rollout_model"]
