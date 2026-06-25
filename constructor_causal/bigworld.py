"""
BigWorld -- a CONFIGURABLE interventional causal world for the scale-up (the grand version).

Generalizes the 5-variable messy world to n_vars in a random acyclic SCM, so the problem can be
made as big as we want (the thing that finally justifies GPUs). Keeps every property the thesis
needs:
  * a true DAG (known ground truth -> auditable),
  * ACTIONS that intervene on actuator variables (active intervention is possible),
  * a CONFOUNDER (a hidden common cause) so observational correlation != causation -- the place a
    prediction-trained net gets fooled and an intervening agent does not,
  * a NOISE KNOB (an action that inflates a variable's variance only -- the pixel-free noisy-TV).

observe() returns the variable vector (optionally scrambled later, for a harder rung). This is
the world the transformer opponent trains on and the experimenter intervenes in.
"""
from __future__ import annotations

import numpy as np


class BigWorld:
    def __init__(self, n_vars=32, n_act=8, rng=None, nonlinear=True, confounder=True,
                 noise_knob=True, p_edge=0.25, edge_scale=0.8):
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.n = int(n_vars); self.na = int(n_act)
        self.nonlinear = bool(nonlinear)
        # random ACYCLIC SCM: edges only i->j with i<j (topological order = index order)
        mask = np.triu((self.rng.random((self.n, self.n)) < p_edge).astype(float), 1)
        self.W = mask * self.rng.normal(0.0, edge_scale, (self.n, self.n))   # true weighted DAG
        self.act_targets = np.arange(self.na) % self.n          # actuators drive the first na vars
        self.noise_var = self.n - 1                             # noise knob inflates this var
        # confounder: a hidden var C drives two observed vars (spurious correlation in passive data)
        self.confounder = bool(confounder)
        self.conf_targets = self.rng.choice(self.n, 2, replace=False) if confounder else ()
        self.c = 0.0
        self.z = np.zeros(self.n)

    def reset(self):
        self.z = np.zeros(self.n); self.c = 0.0
        return self.observe()

    def step(self, action):
        """action: length-na vector. Last entry doubles as the noise knob."""
        a = np.asarray(action, float); z = self.z; zn = np.zeros(self.n)
        drive = self.W.T @ z                                    # parents' weighted sum (acyclic)
        if self.nonlinear:
            drive = np.tanh(drive)
        zn = 0.3 * z + drive
        for k, t in enumerate(self.act_targets):                # actions drive actuator vars
            zn[t] += 0.9 * a[k % self.na]
        if self.confounder:                                     # hidden common cause
            self.c = 0.5 * self.c + self.rng.normal(0, 1.0)
            for t in self.conf_targets:
                zn[t] += 0.7 * self.c
        knob = max(float(a[-1]), 0.0) if self.na else 0.0
        zn[self.noise_var] += self.rng.normal(0.0, 0.1 + 3.0 * knob)   # noise knob (variance only)
        zn += self.rng.normal(0.0, 0.02, self.n)                # small process noise
        self.z = zn
        return self.observe()

    def observe(self):
        return self.z.copy()

    def true_state(self):
        return self.z.copy()

    def clone(self, rng=None):
        c = BigWorld(self.n, self.na, rng, self.nonlinear, self.confounder,
                     True, 0.0, 0.0)
        c.W = self.W.copy(); c.act_targets = self.act_targets.copy()
        c.noise_var = self.noise_var; c.conf_targets = self.conf_targets
        c.reset(); return c


def collect_trajectories(world, n_traj, T, rng, act_scale=2.0):
    """PASSIVE data (random actions) -- the regime a prediction net is trained on. Returns
    states (n_traj, T, n_vars) and actions (n_traj, T, n_act), float32."""
    S, A = [], []
    for _ in range(n_traj):
        world.reset(); s, a = [], []
        for _ in range(T):
            act = rng.uniform(-act_scale, act_scale, world.na).astype(np.float32)
            s.append(world.observe()); a.append(act); world.step(act)
        S.append(s); A.append(a)
    return np.asarray(S, np.float32), np.asarray(A, np.float32)
