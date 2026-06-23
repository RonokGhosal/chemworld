"""
CapabilityWorld -- the empowerment / capability-payoff trial world (commander's order).

Reward-free exploration first; then HELD-OUT goals never seen during exploration test whether
causal understanding turns into CONTROL. One world, every stressor, five goals:

  actuators: a0 (opens gate), a1 (drives the gated chain), aN (controls Var(n), NOT any mean),
             aDec (a DECOY lever: moves `decoy` but nothing causal), aC (moves the HIDDEN
             confounder H -> c1 & c2 together), z0/z1 (INERT -- do nothing).
  sensors:   gate, m1, m2, m3 (deep slow chain m1->m2->m3, gated by gate*a1), decoy
             (driven by m1 AND aDec -> correlates with the chain but isn't on its path),
             c1, c2 (both driven only by hidden H), n (mean fixed; VARIANCE set by aN).

GOALS (target intervals, introduced only at test time):
  deep_chain     -- drive m3 into a band: needs open-gate THEN sustained, decorrelated drive.
  noise_robust   -- drive m1 into a band WHILE keeping Var(n) low: must NOT touch aN.
  decoy_reject   -- drive m2 into a band: the lever is a1 (via gate); aDec only moves the
                    correlated decoy -> a correlational controller is fooled.
  impossible     -- raise c1 WITHOUT moving c2: aC moves H -> both move together, so this is
                    NOT reliably controllable. A good agent ABSTAINS.
  inert_tax      -- any goal, with z0/z1 present: naive controllers waste budget probing them.
"""
from __future__ import annotations

import numpy as np

# actuators
A0, A1, AN, ADEC, AC, Z0, Z1 = range(7)
# sensors
GATE, M1, M2, M3, DECOY, C1, C2, N = range(7, 15)
H = 15                                              # hidden confounder
NAMES = ("a0", "a1", "aN", "aDec", "aC", "z0", "z1",
         "gate", "m1", "m2", "m3", "decoy", "c1", "c2", "n", "H")
ACTUATORS = (A0, A1, AN, ADEC, AC, Z0, Z1)


class CapabilityWorld:
    def __init__(self, rng=None, noise_gain=4.0):
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.noise_gain = float(noise_gain)
        self.d = 16
        self.actuators = ACTUATORS
        self.hidden = (H,)
        self.names = NAMES
        self.A = np.zeros((self.d, self.d))         # placeholder (step overrides)
        self.x = None
        self.command: dict = {}
        self.t = 0

    @property
    def sensors(self):
        return tuple(i for i in range(self.d) if i not in self.actuators)

    @property
    def observed(self):
        return tuple(i for i in range(self.d) if i not in self.hidden)

    def true_edges(self):
        return {(A0, GATE), (GATE, M1), (A1, M1), (M1, M2), (M2, M3),
                (M1, DECOY), (ADEC, DECOY), (AC, H), (H, C1), (H, C2)}

    def reset(self, x0=None):
        self.x = np.zeros(self.d)
        self.command = {}
        self.t = 0
        return self.x.copy()

    def step(self, command=None, noise: bool = True):
        if command:
            for j, v in command.items():
                self.command[j] = float(v)
        xc = self.x.copy()
        for a in self.actuators:
            xc[a] = self.command.get(a, 0.0)
        xn = xc.copy()
        xn[GATE] = 0.20 * xc[GATE] + 0.90 * xc[A0]
        xn[M1] = 0.30 * xc[M1] + 0.60 * xc[GATE] * xc[A1]        # AND-gate
        xn[M2] = 0.60 * xc[M2] + 0.70 * xc[M1]                   # slow
        xn[M3] = 0.70 * xc[M3] + 0.60 * xc[M2]                   # deep
        xn[DECOY] = 0.20 * xc[DECOY] + 0.70 * xc[M1] + 0.80 * xc[ADEC]
        xn[H] = 0.30 * xc[H] + 0.80 * xc[AC]                     # hidden confounder
        xn[C1] = 0.20 * xc[C1] + 0.90 * xc[H]
        xn[C2] = 0.20 * xc[C2] + 0.90 * xc[H]
        xn[N] = 0.30 * xc[N]                                     # mean fixed; variance below
        if noise:
            for i in (GATE, M1, M2, M3, DECOY, C1, C2):
                xn[i] += self.rng.normal(0.0, 0.05)
            xn[H] += self.rng.normal(0.0, 0.30)
            sd_n = 0.10 + self.noise_gain * max(xc[AN], 0.0)    # heteroscedastic noise knob
            xn[N] += self.rng.normal(0.0, sd_n)
        for a in self.actuators:
            xn[a] = self.command.get(a, 0.0)
        self.x = xn
        self.t += 1
        return self.x.copy()

    def clone(self, rng=None):
        return CapabilityWorld(rng if rng is not None else np.random.default_rng(),
                               noise_gain=self.noise_gain)


# ---- held-out goals -----------------------------------------------------------
# Each goal: target sensor(s), a band, optional "keep low" penalty sensor, and whether it is
# reliably achievable at all (impossible -> the right answer is ABSTAIN).
GOALS = {
    "deep_chain":   dict(target=M3,   band=(4.0, 1e9),  penalty=None, achievable=True),
    "noise_robust": dict(target=M1,   band=(2.5, 1e9),  penalty=AN,   achievable=True),
    "decoy_reject": dict(target=M2,   band=(3.0, 1e9),  penalty=None, achievable=True,
                         decoy=DECOY),
    "impossible":   dict(target=C1,   band=(2.0, 1e9),  penalty=C2,   achievable=False),
    # inert_tax is a modifier applied to any goal (z0/z1 already present); measured via cost.
}

__all__ = ["CapabilityWorld", "GOALS", "NAMES", "ACTUATORS",
           "A0", "A1", "AN", "ADEC", "AC", "Z0", "Z1",
           "GATE", "M1", "M2", "M3", "DECOY", "C1", "C2", "N", "H"]
