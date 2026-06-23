"""
THE ACTION NOISY-TV: a knob that controls NOISE, not signal -- the one stressor that should
separate EIG (information about the MECHANISM) from prediction-error curiosity (surprise /
prediction-first), which the kill box could not, because there every informative action was
also a high-surprise action.

  a_sig    -- drives sensor s (the ONLY learnable structure: a_sig -> s)
  a_noise  -- drives the VARIANCE of sensor n, with NO effect on its mean. Driving it high
              makes n swing wildly (huge predictive surprise / prediction error) while
              teaching you NOTHING -- the irreducible-noise trap, but as an ACTION.
  z0..     -- inert distractor knobs.

Expected: EIG spends ~no budget on a_noise (parameter info ~0) and recovers a_sig->s fast;
surprise / prediction-first WASTE budget chasing a_noise and recover a_sig->s slower. If EIG
separates here, its objective specifically buys something dumb-active does not.
"""
from __future__ import annotations

import numpy as np


class NoiseKnobWorld:
    """DynamicalCausalWorld-compatible world with a heteroscedastic (noise-controlling) knob."""

    def __init__(self, n_distract: int = 4, rng=None, noise_gain: float = 4.0):
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.n_distract = int(n_distract)
        self.noise_gain = float(noise_gain)
        self.A_SIG, self.A_NOISE = 0, 1
        self.dist = tuple(range(2, 2 + n_distract))
        self.S = 2 + n_distract                              # signal sensor
        self.N = 3 + n_distract                              # noise sensor
        self.d = 4 + n_distract
        self.actuators = (self.A_SIG, self.A_NOISE) + self.dist
        self.hidden: tuple = ()
        self.names = (("a_sig", "a_noise") + tuple(f"z{i}" for i in range(n_distract))
                      + ("s", "n"))
        self.A = np.zeros((self.d, self.d))                  # placeholder (overridden step)
        self.x = None
        self.command: dict = {}
        self.t = 0

    @property
    def sensors(self):
        return tuple(i for i in range(self.d) if i not in self.actuators)

    @property
    def observed(self):
        return tuple(range(self.d))

    def true_edges(self):
        return {(self.A_SIG, self.S)}                        # the only learnable edge

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
        for j in self.actuators:                             # actuators hold commanded value
            xc[j] = self.command.get(j, 0.0)
        xn = xc.copy()
        xn[self.S] = 0.3 * xc[self.S] + 0.8 * xc[self.A_SIG]
        xn[self.N] = 0.3 * xc[self.N]
        if noise:
            xn[self.S] += self.rng.normal(0.0, 0.1)
            sd_n = 0.1 + self.noise_gain * max(xc[self.A_NOISE], 0.0)   # heteroscedastic trap
            xn[self.N] += self.rng.normal(0.0, sd_n)
        for j in self.actuators:
            xn[j] = self.command.get(j, 0.0)
        self.x = xn
        self.t += 1
        return self.x.copy()

    def clone(self, rng=None):
        return NoiseKnobWorld(self.n_distract,
                              rng if rng is not None else np.random.default_rng(),
                              self.noise_gain)


__all__ = ["NoiseKnobWorld"]
