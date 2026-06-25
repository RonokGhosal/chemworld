"""
MESSY WORLD -- the first rung toward real-world transfer (commander's new mission).

The agent no longer gets clean causal variables. The true latent z (a deep gated chain plus a
noise sensor) is observed ONLY through a random linear MIXING with sensor noise:

        o_t = W z_t + e_t,    e_t ~ N(0, obs_noise)

The agent commands actions (a0, a1, aN) but never sees z -- it must LEARN a state
representation from the raw, higher-dimensional, mixed observation o, then use it to control.

Why this surfaces noise-awareness at the REPRESENTATION level: the noise knob aN inflates the
variance of latent n, so the n-direction DOMINATES the observation's variance. A naive
variance-driven encoder (PCA) spends its capacity on that irreducible-noise direction; a
PREDICTABILITY-driven encoder (keep the directions of o that are predictable from the past +
action) discards it and recovers the controllable chain. Separating learnable structure from
irreducible noise is now a property of the encoder, not just the dynamics.
"""
from __future__ import annotations

import numpy as np

# latent sensor indices
GATE, M1, M2, M3, N = 0, 1, 2, 3, 4
NZ = 5
# action indices
A0, A1, AN = 0, 1, 2
NA = 3


class MessyWorld:
    def __init__(self, rng=None, obs_dim=14, obs_noise=0.05, noise_gain=4.0, nonlinear=True,
                 n_distract=0):
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.obs_dim = int(obs_dim)
        self.obs_noise = float(obs_noise)
        self.noise_gain = float(noise_gain)
        self.nonlinear = bool(nonlinear)
        self.n_distract = int(n_distract)                       # autonomous nuisance latents
        self.nz = NZ + self.n_distract
        self.W = self.rng.normal(0.0, 0.8, (self.obs_dim, self.nz))  # fixed unknown sensor map
        self.b = self.rng.uniform(0.0, 2 * np.pi, self.obs_dim)  # random phases (RFF)
        self.z = np.zeros(self.nz)

    def reset(self):
        self.z = np.zeros(self.nz)
        return self.observe()

    def observe(self):
        # raw sensors: a random nonlinear but MONOTONIC (tanh) map of the latent -- a linear
        # encoder cannot invert it, yet the information is preserved so a nonlinear encoder
        # CAN recover the controllable latent.
        lin = self.W @ self.z
        raw = (np.tanh(lin + 0.3 * self.b) if self.nonlinear else lin)
        return raw + self.rng.normal(0.0, self.obs_noise, self.obs_dim)

    def step(self, action):
        """action: array/dict over (a0, a1, aN)."""
        a0 = float(action[A0]); a1 = float(action[A1]); aN = float(action[AN])
        z = self.z; zn = z.copy()
        zn[GATE] = 0.20 * z[GATE] + 0.90 * a0
        zn[M1] = 0.30 * z[M1] + 0.60 * z[GATE] * a1          # AND-gate (nonlinear in latent)
        zn[M2] = 0.60 * z[M2] + 0.70 * z[M1]
        zn[M3] = 0.70 * z[M3] + 0.60 * z[M2]
        zn[:4] += self.rng.normal(0.0, 0.02, 4)              # small latent process noise
        zn[N] = 0.30 * z[N] + self.rng.normal(0.0, 0.10 + self.noise_gain * max(aN, 0.0))
        if self.n_distract:                                  # autonomous AR nuisance (uncontrolled)
            zn[NZ:] = 0.85 * z[NZ:] + self.rng.normal(0.0, 0.4, self.n_distract)
        self.z = zn
        return self.observe()

    def clone(self, rng=None):
        """A fresh EPISODE on the SAME embodied sensor map: identical W,b, new state + noise
        stream. This is what train/test episodes must share (a NEW random world would be a
        different sensor coordinate system = an unintended domain shift)."""
        c = MessyWorld(rng if rng is not None else np.random.default_rng(),
                       obs_dim=self.obs_dim, obs_noise=self.obs_noise,
                       noise_gain=self.noise_gain, nonlinear=self.nonlinear,
                       n_distract=self.n_distract)
        c.W = self.W.copy(); c.b = self.b.copy(); c.reset()
        return c

    def true_m3(self):
        return float(self.z[M3])

    def m3_readout(self):
        """The linear obs->m3 direction (given to the controller as the goal), so the agent
        controls a readout r.o without being handed the latent."""
        return np.linalg.pinv(self.W)[M3]


# ---------- representation encoders ----------
def variance_pca(O, k):
    """Naive: top-k principal directions of the observation (variance-driven)."""
    Oc = O - O.mean(0)
    U, S, Vt = np.linalg.svd(Oc, full_matrices=False)
    return Vt[:k]                                            # (k, obs_dim)


def predictability_encoder(O, A, k):
    """Noise-aware: keep the directions of o that are PREDICTABLE one step ahead from
    (o_t, action_t). Fit o_{t+1} ~ [o_t, a_t]; the predictable subspace = top-k principal
    directions of the FITTED (predictable) part. The unpredictable noise direction is dropped."""
    Ot, On, At = O[:-1], O[1:], A[:-1]
    X = np.column_stack([Ot, At, np.ones(len(Ot))])
    Bmat, *_ = np.linalg.lstsq(X, On, rcond=None)
    fitted = X @ Bmat                                       # predictable part of o_{t+1}
    Fc = fitted - fitted.mean(0)
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    return Vt[:k]                                            # (k, obs_dim)


__all__ = ["MessyWorld", "variance_pca", "predictability_encoder",
           "GATE", "M1", "M2", "M3", "N", "NZ", "A0", "A1", "AN", "NA"]
