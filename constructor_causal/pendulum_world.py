"""
PendulumWorld -- the standard Gymnasium **Pendulum-v1** dynamics, wrapped to the
DynamicalCausalWorld interface so our reward-free causal agent can be tested on an
environment WE DID NOT BUILD.

Why Pendulum is a good first external test:

  * It is a real, third-party, widely-used control benchmark (OpenAI Gym / Gymnasium).
  * Its **angular-velocity** update is EXACTLY linear in (velocity, sinθ, torque):

        ω_{t+1} = clip( ω_t + dt·(3g/2l · sinθ_t  +  3/ml² · u_t),  ±max_speed )
                = ω_t + 0.75·sinθ_t + 0.15·u_t            (g=10,l=1,m=1,dt=0.05)

    so there is a crisp, FALSIFIABLE causal claim to recover: torque → ω and
    gravity-through-sinθ → ω, but cosθ does NOT directly affect ω. (The cos/sin
    angle updates are a genuine rotation -- nonlinear -- which honestly stresses the
    linear model: that's the known limitation, on display.)

  * A PASSIVE observer (never applies torque) literally cannot identify the torque's
    effect -- u has no variance -- while an INTERVENING agent that pokes the torque
    can. The intervention thesis, on someone else's world.

State vector exposed to the agent (d=4):

        x = [ u(torque) ,  cosθ ,  sinθ ,  ω ]
              index 0       1       2      3
              ACTUATOR      sensor  sensor sensor

The internal truth is (θ, ω); cosθ, sinθ are exact functions of θ. Constants and the
update match Gymnasium's PendulumEnv exactly (g=10, m=1, l=1, dt=0.05, max_torque=2,
max_speed=8); reset draws θ∈[-π,π], ω∈[-1,1] as Gymnasium does.
"""
from __future__ import annotations

import numpy as np

# variable indices
TORQUE, COS, SIN, OMEGA = 0, 1, 2, 3


class PendulumWorld:
    """Gymnasium Pendulum-v1, duck-typed to DynamicalCausalWorld."""

    def __init__(self, rng=None, obs_noise: float = 0.0, g: float = 10.0,
                 m: float = 1.0, l: float = 1.0, dt: float = 0.05,
                 max_torque: float = 2.0, max_speed: float = 8.0):
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.g, self.m, self.l, self.dt = float(g), float(m), float(l), float(dt)
        self.max_torque, self.max_speed = float(max_torque), float(max_speed)
        self.obs_noise = float(obs_noise)
        # ---- DynamicalCausalWorld-compatible surface ----
        self.actuators = (TORQUE,)
        self.hidden: tuple = ()
        self.names = ("torque", "cosθ", "sinθ", "ω")
        self.A = np.zeros((4, 4))        # placeholder (dynamics are overridden below)
        self.b = np.zeros(4)
        self.noise_std = np.array([0.0, obs_noise, obs_noise, obs_noise])
        self.interactions: tuple = ()
        self.nonlinear_terms: tuple = ()
        # internal physical state + the exposed obs vector
        self._theta = 0.0
        self._omega = 0.0
        self.x: np.ndarray | None = None
        self.command: dict = {}
        self.t = 0

    # ---- DynamicalCausalWorld properties -----------------------------------
    @property
    def d(self) -> int:
        return 4

    @property
    def sensors(self) -> tuple:
        return tuple(i for i in range(self.d) if i not in self.actuators)

    @property
    def observed(self) -> tuple:
        return tuple(i for i in range(self.d) if i not in self.hidden)

    # ---- ground-truth causal support (graded against) ----------------------
    def true_edges(self) -> set:
        """One-step cross-variable edges j->i (self-loops excluded).

        VELOCITY (i=OMEGA) is the clean, linearly-identifiable core:
            torque->ω, sinθ->ω    (and NOT cosθ->ω).
        ANGLE (i=COS,SIN) is a rotation: the new angle depends on the old angle AND
        the new velocity (hence on torque, cos, sin, ω). These edges are real but
        NONLINEAR -- included for completeness; the velocity edges are the headline."""
        return {
            (SIN, OMEGA), (TORQUE, OMEGA),                         # ω ← sinθ, torque
            (COS, SIN), (OMEGA, SIN), (TORQUE, SIN),               # sinθ ← cosθ, ω, torque
            (SIN, COS), (OMEGA, COS), (TORQUE, COS),               # cosθ ← sinθ, ω, torque
        }

    def velocity_edges(self) -> dict:
        """The falsifiable headline: which edges into ω SHOULD / SHOULD-NOT exist."""
        return {"present": {(TORQUE, OMEGA), (SIN, OMEGA)},
                "absent":  {(COS, OMEGA)}}

    # ---- dynamics ----------------------------------------------------------
    def _obs(self) -> np.ndarray:
        c, s = np.cos(self._theta), np.sin(self._theta)
        x = np.array([self.command.get(TORQUE, 0.0), c, s, self._omega], float)
        if self.obs_noise > 0.0:
            x[1:] += self.rng.normal(0.0, self.obs_noise, 3)   # sensors only, not the knob
        return x

    def reset(self, x0=None) -> np.ndarray:
        if x0 is None:
            self._theta = float(self.rng.uniform(-np.pi, np.pi))
            self._omega = float(self.rng.uniform(-1.0, 1.0))
        else:                       # accept a state vector [u,cos,sin,ω] (angle from atan2)
            x0 = np.asarray(x0, float)
            self._theta = float(np.arctan2(x0[SIN], x0[COS]))
            self._omega = float(x0[OMEGA])
        self.command = {}
        self.t = 0
        self.x = self._obs()
        return self.x.copy()

    def step(self, command=None, noise: bool = True) -> np.ndarray:
        if command:
            for j, v in command.items():
                self.command[j] = float(v)          # commands persist (like the base world)
        u = float(np.clip(self.command.get(TORQUE, 0.0), -self.max_torque, self.max_torque))
        self.command[TORQUE] = u
        # --- Gymnasium Pendulum-v1 update (exact) ---
        newthdot = self._omega + (
            3.0 * self.g / (2.0 * self.l) * np.sin(self._theta)
            + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        newthdot = float(np.clip(newthdot, -self.max_speed, self.max_speed))
        newth = self._theta + newthdot * self.dt
        self._theta, self._omega = newth, newthdot
        self.t += 1
        self.x = self._obs()
        if not noise:                               # deterministic read (for verification)
            c, s = np.cos(self._theta), np.sin(self._theta)
            self.x = np.array([u, c, s, self._omega], float)
        return self.x.copy()

    # ---- fresh independent copy --------------------------------------------
    def clone(self, rng=None) -> "PendulumWorld":
        return PendulumWorld(rng=rng if rng is not None else np.random.default_rng(),
                             obs_noise=self.obs_noise, g=self.g, m=self.m, l=self.l,
                             dt=self.dt, max_torque=self.max_torque,
                             max_speed=self.max_speed)


__all__ = ["PendulumWorld", "TORQUE", "COS", "SIN", "OMEGA"]
