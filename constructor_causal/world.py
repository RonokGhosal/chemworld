"""
The world: a *dynamical* structural causal model the agent knows nothing about.

This is the ground truth. It is deliberately a generalisation of the static SCMs
in ``causal_dag/scm.py`` to *time*, because Constructor Theory is about
TRANSFORMATIONS, and a transformation needs a before and an after. So the world
is a first-order causal dynamical system

        x_{t+1,i} = sum_j A[i,j] x_{t,j} + b_i + e_{t,i},   e ~ N(0, sigma_i^2)

where ``A[i,j]`` is the (hidden) causal weight of variable j on the next value of
variable i. Across one time-step this is automatically a DAG (nothing at time t+1
feeds back into time t), so causal direction is well defined and recoverable from
interventions.

Two kinds of variables:

  * ACTUATORS  -- knobs the agent can *force* (a constructor primitive: do(x_j=v)).
                  A forced actuator is held at its commanded value until changed,
                  exactly like dialling a setpoint. It ignores its own dynamics.
  * SENSORS    -- everything else. The agent can only *move* a sensor by driving
                  the actuators that feed it through the causal graph. Deep sensors
                  are reachable only by *sequencing* actions over time -- which is
                  precisely where composing constructors earns its keep.

The default world (`DynamicalCausalWorld.default`) bakes in the four classic
traps from PLAN.md so the machinery can be falsified:

    a0 -> chain1 -> chain2      a genuine causal CHAIN; chain2 is SLOW + DEEP,
                                reachable only by holding the chain (composition).
    chain1 -> decoy             a DECOY: correlated with chain2 (shared cause
                                chain1) but NOT a cause of it. Must be rejected.
    static                      pure high-variance NOISE, no parents: the
                                "noisy-TV" trap that hypnotises naive curiosity.
    a1                          a second, near-inert knob (mostly a distractor).

No rewards exist anywhere in this file. The world just *is*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DynamicalCausalWorld:
    """A hidden linear dynamical SCM. ``A[i,j]`` = effect of var j on next var i."""

    A: np.ndarray                       # (d, d) causal weight matrix (target, source)
    b: np.ndarray                       # (d,)   constant drift per variable
    noise_std: np.ndarray               # (d,)   exogenous std per variable
    actuators: tuple[int, ...]          # indices the agent may force
    names: tuple[str, ...]              # human labels, for narration
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    # optional structure -------------------------------------------------------
    interactions: tuple = ()            # ((target_i, src_a, src_b, weight), ...):
                                        #   x_{t+1,i} += weight * x_a * x_b  (a "gate")
    nonlinear_terms: tuple = ()         # ((target_i, src_j, kind, weight, freq), ...):
                                        #   x_{t+1,i} += weight * f(freq * x_j)
    hidden: tuple = ()                  # indices the agent cannot observe

    # set on reset()
    x: np.ndarray = field(default=None, repr=False)
    command: dict = field(default_factory=dict, repr=False)
    t: int = 0

    @property
    def d(self) -> int:
        return self.A.shape[0]

    @property
    def sensors(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.d) if i not in self.actuators)

    @property
    def observed(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.d) if i not in self.hidden)

    # ---- ground-truth causal support (what discovery is graded against) -----
    def true_edges(self, eps: float = 1e-9) -> set[tuple[int, int]]:
        """Cross-variable causal edges j->i (self-loops j==i excluded)."""
        E = set()
        for i in range(self.d):
            for j in range(self.d):
                if i != j and abs(self.A[i, j]) > eps:
                    E.add((j, i))
        return E

    # ---- the dynamics -------------------------------------------------------
    def reset(self, x0: np.ndarray | None = None) -> np.ndarray:
        self.x = np.zeros(self.d) if x0 is None else np.asarray(x0, float).copy()
        self.command = {}
        self.t = 0
        return self.x.copy()

    def step(self, command: dict[int, float] | None = None,
             noise: bool = True) -> np.ndarray:
        """Advance one tick.

        ``command`` maps a variable index -> a value the agent ATTEMPTS to force.
        The agent may attempt to force any variable; only true actuators actually
        clamp (others ignore the attempt and evolve normally). This is what makes
        controllability discoverable: poke a variable and see whether it holds.
        Commands persist until overwritten; forced actuators ignore their dynamics.
        """
        if command:
            for j, v in command.items():
                self.command[j] = float(v)       # record the attempt (any variable)

        x_cur = self.x.copy()
        for j, v in self.command.items():        # only TRUE actuators clamp
            if j in self.actuators:
                x_cur[j] = v

        e = self.rng.normal(0.0, 1.0, self.d) * self.noise_std if noise else 0.0
        x_next = self.A @ x_cur + self.b + e
        for (i, a, b, w) in self.interactions:   # bilinear "gate" terms
            x_next[i] += w * x_cur[a] * x_cur[b]
        for (i, j, kind, w, freq) in self.nonlinear_terms:   # smooth nonlinear edges
            z = freq * x_cur[j]
            f = {"sin": np.sin(z), "cos": np.cos(z), "tanh": np.tanh(z),
                 "sq": z * z}[kind]
            x_next[i] += w * f
        for j, v in self.command.items():        # true knobs stay put, no noise
            if j in self.actuators:
                x_next[j] = v

        self.x = x_next
        self.t += 1
        return self.x.copy()

    # ---- a fresh, independent copy (for repeated verification trials) -------
    def clone(self, rng=None) -> "DynamicalCausalWorld":
        """Same hidden parameters, fresh noise stream. The agent uses this to run
        independent real-world experiments when estimating reliability -- it never
        reads A; it only acts and observes."""
        return DynamicalCausalWorld(
            A=self.A.copy(), b=self.b.copy(), noise_std=self.noise_std.copy(),
            actuators=self.actuators, names=self.names,
            interactions=self.interactions, nonlinear_terms=self.nonlinear_terms,
            hidden=self.hidden,
            rng=rng if rng is not None else np.random.default_rng())

    # ---- factory: the default falsification world ---------------------------
    @classmethod
    def default(cls, rng=None) -> "DynamicalCausalWorld":
        rng = rng if rng is not None else np.random.default_rng(0)
        names = ("a0", "a1", "chain1", "chain2", "decoy", "static")
        A0, A1, C1, C2, DEC, ST = range(6)
        d = 6
        A = np.zeros((d, d))
        # chain1 := 0.3*chain1 + 0.8*a0          (fast, driven by knob a0)
        A[C1, C1], A[C1, A0] = 0.30, 0.80
        # chain2 := 0.5*chain2 + 0.7*chain1      (SLOW + DEEP: needs sustained drive)
        A[C2, C2], A[C2, C1] = 0.50, 0.70
        # decoy  := 0.1*decoy + 0.9*chain1       (correlated w/ chain2, NOT its cause)
        A[DEC, DEC], A[DEC, C1] = 0.10, 0.90
        # static := pure noise, no parents       (the noisy-TV trap)
        b = np.zeros(d)
        noise = np.array([0.0, 0.0, 0.05, 0.05, 0.05, 2.0])  # actuators noiseless
        return cls(A=A, b=b, noise_std=noise, actuators=(A0, A1), names=names, rng=rng)

    @classmethod
    def hard(cls, rng=None) -> "DynamicalCausalWorld":
        """A noisier variant where the OBJECTIVE starts to matter: the a0→chain1
        edge is WEAK (low signal) and must be actively probed to detect, while a1
        is a causally idle distractor. A curiosity agent concentrates its limited
        budget on the informative knob; a random agent squanders half of it on the
        useless one. (The clean ``default`` world is fully observed and so easy
        that even random recovers it -- separation only appears under stress.)"""
        w = cls.default(rng)
        w.A[2, 0] = 0.40                                  # weak a0 -> chain1
        w.noise_std = np.array([0.0, 0.0, 1.2, 0.6, 0.6, 2.0])
        return w

    @classmethod
    def confounded(cls, rng=None) -> "DynamicalCausalWorld":
        """A HIDDEN common cause. H (unobserved) drives both S1 and S2; there is NO
        direct S1→S2 edge. Passive observation sees S1 and S2 move together and
        infers a spurious link. Only INTERVENTION -- forcing S1 so it decorrelates
        from H -- reveals there is no real edge. S1 is forceable yet also has its
        own dynamics (it drifts with H when left alone), so the same world serves
        both the passive observer and the intervening agent."""
        rng = rng if rng is not None else np.random.default_rng(0)
        names = ("S1", "S2", "H")
        S1, S2, H = 0, 1, 2
        A = np.zeros((3, 3))
        A[S1, H] = 0.95                                   # S1 is a CLEAN proxy of H
        A[S2, H] = 0.90                                   # H -> S2 (no S1 -> S2 edge!)
        A[H, H] = 0.90                                    # slow hidden cause
        b = np.zeros(3)
        # S1 low-noise (good H proxy) but S2 idiosyncratically noisy (its own lag is
        # a poor H proxy) -> a passive regression dumps H's signal onto S1 as a big
        # spurious S1->S2 weight; intervention (forcing S1) zeroes it.
        noise = np.array([0.15, 1.00, 1.00])
        return cls(A=A, b=b, noise_std=noise, actuators=(S1,), names=names,
                   hidden=(H,), rng=rng)

    @classmethod
    def gated(cls, rng=None) -> "DynamicalCausalWorld":
        """A GATE (multiplicative interaction). Z rises only when gate AND a1 are
        both high, and gate is opened by a0:   Z := 0.3·Z + 0.5·gate·a1,  gate ← a0.
        So neither knob alone moves Z, and chaining one knob with itself can't
        either. Reaching Z requires composing two DISTINCT constructors IN ORDER:
        first open the gate (a0), then -- only then -- drive a1. a1 looks causally
        idle until the gate is open; its constructor is *context-dependent*."""
        rng = rng if rng is not None else np.random.default_rng(0)
        names = ("a0", "a1", "gate", "Z")
        a0, a1, gate, Z = 0, 1, 2, 3
        A = np.zeros((4, 4))
        A[gate, gate], A[gate, a0] = 0.20, 0.90           # a0 opens the gate
        A[Z, Z] = 0.30                                     # Z has no LINEAR parents
        b = np.zeros(4)
        noise = np.array([0.0, 0.0, 0.05, 0.05])
        interactions = ((Z, gate, a1, 0.50),)             # Z += 0.5 * gate * a1
        return cls(A=A, b=b, noise_std=noise, actuators=(a0, a1), names=names,
                   interactions=interactions, rng=rng)

    @classmethod
    def cascade(cls, rng=None) -> "DynamicalCausalWorld":
        """A TWO-gate cascade -- depth past two. Reaching Z needs THREE distinct
        constructors composed in order:

            gate1 ← a0                       open gate1 with a0
            gate2 := 0.5·gate1·a1            then (a1 | gate1 open) opens gate2
            Z     := 0.5·gate2·a2            then (a2 | gate2 open) drives Z

        Each knob is idle until the previous gate is open, so the agent must stack
        three context-dependent constructors: a0 ≫ (a1|gate1) ≫ (a2|gate2)."""
        rng = rng if rng is not None else np.random.default_rng(0)
        names = ("a0", "a1", "a2", "gate1", "gate2", "Z")
        a0, a1, a2, g1, g2, Z = range(6)
        A = np.zeros((6, 6))
        A[g1, g1], A[g1, a0] = 0.20, 0.90
        A[g2, g2] = 0.20
        A[Z, Z] = 0.30
        b = np.zeros(6)
        noise = np.array([0.0, 0.0, 0.0, 0.05, 0.05, 0.05])
        interactions = ((g2, g1, a1, 0.50),                # gate2 += 0.5 * gate1 * a1
                        (Z, g2, a2, 0.50))                 # Z     += 0.5 * gate2 * a2
        return cls(A=A, b=b, noise_std=noise, actuators=(a0, a1, a2), names=names,
                   interactions=interactions, rng=rng)

    @classmethod
    def wide(cls, k: int = 5, rng=None) -> "DynamicalCausalWorld":
        """A wide world full of DISTRACTORS: one real chain a0→chain1→chain2 (the
        deep target) plus k irrelevant knobs a1..ak, each driving its own dead-end
        sensor d_i. The library is then large and every primitive composes with
        every other, so uninformed BFS branches by the whole library at each step.
        An informed planner that heads toward the target ignores the distractors."""
        rng = rng if rng is not None else np.random.default_rng(0)
        n_act = k + 1
        names = ["a%d" % i for i in range(n_act)] + ["chain1", "chain2"] + \
                ["d%d" % i for i in range(1, k + 1)]
        d = len(names)
        c1, c2 = n_act, n_act + 1
        A = np.zeros((d, d))
        A[c1, c1], A[c1, 0] = 0.30, 0.80                  # a0 -> chain1
        A[c2, c2], A[c2, c1] = 0.50, 0.70                 # chain1 -> chain2 (deep/slow)
        noise = np.zeros(d)
        for i in range(2, n_act + 2):                     # sensor noise
            noise[i] = 0.05
        for i in range(1, k + 1):                          # a_i -> d_i (dead ends)
            di = n_act + 2 + (i - 1)
            A[di, di], A[di, i] = 0.20, 0.80
            noise[di] = 0.05
        return cls(A=A, b=np.zeros(d), noise_std=noise,
                   actuators=tuple(range(n_act)), names=tuple(names), rng=rng)

    @classmethod
    def latent_probe(cls, rng=None) -> "DynamicalCausalWorld":
        """A/B world for higher-lag latent detection. Two sensors that BOTH look
        autoregressive at one lag, but for different reasons:
          zself  := 0.6·zself + 0.8·a0          a GENUINE self-loop (no hidden cause)
          ylatent:= 0.3·ylatent + 0.9·H         a real self-loop PLUS a slow hidden
                                                AR(1) driver H — TWO timescales
        Both look autoregressive at one lag. This world sits NEAR the identifiability
        boundary: empirically the second-lag signal on ylatent is weak (its own lag is
        nearly a sufficient statistic for the slow latent), so higher-lag detection is
        unreliable here — a candid illustration of the hard limit. The confounded()
        world (where the affected sensor carries more idiosyncratic noise, so no single
        lag suffices) is where higher-lag latent detection actually bites."""
        rng = rng if rng is not None else np.random.default_rng(0)
        names = ("a0", "zself", "ylatent", "H")
        a0, z, y, H = 0, 1, 2, 3
        A = np.zeros((4, 4))
        A[z, z], A[z, a0] = 0.60, 0.80                 # genuine self-loop
        A[y, y], A[y, H] = 0.30, 0.90                  # self-loop + hidden slow driver
        A[H, H] = 0.90
        b = np.zeros(4)
        # ylatent needs enough own-noise that no single lag is a sufficient statistic
        # for H — otherwise the self-loop fully absorbs the latent and it is provably
        # indistinguishable from a plain self-loop (the identifiability boundary).
        noise = np.array([0.0, 0.15, 1.00, 1.00])
        return cls(A=A, b=b, noise_std=noise, actuators=(a0,), names=names,
                   hidden=(H,), rng=rng)

    @classmethod
    def nonlinear(cls, rng=None) -> "DynamicalCausalWorld":
        """Two nonlinear edges that defeat a purely linear learner:

          even  := 0.3·even + 1.2·(a0)^2     EVEN in a0 -> zero linear correlation,
                                              so a linear model finds NO edge a0→even;
                                              a product / quadratic feature recovers it.
          sat   := 0.3·sat  + 2.0·tanh(1.5·a1)  a SATURATING curve -- a linear fit
                                              gets the sign but mispredicts the plateau;
                                              a random-Fourier basis predicts it well.
          decoy := 0.3·decoy + 0.9·even       a real linear edge, as a control.
        """
        rng = rng if rng is not None else np.random.default_rng(0)
        names = ("a0", "a1", "even", "sat", "decoy")
        a0, a1, even, sat, decoy = range(5)
        A = np.zeros((5, 5))
        A[even, even] = 0.30
        A[sat, sat] = 0.30
        A[decoy, decoy], A[decoy, even] = 0.30, 0.90      # even -> decoy (linear)
        b = np.zeros(5)
        noise = np.array([0.0, 0.0, 0.05, 0.05, 0.05])
        nonlinear_terms = ((even, a0, "sq", 1.20, 1.0),   # even := (a0)^2
                           (sat, a1, "tanh", 2.0, 1.5))   # sat  := tanh(1.5 a1)
        return cls(A=A, b=b, noise_std=noise, actuators=(a0, a1), names=names,
                   nonlinear_terms=nonlinear_terms, rng=rng)


__all__ = ["DynamicalCausalWorld"]
