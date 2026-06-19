"""
Constructor Theory, made operational on a causal DAG.

Constructor Theory (Deutsch & Marletto) reframes physics around which
TRANSFORMATIONS are possible, rather than around states evolving under laws. The
vocabulary:

  * A TASK is a transformation specified by its endpoints: an input attribute
    (a region of state space the substrate is in) and an output attribute (the
    region it must end in).            T : input_region  ->  output_region
  * A CONSTRUCTOR is a thing that, presented with a substrate in the input region,
    reliably brings it to the output region AND is left able to do it again. It is
    repeatable -- catalytic, not consumed.
  * A task is POSSIBLE iff a constructor for it can be built to arbitrarily high
    reliability. (We approximate "arbitrarily high" by an empirical threshold.)

The bridge to causal inference:  an INTERVENTION do(x_j = v) on the causal DAG is
the simplest possible constructor -- a primitive that reliably forces variable j
to v and can be repeated. A *program* (a schedule of interventions over time) is a
COMPOSITE constructor. Composition is the whole game:

        C1 : P -> Q      C2 : Q' -> R      with  Q  subseteq  Q'
        --------------------------------------------------------
                    C2 . C1 : P -> R         (program = C1.program ++ C2.program)

Composability is a *causal* statement: C2 may assume its precondition because C1
guarantees it as a postcondition. Chaining small, individually-verified
constructors yields big ones that reach deep, slow variables no single primitive
can move -- the constructor library grows itself, with no reward ever defined.

Regions are axis-aligned boxes (intervals per variable); ``subseteq`` and
membership are therefore exact and cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Attribute = a region of state space (an axis-aligned box)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Box:
    """A conjunction of per-variable intervals. Unconstrained vars are absent.

    The empty box (no constraints) is the attribute "any state" -- a valid input
    region for a constructor that works from anywhere.
    """
    bounds: tuple = ()                   # ((var, lo, hi), ...)

    @classmethod
    def any(cls) -> "Box":
        return cls(())

    @classmethod
    def from_dict(cls, d: dict) -> "Box":
        return cls(tuple((int(k), float(lo), float(hi)) for k, (lo, hi) in sorted(d.items())))

    def contains(self, x: np.ndarray) -> bool:
        return all(lo <= x[v] <= hi for (v, lo, hi) in self.bounds)

    def subseteq(self, other: "Box") -> bool:
        """Is every state in self also in other?  (self's box tighter on all of
        other's constrained axes.)  This is the composability test Q subseteq Q'."""
        mine = {v: (lo, hi) for (v, lo, hi) in self.bounds}
        for (v, lo, hi) in other.bounds:
            if v not in mine:
                return False             # self unconstrained where other constrains
            mlo, mhi = mine[v]
            if mlo < lo - 1e-9 or mhi > hi + 1e-9:
                return False
        return True

    def vars(self) -> tuple:
        return tuple(v for (v, _, _) in self.bounds)

    def __str__(self) -> str:
        if not self.bounds:
            return "any"
        return " & ".join(f"x{v}∈[{lo:.2f},{hi:.2f}]" for (v, lo, hi) in self.bounds)


# --------------------------------------------------------------------------- #
# Constructor = task + a program that performs it + measured reliability
# --------------------------------------------------------------------------- #
@dataclass
class Constructor:
    name: str
    precond: Box                         # input region it works from
    effect: Box                          # output region it guarantees
    program: tuple                       # (command_dict, ...) one per time-step
    reliability: float = 0.0             # P(effect | precond), empirical
    n_trials: int = 0
    provenance: str = "primitive"        # "primitive" | "compose(A,B)" | "synth"
    # the program that achieves this constructor's effect FROM REST (==program for a
    # precond='any' constructor; for a conditional it includes the chain that first
    # establishes its precondition). Used as the prefix when minting deeper skills.
    full_program: tuple = ()

    def __post_init__(self):
        if not self.full_program:
            self.full_program = tuple(self.program)

    @property
    def horizon(self) -> int:
        return len(self.program)

    @property
    def reliability_lo(self) -> float:
        """Wilson (~97.7% one-sided) LOWER confidence bound on the success probability,
        from the stored point estimate and trial count."""
        return wilson_lo(self.reliability, self.n_trials)

    @property
    def possible(self) -> bool:
        """Constructor-theoretic 'possible (so far)': the LOWER confidence bound on
        reliability clears tau, over enough trials. Gating on the bound (not the bare
        point estimate) controls the FALSE-possible rate -- e.g. 18/20 successes
        (point 0.90) does NOT certify p>=0.90, whereas 58/60 does."""
        return self.reliability_lo >= POSSIBLE_TAU and self.n_trials >= MIN_TRIALS

    def __str__(self) -> str:
        ok = "✓" if self.possible else ("·" if self.n_trials else "?")
        return (f"[{ok}] {self.name:22s} {str(self.precond):>14s} → {str(self.effect):<18s}"
                f"  r={self.reliability:.2f} (n={self.n_trials}, H={self.horizon})")


def wilson_lo(p: float, n: float, z: float = 2.0) -> float:
    """Wilson score LOWER confidence bound for a binomial success probability given a
    point estimate ``p`` over ``n`` trials (z=2 -> ~97.7% one-sided). Used so that
    'possible' certifies a LOWER bound on reliability, not just a lucky point estimate."""
    n = float(n)
    if n <= 0:
        return 0.0
    p = min(max(float(p), 0.0), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    half = z * np.sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0))
    return max(0.0, (centre - half) / denom)


POSSIBLE_TAU = 0.90          # reliability above which a task counts as "possible"
MIN_TRIALS = 20              # ... and only after this many verification trials


def prog_key(program) -> tuple:
    """Hashable signature of a program (a tuple of command dicts)."""
    return tuple(tuple(sorted(cmd.items())) for cmd in program)


# --------------------------------------------------------------------------- #
# Composition: the operation that grows constructors
# --------------------------------------------------------------------------- #
def compose(c1: Constructor, c2: Constructor) -> Constructor | None:
    """C2 . C1, if c1's guaranteed output satisfies c2's required input.

    Returns a new (untested) composite constructor, or None if not composable.
    Reliability is left at the independent lower bound r1*r2 until verified.
    """
    if not c1.effect.subseteq(c2.precond):
        return None
    return Constructor(
        name=f"({c1.name}≫{c2.name})",
        precond=c1.precond,
        effect=c2.effect,
        program=tuple(c1.program) + tuple(c2.program),
        reliability=c1.reliability * c2.reliability,
        n_trials=0,
        provenance=f"compose({c1.name},{c2.name})",
        full_program=tuple(c1.full_program) + tuple(c2.program),
    )


# --------------------------------------------------------------------------- #
# Reliability estimation by running the program in a world (or learned model)
# --------------------------------------------------------------------------- #
def run_program(env, program, x0=None, noise=True):
    """Execute a program from x0 (or a fresh reset). Returns the trajectory."""
    x = env.reset(x0)
    traj = [x.copy()]
    for cmd in program:
        traj.append(env.step(cmd, noise=noise).copy())
    return np.array(traj)


def estimate_reliability(env_factory, c: Constructor, n: int = 40,
                         init_sampler=None, rng=None) -> float:
    """Fraction of trials in which, starting inside ``precond``, the program lands
    inside ``effect``. ``env_factory`` returns a fresh world each call (so trials
    are independent); ``init_sampler(rng)`` draws a start state in the precond."""
    rng = rng if rng is not None else np.random.default_rng()
    hits = 0
    for _ in range(n):
        env = env_factory()
        x0 = init_sampler(rng) if init_sampler is not None else None
        if x0 is not None and not c.precond.contains(x0):
            continue
        traj = run_program(env, c.program, x0=x0, noise=True)
        if c.effect.contains(traj[-1]):
            hits += 1
    c.reliability = hits / n
    c.n_trials = n
    return c.reliability


# --------------------------------------------------------------------------- #
# The Library: a growing, composable set of constructors
# --------------------------------------------------------------------------- #
class Library:
    """The agent's accumulated know-how: every constructor it has built and
    verified, closed (lazily) under composition. This is the EGTL 'Library' --
    reusable transformations, not episodic memories."""

    def __init__(self):
        self.constructors: list[Constructor] = []
        self._keys: dict = {}                    # semantic key -> constructor

    def add(self, c: Constructor) -> bool:
        if c is None:
            return False
        # de-dup by semantic identity: name (encodes the chain) + program shape.
        # (effect boxes vary by a few percent across re-characterizations, so they
        # must NOT be part of the key, or the same skill gets minted every round.)
        key = (c.name, prog_key(c.program))
        if key in self._keys:
            existing = self._keys[key]
            if c.reliability > existing.reliability:
                existing.reliability, existing.n_trials = c.reliability, c.n_trials
            return False
        self.constructors.append(c)
        self._keys[key] = c
        return True

    def keep(self, predicate):
        """Drop constructors failing ``predicate`` (consolidation / forgetting stale
        skills). Returns the removed ones."""
        removed = [c for c in self.constructors if not predicate(c)]
        self.constructors = [c for c in self.constructors if predicate(c)]
        self._keys = {(c.name, prog_key(c.program)): c for c in self.constructors}
        return removed

    def possible(self) -> list[Constructor]:
        return [c for c in self.constructors if c.possible]

    def reaching(self, target: Box) -> list[Constructor]:
        """Possible constructors whose effect lands inside ``target``."""
        return [c for c in self.possible() if c.effect.subseteq(target)]

    def compose_round(self, verify=None, max_new: int = 50) -> list[Constructor]:
        """One closure step: try composing every ordered pair of *possible*
        constructors. If a ``verify(c)`` callable is given, only verified-reliable
        composites are kept; otherwise the r1*r2 lower bound stands.

        Returns the list of newly added composites."""
        pool = self.possible()
        added = []
        for c1 in pool:
            for c2 in pool:
                if c1 is c2:
                    continue
                cc = compose(c1, c2)
                if cc is None:
                    continue
                if verify is not None:
                    verify(cc)                       # fills reliability/n_trials
                    if not cc.possible:
                        continue
                if self.add(cc):
                    added.append(cc)
                    if len(added) >= max_new:
                        return added
        return added

    def __str__(self) -> str:
        if not self.constructors:
            return "  (empty library)"
        return "\n".join("  " + str(c) for c in self.constructors)

    def __len__(self) -> int:
        return len(self.constructors)


__all__ = ["Box", "Constructor", "Library", "compose", "run_program",
           "estimate_reliability", "POSSIBLE_TAU", "MIN_TRIALS"]
