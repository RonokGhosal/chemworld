"""
THE KILL BOX -- a world engineered to BREAK naive exploration, so that the central claim
(reward-free EIG/causal exploration beats dumb exploration in a meaningful, repeatable way)
is put under real fire instead of demonstrated on a soft pitch.

Every stressor that defeats a different baseline is present at once:

  * CONDITIONAL GATE (defeats single-knob / random pokes): m1 moves only when the gate is
    OPEN *and* a1 is driven  (m1 += w·gate·a1). The gate is opened by a0. So you must do a0
    FIRST, then a1 -- an un-sequenced poke moves nothing. Naive single-variable intervention
    is structurally blind here.
  * DEEP DELAYED CHAIN (defeats impulse / un-sustained action): m1 -> m2 -> m3 are slow
    integrators; a one-step pulse moves only m1. m3 responds only to SUSTAINED drive.
  * HIDDEN CONFOUNDER (defeats passive observation): a hidden H drives both c1 and c2 with
    NO direct edge. Passive observation infers a spurious c1<->c2; only intervention (or
    honest abstention) avoids the wrong arrow.
  * NOISY TV (defeats surprise / prediction-error curiosity): tv is parent-free high-variance
    noise. An agent that chases predictive surprise gets hypnotised by it.
  * INERT DISTRACTOR (taxes random budget): a2 is an actuator that drives NOTHING. Random
    action selection wastes a third of its interventions on it.
  * DECOY (must be rejected): decoy is driven by m1, so it correlates with the chain but is
    not its cause.

True OBSERVED directed edges (the gradable target, hidden H excluded):
    a0->gate, gate->m1, a1->m1, m1->m2, m2->m3, m1->decoy            (6 edges)
plus the HONEST non-edge: c1 -- c2 is CONFOUNDED (mark bidirected, never a directed arrow).

Why EIG should win: information gain about the mechanism is maximised by opening the gate
(huge uncertainty resolved), then driving the gated chain -- and is ~0 for tv (irreducible)
and a2 (inert). If EIG does NOT win here, the predefined tripwires (killbox_experiment.py)
tell us whether it is a representation limit (-> model upgrade) or an objective limit
(-> the thesis itself is in trouble).
"""
from __future__ import annotations

import numpy as np

from .world import DynamicalCausalWorld

# core variable indices (0..10); H (hidden) and any inert distractor knobs are appended
A0, A1, A2, GATE, M1, M2, M3, DECOY, C1, C2, TV = range(11)
CORE_NAMES = ("a0", "a1", "a2", "gate", "m1", "m2", "m3", "decoy", "c1", "c2", "tv")

# the gradable ground truth: observed directed edges (j -> i)
TRUE_OBSERVED_EDGES = {
    (A0, GATE),                       # a0 opens the gate
    (GATE, M1), (A1, M1),             # conditional gate: gate AND a1 drive m1
    (M1, M2), (M2, M3),               # deep slow chain
    (M1, DECOY),                      # decoy (correlated, real edge from m1)
}
CONFOUNDED_PAIR = frozenset((C1, C2))  # H drives both; NO direct edge -> must be bidirected
DEEP_EDGE = (M2, M3)                    # the one that needs SUSTAINED drive to appear


def killbox(rng=None, n_distract: int = 0) -> DynamicalCausalWorld:
    """The kill box. ``n_distract`` appends that many INERT actuator knobs (sparse
    perturbability: most knobs do nothing) so random/surprise WASTE budget on them while
    EIG should learn to ignore them. Layout: core 0..10, then n_distract inert actuators,
    then the hidden confounder H last."""
    rng = rng if rng is not None else np.random.default_rng(0)
    distract = tuple(range(11, 11 + n_distract))   # inert actuators
    Hidx = 11 + n_distract                          # hidden confounder, last
    d = 12 + n_distract
    A = np.zeros((d, d))
    A[GATE, GATE], A[GATE, A0] = 0.20, 0.90        # gate <- a0
    A[M1, M1] = 0.30                                # m1 self; its DRIVE is the gate interaction
    A[M2, M2], A[M2, M1] = 0.60, 0.70              # m2 <- m1 (slow)
    A[M3, M3], A[M3, M2] = 0.70, 0.60              # m3 <- m2 (slow, DEEP)
    A[DECOY, DECOY], A[DECOY, M1] = 0.20, 0.90     # decoy <- m1
    A[C1, C1], A[C1, Hidx] = 0.20, 0.95            # c1 <- H  (clean H proxy)
    A[C2, C2], A[C2, Hidx] = 0.20, 0.90            # c2 <- H  (no c1->c2 edge)
    A[Hidx, Hidx] = 0.90                            # slow hidden confounder
    # a2, distractors (inert), tv (parent-free): no rows
    b = np.zeros(d)
    noise = np.zeros(d)
    noise[[GATE, M1, M2, M3, DECOY]] = 0.05
    noise[C1], noise[C2], noise[TV], noise[Hidx] = 0.15, 1.00, 2.50, 0.50
    interactions = ((M1, GATE, A1, 0.60),)         # m1 += 0.6 * gate * a1   (the AND-gate)
    names = CORE_NAMES + tuple(f"z{i}" for i in range(n_distract)) + ("H",)
    return DynamicalCausalWorld(A=A, b=b, noise_std=noise,
                                actuators=(A0, A1, A2) + distract,
                                names=names, interactions=interactions, hidden=(Hidx,),
                                rng=rng)


__all__ = ["killbox", "CORE_NAMES", "TRUE_OBSERVED_EDGES", "CONFOUNDED_PAIR", "DEEP_EDGE",
           "A0", "A1", "A2", "GATE", "M1", "M2", "M3", "DECOY", "C1", "C2", "TV"]
