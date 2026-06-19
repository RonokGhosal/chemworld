"""
Long-horizon, non-stationary deployment — a driver that runs the agent's existing
continual loop (agent.live_round) across MANY regime changes, with metrics over time,
a discover-once baseline, and localized re-exploration.

constructor_causal already self-detects and adapts to a SINGLE change (live_round +
forgetting + consolidate). What was missing — and is added here — is the *deployment*:
a declarative multi-regime schedule, a long-horizon driver, the baseline that proves
continual learning is necessary, and localized re-exploration that re-checks only the
believed-relevant sub-graph after a localized change.

Metrics tracked per round:
  f1         edge-presence F1 of the recovered graph vs the CURRENT (mutated) world
             — separates on STRUCTURAL changes (add/remove edge).
  belief_mae mean |model weight − true weight| over the live edges — also separates on
             PARAMETRIC changes (sign flips, strength) that leave F1 unchanged.
  worst_z    worst standardized one-step surprise (the change signal).
  changed / pruned / library_size / detection_latency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .agent import ConstructorCausalAgent
from .world import DynamicalCausalWorld

Mutation = Callable[[DynamicalCausalWorld], None]


# --------------------------------------------------------------------------- #
# Mutation factories  (edit the world in place; edge args are (target i, source j),
# matching world.A[i, j] — exactly how the existing demos hard-code changes)
# --------------------------------------------------------------------------- #
def flip_edge(i: int, j: int) -> Mutation:
    def m(w): w.A[i, j] = -w.A[i, j]
    return m


def scale_edge(i: int, j: int, factor: float) -> Mutation:
    def m(w): w.A[i, j] = w.A[i, j] * factor
    return m


def add_edge(i: int, j: int, weight: float) -> Mutation:
    def m(w): w.A[i, j] = weight
    return m


def remove_edge(i: int, j: int) -> Mutation:
    def m(w): w.A[i, j] = 0.0
    return m


def inject_gate(target: int, a: int, b: int, weight: float) -> Mutation:
    def m(w): w.interactions = tuple(w.interactions) + ((target, a, b, weight),)
    return m


def remove_gate(target: int, a: int, b: int) -> Mutation:
    def m(w):
        w.interactions = tuple(t for t in w.interactions if t[:3] != (target, a, b))
    return m


def noise_burst(i: int, factor: float) -> Mutation:
    def m(w): w.noise_std[i] = w.noise_std[i] * factor
    return m


def compose_mutations(*muts: Mutation) -> Mutation:
    def m(w):
        for mu in muts:
            mu(w)
    return m


def identity() -> Optional[Mutation]:
    """A no-op 'mutation' for a regime that changes nothing (e.g. the initial learn
    regime). Returns None so deploy() does NOT treat it as a change-point or arm the
    detection-latency clock for it (a no-op must never be credited with a detection)."""
    return None


# --------------------------------------------------------------------------- #
# Declarative schedule
# --------------------------------------------------------------------------- #
@dataclass
class RegimeSchedule:
    """A sequence of (n_rounds, mutation). The mutation fires ONCE at the regime's
    first round (the change-point); the remaining rounds are stable, to measure
    detection + recovery."""
    regimes: tuple

    @property
    def total_rounds(self) -> int:
        return sum(n for (n, _) in self.regimes)

    @property
    def change_points(self) -> list:
        pts, off = [], 0
        for (n, _) in self.regimes:
            pts.append(off)
            off += n
        return pts

    def regime_of(self, rnd: int) -> int:
        off = 0
        for ri, (n, _) in enumerate(self.regimes):
            if rnd < off + n:
                return ri
            off += n
        return len(self.regimes) - 1

    def mutation_at(self, rnd: int) -> Optional[Mutation]:
        # iterate (don't dict-zip): a zero-length regime makes two regimes share a start
        # offset, and dict-zip would silently clobber one. Return the first REAL (non-None)
        # mutation whose change-point is `rnd`; None means "no change here".
        off = 0
        for (n, m) in self.regimes:
            if off == rnd and m is not None:
                return m
            off += n
        return None

    @classmethod
    def five_regime_default(cls, rounds=5) -> "RegimeSchedule":
        """For DynamicalCausalWorld.default (A0=0,A1=1,chain1=2,chain2=3,decoy=4,static=5)."""
        A0, C1, C2, DEC = 0, 2, 3, 4
        return cls((
            (rounds, identity()),                                  # R0: learn
            (rounds, flip_edge(C1, A0)),                           # R1: sign flip (parametric)
            (rounds, scale_edge(C2, C1, 1.8)),                     # R2: strengthen deep edge
            (rounds, add_edge(DEC, A0, 0.8)),                      # R3: NEW edge a0->decoy
            (rounds, compose_mutations(noise_burst(C1, 3.0),       # R4: noise + remove edge
                                       remove_edge(DEC, A0))),
        ))


@dataclass
class RoundMetrics:
    rnd: int
    regime: int
    f1: float
    belief_mae: float
    worst_z: float
    changed: bool
    pruned: int
    library_size: int
    is_change_point: bool
    detection_latency: Optional[int]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _worst_standardized_surprise(agent) -> float:
    pre = agent.surprise()
    std = [pre[i] / (agent.model.sigma2[i] ** 0.5 + 1e-9) for i in agent.model.sensors]
    return float(max(std)) if std else 0.0


def _belief_mae(agent, world) -> float:
    edges = set(world.true_edges()) | agent.model.recovered_edges()
    if not edges:
        return 0.0
    return float(np.mean([abs(agent.model.weight(i, j) - world.A[i, j])
                          for (j, i) in edges]))


# --------------------------------------------------------------------------- #
# the long-horizon drivers
# --------------------------------------------------------------------------- #
def deploy(agent, world, schedule: RegimeSchedule, steps_per_round: int = 130,
           z_change: float = 4.0, setpoints=(-2.0, 2.0), rediscover: bool = False,
           window: int = 350, localized: bool = False, metrics_cb=None) -> list:
    """Run the continual loop across the schedule; record a per-round timeline."""
    timeline, pending = [], None
    for rnd in range(schedule.total_rounds):
        mut = schedule.mutation_at(rnd)
        if mut is not None:
            mut(world)
            pending = rnd
        rep = agent.live_round(steps=steps_per_round, z_change=z_change,
                               setpoints=setpoints, rediscover=rediscover, window=window)
        if localized and rep["changed"]:
            worst = max(agent.surprise().items(), key=lambda kv: kv[1])[0]
            agent.explore_localized(steps_per_round, worst)
        sc = agent.recovered_dag()
        latency = None
        if rep["changed"] and pending is not None:
            latency = rnd - pending
            pending = None
        m = RoundMetrics(rnd=rnd, regime=schedule.regime_of(rnd), f1=sc["f1"],
                         belief_mae=_belief_mae(agent, world), worst_z=rep["z_surprise"],
                         changed=rep["changed"], pruned=rep["pruned"],
                         library_size=rep["skills"], is_change_point=(mut is not None),
                         detection_latency=latency)
        timeline.append(m)
        if metrics_cb:
            metrics_cb(m)
    return timeline


def deploy_baseline(world, schedule: RegimeSchedule, seed: int = 0, warmup_steps: int = 500,
                    setpoints=(-2.0, 2.0)) -> list:
    """Discover-once control: learn regime 0 thoroughly, then FREEZE — never explore,
    consolidate, or rebuild again. Metrics are still recorded so the belief's decay
    against the drifting world is visible. (Honest baseline: no model updates after
    warmup; forget=1.0 so even the warmup belief is the full-history average.)"""
    agent = ConstructorCausalAgent(world, seed=seed, forget=1.0)
    agent.explore(warmup_steps)
    agent.build_library(setpoints=setpoints)
    timeline = []
    for rnd in range(schedule.total_rounds):
        mut = schedule.mutation_at(rnd)
        if mut is not None:
            mut(world)
        sc = agent.recovered_dag()                       # frozen belief vs current world
        timeline.append(RoundMetrics(
            rnd=rnd, regime=schedule.regime_of(rnd), f1=sc["f1"],
            belief_mae=_belief_mae(agent, world),
            worst_z=_worst_standardized_surprise(agent), changed=False, pruned=0,
            library_size=len(agent.library.possible()),
            is_change_point=(mut is not None), detection_latency=None))
    return timeline


def measure_localized_saving(world_factory, target_i: int, source_j: int,
                             warmup: int = 400, recover_steps: int = 40,
                             setpoints=(-2.0, 0.0, 2.0), seed: int = 0) -> dict:
    """After a sign flip of edge (target_i <- source_j), recover with FULL vs
    LOCALIZED exploration and compare the weight error on the changed edge for the
    same budget, plus the candidate-grid sizes."""
    def run(localized):
        w = world_factory()
        agent = ConstructorCausalAgent(w, seed=seed, forget=0.9)
        agent.explore(warmup)
        w.A[target_i, source_j] = -w.A[target_i, source_j]    # the localized change
        true_w = w.A[target_i, source_j]
        if localized:
            acts = agent.explore_localized(recover_steps, target_i)
        else:
            agent.explore(recover_steps)
            acts = agent.actuators
        err = abs(agent.model.weight(target_i, source_j) - true_w)
        grid = len(setpoints) ** len(acts)
        return err, grid, acts
    err_f, grid_f, acts_f = run(False)
    err_l, grid_l, acts_l = run(True)
    return {"full_weight_err": err_f, "localized_weight_err": err_l,
            "full_grid": grid_f, "localized_grid": grid_l,
            "full_actuators": acts_f, "localized_actuators": acts_l}
