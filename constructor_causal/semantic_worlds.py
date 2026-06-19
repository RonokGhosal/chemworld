"""
Semantic dynamical worlds — named-variable DynamicalCausalWorlds whose causal
directions a real LLM genuinely knows. The abstract worlds (a0, chain1, ...) give an
LLM nothing to reason about; these do, so they are where the REAL-LLM fusion is tested.

Each is a clean per-step DAG (no cycles), with two actuators an agent can force.
Convention: A[target, source] = weight; true_edges() returns (source, target).

heater:  heater_power -> room_temp <- window_open ;  room_temp -> thermostat_reading
tank:    inflow_valve -> tank_level <- drain_valve ; tank_level -> outflow_rate
"""
from __future__ import annotations

import numpy as np

from .world import DynamicalCausalWorld


def heater_world(rng=None) -> DynamicalCausalWorld:
    """H=heater_power(act), T=room_temp, R=thermostat_reading, W=window_open(act).
        T' = 0.5 T + 0.8 H - 0.3 W      heater warms, open window cools the room
        R' = 0.4 R + 0.9 T              the thermostat reads (lags) the temperature
    True cross-edges: H->T, W->T, T->R. An LLM knows heater->temp, window->temp,
    temp->thermostat (and would NEVER say thermostat->temp)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    H, T, R, W = 0, 1, 2, 3
    names = ("heater_power", "room_temp", "thermostat_reading", "window_open")
    A = np.zeros((4, 4))
    A[T, T], A[T, H], A[T, W] = 0.5, 0.8, -0.3
    A[R, R], A[R, T] = 0.4, 0.9
    b = np.zeros(4)
    noise = np.array([0.1, 0.3, 0.3, 0.1])
    return DynamicalCausalWorld(A=A, b=b, noise_std=noise, actuators=(H, W),
                                names=names, rng=rng)


def tank_world(rng=None) -> DynamicalCausalWorld:
    """I=inflow_valve(act), L=tank_level, O=outflow_rate, D=drain_valve(act).
        L' = 0.7 L + 0.6 I - 0.5 D      inflow raises level, open drain lowers it
        O' = 0.3 O + 0.8 L              outflow rate is set by the level (pressure)
    True cross-edges: I->L, D->L, L->O. An LLM knows inflow->level, drain->level,
    level->outflow (and would NEVER say outflow->level)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    I, L, O, D = 0, 1, 2, 3
    names = ("inflow_valve", "tank_level", "outflow_rate", "drain_valve")
    A = np.zeros((4, 4))
    A[L, L], A[L, I], A[L, D] = 0.7, 0.6, -0.5
    A[O, O], A[O, L] = 0.3, 0.8
    b = np.zeros(4)
    noise = np.array([0.1, 0.3, 0.3, 0.1])
    return DynamicalCausalWorld(A=A, b=b, noise_std=noise, actuators=(I, D),
                                names=names, rng=rng)


SEMANTIC = {"heater": heater_world, "tank": tank_world}


def question_list(world) -> list[str]:
    """The 'NameA -- NameB' orientation questions over the world's TRUE observed
    adjacencies (the orientation task the prior fills), in llm_experiment format."""
    true_e = world.true_edges()
    obs = world.observed
    qs = []
    for ia in range(len(obs)):
        for ib in range(ia + 1, len(obs)):
            a, b = obs[ia], obs[ib]
            if (a, b) in true_e or (b, a) in true_e:
                qs.append(f"{world.names[a]} -- {world.names[b]}")
    return qs


def ground_truth_answers(world) -> dict:
    """Perfect answers (cause = true parent) in the LLM answer format — a stand-in
    for a flawless expert so the from_llm_answers path is CI-testable offline."""
    true_e = world.true_edges()
    ans = {}
    for q in question_list(world):
        a_name, b_name = [s.strip() for s in q.split("--")]
        idx = {nm: i for i, nm in enumerate(world.names)}
        a, b = idx[a_name], idx[b_name]
        ans[q] = a_name if (a, b) in true_e else b_name
    return ans
