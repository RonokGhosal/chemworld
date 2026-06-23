"""
MACRO-ACTIONS (temporally-extended experiments) for causal discovery.

The kill box exposed a myopic-policy failure: a deep slow chain m1->m2->m3 needs SUSTAINED
drive, and a conditional gate needs an OPEN-then-DRIVE sequence -- neither is reachable by a
one-step command, so per-step EIG (and every per-step baseline) misses them. The fix is to
make the unit of experiment a MACRO: a schedule of (command, duration) segments. EIG (and
every baseline) then plans over macros, so "hold a0+a1 for 8 steps" or "open the gate, then
drive" become single candidate experiments.

Crucially, ALL policies draw from the SAME macro vocabulary -- only the SCORING differs:
  EIG       -- total expected information gain about the mechanism over the macro's rollout
  surprise  -- total predictive ENTROPY over the rollout (noisy-TV trap)
  predfirst -- total expected predictive ERROR over the rollout (ICM / prediction-first)
  random    -- uniform pick
  passive   -- no intervention
This keeps the comparison about the OBJECTIVE, not the action vocabulary.
"""
from __future__ import annotations

import itertools

import numpy as np

EPS = 1e-12


def macro_vocabulary(actuators, setpoints=(-2.0, 2.0), durations=(1, 4, 8),
                     max_pairs=24, rng=None):
    """Build the shared macro vocabulary:
      * SINGLE HOLD    -- hold one actuator at a setpoint for k steps
      * JOINT HOLD     -- hold a pair of actuators high together for 8 steps
      * OPEN-THEN-DRIVE-- hold a for 4 steps, THEN a+b together for 8 (the gate sequence)
    Pairs are sampled when there are too many actuators (sparse perturbability)."""
    A = list(actuators)
    macros = [[({a: v}, k)] for a in A for v in setpoints for k in durations]
    pairs = list(itertools.combinations(A, 2))
    if len(pairs) > max_pairs:
        idx = (rng or np.random.default_rng(0)).choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    for (a, b) in pairs:
        macros.append([({a: 2.0, b: 2.0}, 8)])                       # joint sustained hold
        macros.append([({a: 2.0}, 4), ({a: 2.0, b: 2.0}, 8)])        # open-then-drive
    return macros


_ROLL_CLIP = 30.0   # cap the look-ahead rollout so an early mis-estimated coef can't diverge


def full_command(partial, actuators):
    """An intervention is a do() on the WHOLE actuator vector: every actuator gets an
    explicit value, unspecified ones reset to NEUTRAL (0.0). This is what prevents a stale
    actuator from a previous macro silently persisting into the next one. Used identically
    for rollout scoring and real execution, so they can never diverge."""
    f = {int(a): 0.0 for a in actuators}
    f.update({int(j): float(v) for j, v in partial.items()})
    return f


def rollout_states(model, x0, macro, actuators):
    """Deterministic model rollout of a macro using FULL commands (every actuator set each
    step); return the (clamped_state, full_command) pairs it would visit. The propagated
    state is clipped so a not-yet-learned unstable estimate cannot blow the rollout to nan."""
    x = np.clip(np.asarray(x0, float), -_ROLL_CLIP, _ROLL_CLIP)
    out = []
    for cmd, k in macro:
        fc = full_command(cmd, actuators)
        for _ in range(int(k)):
            xc = x.copy()
            for j, v in fc.items():
                xc[j] = v
            out.append((xc, fc))
            mu, _ = model.predict_next(xc, fc)
            x = np.clip(np.nan_to_num(mu, nan=0.0, posinf=_ROLL_CLIP, neginf=-_ROLL_CLIP),
                        -_ROLL_CLIP, _ROLL_CLIP)
    return out


def score_eig(model, states):
    """Total expected information gain about the PARAMETERS over the macro (chain-rule)."""
    return model.seq_info_gain([model._phi(xc) for xc, _ in states])


def score_surprise(model, states):
    """Total predictive ENTROPY over the macro (the surprise / max-entropy objective)."""
    s = 0.0
    for xc, cmd in states:
        _, sd = model.predict_next(xc, cmd)
        var = np.array([sd[i] ** 2 for i in model.sensors])
        s += 0.5 * float(np.sum(np.log(2 * np.pi * np.e * np.clip(var, EPS, None))))
    return s


def score_predfirst(model, states):
    """Total expected predictive ERROR over the macro (ICM / prediction-first: go where the
    model mispredicts to improve its next-state prediction loss)."""
    s = 0.0
    for xc, cmd in states:
        _, sd = model.predict_next(xc, cmd)
        s += float(np.sum([sd[i] for i in model.sensors]))
    return s


_SCORERS = {"EIG": score_eig, "surprise": score_surprise, "predfirst": score_predfirst}


def macro_explore(world, model, policy, budget, actuators, rng,
                  epsilon=0.1, setpoints=(-2.0, 2.0), durations=(1, 4, 8), max_pairs=24):
    """Run a policy over macro-actions until the step budget is spent. Updates `model`
    on every transition. `passive` never intervenes. Returns the number of macros chosen."""
    steps = 0
    n_macros = 0
    while steps < budget:
        if policy == "passive":
            macro = [({}, 1)]               # full_command -> all-zero -> CLEARS every actuator
        else:
            vocab = macro_vocabulary(actuators, setpoints, durations, max_pairs, rng)
            if policy == "random" or rng.random() < epsilon:
                macro = vocab[int(rng.integers(len(vocab)))]
            else:
                x0 = world.x.copy()
                scorer = _SCORERS[policy]
                scores = np.nan_to_num(
                    np.array([scorer(model, rollout_states(model, x0, m, actuators)) for m in vocab]),
                    nan=-1e18, posinf=1e18, neginf=-1e18)
                best = np.flatnonzero(scores >= np.max(scores) - 1e-9)
                macro = vocab[int(rng.choice(best)) if len(best) else int(rng.integers(len(vocab)))]
        for cmd, k in macro:
            fc = full_command(cmd, actuators)            # SAME full command as the rollout used
            for _ in range(int(k)):
                if steps >= budget:
                    break
                xc = world.x.copy()
                for j, v in fc.items():
                    xc[j] = v
                xn = world.step(fc)
                model.update(xc, xn)
                steps += 1
        n_macros += 1
    return n_macros


__all__ = ["macro_vocabulary", "rollout_states", "macro_explore", "full_command",
           "score_eig", "score_surprise", "score_predfirst"]
