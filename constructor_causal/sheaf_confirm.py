"""
INTERVENTIONAL gate confirmation -- overcoming the 5-gate plateau by RECOVERING STRUCTURE BY ACTING.

The plateau (measured in sheaf_active/sheaf_frontier): the sheaf model recovers gates OBSERVATIONALLY,
reading each off an L1-sparsified sigmoid(gate2) weight. On a deep multiplicative cascade the variables
are heavily correlated, so a deep gate is predictively redundant with a shallow proxy and the L1 zeros
the true deep gate (g1=0.55 kept, g2=0.29, g3/g4/Z=0.00). Observation cannot tell the deep gate from the
proxy -- exactly the "watching" limit the six rungs proved.

The fix, on-thesis: CONFIRM each candidate gate {g_k, a_{k+1}} -> g_{k+1} by ACTING. Reach the
precondition (g_k open) with a verified constructor, then do(a_{k+1}=HIGH vs LOW) from that deep state
and check the target moves NET of what it does from rest (difference-in-differences). With g_k held open
by the reach, every shallow proxy is held identical across HIGH/LOW and CANCELS -- only the variable
genuinely gated by a_{k+1} given g_k open can move. The rest-contrast rejects a plain linear edge
(a multiplicative gate is inert from rest; a linear edge fires from rest). This is the project's own
thesis (acting beats watching) turned on the model's weakness.

World-graded: the confirmation scores real interventions (synth._finals -> world.step) and never reads
the world's hidden ground-truth structure -- it cannot see the answer, only measure responses (the
selftest greps this module to prove it). Reuses planner._finals for the reach-then-do rollout,
model._t_crit for significance, constructor.Library for precondition targeting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import _t_crit


@dataclass
class GateVerdict:
    source: int
    actuator: int
    target: int
    effect: float          # the difference-in-differences effect (deep - rest)
    tstat: float
    deep: float            # do(a) response GIVEN source open
    rest: float            # do(a) response from rest (source = 0)
    prereq: str
    n: int


def find_prereq(source_gate, library, open_lo=0.5):
    """The verified constructor whose full_program opens `source_gate` the most (either sign), or None
    (then the source is reachable from rest -- e.g. g0 opened linearly by a0 has its own primitive)."""
    best, best_mag = None, 0.0
    for c in library.possible():
        for (v, lo, hi) in c.effect.bounds:
            if v == source_gate:
                mid = 0.5 * (lo + hi)
                if abs(mid) >= open_lo and abs(mid) > best_mag:
                    best_mag, best = abs(mid), c
    return best


def confirm_gate(source_gate, actuator, synth, library, sensors, *, hi=2.0, lo=-2.0, h=3, n=60,
                 min_effect=0.3, gate_ratio=0.35, z=4.0, open_lo=0.5, rest_cache=None):
    """Difference-in-differences interventional test for the gate {source_gate, actuator} -> target.
    Returns a GateVerdict for the confirmed target (max |did|), or None."""
    prereq = find_prereq(source_gate, library, open_lo=open_lo)
    prefix = tuple(prereq.full_program) if prereq is not None else ()
    if actuator in {j for cmd in prefix for j in cmd}:
        return None                                              # can't independently do() a held actuator

    prog_hi = tuple({actuator: float(hi)} for _ in range(h))
    prog_lo = tuple({actuator: float(lo)} for _ in range(h))
    F_hi = synth._finals(prog_hi, n, prefix=prefix)              # REACH (source open) then do(a=hi)
    F_lo = synth._finals(prog_lo, n, prefix=prefix)

    if prefix:                                                   # precondition-fired: did the reach open it?
        src = np.concatenate([F_hi[:, source_gate], F_lo[:, source_gate]]).mean()
        if abs(src) < open_lo:
            return None

    if rest_cache is not None and actuator in rest_cache:        # rest contrast (cache per actuator)
        R_hi, R_lo = rest_cache[actuator]
    else:
        R_hi = synth._finals(prog_hi, n, prefix=())             # do(a=hi) from REST (source = 0)
        R_lo = synth._finals(prog_lo, n, prefix=())
        if rest_cache is not None:
            rest_cache[actuator] = (R_hi, R_lo)

    best = None
    for t in sensors:
        if t in (source_gate, actuator):
            continue
        deep = float(F_hi[:, t].mean() - F_lo[:, t].mean())     # effect of do(a) GIVEN source open
        rest = float(R_hi[:, t].mean() - R_lo[:, t].mean())     # effect of do(a) from rest
        did = deep - rest                                       # net of any DIRECT linear a->t effect
        se = float(np.sqrt(F_hi[:, t].var() / n + F_lo[:, t].var() / n
                           + R_hi[:, t].var() / n + R_lo[:, t].var() / n))
        tstat = abs(did) / max(se, 1e-12)
        gate_sig = abs(deep) > 1e-9 and abs(rest) < gate_ratio * abs(deep)   # multiplicative signature
        if abs(did) > min_effect and tstat > _t_crit(z, 2 * n - 2) and gate_sig:
            if best is None or abs(did) > abs(best.effect):
                best = GateVerdict(source=int(source_gate), actuator=int(actuator), target=int(t),
                                   effect=did, tstat=float(tstat), deep=deep, rest=rest,
                                   prereq=(prereq.name if prereq else "rest"), n=n)
    return best


def _residual_candidates(ens, sensors, min_effect, z):
    """Residual-scan on the ensemble's raw buffer (mirrors agent.discover_interactions): pairs whose
    centered product explains a target's residual beyond the linear fit. High recall -- the world test
    supplies precision. This catches deep gates the L1 readout zeroed out."""
    Xc = np.asarray(ens._buf_c, float)
    Xn = np.asarray(ens._buf_n, float)
    n, d = Xc.shape
    Phi = np.column_stack([Xc, np.ones(n)])
    out = set()
    for t in sensors:
        beta, *_ = np.linalg.lstsq(Phi, Xn[:, t], rcond=None)
        resid = Xn[:, t] - Phi @ beta
        for a in range(d):
            for b in range(a + 1, d):
                p = Xc[:, a] * Xc[:, b]
                p = p - p.mean()
                denom = float(p @ p)
                if denom < 1e-9:
                    continue
                slope = float(p @ resid) / denom
                r2 = resid - slope * p
                dof = max(n - Phi.shape[1] - 1, 1)
                se = float(np.sqrt(max(float(r2 @ r2), 1e-12) / dof / denom))
                if abs(slope) > min_effect and abs(slope) / max(se, 1e-12) > _t_crit(z, dof):
                    out.add((a, b))
    return out


def recover_structure_interventional(ensemble, library, synth, sensors, actuators, *,
                                     low_thresh=0.05, n=60, min_effect=0.3, gate_ratio=0.35,
                                     z=4.0, max_pairs=80):
    """Drop-in replacement for SheafEnsemble.recovered_hyperedges: PERMISSIVE candidate generation +
    INTERVENTIONAL confirmation. Returns [((source, actuator), target, effect), ...] (same shape)."""
    sensors = tuple(sensors)
    actuators = set(actuators)
    reachable = {v for c in library.possible() for v in c.effect.vars() if v in sensors}

    cands = set()
    for H, i, w in ensemble.recovered_hyperedges(thresh=low_thresh):     # (a) low-threshold readout
        if len(H) == 2:
            cands.add(frozenset(H))
    for pair in _residual_candidates(ensemble, sensors, min_effect, z):  # (b) residual scan (deep signal)
        cands.add(frozenset(pair))
    for g in reachable:                                                  # (c) structural fallback
        for a in actuators:
            cands.add(frozenset((g, a)))

    confirmed, rest_cache = [], {}
    for fs in list(cands)[:max_pairs]:
        members = tuple(fs)
        if len(members) != 2:
            continue
        acts_in = [m for m in members if m in actuators]
        srcs_in = [m for m in members if m in sensors and m in reachable]
        if len(acts_in) != 1 or len(srcs_in) != 1:                       # need one reachable source + one actuator
            continue
        res = confirm_gate(srcs_in[0], acts_in[0], synth, library, sensors, n=n,
                           min_effect=min_effect, gate_ratio=gate_ratio, z=z, rest_cache=rest_cache)
        if res is not None:
            confirmed.append(((res.source, res.actuator), res.target, res.effect))
    return confirmed
