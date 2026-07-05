"""
FRONTIER MAP -- the honest operating envelope of the interventional gate-confirmation method.

The 5/7/8-gate wins (sheaf_confirm.py) are on `multigate_world`, a cascade ENGINEERED to satisfy the
method's preconditions: one dedicated labeled actuator per gate, strong couplings (0.6-0.9), tiny noise
(SNR~12), no un-actuatable confounder. This module measures where the method actually BREAKS by sweeping
the two walls the real results (faithful Sachs) live behind:

  WALL B -- WEAK SIGNAL / SNR:  sweep gate coupling w down from 0.6. How far past observation does ACTING
            push recovery, and where is the floor? A LOWERED-threshold pass separates an ENGINEERING limit
            (fixed min_effect gives up -> lowering it recovers more, cleanly) from an INFORMATION limit
            (signal below noise -> lowering it recovers nothing / floods false+).

  WALL A -- UN-ACTUATABLE CONFOUNDER:  a fork C->X, C->Y (X gated by a_X given C, Y by a_Y given C, NO
            X->Y edge). Flip ONLY whether C is REACHABLE (arm 1 gives C a real opener a_C->C, mirroring
            how g0 is opened; arm 2 drops it so C is noise-driven and un-reachable). Wall A: no do()- or
            bridge-based method can stand on a cause it cannot reach.

Measured [EDGE-level scoring, seeds 0-2, gates=3; see selftest_frontier_map for exact counts]:
  Wall B: at the DEFAULT observational threshold, interventional recovers the deep multiplicative gates
          while the observational readout gets only the shallow LINEAR gate (~1/3 of targets). CAVEAT
          (pass-2 finding 2): this is a THRESHOLD asymmetry, not a detectability gap -- observation is
          scored only at its default threshold while the interventional pipeline uses a low (0.05)
          threshold; scored at the SAME low threshold, observation reportedly recovers the deep gates too
          at much lower PRECISION (~0.38 vs ~0.8). So the honest win is ~2x PRECISION, not "observation
          never recovers them" -- NOT yet re-measured here (add an obs_low column; deferred). The
          interventional floor below w~=0.15-0.20 is a reachability/do-magnitude limit (the deep source
          never opens, src_open~0.01), NOT a noise-SNR ratio -- do not read it as an "Nx SNR" number.
  Wall A: C reachable -> both fork edges recovered; C un-reachable -> recall collapses to 0. The self-loop
          A[C,C] is MATCHED across arms so the collapse isolates un-actuatability (finding 9). Precision is
          scored at EDGE level (finding 2) and is NOT perfect: in the reachable arm the method emits ~0.5
          spurious edges/run because reaching X entails reaching C, so do(a_Y) moves Y and the SOURCE is
          misattributed (X vs C) -- precision ~0.8, not the "0 false+" the old target-only metric reported.
          Sachs in miniature -- recall capped by un-actuatability, precision limited by source-confounding.
          CAVEAT (finding 6): no linear distractor, so false-positive counts are partly by-construction;
          n=3 seeds, no CI (finding 11). CAVEAT (pass-2 finding 10): the un-reachable recall=0 is partly
          STRUCTURAL -- recover_structure_interventional only proposes SOURCES from library.possible(),
          which cannot contain an un-reachable C, so the pipeline cannot even REPRESENT an un-reachable-
          source edge. 0/2 is consistent with the un-actuatability wall but is NOT an independent
          measurement of it -- reframe as a design property, not evidence.

CLI:  python -m constructor_causal.frontier_map [--seeds 3] [--gates 3]
"""
from __future__ import annotations

import argparse

import numpy as np

from .world import DynamicalCausalWorld
from .sheaf_active import SheafEnsemble
from .sheaf_frontier import random_collect, frontier_collect, build_library, gates_recovered
from .sheaf_confirm import recover_structure_interventional
from .planner import ConstructorSynthesizer


# --------------------------------------------------------------------------- parametrized worlds
def cascade_world(n_gates, rng, w=0.6, sigma=0.05):
    """multigate_world with coupling `w` and noise `sigma` as the SNR knobs. True gate targets = the
    n_gates cascade variables g1..g_{n-1} and Z (g0 is the shallow LINEAR-opened gate)."""
    n_act = n_gates + 1
    gates = list(range(n_act, n_act + n_gates))
    Z = n_act + n_gates
    d = Z + 1
    names = tuple([f"a{i}" for i in range(n_act)] + [f"g{i}" for i in range(n_gates)] + ["Z"])
    A = np.zeros((d, d))
    A[gates[0], gates[0]] = 0.2; A[gates[0], 0] = 0.9
    inter = []
    for i in range(1, n_gates):
        A[gates[i], gates[i]] = 0.2
        inter.append((gates[i], gates[i - 1], i, w))
    A[Z, Z] = 0.3
    inter.append((Z, gates[-1], n_gates, w))
    noise = np.full(d, sigma); noise[:n_act] = 0.0
    return DynamicalCausalWorld(A=A, b=np.zeros(d), noise_std=noise, actuators=tuple(range(n_act)),
                               names=names, interactions=tuple(inter), rng=rng), Z, names


def confounder_world(rng, w=0.6, c_reachable=True):
    """Fork C->X, C->Y (no X->Y). C is a SENSOR either way; the self-loop A[C,C]=0.2 is MATCHED across
    arms. arm1: a_C opens C linearly (reachable, mirrors g0). arm2: no a_C, C driven by exogenous noise
    (un-reachable). The only differences are (i) the actuator on C and (ii) C's driver (knob vs exogenous
    noise) -- the latter is intrinsic to what "actuatable" means, not an incidental confound. (Audit pass
    1, finding 9: earlier arms also differed in A[C,C] 0.2->0.5, now matched.) True gate targets = {X, Y}."""
    if c_reachable:
        aC, aX, aY, C, X, Y = 0, 1, 2, 3, 4, 5
        d = 6; acts = (aC, aX, aY); names = ("a_C", "a_X", "a_Y", "C", "X", "Y")
        A = np.zeros((d, d)); A[C, aC] = 0.9; A[C, C] = 0.2
        noise = np.zeros(d); noise[[C, X, Y]] = 0.05
    else:
        aX, aY, C, X, Y = 0, 1, 2, 3, 4
        d = 5; acts = (aX, aY); names = ("a_X", "a_Y", "C", "X", "Y")
        A = np.zeros((d, d)); A[C, C] = 0.2                          # MATCHED to reachable arm
        noise = np.zeros(d); noise[C] = 1.0; noise[[X, Y]] = 0.05     # C exogenous (its only driver) => un-reachable
    inter = ((X, C, aX, w), (Y, C, aY, w))                            # TRUE edges: {C,a_X}->X, {C,a_Y}->Y
    return DynamicalCausalWorld(A=A, b=np.zeros(d), noise_std=noise, actuators=acts,
                               names=names, interactions=inter, rng=rng), names, inter


# --------------------------------------------------------------------------- build + score
def build(make, seed, max_rounds, rounds=2, epochs=250, ep=120, K=3):
    """random_collect + fit + climb the library + a couple frontier rounds. Returns (ens, lib, synth, world)."""
    rng = np.random.default_rng(seed)
    world = make(np.random.default_rng(seed))[0]
    wf = lambda: make(rng)[0]
    ens = SheafEnsemble(world.d, world.actuators, K=K, device="cpu")
    synth = ConstructorSynthesizer(model=ens, world_factory=wf, actuators=world.actuators,
                                   sensors=world.sensors, d=world.d, rng=rng)
    random_collect(wf, ens, ep, 6, rng); ens.fit(epochs=epochs)
    lib = build_library(synth, max_rounds=max_rounds)
    for _ in range(rounds):
        frontier_collect(wf, ens, lib, ep, 3, rng); ens.fit(epochs=epochs)
        lib = build_library(synth, max_rounds=max_rounds)
    return ens, lib, synth, world


def interventional_recall(ens, lib, synth, world, **kw):
    """(true EDGES recovered, total true edges, false-positive edges) via world-graded confirmation.

    Scored at EDGE granularity, not target-node: a recovered ((source, actuator), target) counts as recall
    ONLY if {source, actuator} equals the true gate's two sources for that target; every other confirmed
    edge is a false positive. (Target-only scoring -- the earlier version -- let a wrong-source edge count
    as recall and made a spurious edge into a true target un-flaggable; audit pass 1, finding 2.)"""
    hyper = recover_structure_interventional(ens, lib, synth, world.sensors, world.actuators, **kw)
    true_edges = {(i, frozenset((a, b))) for (i, a, b, w) in world.interactions}
    rec_edges = {(t, frozenset(H)) for H, t, _ in hyper if len(H) == 2}
    return len(rec_edges & true_edges), len(true_edges), len(rec_edges - true_edges)


# --------------------------------------------------------------------------- the two sweeps
def sweep_wall_b(seeds, gates=3, couplings=(0.60, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05),
                 low_min_effect=0.12, low_z=3.0, **build_kw):
    """Return {w: dict(obs, interv_def, interv_low, fp_def, fp_low, total)} averaged over seeds."""
    out = {}
    for w in couplings:
        ob, itd, itl, fpd, fpl, tot = [], [], [], [], [], []
        for s in seeds:
            ens, lib, synth, world = build(lambda r, w=w: cascade_world(gates, r, w=w), s,
                                           max_rounds=gates + 1, **build_kw)
            og, t = gates_recovered(ens, world)
            rd, td, fd = interventional_recall(ens, lib, synth, world)
            rl, _, fl = interventional_recall(ens, lib, synth, world, min_effect=low_min_effect, z=low_z)
            ob.append(og); itd.append(rd); itl.append(rl); fpd.append(fd); fpl.append(fl); tot.append(td)
        out[w] = dict(obs=float(np.mean(ob)), interv_def=float(np.mean(itd)), interv_low=float(np.mean(itl)),
                      fp_def=float(np.mean(fpd)), fp_low=float(np.mean(fpl)), total=int(np.mean(tot)))
    return out


def sweep_wall_a(seeds, w=0.6, **build_kw):
    """Return {c_reachable: dict(recall, total, fp, c_reachable_in_lib)} averaged over seeds."""
    out = {}
    for c_ok in (True, False):
        rt, tot, fp, creach = [], [], [], []
        for s in seeds:
            ens, lib, synth, world = build(lambda r, c=c_ok: confounder_world(r, w=w, c_reachable=c), s,
                                           max_rounds=3, **build_kw)
            C_idx = world.names.index("C")
            reach = {v for cc in lib.possible() for v in cc.effect.vars()}
            creach.append(C_idx in reach)
            a, b, c = interventional_recall(ens, lib, synth, world)
            rt.append(a); tot.append(b); fp.append(c)
        out[c_ok] = dict(recall=float(np.mean(rt)), total=int(np.mean(tot)), fp=float(np.mean(fp)),
                         c_reachable_in_lib=bool(np.mean(creach) > 0.5))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gates", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    print("=" * 92)
    print(f"AXIS B -- weak-signal frontier: cascade gates={args.gates}, seeds {seeds}. mean over seeds.")
    print("  does ACTING buy SNR headroom over observation, and where does each fall off?")
    print("=" * 92)
    print(f"  {'w':>5} | {'OBS':>7} | {'INTERV-def':>16} | {'INTERV-lowthr':>16}   (true, +false)")
    wb = sweep_wall_b(seeds, gates=args.gates)
    for w, r in wb.items():
        print(f"  {w:>5.2f} | {r['obs']:>4.1f}/{r['total']} | {r['interv_def']:>4.1f}/{r['total']}"
              f"  (+{r['fp_def']:>3.1f}) | {r['interv_low']:>4.1f}/{r['total']}  (+{r['fp_low']:>3.1f})", flush=True)

    print("=" * 92)
    print(f"AXIS A -- un-actuatable-confounder frontier: fork C->X, C->Y. seeds {seeds}.")
    print("  SAME structure; flip ONLY whether C is reachable. Wall A: no compute crosses un-actuatability.")
    print("=" * 92)
    print(f"  {'C reachable?':>13} | {'lib reached C?':>14} | {'INTERV recall':>14} | {'false+':>7}")
    wa = sweep_wall_a(seeds)
    for c_ok, r in wa.items():
        print(f"  {str(c_ok):>13} | {str(r['c_reachable_in_lib']):>14} | {r['recall']:>5.2f}/{r['total']}"
              f"      | {r['fp']:>7.2f}", flush=True)
    print("=" * 92)


if __name__ == "__main__":
    main()
