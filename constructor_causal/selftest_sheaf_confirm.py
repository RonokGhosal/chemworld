"""
Selftest for interventional gate confirmation (sheaf_confirm.py) -- the honest validation that it
breaks the 5-gate plateau WITHOUT fooling us. Exits nonzero on any failed check.

Checks:
  1. FIREWALL   -- sheaf_confirm.py never reads world.interactions / world.A (grep). It can only ACT
                   (via synth._finals -> world.step). This is the anti-circularity proof: it cannot
                   read the answer, only measure interventional responses.
  2. RECALL     -- across seeds, INTERVENTIONAL confirmation recovers >= OBSERVATIONAL readout, and
                   reaches all/most of the 5 gates (the plateau the observational readout cannot cross).
  3. PRECISION  -- interventional confirmation emits ZERO false gates (no non-true target) across seeds.
  4. CONTROLS   -- confirm_gate REJECTS wrong-actuator pairs and a0-linear candidates (a multiplicative
                   gate is inert from rest; the rest-contrast/DiD guard rejects a linear effect).
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

from .sheaf_active import SheafEnsemble, multigate_world
from .sheaf_frontier import random_collect, frontier_collect, build_library, gates_recovered
from .sheaf_confirm import confirm_gate, recover_structure_interventional
from .planner import ConstructorSynthesizer

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def build(seed, gates=5, K=3, rounds=3, epochs=250, ep=120):
    rng = np.random.default_rng(seed)
    world, Z, names = multigate_world(gates, np.random.default_rng(seed))
    wf = lambda: multigate_world(gates, rng)[0]
    ens = SheafEnsemble(world.d, world.actuators, K=K, device="cpu")
    synth = ConstructorSynthesizer(model=ens, world_factory=wf, actuators=world.actuators,
                                   sensors=world.sensors, d=world.d, rng=rng)
    random_collect(wf, ens, ep, 6, rng); ens.fit(epochs=epochs)
    lib = build_library(synth, max_rounds=gates + 1)
    for _ in range(rounds):
        frontier_collect(wf, ens, lib, ep, 3, rng); ens.fit(epochs=epochs)
        lib = build_library(synth, max_rounds=gates + 1)
    return ens, lib, synth, world, names


def main():
    print("=" * 84)
    print("SELFTEST -- interventional gate confirmation breaks the 5-gate plateau (honestly)")
    print("=" * 84)

    # 1. FIREWALL: the confirmation module must not read ground-truth structure.
    src = (pathlib.Path(__file__).parent / "sheaf_confirm.py").read_text()
    leak = [tok for tok in (".interactions", ".A ", ".A\n", ".A[", "world.A") if tok in src]
    check("firewall: sheaf_confirm never reads world.interactions/A (world-graded, not model-graded)",
          not leak, f"forbidden tokens found: {leak}" if leak else "clean")

    # 2-3. RECALL + PRECISION across seeds
    seeds = [0, 1, 2]
    obs, intr, false_pos = [], [], []
    for s in seeds:
        ens, lib, synth, world, names = build(s)
        true_t = {i for (i, a, b, w) in world.interactions}
        og, tot = gates_recovered(ens, world)
        hyper = recover_structure_interventional(ens, lib, synth, world.sensors, world.actuators)
        rec = {i for H, i, _ in hyper if len(H) == 2}
        obs.append(og); intr.append(len(rec & true_t)); false_pos.append(len(rec - true_t))
        print(f"  seed {s}: observational {og}/{tot}   interventional {len(rec & true_t)}/{tot}   "
              f"false+ {len(rec - true_t)}")
    check("recall: interventional >= observational on every seed",
          all(i >= o for i, o in zip(intr, obs)), f"obs={obs} intr={intr}")
    check("recall: interventional mean >= 4.5/5 (crosses the ~3/5 plateau)",
          float(np.mean(intr)) >= 4.5, f"mean interventional = {np.mean(intr):.2f}/5")
    check("precision: interventional emits ZERO false gates across seeds",
          sum(false_pos) == 0, f"total false positives = {sum(false_pos)}")

    # 4. NEGATIVE CONTROLS (seed 0): rejections prove the confirm can say "no"
    ens, lib, synth, world, names = build(0)
    # indices: actuators 0..gates, gates next (g0 at gates+1), Z last
    g0, g1 = world.sensors[0], world.sensors[1]
    controls = [(g0, 3, "wrong-actuator {g0,a3}"),         # a3 gates a deeper var, not reachable here
                (g1, 4, "wrong-actuator {g1,a4}"),
                (g1, 0, "a0-linear {g1,a0} (must fail the rest-contrast/DiD guard)")]
    rejected = []
    for src_g, act, lbl in controls:
        v = confirm_gate(src_g, act, synth, lib, world.sensors)
        rejected.append(v is None)
        print(f"  control {lbl}: {'REJECTED' if v is None else 'CONFIRMED(BAD) '+names[v.target]}")
    check("controls: all wrong-actuator / a0-linear candidates REJECTED",
          all(rejected), f"{sum(rejected)}/{len(rejected)} rejected")

    print("=" * 84)
    if FAILS:
        print(f"FAILED: {FAILS}")
        sys.exit(1)
    print("ALL CHECKS PASSED -- interventional confirmation breaks the plateau, world-graded, no false gates.")


if __name__ == "__main__":
    main()
