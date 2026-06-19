"""
Falsifiable checks for the advanced cases: hidden confounder, distinct-constructor
composition through a gate, and the observation-gating noisy-TV trap.

Run:  ./.venv/bin/python -m constructor_causal.selftest_advanced
"""
from __future__ import annotations

import sys

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import POSSIBLE_TAU, Box, Constructor, compose
from .observation_gating import compare as gating_compare
from .world import DynamicalCausalWorld

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def observed_pairs(world):
    obs = world.observed
    return [(a, b) for i, a in enumerate(obs) for b in obs[i + 1:]]


def main():
    print("=" * 78 + "\nconstructor_causal — ADVANCED selftest\n" + "=" * 78)

    # ===== 1. HIDDEN CONFOUNDER ============================================
    print("\n-- hidden confounder (H → S1, S2; no S1→S2) --")
    world = DynamicalCausalWorld.confounded(np.random.default_rng(0))
    wp, wa, ep, ea = [], [], [], []
    for s in range(5):
        pa = ConstructorCausalAgent(world.clone(np.random.default_rng(s)), seed=s,
                                    experimenter="passive"); pa.explore(300)
        ac = ConstructorCausalAgent(world.clone(np.random.default_rng(s)), seed=s,
                                    experimenter="epistemic"); ac.explore(300)
        wp.append(abs(pa.model.weight(1, 0))); wa.append(abs(ac.model.weight(1, 0)))
        ep.append((0, 1) in pa.model.recovered_edges())
        ea.append((0, 1) in ac.model.recovered_edges())
    wp_m, wa_m = float(np.mean(wp)), float(np.mean(wa))
    check("passive observer infers a strong spurious S1→S2 weight (>0.4)",
          wp_m > 0.4, f"|w_passive|={wp_m:.2f}")
    check("intervention collapses that weight (<0.2)", wa_m < 0.2, f"|w_interv|={wa_m:.2f}")
    check("intervention << passive (debiased by >0.3)", wp_m - wa_m > 0.3,
          f"{wp_m:.2f} vs {wa_m:.2f}")
    check("passive infers the spurious EDGE; intervention does not",
          np.mean(ep) >= 0.8 and np.mean(ea) <= 0.2,
          f"passive {100*np.mean(ep):.0f}% / interv {100*np.mean(ea):.0f}% of seeds")

    # ===== 2. DISTINCT-CONSTRUCTOR COMPOSITION (gate) ======================
    print("\n-- distinct-constructor composition through a gate --")
    gw = DynamicalCausalWorld.gated(np.random.default_rng(0))
    ag = ConstructorCausalAgent(gw, seed=0, interaction_pairs=observed_pairs(gw))
    ag.explore(400)
    inter = ag.model.recovered_interactions()
    check("discovers the GATE interaction a1·gate→Z",
          ((1, 2), 3) in inter, f"interactions={sorted(inter)}")
    check("discovers linear edge a0→gate", (0, 2) in ag.model.recovered_edges(),
          f"linear={sorted(ag.model.recovered_edges())}")

    ag.build_library(setpoints=(-2.0, 2.0), conditional=True)
    prims = [c for c in ag.library.possible() if "conditional" not in c.provenance]
    check("a1 is idle from rest (no primitive moves Z or anything via a1)",
          all(3 not in c.effect.vars() for c in prims),
          f"primitive effects={[sorted(c.effect.vars()) for c in prims]}")
    check("a CONDITIONAL constructor (a1 | gate) that controls Z is discovered",
          any(3 in c.effect.vars() for c in ag.conditionals),
          f"conditionals={[c.name for c in ag.conditionals]}")

    Zhigh = Box.from_dict({3: (2.5, 3.6)})
    a0c = next(c for c in prims if c.program[0].get(0) == 2.0)
    sc = a0c
    for _ in range(2):
        sc = compose(sc, a0c)
    r_self = ag.synth.verify(Constructor(name="a0^3→Z", precond=a0c.precond,
                             effect=Zhigh, program=sc.program, provenance="self"), Zhigh)
    check("one knob alone (even self-chained) CANNOT reach Z", r_self < 0.5,
          f"reliability={r_self:.2f}")
    c, r = ag.achieve(Zhigh)
    check("DISTINCT composition reaches Z reliably (≥ τ)",
          c is not None and r >= POSSIBLE_TAU, f"reliability={r:.2f}")
    uses_both = c is not None and "x0" in c.provenance and "x1" in c.provenance
    check("the reaching constructor composes BOTH knobs (a0 ≫ a1|gate)", uses_both,
          f"provenance={c.provenance if c else None}")

    # ===== 3. OBSERVATION-GATING (noisy-TV trap bites) =====================
    print("\n-- observation-gating: noisy-TV trap --")
    g = gating_compare(budget=30, seeds=range(12))
    eig, nai = g["eig"], g["surprise"]
    check("naive surprise wastes far more budget on the noise channel",
          nai["frac_static"] > eig["frac_static"] + 0.2,
          f"naive {100*nai['frac_static']:.0f}% vs eig {100*eig['frac_static']:.0f}%")
    check("info-gain learns chain1's law better than naive surprise",
          eig["weight_err"] < nai["weight_err"] - 0.03,
          f"eig err={eig['weight_err']:.2f} vs naive err={nai['weight_err']:.2f}")
    check("info-gain learns chain1 more reliably than naive",
          eig["learned"] >= nai["learned"] + 0.1,
          f"eig {100*eig['learned']:.0f}% vs naive {100*nai['learned']:.0f}%")

    n_pass = sum(ok for _, ok, _ in CHECKS)
    print("\n" + "=" * 78 + f"\n{n_pass}/{len(CHECKS)} checks passed\n" + "=" * 78)
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
