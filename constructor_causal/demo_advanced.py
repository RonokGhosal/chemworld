"""
Advanced demo: the three hard cases the basic demo flagged as open.

  1. HIDDEN CONFOUNDER  -- a common cause makes two variables move together with no
     edge between them. Passive observation infers a spurious edge; only INTERVENTION
     reveals there is none.
  2. DISTINCT-CONSTRUCTOR COMPOSITION -- a gate (Z := gate·a1, gate←a0). No single
     knob, and no self-chaining of one knob, can move Z. The agent must compose two
     DIFFERENT constructors in order: open the gate, then drive the gated variable.
     a1 looks idle until the gate is open -- its constructor is context-dependent.
  3. OBSERVATION-GATING -- when the action chooses WHAT to observe, the noisy-TV trap
     finally bites: naive surprise stares at noise, the info-gain objective doesn't.

Run:  ./.venv/bin/python -m constructor_causal.demo_advanced
"""
from __future__ import annotations

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import Box
from .observation_gating import compare as gating_compare
from .world import DynamicalCausalWorld


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def observed_pairs(world):
    obs = world.observed
    return [(a, b) for i, a in enumerate(obs) for b in obs[i + 1:]]


def edge_names(world, E):
    return sorted(f"{world.names[j]}→{world.names[i]}" for (j, i) in E)


# --------------------------------------------------------------------------- #
def demo_confounder():
    section("CASE 1 — a HIDDEN confounder: intervention vs passive observation")
    world = DynamicalCausalWorld.confounded(np.random.default_rng(0))
    print("  Hidden H drives both S1 and S2.  There is NO S1→S2 edge.")
    print("  S1 is forceable; S2 is only observed.\n")

    passive = ConstructorCausalAgent(world.clone(np.random.default_rng(0)), seed=0,
                                     experimenter="passive")
    actor = ConstructorCausalAgent(world.clone(np.random.default_rng(0)), seed=0,
                                   experimenter="epistemic")
    passive.explore(300)
    actor.explore(300)
    wp = passive.model.weight(1, 0)        # S2's inferred dependence on S1
    wa = actor.model.weight(1, 0)
    print(f"  PASSIVE observer : S2←S1 weight = {wp:+.2f}   edges = "
          f"{edge_names(world, passive.model.recovered_edges())}")
    print(f"     → fooled: S1 and S2 co-vary through H, so it 'sees' a link.")
    print(f"  INTERVENING agent: S2←S1 weight = {wa:+.2f}   edges = "
          f"{edge_names(world, actor.model.recovered_edges())}")
    print(f"     → forcing S1 decorrelates it from H; the spurious weight collapses.")
    print("  Only intervention tells a shared hidden cause from a real edge.")


def demo_gate():
    section("CASE 2 — DISTINCT-constructor composition through a gate")
    world = DynamicalCausalWorld.gated(np.random.default_rng(0))
    print("  Z := 0.3·Z + 0.5·gate·a1 ,  gate ← a0.  Neither knob alone moves Z;")
    print("  nor does chaining one knob with itself. Z needs a0 THEN a1, in order.\n")

    agent = ConstructorCausalAgent(world, seed=0, interaction_pairs=observed_pairs(world))
    agent.explore(400)
    lin = edge_names(world, agent.model.recovered_edges())
    inter = sorted(f"{world.names[a]}·{world.names[b]}→{world.names[i]}"
                   for ((a, b), i) in agent.model.recovered_interactions())
    print(f"  recovered linear edges : {lin}")
    print(f"  recovered interactions : {inter}   ← the GATE, discovered")

    agent.build_library(setpoints=(-2.0, 2.0), conditional=True)
    prims = [c for c in agent.library.possible() if "conditional" not in c.provenance]
    print(f"\n  primitive constructors (work from rest):")
    for c in prims:
        print("    " + str(c))
    print(f"  → a1 is idle from rest (it moves nothing until the gate is open).")
    print(f"\n  CONDITIONAL constructors (unlocked by another constructor):")
    for c in agent.conditionals:
        print("    " + str(c))

    Zhigh = Box.from_dict({3: (2.5, 3.6)})
    print(f"\n  goal: drive Z into {Zhigh}")

    # show a0 alone (even self-chained) cannot reach Z
    a0c = next(c for c in prims if c.program[0].get(0) == 2.0)
    from .constructor import Constructor, compose
    selfchain = a0c
    for _ in range(2):
        selfchain = compose(selfchain, a0c)
    r_a0 = agent.synth.verify(
        Constructor(name="a0^3→Z?", precond=a0c.precond, effect=Zhigh,
                    program=selfchain.program, provenance="self-chain"), Zhigh)
    print(f"  hold a0 for {len(selfchain.program)} steps, aimed at Z: reliability = {r_a0:.2f}")
    print(f"    → opening the gate wider does nothing; a1 was never driven.")

    c, r = agent.achieve(Zhigh)
    print(f"\n  COMPOSED (distinct) constructor: {c.name}")
    print(f"    provenance : {c.provenance}")
    print(f"    = open-the-gate (a0)  ≫  drive-the-gated-variable (a1 | gate)")
    print(f"    reliability: {r:.2f}   → Z reached by composing TWO different skills.")
    env = world.clone(np.random.default_rng(7)); x = env.reset()
    for cmd in c.program:
        x = env.step(cmd)
    print(f"  executed once: final Z = {x[3]:.2f}  → {'HIT' if Zhigh.contains(x) else 'miss'}")


def demo_observation_gating():
    section("CASE 3 — observation-gating: the noisy-TV trap finally bites")
    print("  The agent forces the knob but may record only ONE channel per step.")
    print("  chain1 has a learnable law; static is parent-free large noise.\n")
    for obj, m in gating_compare().items():
        label = "EIG (curiosity)" if obj == "eig" else "naive surprise"
        print(f"  {label:16s} watched static {100*m['frac_static']:4.0f}% of looks   "
              f"chain1 law err={m['weight_err']:.2f}   learned chain1: {100*m['learned']:3.0f}%")
    print("\n  Naive surprise is hypnotised by the noise and never reliably learns the")
    print("  law. Info-gain spends two looks pinning the noise, then learns chain1.")


def main():
    demo_confounder()
    demo_gate()
    demo_observation_gating()
    section("VERDICT")
    print("  Three things the basic system couldn't yet do, now done — all reward-free:")
    print("   1. reject a spurious link from a HIDDEN common cause (via intervention),")
    print("   2. compose two DISTINCT constructors in order to crack a gate that no")
    print("      single skill (or repeated single skill) can,")
    print("   3. resist the noisy-TV trap once actions gate what is observed.")


if __name__ == "__main__":
    main()
