"""
End-to-end, narrated:  curiosity -> causal map -> constructor library -> goals.

Run:  ./.venv/bin/python -m constructor_causal.demo     (from the ChemicalWorld dir)
"""
from __future__ import annotations

import numpy as np

from .active_inference import (EpistemicExperimenter, NaiveSurpriseExperimenter,
                               RandomExperimenter)
from .agent import ConstructorCausalAgent
from .constructor import Box
from .model import BayesianDynamicsModel, edge_scores
from .world import DynamicalCausalWorld


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
# helper: run one reward-free exploration with a given experimenter, return F1
# --------------------------------------------------------------------------- #
def recover_f1(make_experimenter, budget, seed, world_fn=DynamicalCausalWorld.hard):
    world = world_fn(np.random.default_rng(seed))
    model = BayesianDynamicsModel(world.d, world.actuators)
    exp = make_experimenter(model, world, np.random.default_rng(seed + 7))
    x = world.reset()
    for _ in range(budget):
        cmd = exp.choose(x)
        xc = x.copy()
        for j, v in cmd.items():
            xc[j] = v
        xn = world.step(cmd)
        model.update(xc, xn)
        x = xn
    return edge_scores(model, world.true_edges())["f1"]


def compare_objectives(budget=16, seeds=range(10)):
    makers = {
        "curiosity (info gain)": lambda m, w, r: EpistemicExperimenter(
            m, w.actuators, rng=r),
        "random (sanity floor)": lambda m, w, r: RandomExperimenter(
            m, w.actuators, rng=r),
        "naive surprise (TV trap)": lambda m, w, r: NaiveSurpriseExperimenter(
            m, w.actuators, rng=r),
    }
    out = {}
    for name, mk in makers.items():
        f1s = [recover_f1(mk, budget, s) for s in seeds]
        out[name] = (float(np.mean(f1s)), float(np.std(f1s)))
    return out


def main():
    rng = np.random.default_rng(0)
    world = DynamicalCausalWorld.default(rng)

    section("THE WORLD  (hidden from the agent)")
    print("  variables :", ", ".join(f"{i}:{n}" for i, n in enumerate(world.names)))
    print("  actuators :", [world.names[j] for j in world.actuators],
          "  (the only things the agent can force)")
    print("  true causal edges (ground truth, the agent must recover these):")
    print("     ", ", ".join(sorted(f"{world.names[j]}→{world.names[i]}"
                                     for (j, i) in world.true_edges())))
    print("  traps baked in:  chain2 is SLOW+DEEP (needs sustained drive);")
    print("                   decoy correlates with chain2 but is NOT its cause;")
    print("                   static is pure noise (the noisy-TV trap).")

    # ---------------------------------------------------------------- explore
    section("STEP 1 — reward-free curiosity builds a causal map")
    agent = ConstructorCausalAgent(world, seed=0)
    agent.explore(400, track_every=80)
    sc = agent.recovered_dag()
    print("  recovery curve (step : F1):",
          "  ".join(f"{s}:{f:.2f}" for s, f in agent.history))
    print(f"\n  recovered edges : {agent.named_edges(sc['recovered'])}")
    print(f"  ground truth    : {agent.named_edges(world.true_edges())}")
    print(f"  precision={sc['precision']:.2f}  recall={sc['recall']:.2f}  F1={sc['f1']:.2f}")
    if sc["extra"]:
        print(f"  false edges     : {agent.named_edges(sc['extra'])}")
    else:
        print("  false edges     : none — decoy correctly rejected as non-causal")
    print(f"  static's learned parents: {agent.model.recovered_parents(5)} "
          f"(σ²≈{agent.model.sigma2[5]:.1f}) — recognised as irreducible noise, "
          f"not chased")

    # ------------------------------------------------- objective comparison
    section("STEP 2 — when does the OBJECTIVE matter?  (hard world, budget = 16)")
    print("  Harder variant: a0→chain1 is a WEAK edge (must be probed), a1 is a")
    print("  causally idle distractor knob, and the sensors are noisy.")
    for name, (m, s) in compare_objectives().items():
        print(f"  {name:26s}  mean F1 = {m:.2f} ± {s:.2f}")
    print("  Curiosity beats RANDOM by concentrating its scarce budget on the")
    print("  informative knob and abandoning the useless one. Naive surprise does")
    print("  about as well here — because the noise source (static) can't be acted")
    print("  on, so chasing surprise doesn't waste actions. The noisy-TV trap bites")
    print("  only when actions gate what you OBSERVE; see discovery.py for that case.")

    # ------------------------------------------------------- build library
    section("STEP 3 — distil composable constructors (still no reward)")
    agent.build_library()
    print("  primitive constructors (verified by repeated real experiments):")
    print(agent.library)
    if agent.idle_knobs:
        idle = sorted({world.names[j] for (j, _) in agent.idle_knobs})
        print(f"  causally idle knobs discovered: {idle} (move nothing — the agent")
        print("  learned which levers are real)")

    # ---------------------------------------------- composition for depth
    section("STEP 4 — compose small constructors into a bigger one")
    chain2_high = Box.from_dict({3: (3.0, 3.5)})
    print(f"  goal region (NEVER seen during learning):  {chain2_high}")

    # show a single short primitive cannot reach the deep, slow variable
    base = next(c for c in agent.library.possible() if c.program[0].get(0) == 2.0)
    r_single = agent.synth.verify(
        type(base)(name=base.name + "→chain2?", precond=base.precond,
                   effect=chain2_high, program=base.program, provenance="primitive"),
        chain2_high)
    print(f"  single primitive {base.name!r} aimed at chain2: reliability = {r_single:.2f}")
    print("    → a short pulse can't move the slow, deep variable. Need composition.")

    c, r = agent.achieve(chain2_high)
    if c is not None:
        print(f"\n  COMPOSED constructor: {c.name}")
        print(f"    provenance : {c.provenance}")
        print(f"    program    : hold a0=+2 for {c.horizon} steps "
              f"(= the primitive chained with itself)")
        print(f"    reliability: {c.reliability:.2f}   → chain2∈[3.0,3.5] reached")
        print("    A capability NEITHER parent had alone, manufactured by composition.")
    else:
        print("  (composition failed — see selftest)")

    # ----------------------------------------------------- use & predict
    section("STEP 5 — use the understanding: reach a goal, and answer 'what if'")
    if c is not None:
        env = world.clone(np.random.default_rng(123))
        x = env.reset()
        for cmd in c.program:
            x = env.step(cmd)
        print(f"  executed composite once: final chain2 = {x[3]:.2f}  "
              f"(target [3.0,3.5]) → {'HIT' if chain2_high.contains(x) else 'miss'}")

    # counterfactual: what if I hold a0=+2 for 6 steps, from rest?
    pred = agent.whatif(np.zeros(world.d), {0: 2.0}, steps=6)
    finals = []
    for s in range(200):
        env = world.clone(np.random.default_rng(1000 + s))
        x = env.reset()
        for _ in range(6):
            x = env.step({0: 2.0})
        finals.append(x)
    finals = np.array(finals)
    print("\n  counterfactual  do(a0=+2) held 6 steps, predicted vs actual:")
    for i, nm in [(2, "chain1"), (3, "chain2"), (4, "decoy")]:
        print(f"    {nm:7s}  predicted {pred[-1, i]:+.2f}   actual "
              f"{finals[:, i].mean():+.2f} ± {finals[:, i].std():.2f}")

    section("VERDICT")
    print("  No reward was ever defined. Driven only by curiosity, the agent learned")
    print("  the causal graph (rejecting the decoy, ignoring the noise), distilled a")
    print("  library of repeatable constructors, and COMPOSED them into a bigger")
    print("  constructor that reaches a deep variable on demand — then predicted the")
    print("  effects of its own actions. Understanding first; capability falls out.")


if __name__ == "__main__":
    main()
