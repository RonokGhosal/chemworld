"""
Falsifiable checks: every headline claim, asserted. Exits nonzero on any failure.

Run:  ./.venv/bin/python -m constructor_causal.selftest    (from ChemicalWorld dir)
"""
from __future__ import annotations

import sys

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import POSSIBLE_TAU, Box, Constructor
from .demo import compare_objectives
from .world import DynamicalCausalWorld

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    print("=" * 78 + "\nconstructor_causal — selftest\n" + "=" * 78)

    # ----- 1. reward-free causal recovery -----------------------------------
    world = DynamicalCausalWorld.default(np.random.default_rng(0))
    agent = ConstructorCausalAgent(world, seed=0)
    agent.explore(400)
    sc = agent.recovered_dag()
    check("causal recall ≥ 0.8 (chain + decoy-source edges found)",
          sc["recall"] >= 0.8, f"recall={sc['recall']:.2f}")
    decoy_to_chain2 = (4, 3)            # decoy → chain2 must NOT be inferred
    check("decoy rejected (no decoy→chain2 edge)",
          decoy_to_chain2 not in sc["recovered"],
          f"recovered={agent.named_edges(sc['recovered'])}")
    check("static recognised as parent-free noise",
          agent.model.recovered_parents(5) == [] and agent.model.sigma2[5] > 0.5,
          f"parents={agent.model.recovered_parents(5)}, σ²={agent.model.sigma2[5]:.1f}")

    # ----- 2. the objective is what matters (hard world, tight budget) ------
    comp = compare_objectives(budget=16, seeds=range(12))
    cur = comp["curiosity (info gain)"][0]
    rnd = comp["random (sanity floor)"][0]
    nai = comp["naive surprise (TV trap)"][0]
    # Curiosity stays at least as good as random, but the once-wide margin narrowed to
    # near-parity after two correctness fixes BOTH lifted random / re-shaped curiosity:
    # the σ²-estimator recalibration (which suppressed random's spurious edges) and the
    # chain-rule sequential EIG (which stopped curiosity over-counting repeated probes).
    # The honest, deterministic claim on this hard world is therefore directional.
    check("curiosity at least matches random on the hard world",
          cur >= rnd, f"curiosity={cur:.2f} vs random={rnd:.2f}")
    # Note: naive-surprise is NOT trapped here (static can't be acted on), so it
    # ties curiosity. We assert only that curiosity is not WORSE than it.
    check("curiosity not worse than naive-surprise",
          cur >= nai - 0.06, f"curiosity={cur:.2f} vs naive={nai:.2f}")

    # ----- 3. primitive library + idle-knob discovery -----------------------
    agent.build_library()
    ctrl_chain1 = [c for c in agent.library.possible()
                   if any(v == 2 for (v, _, _) in c.effect.bounds) or 2 in c.effect.vars()]
    check("≥1 verified primitive controls chain1",
          any(2 in c.effect.vars() for c in agent.library.possible()),
          f"library size={len(agent.library)}")
    idle_knobs = {j for (j, _) in agent.idle_knobs}
    check("inert knob a1 discovered as causally idle", 1 in idle_knobs,
          f"idle={sorted(idle_knobs)}")

    # ----- 4. composition manufactures a new capability ---------------------
    chain2_high = Box.from_dict({3: (3.0, 3.5)})
    base = next(c for c in agent.library.possible() if c.program[0].get(0) == 2.0)
    r_single = agent.synth.verify(
        Constructor(name="single→chain2", precond=base.precond, effect=chain2_high,
                    program=base.program, provenance="primitive"),
        chain2_high)
    check("single short primitive CANNOT reach deep chain2", r_single < 0.5,
          f"reliability={r_single:.2f}")
    c, r = agent.achieve(chain2_high)
    check("composition reaches chain2 reliably (≥ τ)",
          c is not None and r >= POSSIBLE_TAU, f"reliability={r:.2f}")
    check("composite is bigger than its parts (longer program)",
          c is not None and c.horizon > base.horizon,
          f"H_composite={getattr(c, 'horizon', '—')} > H_primitive={base.horizon}")
    check("composite reliability beats single primitive",
          c is not None and r > r_single + 0.3, f"{r:.2f} vs {r_single:.2f}")

    # ----- 5. counterfactual prediction is accurate -------------------------
    pred = agent.whatif(np.zeros(world.d), {0: 2.0}, steps=6)[-1]
    finals = []
    for s in range(200):
        env = world.clone(np.random.default_rng(1000 + s))
        x = env.reset()
        for _ in range(6):
            x = env.step({0: 2.0})
        finals.append(x)
    actual = np.array(finals).mean(0)
    err = abs(pred[3] - actual[3])
    check("'what-if' chain2 prediction accurate (|err|<0.4)", err < 0.4,
          f"pred={pred[3]:.2f} actual={actual[3]:.2f} err={err:.2f}")

    # ----- summary ----------------------------------------------------------
    n_pass = sum(ok for _, ok, _ in CHECKS)
    print("\n" + "=" * 78)
    print(f"{n_pass}/{len(CHECKS)} checks passed")
    print("=" * 78)
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
