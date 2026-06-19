"""
Falsifiable checks for the knowledge-prior fusion (CI-safe — no live LLM; a perfect
prior and ground-truth-as-LLM answers stand in for a flawless expert).

Run:  ./.venv/bin/python -m constructor_causal.selftest_prior
"""
from __future__ import annotations

import sys

import numpy as np

from .agent import ConstructorCausalAgent
from .world import DynamicalCausalWorld
from .prior import CausalPrior, world_to_B, raw_orientation_accuracy
from .semantic_worlds import heater_world, tank_world, question_list, ground_truth_answers

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _is_dag(edges, d):
    """Kahn over (source,target) edges."""
    indeg = {i: 0 for i in range(d)}
    for (s, t) in edges:
        indeg[t] += 1
    ready = [i for i in indeg if indeg[i] == 0]
    seen = 0
    while ready:
        n = ready.pop(); seen += 1
        for (s, t) in edges:
            if s == n:
                indeg[t] -= 1
                if indeg[t] == 0:
                    ready.append(t)
    return seen == d


def f1_at(make_agent, prior, budget):
    agent = make_agent()
    if budget:
        agent.explore(budget)
    return (prior.fused_scores(agent.model, agent.world.true_edges())["f1"] if prior
            else agent.recovered_dag()["f1"])


def main():
    print("=" * 78 + "\nconstructor_causal — PRIOR-FUSION selftest\n" + "=" * 78)

    print("\n-- the convention adapter (transpose) is correct --")
    B = world_to_B(DynamicalCausalWorld.default(np.random.default_rng(0)))
    check("world_to_B maps a0->chain1 to B[0,2]=1, B[2,0]=0",
          B[0, 2] == 1 and B[2, 0] == 0)

    print("\n-- a PERFECT prior gives zero-experiment correctness --")
    for name, W in [("heater", heater_world), ("tank", tank_world)]:
        prior = CausalPrior.from_true_edges(W(np.random.default_rng(0)))
        f0 = f1_at(lambda: ConstructorCausalAgent(W(np.random.default_rng(0)), seed=0, prior=prior),
                   prior, 0)
        check(f"{name}: F1 = 1.0 at budget 0 with a perfect prior", f0 >= 0.999, f"F1={f0}")
        f0_no = f1_at(lambda: ConstructorCausalAgent(W(np.random.default_rng(0)), seed=0), None, 0)
        check(f"{name}: no-prior F1 at budget 0 is far lower (needs experiments)",
              f0_no < 0.5, f"F1={f0_no}")

    print("\n-- a WRONG prior is OVERRIDDEN by interventions (soft-prior safety) --")
    for name, W in [("heater", heater_world)]:
        bad = CausalPrior.from_oracle(W(np.random.default_rng(0)), accuracy=0.0, abstain=0.0,
                                      rng=np.random.default_rng(1))
        f_bad0 = f1_at(lambda: ConstructorCausalAgent(W(np.random.default_rng(0)), seed=0, prior=bad),
                       bad, 0)
        f_bad = f1_at(lambda: ConstructorCausalAgent(W(np.random.default_rng(0)), seed=0, prior=bad),
                      bad, 40)
        check(f"{name}: an accuracy-0 prior starts wrong (low F1 at budget 0)",
              f_bad0 < 0.5, f"F1={f_bad0}")
        check(f"{name}: interventions override it back to F1 = 1.0 by budget 40",
              f_bad >= 0.999, f"F1={f_bad}")

    print("\n-- the semantic worlds are valid, fully-covered per-step DAGs --")
    for name, W in [("heater", heater_world), ("tank", tank_world)]:
        w = W(np.random.default_rng(0))
        check(f"{name}: true_edges form a DAG (acyclic per step)", _is_dag(w.true_edges(), w.d))
        n_edges = len(w.true_edges())
        check(f"{name}: question_list covers every true adjacency",
              len(question_list(w)) == n_edges, f"{len(question_list(w))} q / {n_edges} edges")

    print("\n-- the real-LLM answer path works end-to-end (ground-truth stand-in) --")
    for name, W in [("heater", heater_world), ("tank", tank_world)]:
        w = W(np.random.default_rng(0))
        gt = ground_truth_answers(w)
        acc = raw_orientation_accuracy(w, gt)
        check(f"{name}: ground-truth answers score 100% orientation accuracy",
              acc["accuracy"] >= 0.999, f"{acc['correct']}/{acc['total']}")
        prior = CausalPrior.from_llm_answers(w, gt)
        f0 = f1_at(lambda: ConstructorCausalAgent(W(np.random.default_rng(0)), seed=0, prior=prior),
                   prior, 0)
        check(f"{name}: from_llm_answers prior gives F1 = 1.0 at budget 0", f0 >= 0.999, f"F1={f0}")

    n_pass = sum(CHECKS)
    print("\n" + "=" * 78 + f"\n{n_pass}/{len(CHECKS)} checks passed\n" + "=" * 78)
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
