"""
Knowledge-prior fusion — a causal_dag RAG/LLM prior seeds the constructor_causal agent.

Honest value (the agent already gets edge DIRECTION for free by intervening, so the
prior fills no Markov-equivalence gap here):
  * ZERO-EXPERIMENT correctness: a good prior hands over a correct causal graph before
    any action (F1 high at budget 0); active intervention alone starts at F1 0.
  * OVERRIDE safety: interventions are ground truth — a WRONG prior is corrected (its
    reverse-recovered edges drop it), so a bad prior is never permanent.
  * On efficiently-recoverable worlds, active inference is already so fast that the
    prior's benefit is this head-start + safety, not "fewer experiments" — stated honestly.

Two paths:
  (a) mechanism (simulated oracle, abstract + semantic worlds): F1-vs-budget by accuracy.
  (b) real-LLM (semantic worlds): a Claude subagent orients heater/tank from NAMES alone.

Run:  ./.venv/bin/python -m constructor_causal.demo_prior              # mechanism demo
      ./.venv/bin/python -m constructor_causal.demo_prior prepare      # print LLM questions
      ./.venv/bin/python -m constructor_causal.demo_prior score '<json>'   # score LLM answers
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .agent import ConstructorCausalAgent
from .world import DynamicalCausalWorld
from .prior import CausalPrior, world_to_B, raw_orientation_accuracy
from .semantic_worlds import (heater_world, tank_world, SEMANTIC, question_list,
                              ground_truth_answers)

STATE = "constructor_causal/_prior_state.json"


def _f1(agent, prior):
    return (prior.fused_scores(agent.model, agent.world.true_edges())["f1"] if prior
            else agent.recovered_dag()["f1"])


def f1_at_budgets(make_agent, prior, budgets=(0, 8, 20, 40)):
    agent = make_agent()
    out, cum = [], 0
    for b in budgets:
        if b > cum:
            agent.explore(b - cum); cum = b
        out.append(round(_f1(agent, prior), 2))
    return out


# =========================================================================== (a)
def mechanism_demo():
    print("=" * 80)
    print("A.  MECHANISM (simulated oracle) — F1 vs experiment budget, by prior accuracy")
    print("=" * 80)
    budgets = (0, 8, 20, 40)
    for name, W in [("heater (semantic)", heater_world), ("tank (semantic)", tank_world),
                    ("wide(k=5) (abstract)", lambda r: DynamicalCausalWorld.wide(k=5, rng=r))]:
        print(f"\n[{name}]   F1 at experiment budgets {budgets}")
        no = f1_at_budgets(lambda: ConstructorCausalAgent(W(np.random.default_rng(0)), seed=0),
                           None, budgets)
        print(f"    {'no prior':>16}: {no}")
        for acc in (1.0, 0.7, 0.0):
            prior = CausalPrior.from_oracle(W(np.random.default_rng(0)), accuracy=acc,
                                            abstain=0.0, rng=np.random.default_rng(1))
            row = f1_at_budgets(
                lambda: ConstructorCausalAgent(W(np.random.default_rng(0)), seed=0, prior=prior),
                prior, budgets)
            print(f"    {('prior acc=%.1f' % acc):>16}: {row}")
    print("\n  => a perfect prior gives F1=1.0 at ZERO experiments (knowledge, not data).")
    print("     A WRONG prior (acc 0.0) starts low but interventions OVERRIDE it back to 1.0.")
    print("     Active inference is efficient, so the win is the head-start + safety.")


# =========================================================================== (b)
def prepare():
    print("=" * 80)
    print("ORIENTATION QUESTIONS FOR A REAL LLM (semantic worlds; it sees only NAMES)")
    print("=" * 80)
    state = {}
    for key, W in SEMANTIC.items():
        w = W(np.random.default_rng(0))
        qs = question_list(w)
        state[key] = {"names": list(w.names)}
        print(f"\n[{key}]  variables: {', '.join(w.names)}")
        print("  For each pair, which variable is the CAUSE?")
        for q in qs:
            print(f"    {q}")
    with open(STATE, "w") as f:
        json.dump(state, f)
    print(f"\n[wrote {STATE}]   answer as JSON: "
          '{"heater": {"heater_power -- room_temp": "heater_power", ...}, "tank": {...}}')


def run_with_answers(answers_by_world: dict):
    print("=" * 80)
    print("REAL-LLM PRIOR — orient semantic worlds from knowledge, then verify by acting")
    print("=" * 80)
    budgets = (0, 8, 20, 40)
    for key, W in SEMANTIC.items():
        w = W(np.random.default_rng(0))
        ans = answers_by_world.get(key, {})
        acc = raw_orientation_accuracy(w, ans)
        prior = CausalPrior.from_llm_answers(w, ans)
        no = f1_at_budgets(lambda: ConstructorCausalAgent(W(np.random.default_rng(0)), seed=0),
                           None, budgets)
        yes = f1_at_budgets(
            lambda: ConstructorCausalAgent(W(np.random.default_rng(0)), seed=0, prior=prior),
            prior, budgets)
        print(f"\n[{key}] LLM raw orientation accuracy: {100*acc['accuracy']:.0f}% "
              f"({acc['correct']}/{acc['total']})")
        print(f"    F1 at budgets {budgets}:  no prior {no}   with LLM prior {yes}")
    print("\n  => the LLM prior hands the agent a correct graph at zero experiments; the")
    print("     agent's interventions then verify it (and would override any wrong edge).")


def main(argv):
    if argv and argv[0] == "prepare":
        prepare()
    elif argv and argv[0] == "score":
        run_with_answers(json.loads(argv[1]))
    else:
        mechanism_demo()


if __name__ == "__main__":
    main(sys.argv[1:])
