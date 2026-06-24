"""
Serious baselines on CONTROL (commander Order 3): does the causal advantage survive against
stronger curiosity explorers and a model-free learner, on the held-out goals?

  ensemble / hetero-disagreement / learning-progress  -- reward-free explorers -> frozen
        WeightedHeteroModel -> shared MPC (same as causal/prediction).
  random_shooting  -- NO pre-exploration model; random-shooting control directly on the goal world
        with only the post-goal budget (tests low-shot learning from scratch).
  oracle      -- true dynamics (upper bound).
"""
from __future__ import annotations

import numpy as np

import constructor_causal.capability_trial as ct
from .capability_world import CapabilityWorld, ACTUATORS, GOALS
from .curiosity_baselines import Ensemble, LPTracker
from .hetero import WeightedHeteroModel
from .macro import full_command, macro_vocabulary, rollout_states


def _setup():
    ct.PRODUCTS = [(a, s) for a in ACTUATORS for s in CapabilityWorld().sensors]


def _score(policy, primary, ens, lp, macro, states):
    S = primary.sensors
    if policy == "eig":
        return primary.seq_eig(states)
    if policy == "pred_error":
        return sum(np.sqrt(primary.sigma2(xc, i)) for xc, _ in states for i in S)
    if policy == "disagreement":
        return sum(sum(ens.disagreement(xc).values()) for xc, _ in states)
    if policy == "hetero_disagreement":
        return sum(d / primary.sigma2(xc, i)
                   for xc, _ in states for i, d in ens.disagreement(xc).items())
    if policy == "learning_progress":
        return lp.score(macro)
    return 0.0


def explore_baseline(world, policy, budget, rng):
    acts = world.actuators
    primary = WeightedHeteroModel(world.d, acts, hidden=world.hidden, interaction_pairs=ct.PRODUCTS,
                                  rng=np.random.default_rng(int(rng.integers(1 << 31))), bayes_head=True)
    ens = (Ensemble(world.d, acts, hidden=world.hidden, interaction_pairs=ct.PRODUCTS, K=4, rng=rng)
           if policy in ("disagreement", "hetero_disagreement") else None)
    lp = LPTracker(acts) if policy == "learning_progress" else None
    steps = 0
    while steps < budget:
        vocab = macro_vocabulary(acts, rng=rng)
        if policy == "random" or rng.random() < 0.1:
            macro = vocab[int(rng.integers(len(vocab)))]
        else:
            x0 = world.x.copy()
            sc = np.nan_to_num(np.array([_score(policy, primary, ens, lp, m, rollout_states(primary, x0, m, acts))
                                         for m in vocab]), nan=-1e18, posinf=1e18, neginf=-1e18)
            macro = vocab[int(np.argmax(sc))]
        for cmd, k in macro:
            fc = full_command(cmd, acts)
            for _ in range(int(k)):
                if steps >= budget:
                    break
                xc = world.x.copy()
                for j, v in fc.items():
                    xc[j] = v
                mu = primary.predict_next(xc)[0] if lp is not None else None
                xn = world.step(fc)
                if lp is not None:
                    lp.observe(xc, float(sum(abs(xn[i] - mu[i]) for i in primary.sensors)))
                primary.update(xc, xn)
                if ens is not None:
                    ens.update(xc, xn)
                steps += 1
    return primary


def random_shooting_control(world, goal, budget, rng, n_cand=5):
    """No pre-built model: random-shooting constant actions on the real goal world."""
    target, (lo, hi), _ = goal["target"], goal["band"], None
    steps, best_a, best_v = 0, None, -1e9
    per = max(budget // (n_cand + 1), 3)
    for _ in range(n_cand):
        a = {j: float(rng.choice([-2.0, 0.0, 2.0])) for j in ACTUATORS}
        for _ in range(per):
            if steps >= budget:
                break
            world.step(a); steps += 1
            if world.x[target] >= lo:
                return dict(reached=True, steps=steps)
        if world.x[target] > best_v:
            best_v, best_a = float(world.x[target]), a
    while steps < budget and best_a is not None:
        world.step(best_a); steps += 1
        if world.x[target] >= lo:
            return dict(reached=True, steps=steps)
    return dict(reached=False, steps=budget)


def run(agent, goal_name, seed, eb=300, gb=40):
    rng = np.random.default_rng(seed)
    gw = CapabilityWorld(np.random.default_rng(seed + 999)); gw.reset()
    if agent == "random_shooting":
        return random_shooting_control(gw, GOALS[goal_name], gb, rng)
    ew = CapabilityWorld(np.random.default_rng(seed)); ew.reset()
    if agent == "oracle":
        model = ct.OracleModel(ew)
    elif agent in ("causal", "prediction"):
        model = ct.explore(ew, {"causal": "eig", "prediction": "pred_error"}[agent], eb, rng)
    else:
        model = explore_baseline(ew, agent, eb, rng)
    return ct.mpc_control(gw, model, GOALS[goal_name], gb, ACTUATORS, rng)


def main(goal="deep_chain", seeds=range(8)):
    _setup()
    print("=" * 72)
    print(f"SERIOUS BASELINES on CONTROL -- goal={goal}  ({len(list(seeds))} seeds)")
    print("=" * 72)
    agents = ["causal", "prediction", "disagreement", "hetero_disagreement",
              "learning_progress", "random_shooting", "oracle"]
    res = {a: [] for a in agents}
    for s in seeds:
        for a in agents:
            res[a].append(run(a, goal, s)["reached"])
    print(f"  {'agent':>20} {'zero-shot success':>18}")
    for a in agents:
        print(f"  {a:>20} {100*np.mean(res[a]):>15.0f}%")
    print("=" * 72)
    return res


if __name__ == "__main__":
    import sys
    g = sys.argv[1] if len(sys.argv) > 1 else "deep_chain"
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(goal=g, seeds=range(ns))
