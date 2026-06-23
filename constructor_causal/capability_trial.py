"""
THE CAPABILITY-PAYOFF TRIAL (commander's decisive experiment): after REWARD-FREE exploration,
does causal understanding turn into CONTROL on HELD-OUT goals?

Protocol per agent:
  1. EXPLORE the world reward-free with the agent's own objective (fixed budget); FREEZE model.
  2. GOAL PHASE on a fresh world: a shared model-predictive controller (MPC) uses the frozen
     model to pursue a goal it never saw. No model updates before the first attempt (zero-shot).

All model-based agents share the SAME MPC, so the only thing that differs is the MODEL quality
their exploration produced. Baselines: causal-EIG (ours), prediction-first (surprise/pred-error),
random, and an ORACLE that plans with the true dynamics (upper bound).
"""
from __future__ import annotations

import numpy as np

from .capability_world import CapabilityWorld, GOALS, ACTUATORS, AN
from .hetero import WeightedHeteroModel
from .macro import full_command, macro_vocabulary, rollout_states

PRODUCTS = []   # filled in main(): pairwise products so the conditional gate is representable


# ---- exploration policies (reward-free) ---------------------------------------
def _score(policy, model, states):
    S = model.sensors
    if policy == "eig":
        return model.seq_eig(states)
    if policy == "surprise":
        return sum(0.5 * np.log(2 * np.pi * np.e * model.sigma2(xc, i)) for xc, _ in states for i in S)
    if policy == "pred_error":
        return sum(np.sqrt(model.sigma2(xc, i)) for xc, _ in states for i in S)
    return 0.0


def explore(world, policy, budget, rng, epsilon=0.1):
    acts = world.actuators
    model = WeightedHeteroModel(world.d, acts, hidden=world.hidden, interaction_pairs=PRODUCTS,
                                rng=np.random.default_rng(int(rng.integers(1 << 31))), bayes_head=True)
    steps = 0
    while steps < budget:
        vocab = macro_vocabulary(acts, rng=rng)
        if policy == "random" or rng.random() < epsilon:
            macro = vocab[int(rng.integers(len(vocab)))]
        else:
            x0 = world.x.copy()
            sc = np.nan_to_num(np.array([_score(policy, model, rollout_states(model, x0, m, acts))
                                         for m in vocab]), nan=-1e18, posinf=1e18, neginf=-1e18)
            best = np.flatnonzero(sc >= sc.max() - 1e-9)
            macro = vocab[int(rng.choice(best)) if len(best) else int(rng.integers(len(vocab)))]
        for cmd, k in macro:
            fc = full_command(cmd, acts)
            for _ in range(int(k)):
                if steps >= budget:
                    break
                xc = world.x.copy()
                for j, v in fc.items():
                    xc[j] = v
                model.update(xc, world.step(fc))
                steps += 1
    return model


class OracleModel:
    """Plans with the TRUE dynamics (upper bound). predict_next = deterministic true step."""
    def __init__(self, world):
        self._w = CapabilityWorld(noise_gain=world.noise_gain)
        self.sensors = world.sensors
        self.actuators = world.actuators

    def predict_next(self, xc, command=None):
        self._w.x = np.asarray(xc, float).copy()
        self._w.command = {}
        return self._w.step(command or {}, noise=False), None


# ---- shared MPC controller ----------------------------------------------------
def mpc_control(world, model, goal, budget, acts, rng, pen_weight=0.6):
    target, (lo, hi), pen = goal["target"], goal["band"], goal["penalty"]
    steps, noise_exp, reached = 0, 0.0, False
    while steps < budget and not reached:
        vocab = macro_vocabulary(acts, rng=rng)
        x0 = world.x.copy()
        scores = []
        for m in vocab:
            st = rollout_states(model, x0, m, acts)
            prog = max(float(model.predict_next(xc, cmd)[0][target]) for xc, cmd in st)  # reach high
            penc = sum(abs(cmd.get(pen, 0.0)) for _, cmd in st) if pen is not None else 0.0
            scores.append(prog - pen_weight * penc)
        macro = vocab[int(np.argmax(scores))]
        for cmd, k in macro:
            fc = full_command(cmd, acts)
            for _ in range(int(k)):
                if steps >= budget:
                    break
                world.step(fc); steps += 1
                noise_exp += max(world.x[AN], 0.0)
                if world.x[target] >= lo:
                    reached = True; break
            if reached:
                break
    return dict(reached=reached, steps=steps if reached else budget,
                final=float(world.x[target]), noise=noise_exp)


def run_agent(kind, goal_name, seed, explore_budget, goal_budget):
    rng = np.random.default_rng(seed)
    ew = CapabilityWorld(np.random.default_rng(seed)); ew.reset()
    if kind == "oracle":
        model = OracleModel(ew)
    else:
        pol = {"causal": "eig", "prediction": "pred_error", "random": "random"}[kind]
        model = explore(ew, pol, explore_budget, rng)
    gw = CapabilityWorld(np.random.default_rng(seed + 999)); gw.reset()
    return mpc_control(gw, model, GOALS[goal_name], goal_budget, ACTUATORS, rng)


def main(goal="deep_chain", seeds=range(8), explore_budget=300, goal_budget=40):
    global PRODUCTS
    w = CapabilityWorld()
    # actuator x sensor products: a non-peeking prior that "an actuator may GATE a sensor's
    # dynamics" -- includes the true gate (a1 * gate) without enumerating all O(d^2) pairs.
    PRODUCTS = [(a, s) for a in ACTUATORS for s in w.sensors]
    print("=" * 76)
    print(f"CAPABILITY TRIAL -- goal={goal}  (explore {explore_budget}, goal budget {goal_budget}, "
          f"{len(list(seeds))} seeds)")
    print("=" * 76)
    agents = ["causal", "prediction", "random", "oracle"]
    res = {a: {"reached": [], "steps": [], "final": []} for a in agents}
    for s in seeds:
        for a in agents:
            r = run_agent(a, goal, s, explore_budget, goal_budget)
            res[a]["reached"].append(r["reached"]); res[a]["steps"].append(r["steps"])
            res[a]["final"].append(r["final"])
    band_lo = GOALS[goal]["band"][0]
    print(f"  {'agent':>12} {'zero-shot success':>18} {'steps-to-goal':>14} {'final target':>14}  (band>={band_lo})")
    for a in agents:
        sr = 100 * np.mean(res[a]["reached"])
        st = np.mean(res[a]["steps"]); fi = np.mean(res[a]["final"])
        print(f"  {a:>12} {sr:>15.0f}% {st:>14.1f} {fi:>14.2f}")
    print("=" * 76)
    return res


if __name__ == "__main__":
    import sys
    g = sys.argv[1] if len(sys.argv) > 1 else "deep_chain"
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(goal=g, seeds=range(ns))
