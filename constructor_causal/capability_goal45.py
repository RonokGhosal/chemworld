"""
Goals 4 & 5 (commander Order 1):
  GOAL 4 -- IMPOSSIBLE / abstention: "raise c1 WITHOUT moving c2". c1,c2 share the hidden
            confounder H, so no action moves one without the other. A good agent must ABSTAIN
            ("not reliably controllable"), not thrash. Metric: abstention ACCURACY (abstain on
            the impossible goal AND attempt the achievable one).
  GOAL 5 -- INERT TAX: z0/z1 do nothing. A good model never wastes control budget on them.
            Metric: inert-drive% during control (cost) and steps-to-goal regret vs oracle.

Abstention rule (uses the agent's OWN frozen model): the goal is feasible iff some candidate
macro is predicted to move the target meaningfully toward the band while the penalty variable's
change stays below tol. A correct causal model finds NO such macro for the impossible goal; an
interventionally-unstable model may hallucinate one (fail to abstain).
"""
from __future__ import annotations

import numpy as np

import constructor_causal.capability_trial as ct
from .capability_world import (CapabilityWorld, ACTUATORS, GOALS, Z0, Z1, M1, AN)
from .macro import full_command, macro_vocabulary, rollout_states


def _setup():
    ct.PRODUCTS = [(a, s) for a in ACTUATORS for s in CapabilityWorld().sensors]


def feasible(model, goal, rng, tol=0.6, samples=4):
    """Does the frozen model believe the goal is reachable without violating the penalty?"""
    target, (lo, hi), pen = goal["target"], goal["band"], goal["penalty"]
    x0 = np.zeros(16)
    for _ in range(samples):
        for m in macro_vocabulary(ACTUATORS, rng=rng):
            preds = [model.predict_next(xc, cmd)[0] for xc, cmd in rollout_states(model, x0, m, ACTUATORS)]
            tgt_gain = max(p[target] for p in preds) - x0[target]
            pen_chg = max(abs(p[pen] - x0[pen]) for p in preds) if pen is not None else 0.0
            if tgt_gain > 0.5 and pen_chg < tol:
                return True
    return False


def control_with_abstain(world, model, goal, budget, rng, tol=0.6):
    if not feasible(model, goal, rng, tol):
        return dict(abstained=True, reached=False, steps=0, inert=0.0)
    target, (lo, hi), pen = goal["target"], goal["band"], goal["penalty"]
    steps, inert_hi, reached = 0, 0, False
    while steps < budget and not reached:
        vocab = macro_vocabulary(ACTUATORS, rng=rng); x0 = world.x.copy()
        sc = []
        for m in vocab:
            st = rollout_states(model, x0, m, ACTUATORS)
            prog = max(float(model.predict_next(xc, cmd)[0][target]) for xc, cmd in st)
            penc = sum(abs(cmd.get(pen, 0.0)) for _, cmd in st) if pen is not None else 0.0
            sc.append(prog - 0.6 * penc)
        macro = vocab[int(np.argmax(sc))]
        for cmd, k in macro:
            fc = full_command(cmd, ACTUATORS)
            for _ in range(int(k)):
                if steps >= budget:
                    break
                if fc.get(Z0, 0.0) != 0 or fc.get(Z1, 0.0) != 0:
                    inert_hi += 1
                world.step(fc); steps += 1
                if world.x[target] >= lo:
                    reached = True; break
            if reached:
                break
    return dict(abstained=False, reached=reached, steps=steps if reached else budget,
                inert=inert_hi / max(steps, 1))


def model_for(agent, seed, eb=300):
    ew = CapabilityWorld(np.random.default_rng(seed)); ew.reset()
    if agent == "oracle":
        return ct.OracleModel(ew)
    return ct.explore(ew, {"causal": "eig", "prediction": "pred_error", "random": "random"}[agent],
                      eb, np.random.default_rng(seed))


def main(seeds=range(10)):
    _setup()
    print("=" * 78)
    print(f"GOALS 4-5 -- abstention accuracy + inert-actuator tax  ({len(list(seeds))} seeds)")
    print("=" * 78)
    agents = ["causal", "prediction", "random", "oracle"]
    # abstention: SHOULD abstain on 'impossible'; should ATTEMPT 'noise_robust' (achievable+penalty)
    res = {a: dict(abst_imposs=[], abst_achiev=[], inert=[], steps=[]) for a in agents}
    for s in seeds:
        for a in agents:
            m = model_for(a, s)
            iw = CapabilityWorld(np.random.default_rng(s + 999)); iw.reset()
            r_imp = control_with_abstain(iw, m, GOALS["impossible"], 40, np.random.default_rng(s))
            gw = CapabilityWorld(np.random.default_rng(s + 777)); gw.reset()
            r_ach = control_with_abstain(gw, m, GOALS["noise_robust"], 40, np.random.default_rng(s + 1))
            res[a]["abst_imposs"].append(r_imp["abstained"])
            res[a]["abst_achiev"].append(not r_ach["abstained"])
            res[a]["inert"].append(r_ach["inert"]); res[a]["steps"].append(r_ach["steps"])
    print(f"  {'agent':>11} {'abstain on IMPOSSIBLE':>22} {'attempt ACHIEVABLE':>20} "
          f"{'abst-accuracy':>14} {'inert-drive%':>13}")
    for a in agents:
        ai = 100 * np.mean(res[a]["abst_imposs"]); aa = 100 * np.mean(res[a]["abst_achiev"])
        acc = 100 * np.mean([0.5 * (i + j) for i, j in zip(res[a]["abst_imposs"], res[a]["abst_achiev"])])
        inert = 100 * np.mean(res[a]["inert"])
        print(f"  {a:>11} {ai:>20.0f}% {aa:>18.0f}% {acc:>12.0f}% {inert:>12.0f}%")
    print("=" * 78)
    return res


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(seeds=range(ns))
