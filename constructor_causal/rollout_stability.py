"""
ROLLOUT-STABILITY metrics (now the central evidence): control uses multi-step COUNTERFACTUAL
rollouts, so the question isn't one-step prediction accuracy -- it's whether a learned model
stays accurate when rolled forward under a sustained intervention. Causal exploration should
yield intervention-STABLE models; prediction-first can be one-step accurate yet diverge.

For each agent's frozen model we roll it forward under the control-relevant intervention
do(a0=2, a1=2) and compare the predicted m3 trajectory to the TRUE one:
  err(h)         -- |predicted m3 - true m3| at horizon h
  compound slope -- growth rate of err with h (instability)
  sign-correct   -- does it even get the DIRECTION of m3 right at the horizon?
  a1-drive%      -- during control, how often the MPC drives the true lever a1 (oracle: always)
"""
from __future__ import annotations

import numpy as np

import constructor_causal.capability_trial as ct
from .capability_world import CapabilityWorld, ACTUATORS, A0, A1, M3, GOALS
from .macro import full_command, macro_vocabulary, rollout_states


def _setup():
    ct.PRODUCTS = [(a, s) for a in ACTUATORS for s in CapabilityWorld().sensors]


def _true_traj(interv, H):
    w = CapabilityWorld(); w.reset(); out = []
    for _ in range(H):
        w.step(interv, noise=False); out.append(w.x[M3])
    return np.array(out)


def _model_traj(model, interv, H):
    x = np.zeros(16); out = []
    fc = full_command(interv, ACTUATORS)
    for _ in range(H):
        xc = x.copy()
        for j, v in fc.items():
            xc[j] = v
        mu, _ = model.predict_next(xc, fc)
        out.append(float(mu[M3]))
        x = np.clip(mu, -50, 50)
    return np.array(out)


def stability(model, H=12):
    interv = {A0: 2.0, A1: 2.0}
    tt = _true_traj(interv, H); mt = _model_traj(model, interv, H)
    err = np.abs(mt - tt)
    slope = float(np.polyfit(np.arange(1, H + 1), err, 1)[0])
    sign_ok = bool(mt[-1] > 1.0 and tt[-1] > 1.0)        # predicts m3 genuinely rises
    return dict(err3=float(err[2]), err12=float(err[-1]), slope=slope, sign_ok=sign_ok,
                pred12=float(mt[-1]))


def a1_drive_fraction(model, seed, budget=40):
    """During zero-shot control of the deep-chain goal, how often does the MPC drive the TRUE
    lever a1? (oracle drives it ~always.)"""
    gw = CapabilityWorld(np.random.default_rng(seed + 999)); gw.reset()
    rng = np.random.default_rng(seed)
    steps, a1_hi = 0, 0
    while steps < budget:
        vocab = macro_vocabulary(ACTUATORS, rng=rng); x0 = gw.x.copy()
        sc = [max(float(model.predict_next(xc, cmd)[0][M3]) for xc, cmd in rollout_states(model, x0, mm, ACTUATORS))
              for mm in vocab]
        macro = vocab[int(np.argmax(sc))]
        for cmd, k in macro:
            fc = full_command(cmd, ACTUATORS)
            for _ in range(int(k)):
                if steps >= budget:
                    break
                if fc.get(A1, 0.0) > 0:
                    a1_hi += 1
                gw.step(fc); steps += 1
    return a1_hi / max(steps, 1)


def main(seeds=range(8), explore_budget=300):
    _setup()
    print("=" * 80)
    print(f"ROLLOUT STABILITY -- model rolled under do(a0=2,a1=2); true m3@12 ~= +12  "
          f"({len(list(seeds))} seeds)")
    print("=" * 80)
    agents = ["causal", "prediction", "random", "oracle"]
    res = {a: {"err12": [], "slope": [], "sign": [], "pred12": [], "a1": []} for a in agents}
    for s in seeds:
        for a in agents:
            ew = CapabilityWorld(np.random.default_rng(s)); ew.reset()
            if a == "oracle":
                m = ct.OracleModel(ew)
            else:
                m = ct.explore(ew, {"causal": "eig", "prediction": "pred_error",
                                    "random": "random"}[a], explore_budget, np.random.default_rng(s))
            st = stability(m)
            res[a]["err12"].append(st["err12"]); res[a]["slope"].append(st["slope"])
            res[a]["sign"].append(st["sign_ok"]); res[a]["pred12"].append(st["pred12"])
            res[a]["a1"].append(a1_drive_fraction(m, s))
    print(f"  {'agent':>11} {'|err| m3@h12':>13} {'compound slope':>15} {'sign-correct':>13} "
          f"{'a1-drive% (ctrl)':>17}")
    for a in agents:
        print(f"  {a:>11} {np.mean(res[a]['err12']):>13.1f} {np.mean(res[a]['slope']):>15.2f} "
              f"{100*np.mean(res[a]['sign']):>11.0f}% {100*np.mean(res[a]['a1']):>15.0f}%")
    print("=" * 80)
    return res


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    main(seeds=range(ns))
