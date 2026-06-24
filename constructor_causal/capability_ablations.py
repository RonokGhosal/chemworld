"""
Closing the capability battery (commander's remaining orders):
  1. NO-DECOY and NO-CONFOUNDER ablations -- are those stressors essential to the causal vs
     prediction-first gap, or just battlefield decoration?
  2. AUDIT hetero-disagreement -- does it win by learning the SAME stable causal model as
     causal-EIG (rollout stability + gate/chain coefficients), or by a different route?
  3. ABSTENTION CALIBRATION -- sweep the feasibility threshold; false-act (act on the
     impossible goal) vs false-abstain (abstain on an achievable one).
"""
from __future__ import annotations

import numpy as np

import constructor_causal.capability_baselines as cb
import constructor_causal.capability_trial as ct
from .capability_goal45 import feasible
from .capability_world import CapabilityWorld, ACTUATORS, GOALS, A1, GATE, M1, M2, M3
from .rollout_stability import stability


def _setup():
    ct.PRODUCTS = [(a, s) for a in ACTUATORS for s in CapabilityWorld().sensors]


def _model(agent, ew, rng):
    if agent == "oracle":
        return ct.OracleModel(ew)
    if agent == "causal":
        return ct.explore(ew, "eig", 300, rng)
    if agent == "prediction":
        return ct.explore(ew, "pred_error", 300, rng)
    return cb.explore_baseline(ew, agent, 300, rng)


def ablations(seeds=range(8), gb=22):
    print("=" * 80)
    print(f"ABLATIONS -- deep-chain control @ budget {gb} (low-shot regime), {len(list(seeds))} seeds")
    print("=" * 80)
    configs = [("full", True, True), ("no-decoy", False, True),
               ("no-confounder", True, False), ("neither", False, False)]
    agents = ["causal", "prediction", "hetero_disagreement", "oracle"]
    print(f"  {'config':>15} " + " ".join(f"{a:>20}" for a in agents))
    for name, dec, conf in configs:
        row = {a: [] for a in agents}
        for s in seeds:
            for a in agents:
                rng = np.random.default_rng(s)
                ew = CapabilityWorld(np.random.default_rng(s), decoy_lever=dec, confounder=conf); ew.reset()
                gw = CapabilityWorld(np.random.default_rng(s + 999), decoy_lever=dec, confounder=conf); gw.reset()
                m = _model(a, ew, rng)
                row[a].append(ct.mpc_control(gw, m, GOALS["deep_chain"], gb, ACTUATORS, rng)["reached"])
        print(f"  {name:>15} " + " ".join(f"{100*np.mean(row[a]):>19.0f}%" for a in agents))
    print("=" * 80)


def hetero_audit(seeds=range(8)):
    print("\n" + "=" * 80)
    print("HETERO-DISAGREEMENT AUDIT -- same stable causal model, or a different route?")
    print("=" * 80)
    print(f"  {'agent':>20} {'rollout |err|@12':>16} {'slope':>8} {'gate*a1->m1':>12} "
          f"{'m1->m2':>8} {'m2->m3':>8}")
    for a in ["causal", "hetero_disagreement", "prediction"]:
        E, SL, G, AB, BC = [], [], [], [], []
        for s in seeds:
            ew = CapabilityWorld(np.random.default_rng(s)); ew.reset()
            m = _model(a, ew, np.random.default_rng(s))
            st = stability(m); E.append(st["err12"]); SL.append(st["slope"])
            G.append(float(m._mean(M1)[m.base._inter_k(A1, GATE)]))
            AB.append(float(m._mean(M2)[m.base._lin_k(M1)])); BC.append(float(m._mean(M3)[m.base._lin_k(M2)]))
        print(f"  {a:>20} {np.mean(E):>16.1f} {np.mean(SL):>8.2f} {np.mean(G):>12.2f} "
              f"{np.mean(AB):>8.2f} {np.mean(BC):>8.2f}   (true 0.60/0.70/0.60)")
    print("=" * 80)


def abstention_calibration(seeds=range(10)):
    print("\n" + "=" * 80)
    print("ABSTENTION CALIBRATION (causal) -- threshold sweep: false-act vs false-abstain")
    print("=" * 80)
    print(f"  {'tol':>6} {'false-ACT (impossible)':>24} {'false-ABSTAIN (achievable)':>28}")
    for tol in (0.2, 0.4, 0.6, 1.0, 1.5):
        fa, fab = [], []
        for s in seeds:
            ew = CapabilityWorld(np.random.default_rng(s)); ew.reset()
            m = ct.explore(ew, "eig", 300, np.random.default_rng(s))
            # impossible: should ABSTAIN -> feasible()==True is a false-ACT
            fa.append(feasible(m, GOALS["impossible"], np.random.default_rng(s), tol))
            # noise_robust: should ATTEMPT -> feasible()==False is a false-ABSTAIN
            fab.append(not feasible(m, GOALS["noise_robust"], np.random.default_rng(s + 1), tol))
        print(f"  {tol:>6.1f} {100*np.mean(fa):>22.0f}% {100*np.mean(fab):>26.0f}%")
    print("=" * 80)


def main():
    _setup()
    ablations()
    hetero_audit()
    abstention_calibration()


if __name__ == "__main__":
    main()
