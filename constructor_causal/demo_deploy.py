"""
Long-horizon non-stationary DEPLOYMENT — the continual agent across many regimes,
versus a discover-once baseline, plus localized re-exploration.

Run:  ./.venv/bin/python -m constructor_causal.demo_deploy
"""
from __future__ import annotations

import statistics as st

import numpy as np

from .agent import ConstructorCausalAgent
from .world import DynamicalCausalWorld
from .deploy import (RegimeSchedule, deploy, deploy_baseline, measure_localized_saving)


def _by_regime(tl, attr):
    out = {}
    for r in sorted({m.regime for m in tl}):
        out[r] = st.mean([getattr(m, attr) for m in tl if m.regime == r])
    return out


def main():
    print("=" * 80)
    print("LONG-HORIZON NON-STATIONARY DEPLOYMENT — continual agent vs discover-once")
    print("=" * 80)
    print("5 regimes x 5 rounds on the `default` world. The world changes at the start")
    print("of each regime; nothing tells the agent when. Metrics graded vs the CURRENT")
    print("(mutated) world: F1 (edge presence), belief_mae (|weight error|), worst_z")
    print("(standardized surprise = the change signal).\n")
    print("  Regimes:  R0 learn · R1 flip a0->chain1 (sign) · R2 strengthen chain1->chain2")
    print("            · R3 ADD a0->decoy (structural) · R4 noise burst + REMOVE a0->decoy\n")

    sched = RegimeSchedule.five_regime_default(rounds=5)
    w = DynamicalCausalWorld.default(np.random.default_rng(0))
    agent = ConstructorCausalAgent(w, seed=0, forget=0.9)
    cont = deploy(agent, w, sched, steps_per_round=130, z_change=4.0)
    wb = DynamicalCausalWorld.default(np.random.default_rng(0))
    base = deploy_baseline(wb, sched, seed=0)

    print(f"  {'rnd':>3} {'reg':>3} | {'CONT f1':>7} {'mae':>5} {'z':>5} {'chg':>3} | "
          f"{'BASE f1':>7} {'mae':>5} {'z':>5}")
    for c, b in zip(cont, base):
        cp = "*" if c.is_change_point else " "
        print(f"  {c.rnd:>3} {('R%d'%c.regime):>3}{cp}| {c.f1:>7.2f} {c.belief_mae:>5.2f} "
              f"{c.worst_z:>5.1f} {int(c.changed):>3} | {b.f1:>7.2f} {b.belief_mae:>5.2f} {b.worst_z:>5.1f}")

    print("\n  per-regime means:")
    print(f"    belief_mae  continual {[round(v,2) for v in _by_regime(cont,'belief_mae').values()]}")
    print(f"                baseline  {[round(v,2) for v in _by_regime(base,'belief_mae').values()]}")
    print(f"    worst_z     continual {[round(v,1) for v in _by_regime(cont,'worst_z').values()]}")
    print(f"                baseline  {[round(v,1) for v in _by_regime(base,'worst_z').values()]}")
    lat = [(m.regime, m.detection_latency) for m in cont if m.detection_latency is not None]
    print(f"    change detection latency (rounds): {lat}")
    print("  => the continual agent tracks every change (mae~0, surprise settles); the")
    print("     frozen baseline's belief diverges (mae up to ~0.7, surprise pinned ~55+).")

    print("\n" + "=" * 80)
    print("LOCALIZED RE-EXPLORATION — re-check only the believed-relevant sub-graph")
    print("=" * 80)
    print("After a sign flip, recover with FULL vs LOCALIZED exploration (same budget).")
    for label, factory, C1, na in [
            ("wide (6 actuators)", lambda: DynamicalCausalWorld.wide(k=5, rng=np.random.default_rng(0)), 6, 6),
            ("default (2 actuators)", lambda: DynamicalCausalWorld.default(np.random.default_rng(0)), 2, 2)]:
        r = measure_localized_saving(factory, target_i=C1, source_j=0, warmup=400, recover_steps=30)
        print(f"  {label}: weight error  full {r['full_weight_err']:.3f}  vs  "
              f"localized {r['localized_weight_err']:.3f}   "
              f"(candidate grid {r['full_grid']} -> {r['localized_grid']}; "
              f"focused on {r['localized_actuators']})")
    print("  => localization restricts experiments to the changed edge's believed")
    print("     ancestors; the saving grows with the number of irrelevant knobs.")
    print("     (Honest: a NEW edge from a previously-idle knob isn't a believed ancestor")
    print("      yet, so localization correctly falls back to all actuators.)")


if __name__ == "__main__":
    main()
