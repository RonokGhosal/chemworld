"""
Selftest for the frontier map (frontier_map.py) -- the honest operating envelope. Exits nonzero on any
failed check. Runs a REDUCED config (2 seeds, low epochs, coupling endpoints only) that still reproduces
the qualitative findings; the full curves are in `python -m constructor_causal.frontier_map`.

Checks:
  WALL B (strong coupling) -- ACTING beats WATCHING: at w=0.4 interventional recovers most gates and
                              strictly more than the observational readout.
  WALL B (weak coupling)   -- INFORMATION limit, not tuning: at w=0.05 interventional DEGRADES vs strong
                              coupling, AND lowering the detection threshold recovers few extra gates with
                              ZERO false positives (a tuning limit would recover many more; noise would
                              flood false+). This is the honest engineering-vs-information distinction.
  WALL A                   -- UN-ACTUATABILITY caps recall: same fork, C reachable -> full recall; C
                              un-reachable -> recall collapses to ~0. Precision holds both arms (0 false+).
"""
from __future__ import annotations

import sys

from .frontier_map import sweep_wall_b, sweep_wall_a

FAILS = []
CHECKS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    CHECKS.append(bool(cond))
    if not cond:
        FAILS.append(name)


def main():
    print("=" * 84)
    print("SELFTEST -- frontier map: where the interventional method actually breaks (reduced config)")
    print("=" * 84)
    seeds = [0, 1]
    cfg = dict(epochs=150, ep=80, rounds=1)

    # WALL B: strong + weak coupling endpoints
    wb = sweep_wall_b(seeds, gates=3, couplings=(0.40, 0.05), **cfg)
    hi, lo = wb[0.40], wb[0.05]
    print(f"  Wall B w=0.40: obs {hi['obs']:.1f}/{hi['total']}  interv {hi['interv_def']:.1f}/{hi['total']} "
          f"(+{hi['fp_def']:.1f})   |  w=0.05: interv {lo['interv_def']:.1f}/{lo['total']} "
          f"(+{lo['fp_def']:.1f})  lowthr {lo['interv_low']:.1f}/{lo['total']} (+{lo['fp_low']:.1f})")

    check("Wall B strong-coupling: ACTING beats WATCHING (interv >= 2 AND interv > obs at w=0.4)",
          hi['interv_def'] >= 2.0 and hi['interv_def'] > hi['obs'] + 0.5,
          f"interv={hi['interv_def']:.1f} obs={hi['obs']:.1f}")
    check("Wall B weak-coupling: recovery DEGRADES vs strong (interv@0.05 <= interv@0.4 - 1)",
          lo['interv_def'] <= hi['interv_def'] - 1.0,
          f"interv@0.05={lo['interv_def']:.1f} vs interv@0.4={hi['interv_def']:.1f}")
    check("Wall B weak-coupling: INFORMATION limit not tuning (lowering thr adds <=1.5 gate AND 0 false+)",
          (lo['interv_low'] - lo['interv_def']) <= 1.5 and lo['fp_low'] == 0.0,
          f"gain={lo['interv_low'] - lo['interv_def']:.1f} false+_low={lo['fp_low']:.1f}")

    # WALL A: un-actuatable confounder
    wa = sweep_wall_a(seeds, **cfg)
    yes, no = wa[True], wa[False]
    print(f"  Wall A: C reachable -> recall {yes['recall']:.2f}/{yes['total']} (+{yes['fp']:.2f})   |  "
          f"C un-reachable -> recall {no['recall']:.2f}/{no['total']} (+{no['fp']:.2f})")

    check("Wall A: C reachable -> FULL recall (>= total - 0.5)",
          yes['recall'] >= yes['total'] - 0.5, f"recall={yes['recall']:.2f}/{yes['total']}")
    check("Wall A: C un-reachable -> recall COLLAPSES (<= 0.5)",
          no['recall'] <= 0.5, f"recall={no['recall']:.2f}/{no['total']}")
    check("Wall A: precision holds BOTH arms (0 false+ -- never invents a spurious edge)",
          yes['fp'] == 0.0 and no['fp'] == 0.0, f"fp_reachable={yes['fp']:.2f} fp_unreachable={no['fp']:.2f}")

    print("=" * 84)
    print(f"{sum(CHECKS)}/{len(CHECKS)} checks passed")
    if FAILS:
        print(f"FAILED: {FAILS}")
        sys.exit(1)
    print("ALL CHECKS PASSED -- acting buys SNR headroom (Wall B), floored by an information limit; "
          "un-actuatability caps recall (Wall A); precision intact.")


if __name__ == "__main__":
    main()
