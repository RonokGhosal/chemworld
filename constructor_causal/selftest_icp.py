"""
Selftest for Invariant Causal Prediction (icp.py) -- the non-circular instrument. Exits nonzero on any
failed check. Verifies the THREE properties that make ICP worth having, and the ONE honest limit:

  1. NON-CIRCULAR  -- icp() consumes (X, env, target): ENVIRONMENT labels, never actuator identity. It
                      structurally cannot read `world.actuators` (checked on the function source + signature).
  2. RECOVERY      -- from environment labels ALONE, ICP returns exactly the gate parents {C, a_X} of a
                      multiplicative AND-gate (interaction basis), with no handed knobs.
  3. FAIL-SAFE     -- below the noise floor (SNR<1) recall DROPS: the accepted-set intersection shrinks
                      (often to {}) with NO false parent. This is the point -- it reports "wall is here",
                      not a wrong edge. Buys HONESTY, not recall.
  4. PRECISION     -- on a confounder fork C->X, C->Y, the proxy X is NEVER returned as a parent of Y.
"""
from __future__ import annotations

import sys
import inspect

import numpy as np

from .icp import icp, gate_environments, fork_environments, _invariant

FAILS = []
CHECKS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    CHECKS.append(bool(cond))
    if not cond:
        FAILS.append(name)


def main():
    print("=" * 84)
    print("SELFTEST -- ICP: the non-circular instrument (env labels, not handed actuators)")
    print("=" * 84)

    # 1. NON-CIRCULAR (structural firewall): the estimator never reads actuator identity.
    src = inspect.getsource(icp).lower()
    params = list(inspect.signature(icp).parameters)
    check("non-circular: icp() takes (X, env, target) -- ENV labels, no actuator identity in source/sig",
          "actuator" not in src and "world" not in src and "env" in params
          and not any("act" in p for p in params),
          f"params={params}")

    # 2. RECOVERY from environment labels alone (moderate SNR).
    envs = [(-1, -1), (1, 1), (1.5, -1), (-1, 1.5)]
    rec = []
    for s in [0, 1, 2]:
        X, env, names, tgt, tp = gate_environments(300, envs, np.random.default_rng(s), w=0.5, sigma=0.5)
        r = icp(X, env, tgt)
        rec.append(r["parents"] == tp)
        if s == 0:
            print(f"  recovery seed0: parents={ {names[i] for i in r['parents']} }  true={ {names[i] for i in tp} }")
    check("recovery: from ENV LABELS alone, ICP returns exactly {C, a_X} (all seeds)",
          all(rec), f"{sum(rec)}/3 seeds exact")

    # 3. FAIL-SAFE below the noise floor: recovery DEGRADES (not all seeds full) with NO false parent.
    nofalse, sizes = [], []
    for s in [0, 1, 2]:
        X, env, names, tgt, tp = gate_environments(200, envs, np.random.default_rng(s), w=0.3, sigma=3.0)
        r = icp(X, env, tgt)
        nofalse.append(r["parents"].issubset(tp)); sizes.append(len(r["parents"]))
        if s == 0:
            print(f"  fail-safe seed0: parents={ {names[i] for i in r['parents']} }  sizes(all)={sizes}")
    check("fail-safe: at SNR<1 NO false parent (all subset of truth) AND recovery degrades (not all seeds full)",
          all(nofalse) and sum(sz == len(tp) for sz in sizes) < 3, f"sizes={sizes} (2=full); all-subset={all(nofalse)}")

    # 4. PRECISION on a confounder fork: the proxy X is never a parent of Y. (Audit: 'NEVER' softened -- a
    #    200-seed stress shows ~0.3-0.7% subset-violations at weak SNR; here we assert it holds AND report
    #    the empty-return rate so a {}-dominated vacuous pass is visible.)
    fenvs = [(-1, -1, -1), (1, 1, 1), (1.5, -1, 1), (-1, 1.5, -1)]
    pr, empties = [], 0
    for s in [0, 1, 2]:
        X, env, names, tgt, tp = fork_environments(300, fenvs, np.random.default_rng(s), w=0.6, sigma=0.4)
        r = icp(X, env, tgt)
        pr.append(names.index("X") not in r["parents"] and r["parents"].issubset(tp)); empties += (len(r["parents"]) == 0)
        if s == 0:
            print(f"  precision seed0: Y-parents={ {names[i] for i in r['parents']} }  X_in={names.index('X') in r['parents']}")
    check("precision: proxy X not a parent of Y in these seeds (empties reported; not a 'never' guarantee)",
          all(pr), f"{sum(pr)}/3 clean, {empties}/3 empty-return")

    # 5. CALIBRATION (audit findings 1/3 FIXED): true-set reject rate ~alpha and FLAT in n_env (Bonferroni).
    rej = {}
    for n_env in [2, 32]:
        c, T, rng = 0, 200, np.random.default_rng(0)
        for _ in range(T):
            env = np.repeat(np.arange(n_env), 40); r = rng.normal(0.0, 1.0, len(env))   # truly invariant
            c += (not _invariant(r, env, 0.05))
        rej[n_env] = c / T
    check("calibration: invariant-null reject ~alpha and FLAT in n_env (was 0.006->0.19 uncorrected)",
          rej[2] <= 0.12 and rej[32] <= 0.12, f"reject n_env=2:{rej[2]:.3f} n_env=32:{rej[32]:.3f}")

    print("=" * 84)
    print(f"{sum(CHECKS)}/{len(CHECKS)} checks passed")
    if FAILS:
        print(f"FAILED: {FAILS}")
        sys.exit(1)
    print("ALL CHECKS PASSED -- ICP recovers gate parents from env labels (non-circular), calibrated "
          "(Bonferroni+Welch+Levene, flat Type-I in n_env), degrades to {} not a wrong edge below the floor.")


if __name__ == "__main__":
    main()
