"""
Falsifiable checks for the frontier capabilities: interaction DISCOVERY (no candidate
products supplied) and DEEP composition (a two-gate cascade needing three distinct
constructors in order).

Run:  ./.venv/bin/python -m constructor_causal.selftest_frontier
"""
from __future__ import annotations

import sys

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import POSSIBLE_TAU, Box
from .world import DynamicalCausalWorld

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    print("=" * 78 + "\nconstructor_causal — FRONTIER selftest\n" + "=" * 78)

    # ===== A. INTERACTION DISCOVERY (gate, nothing supplied) ===============
    print("\n-- interaction discovery: find the gate from residuals --")
    gw = DynamicalCausalWorld.gated(np.random.default_rng(0))
    ag = ConstructorCausalAgent(gw, seed=0)              # NO interaction_pairs given
    ag.explore(400)
    found = ag.discover_interactions()
    check("discovers the gate product gate·a1 (= pair (1,2)) unaided",
          (1, 2) in found, f"found={sorted(found)}")
    check("does not hallucinate many spurious interactions", len(found) <= 2,
          f"n_found={len(found)}")
    check("the discovered model confirms a1·gate→Z",
          ((1, 2), 3) in ag.model.recovered_interactions(),
          f"interactions={sorted(ag.model.recovered_interactions())}")

    # ===== B. DEEP COMPOSITION (two-gate cascade, depth 3) =================
    print("\n-- deep composition: two-gate cascade (three distinct skills) --")
    cw = DynamicalCausalWorld.cascade(np.random.default_rng(0))
    cag = ConstructorCausalAgent(cw, seed=0)
    cag.explore(600)
    found = cag.discover_interactions()
    check("discovers BOTH cascade gates (a1·gate1, a2·gate2)",
          (1, 3) in found and (2, 4) in found, f"found={sorted(found)}")

    cag.build_library(setpoints=(-2.0, 2.0), conditional=True)
    prims = [c for c in cag.library.possible() if "conditional" not in c.provenance]
    check("only a0 works from rest (a1, a2 are idle until their gate opens)",
          all(set(c.effect.vars()) <= {3} for c in prims),   # primitives touch only gate1 (idx 3)
          f"primitive effects={[sorted(c.effect.vars()) for c in prims]}")
    depth2 = [c for c in cag.conditionals if c.name.count("|") == 2]
    check("conditional minting reaches depth 2 (a2 | gate2 | gate1)",
          len(depth2) > 0, f"n_depth2={len(depth2)}")
    check("no duplicate skills minted (clean library)",
          len(cag.conditionals) == len(set(c.name for c in cag.conditionals)),
          f"n={len(cag.conditionals)}")

    Zt = Box.from_dict({5: (3.0, 4.5)})
    c, r = cag.achieve(Zt)
    check("a THREE-deep chain reaches Z reliably (≥ τ)",
          c is not None and r >= POSSIBLE_TAU, f"reliability={r:.2f}")
    uses_all = c is not None and all(f"x{k}" in c.provenance for k in (0, 1, 2))
    check("the chain composes all THREE distinct knobs (a0 ≫ a1|g1 ≫ a2|g2)",
          uses_all, f"provenance={c.provenance if c else None}")
    check("the chain is genuinely deep (horizon ≥ 9 = three 3-step skills)",
          c is not None and c.horizon >= 9, f"horizon={c.horizon if c else None}")

    n_pass = sum(ok for _, ok, _ in CHECKS)
    print("\n" + "=" * 78 + f"\n{n_pass}/{len(CHECKS)} checks passed\n" + "=" * 78)
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
