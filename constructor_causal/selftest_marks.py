"""
constructor_causal -- DIRECTED vs BIDIRECTED honesty selftest (census flaw #7).

When a latent common cause drives two sensors the agent CANNOT reach by intervention,
a confident directed edge S1->S2 is a lie (it's identical to a passive observer's
mistake). The honest output is a BIDIRECTED mark S1<->S2 ("possibly confounded").
recovered_marks() emits directed edges only where they are genuinely do-identified
(actuator source, or a sensor with an actuator INSTRUMENT) and marks everything else
bidirected. This test shows: (a) when intervention DOES reach the structure (default
world), every edge is directed; (b) when it cannot (un-actuable hidden common cause),
the spurious link is emitted bidirected, not asserted as a confident cause.
"""
from __future__ import annotations

import numpy as np

from .agent import ConstructorCausalAgent
from .world import DynamicalCausalWorld as W

R = []
def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _latent_world(rng):
    """a0 drives d0 (a real job for the only actuator); a HIDDEN H drives BOTH S1 and
    S2 with NO direct S1->S2 edge; neither S1 nor S2 is actuable. So the recovered
    S1->S2 association is pure confounding the agent cannot orient or break."""
    a0, d0, S1, S2, H = 0, 1, 2, 3, 4
    d = 5
    A = np.zeros((d, d))
    A[d0, d0], A[d0, a0] = 0.20, 0.80         # a0 -> d0
    A[S1, S1], A[S1, H] = 0.20, 0.95          # H -> S1  (clean proxy, low noise)
    A[S2, S2], A[S2, H] = 0.20, 0.90          # H -> S2  (noisy)  -- NO direct S1->S2
    A[H, H] = 0.90                            # slow hidden common cause
    noise = np.array([0.0, 0.05, 0.15, 1.00, 0.50])
    names = ("a0", "d0", "S1", "S2", "H")
    return W(A=A, b=np.zeros(d), noise_std=noise, actuators=(a0,), names=names,
             hidden=(H,), rng=rng)


def main():
    print("=" * 78)
    print("constructor_causal -- DIRECTED vs BIDIRECTED honesty (flaw #7)")
    print("=" * 78)

    # (1) intervention REACHES the structure -> everything do-identified (directed)
    ag0 = ConstructorCausalAgent(W.default(np.random.default_rng(0)), seed=0)
    ag0.explore(700)
    m0 = ag0.named_marks()
    print(f"\n  default world : directed={m0['directed']}  bidirected={m0['bidirected']}")
    check("default world: edges are DIRECTED (actuator a0 instruments the chain)",
          "a0→chain1" in m0["directed"] and "chain1→chain2" in m0["directed"])
    check("default world: NO spurious bidirected marks", len(m0["bidirected"]) == 0,
          str(m0["bidirected"]))

    # (2) un-actuable hidden common cause -> honest bidirected mark, not a wrong edge
    agl = ConstructorCausalAgent(_latent_world(np.random.default_rng(1)), seed=1)
    agl.explore(1500)
    raw = agl.named_edges(agl.model.recovered_edges())
    ml = agl.named_marks()
    print(f"\n  latent world  : raw recovered={raw}")
    print(f"                  directed={ml['directed']}  bidirected={ml['bidirected']}")
    check("a confounded S1-S2 association IS recovered (the trap)",
          ("S1→S2" in raw) or ("S2→S1" in raw), str(raw))
    check("...but emitted BIDIRECTED (possibly-confounded), not a confident directed edge",
          "S1↔S2" in ml["bidirected"]
          and "S1→S2" not in ml["directed"] and "S2→S1" not in ml["directed"],
          f"directed={ml['directed']} bidirected={ml['bidirected']}")
    check("the do-identified actuator edge a0->d0 stays DIRECTED",
          "a0→d0" in ml["directed"], str(ml["directed"]))

    print("=" * 78)
    print(f"{sum(R)}/{len(R)} checks passed")
    print("=" * 78)
    return all(R)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
