"""
constructor_causal -- EMBODIED single-trajectory selftest.

Closes census flaw #1 (the critical one): verification used to REQUIRE a cloneable,
resettable world -- one build_library() spawned ~360 fresh clones, each reset() -- so
the agent could not be deployed in a real body that lives ONE trajectory and cannot
rewind. Here the agent learns, builds verified skills, reaches goals, and runs the
cloning-free anytime-valid certificate -- all in ONE ongoing life, with NO clone() and
only the single birth reset(). We PROVE it by counting clone()/reset() calls.
"""
from __future__ import annotations

import numpy as np

from .agent import ConstructorCausalAgent
from .constructor import Box
from .world import DynamicalCausalWorld as W

R = []
def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _count_calls(world):
    """Monkeypatch the instance to COUNT clone()/reset() -- the proof of one trajectory."""
    c = {"clone": 0, "reset": 0}
    o_reset, o_clone = world.reset, world.clone
    def reset(x0=None):
        c["reset"] += 1
        return o_reset(x0)
    def clone(rng=None):
        c["clone"] += 1
        return o_clone(rng)
    world.reset = reset
    world.clone = clone
    return c


def main():
    print("=" * 78)
    print("constructor_causal -- EMBODIED selftest (one life: no clone, no reset)")
    print("=" * 78)

    # --- a clone-based agent DOES clone (the contrast) ---
    cb_world = W.default(np.random.default_rng(0))
    cb_counts = _count_calls(cb_world)
    cb = ConstructorCausalAgent(cb_world, seed=0)          # default: clone-based
    cb.explore(300)
    cb.build_library()

    # --- the embodied agent: ONE ongoing trajectory ---
    world = W.default(np.random.default_rng(0))
    counts = _count_calls(world)
    ag = ConstructorCausalAgent(world, seed=0, embodied=True)
    birth_resets = counts["reset"]                         # exactly the birth reset
    ag.explore(700)                                        # learn on the live stream
    ag.build_library()                                     # verify skills in-world (no clone)
    C1 = world.names.index("chain1")
    edges = ag.named_edges(ag.model.recovered_edges())
    lib = list(ag.library.possible())
    target = Box.from_dict({C1: (1.0, 5.0)})
    c_goal, r_goal = ag.achieve(target)                   # reach a goal single-trajectory
    rep = ag.synth.certify(c_goal, target) if c_goal is not None else {"verdict": "n/a"}

    print(f"\n  clone-based agent: clones={cb_counts['clone']}, resets={cb_counts['reset']}")
    print(f"  embodied agent:    clones={counts['clone']}, resets={counts['reset']}"
          f"  (recovered edges: {edges})\n")

    # ----- proof of single-trajectory operation -----
    check("clone-based agent DOES clone (shows the old privilege)", cb_counts["clone"] > 0,
          f"{cb_counts['clone']} clones")
    check("embodied agent NEVER clones the world", counts["clone"] == 0,
          f"{counts['clone']} clones")
    check("embodied agent resets ONLY at birth (one ongoing trajectory)",
          counts["reset"] == birth_resets == 1, f"{counts['reset']} resets")
    # ----- and it still works, entirely single-trajectory -----
    check("learned a0->chain1 from one trajectory", "a0→chain1" in edges, str(edges))
    check("built >=1 verified primitive in-world", len(lib) >= 1, f"library={len(lib)}")
    check("reached a goal single-trajectory (reliability >= 0.9)",
          c_goal is not None and r_goal >= 0.9, f"rel {r_goal:.2f}")
    check("cloning-free betting certificate decides POSSIBLE",
          rep.get("verdict") == "POSSIBLE", f"verdict {rep.get('verdict')}, n={rep.get('n')}")

    print("=" * 78)
    print(f"{sum(R)}/{len(R)} checks passed")
    print("=" * 78)
    return all(R)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
