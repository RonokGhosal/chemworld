"""
constructor_causal -- THE NONLINEAR FIX, demonstrated on Pendulum-v1.

The plain edge test asks "is variable X's LINEAR coefficient on Y non-zero?" -- and so
misses the pendulum's 6 angle-rotation edges, because the angle update is a ROTATION:
the influence of (say) sinθ on next-cosθ is carried by the product sinθ*ω and FLIPS SIGN
with the spin, so its linear coefficient averages to ~0.

The fix (model.recovered_edges_grouped): an edge X->Y exists iff the WHOLE BLOCK of
features derived from X (its linear term + every product with X + X's nonlinear basis)
JOINTLY explains Y -- a Mahalanobis test on the block of the multivariate-t posterior,
with an effect-size floor. With a pairwise-product basis the rotation's sinθ*ω / cosθ*ω
features exist, and the block test reads them.

Claims (multi-seed):
  * BEFORE: linear basis + plain edge test -> recall ~0.25 (only the 2 velocity edges).
  * AFTER : product basis + grouped block test -> recall ~0.75, recovering all 4 strong
    ROTATION edges (cos<->sin, ω->cos, ω->sin), precision still high.
  * The only 2 still missed (torque->cosθ, torque->sinθ) are HONESTLY negligible: their
    one-step effect is O(dt^2)~0.0075, ~20x weaker than the rotation edges -- excluded by
    the effect-size floor, not by a nonlinearity failure.
"""
from __future__ import annotations

import numpy as np

from .agent import ConstructorCausalAgent
from .pendulum_world import PendulumWorld, TORQUE, COS, SIN, OMEGA

SEEDS = list(range(5))
N = 5000
COLS = (TORQUE, COS, SIN, OMEGA)
PAIRS = [(a, b) for i, a in enumerate(COLS) for b in COLS[i:]]   # all pairwise products + squares
NM = ("torque", "cosθ", "sinθ", "ω")

R = []
def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _agent(seed, pairs):
    w = PendulumWorld(rng=np.random.default_rng(seed), obs_noise=0.01)
    ag = ConstructorCausalAgent(w, seed=seed, experimenter="random",
                                interaction_pairs=pairs)
    ag.explore(N)
    return ag, w


def _pr(E, true):
    tp = len(E & true)
    return (tp / len(E) if E else 1.0), tp / len(true)


def _sigB(model, j, i):
    """RMS predictive contribution of source j's feature block to target i (target units)."""
    mean, _ = model._post_light(i)
    B = model._source_block(j)
    mB = mean[B]; SBB = model.S[np.ix_(B, B)]
    return float(np.sqrt(max(float(mB @ SBB @ mB) / max(model.n, 1.0), 0.0)))


def _named(E):
    return sorted(f"{NM[j]}→{NM[i]}" for (j, i) in E)


def main():
    print("=" * 78)
    print("constructor_causal -- THE NONLINEAR FIX (group conditional-dependence) on Pendulum")
    print("=" * 78)
    true = None
    lin_p, lin_r, grp_p, grp_r = [], [], [], []
    rot_hits = []      # how many of the 4 rotation edges the group test recovers
    ROT = {(COS, SIN), (SIN, COS), (OMEGA, COS), (OMEGA, SIN)}
    WEAK = {(TORQUE, COS), (TORQUE, SIN)}
    for s in SEEDS:
        agL, w = _agent(s, ())                  # linear basis: no product vocabulary
        true = w.true_edges()
        # the DEFAULT recovered_edges() is now the grouped test -- both calls below use it,
        # so the only thing that changes between BEFORE and AFTER is the agent's BASIS.
        p, r = _pr(agL.model.recovered_edges(), true); lin_p.append(p); lin_r.append(r)
        agG, _ = _agent(s, PAIRS)               # product basis: sinθ*ω, cosθ*ω, ... available
        Eg = agG.model.recovered_edges()        # DEFAULT path -- grouped wiring proven here
        p, r = _pr(Eg, true); grp_p.append(p); grp_r.append(r)
        rot_hits.append(len(Eg & ROT))
        if s == 0:
            seed0 = (agL, agG, Eg)

    print(f"\n  (the grouped block test is now the DEFAULT recovered_edges; only the BASIS differs)")
    print(f"  BEFORE (linear basis -- no product features): "
          f"precision={np.mean(lin_p):.2f}  recall={np.mean(lin_r):.2f}  "
          f"({int(round(np.mean(lin_r)*8))}/8 edges)")
    print(f"  AFTER  (product basis -- same default test): "
          f"precision={np.mean(grp_p):.2f}  recall={np.mean(grp_r):.2f}  "
          f"({int(round(np.mean(grp_r)*8))}/8 edges)")
    agL, agG, Eg = seed0
    print(f"\n  seed-0 recovered (grouped): {_named(Eg)}")

    check("BEFORE: linear test recovers only the ~2 velocity edges (recall≈0.25)",
          abs(np.mean(lin_r) - 0.25) < 0.1, f"recall={np.mean(lin_r):.2f}")
    check("AFTER: group test recovers ALL 4 rotation edges (cos↔sin, ω→cos, ω→sin)",
          np.mean(rot_hits) >= 3.8, f"{np.mean(rot_hits):.1f}/4 rotation edges per seed")
    check("AFTER: recall jumps to ≈0.75 (6/8 edges)", np.mean(grp_r) >= 0.7,
          f"recall={np.mean(grp_r):.2f}")
    check("AFTER: precision stays high (no false edges from the richer basis)",
          np.mean(grp_p) >= 0.9, f"precision={np.mean(grp_p):.2f}")

    # honest: the 2 still-missed edges are genuinely negligible (effect size), not nonlinear-blind
    print(f"\n  effect size σ (RMS predictive contribution into the angle, target units):")
    for (j, i) in [(SIN, COS), (OMEGA, COS), (TORQUE, COS), (TORQUE, SIN)]:
        tag = "ROTATION (recovered)" if (j, i) in ROT else "weak O(dt²) (excluded)"
        print(f"    {NM[j]:>6}→{NM[i]:<5} σ={_sigB(agG.model, j, i):.4f}   {tag}")
    weak_sig = max(_sigB(agG.model, j, i) for (j, i) in WEAK)
    rot_sig = min(_sigB(agG.model, j, i) for (j, i) in ROT)
    check("the 2 missed edges (torque→angle) are HONESTLY negligible: their effect is an "
          "order below the rotation edges (excluded by effect-size floor, not nonlinearity)",
          weak_sig < 0.02 and rot_sig > 3 * weak_sig,
          f"weak σ≤{weak_sig:.4f} vs rotation σ≥{rot_sig:.3f}")

    print("=" * 78)
    print(f"{sum(R)}/{len(R)} checks passed")
    print("=" * 78)
    return all(R)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
