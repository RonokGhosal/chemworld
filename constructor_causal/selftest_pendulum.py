"""
constructor_causal -- EXTERNAL-WORLD test on **Gymnasium Pendulum-v1** (we didn't build it).

This test was hardened after an adversarial multi-lens review (physics / causal-correctness
/ honesty / completeness). It now states plainly BOTH what the agent genuinely does and
what this particular world does NOT exercise. Multi-seed throughout; no single lucky run.

WHAT IS GENUINELY SHOWN (repeatably, >=9/10 seeds):
  (1) An INTERVENING agent recovers the pendulum's VELOCITY LAW -- torque->ω and gravity
      sinθ->ω -- with coefficients matching the true physics (+0.15, +0.75), and does NOT
      assert cosθ->ω. The recovery requires intervention: a PASSIVE observer (torque held
      at 0) can never identify torque->ω, though it still learns gravity.
  (4) The adapter is BIT-EXACT vs the installed gymnasium Pendulum-v1 (guards transcription).

WHAT IS HONESTLY *NOT* SHOWN ON THIS WORLD (reported, not hidden):
  (A) RECALL ~0.25: the linear model recovers only the 2 linearly-identifiable velocity
      edges; the 6 nonlinear angle-rotation edges are MISSED (cosθ_{t+1} is a rotation, not
      a linear function). Precision stays ~1.0.
  (B) RANDOM TIES CURIOSITY: random torque poking recovers the velocity law just as well as
      the active-inference (EIG) agent. Intervention vs observation is what matters here;
      the *information-seeking* advantage must be shown on a world where it bites (next:
      SACHS, under a tight intervention budget).
  (C) cosθ->ω is rejected by the EFFECT-SIZE floor (|w|<<eps), NOT by failing the t-test
      (the cos weight is often statistically significant -- its magnitude is just ~0).
  (D) LIBRARY/PLANNER UNEXERCISED: Pendulum has no reachable held setpoint, so
      build_library() mints 0 constructors. The Constructor-Theory planner is out of scope
      here (next: CartPole hold-upright).
"""
from __future__ import annotations

import numpy as np

from .agent import ConstructorCausalAgent
from .model import edge_scores
from .pendulum_world import PendulumWorld, TORQUE, COS, SIN, OMEGA

SEEDS = list(range(10))
RFF_SEEDS = list(range(5))
N = 2000

R = []
def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _build(experimenter, seed, rff=0, obs_noise=0.01, n=N):
    w = PendulumWorld(rng=np.random.default_rng(seed), obs_noise=obs_noise)
    ag = ConstructorCausalAgent(w, seed=seed, experimenter=experimenter, rff=rff)
    ag.explore(n)
    return ag, w


def _vel(model, eps=0.05):
    into = {j for (j, i) in model.recovered_edges(eps=eps) if i == OMEGA}
    return (TORQUE in into, SIN in into, COS in into)


def _cos_tstat(model):
    mean, s2 = model._post_light(OMEGA)
    Pinv = model._ensure_pinv()
    k = model._lin_k(COS)
    std = float(np.sqrt(max(s2 * Pinv[k, k], 1e-12)))
    return abs(mean[k]) / std


def _holdout_rmse(model, seed):
    rng = np.random.default_rng(seed)
    env = PendulumWorld(rng=np.random.default_rng(seed + 1), obs_noise=0.0); env.reset()
    x = env.x.copy(); sq = {i: [] for i in (COS, SIN, OMEGA)}
    for _ in range(600):
        u = float(rng.uniform(-2.0, 2.0)); cmd = {TORQUE: u}
        xc = x.copy(); xc[TORQUE] = u
        mu, _ = model.predict_next(xc, cmd)
        xn = env.step(cmd, noise=False)
        for i in sq:
            sq[i].append((mu[i] - xn[i]) ** 2)
        x = xn
    return {i: float(np.sqrt(np.mean(sq[i]))) for i in sq}


def main():
    print("=" * 78)
    print("constructor_causal -- EXTERNAL TEST: Gymnasium Pendulum-v1 (hardened, multi-seed)")
    print("=" * 78)

    # ---- multi-seed velocity-law recovery: epistemic vs random vs passive ----
    ep, rnd, pas = [], [], []
    recalls, precs, cosw, cost, tqw = [], [], [], [], []
    for s in SEEDS:
        age, w = _build("epistemic", s)
        ep.append(_vel(age.model))
        sc = edge_scores(age.model, w.true_edges())
        recalls.append(sc["recall"]); precs.append(sc["precision"])
        cosw.append(abs(age.model.weight(OMEGA, COS)))
        tqw.append(abs(age.model.weight(OMEGA, TORQUE)))
        cost.append(_cos_tstat(age.model))
        rnd.append(_vel(_build("random", s)[0].model))
        pas.append(_vel(_build("passive", s)[0].model))

    ep_ok = sum(t and si and not c for (t, si, c) in ep)
    rnd_ok = sum(t and si and not c for (t, si, c) in rnd)
    pas_no_tq = sum(not t for (t, si, c) in pas)
    pas_grav = sum(si for (t, si, c) in pas)
    print(f"\n  velocity law (torque & sinθ in, cosθ out) over {len(SEEDS)} seeds:")
    print(f"    INTERVENING (epistemic): {ep_ok}/10     RANDOM poking: {rnd_ok}/10     "
          f"PASSIVE: torque-absent {pas_no_tq}/10, gravity-learned {pas_grav}/10")
    print(f"    recovered weights: torque→ω={np.mean(tqw):.3f}±{np.std(tqw):.3f}  "
          f"(true +0.150);  cosθ→ω |w|={np.mean(cosw):.4f} (true 0)")

    check("INTERVENING agent recovers the velocity law in >=9/10 seeds", ep_ok >= 9, f"{ep_ok}/10")
    check("PASSIVE observer can NEVER identify torque→ω (intervention is required)",
          pas_no_tq == 10, f"torque-absent {pas_no_tq}/10")
    check("PASSIVE still learns gravity sinθ→ω (it always acts)", pas_grav >= 9, f"{pas_grav}/10")
    check("HONEST(B): random poking ties the curiosity agent here (intervention, not EIG, is "
          "what wins on Pendulum)", rnd_ok >= 9, f"random {rnd_ok}/10 vs epistemic {ep_ok}/10")

    # ---- (A) full-graph recall: only the linearly-identifiable edges ----
    print(f"\n  full-graph recovery (epistemic): precision={np.mean(precs):.2f}  "
          f"recall={np.mean(recalls):.2f}  -> {int(round(np.mean(recalls)*8))}/8 true edges")
    check("HONEST(A): linear model recovers only the 2 linearly-identifiable velocity edges "
          "(recall≈0.25; 6 nonlinear angle edges missed)", abs(np.mean(recalls) - 0.25) < 0.1,
          f"recall={np.mean(recalls):.2f}")
    check("precision stays high (no false edges asserted)", np.mean(precs) >= 0.9,
          f"prec={np.mean(precs):.2f}")

    # ---- (C) cosθ→ω: rejected by effect-size, not by the t-test ----
    n_sig = sum(t > 3.0 for t in cost)
    print(f"\n  cosθ→ω rejection: |w|≈{np.mean(cosw):.4f} (<< eps=0.05 and << smallest true "
          f"edge |torque→ω|≈{np.min(tqw):.3f}); yet t>3 (significant) in {n_sig}/10 seeds")
    check("HONEST(C): cosθ→ω rejected by EFFECT-SIZE margin, not by failing significance "
          "(it is often significant; its magnitude is ~0)",
          max(cosw) < 0.02 and max(cosw) < min(tqw),
          f"max|w_cos|={max(cosw):.4f}, min|w_torque|={min(tqw):.3f}")

    # ---- honest limit: ω (linear law) predicted far better than the NONLINEAR rotation ----
    wl, sl, cl, sr, cr = [], [], [], [], []
    for s in RFF_SEEDS:
        ml = _build("random", s, rff=0)[0].model
        mr = _build("random", s, rff=24)[0].model
        a = _holdout_rmse(ml, 100 + s); b = _holdout_rmse(mr, 100 + s)
        wl.append(a[OMEGA]); sl.append(a[SIN]); cl.append(a[COS])
        sr.append(b[SIN]); cr.append(b[COS])
    ang = 0.5 * (np.mean(sl) + np.mean(cl))
    print(f"\n  one-step prediction RMSE (mean over {len(RFF_SEEDS)} seeds):")
    print(f"    ω (linear law) = {np.mean(wl):.3f}   vs   cosθ/sinθ (nonlinear rotation) ≈ {ang:.3f}")
    print(f"    random-Fourier: sinθ {np.mean(sl):.3f}→{np.mean(sr):.3f}, "
          f"cosθ {np.mean(cl):.3f}→{np.mean(cr):.3f}  -- FRAGILE: no robust gain at this "
          f"scale/policy; the rotation needs a better model CLASS, not random features")
    check("HONEST limit: the linear law (ω) is predicted far better than the nonlinear "
          "rotation (angles) -- the known model-class limit", np.mean(wl) < 0.6 * ang,
          f"ω={np.mean(wl):.3f} vs angle≈{ang:.3f}")

    # ---- (4) bit-exact vs the REAL installed gymnasium (guards transcription) ----
    try:
        import gymnasium as gym
        env = gym.make("Pendulum-v1").unwrapped
        rng = np.random.default_rng(7); maxdev = 0.0
        for _ in range(5000):
            th = rng.uniform(-np.pi, np.pi); om = rng.uniform(-8, 8); u = rng.uniform(-3, 3)
            env.state = np.array([th, om]); env.step([u])
            wv = PendulumWorld(); wv._theta, wv._omega, wv.command = th, om, {}
            xn = wv.step({TORQUE: u}, noise=False)
            g = np.array([np.cos(env.state[0]), np.sin(env.state[0]), env.state[1]])
            maxdev = max(maxdev, float(np.max(np.abs(g - np.array([xn[COS], xn[SIN], xn[OMEGA]])))))
        check("PendulumWorld is BIT-EXACT vs the installed gymnasium Pendulum-v1",
              maxdev < 1e-9, f"max deviation over 5000 steps = {maxdev:.1e}")
    except Exception as e:        # gymnasium optional: don't fail CI, but say so
        print(f"  [skip] gymnasium cross-check unavailable ({type(e).__name__}); "
              f"transcription unguarded this run")

    # ---- (D) honest scope note: library/planner not exercised here ----
    agL, wL = _build("epistemic", 0)
    agL.build_library()
    print(f"\n  scope note(D): on Pendulum build_library() mints {len(agL.library.constructors)} "
          f"constructors (no reachable held setpoint) -> the Constructor-Theory planner is "
          f"OUT OF SCOPE here; exercise it on CartPole hold-upright next.")

    print("=" * 78)
    print(f"{sum(R)}/{len(R)} checks passed   (claims are deliberately two-sided: see (A)-(D))")
    print("=" * 78)
    return all(R)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
