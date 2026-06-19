"""
Falsifiable checks for the two researched frontiers (see RESEARCH.md):
  Q2 — cloning-free, anytime-valid reliability certification (the strong result), and
  Q1 — higher-lag latent detection (a real but data-hungry signal: strong specificity,
       majority sensitivity — honest about the identifiability limit).

Run:  ./.venv/bin/python -m constructor_causal.selftest_certify
"""
from __future__ import annotations

import sys

import numpy as np

from .agent import ConstructorCausalAgent
from .certify import (AnytimeCS, BettingCS, DriftDetector, calibrate_passive,
                      certify_library, certify_modelfree, certify_modelfree_continuous,
                      certify_passive, certify_reliability, detect_latent_lag)
from .constructor import Box, Constructor, POSSIBLE_TAU
from .world import DynamicalCausalWorld

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    print("=" * 78 + "\nconstructor_causal — CERTIFY selftest\n" + "=" * 78)

    # ===== Q2a: betting CS is valid AND tighter than the mixture CS ==========
    print("\n-- Q2: anytime-valid confidence sequences (cloning-free certificate) --")
    for p in (0.97, 0.70):
        N, T = 800, 400
        widths = {}
        for Cls, nm in [(BettingCS, "betting"), (AnytimeCS, "mixture")]:
            miss, last_w, rng = 0, 0.0, np.random.default_rng(0)
            for _ in range(N):
                cs, out = Cls(alpha=0.05), False
                for _ in range(T):
                    cs.update(1.0 if rng.random() < p else 0.0)
                    lo, hi = cs.interval()
                    if not (lo <= p <= hi):
                        out = True
                miss += out
                last_w = hi - lo
            widths[nm] = last_w
            check(f"{nm} CS covers the true mean at all t (p={p}; cov ≥ 0.94)",
                  1 - miss / N >= 0.94, f"coverage={1-miss/N:.3f}, width@{T}={last_w:.2f}")
        check(f"betting CS is tighter than the mixture CS (p={p})",
              widths["betting"] < widths["mixture"],
              f"betting={widths['betting']:.2f} vs mixture={widths['mixture']:.2f}")

    # ===== Q2b: cloning-free certify — POSSIBLE vs IMPOSSIBLE ================
    prog = tuple({0: 2.0} for _ in range(4))                 # hold a0=+2 ×4 (one trajectory)
    env = DynamicalCausalWorld.default(np.random.default_rng(7))
    good = certify_reliability(env, prog, Box.from_dict({2: (1.5, 3.0)}), tau=0.9,
                               rng=np.random.default_rng(3))
    env2 = DynamicalCausalWorld.default(np.random.default_rng(8))
    bad = certify_reliability(env2, prog, Box.from_dict({3: (3.0, 3.5)}), tau=0.9,
                              rng=np.random.default_rng(4))
    check("a reachable effect is certified POSSIBLE in-stream (no reset/clone)",
          good["verdict"] == "POSSIBLE", f"interval={tuple(round(v,2) for v in good['interval'])}, n={good['n']}")
    check("an unreachable effect is certified IMPOSSIBLE in-stream",
          bad["verdict"] == "IMPOSSIBLE", f"interval={tuple(round(v,2) for v in bad['interval'])}, n={bad['n']}")

    # ===== Q2c: drift detector (re-verification trigger) =====================
    print("\n-- Q2: drift e-process triggers re-verification when the world changes --")
    rng = np.random.default_rng(0)
    d, fired = DriftDetector(p0=0.95), None
    for t in range(400):
        p = 0.95 if t < 150 else 0.50                # certified skill breaks at t=150
        if d.update(1.0 if rng.random() < p else 0.0) and fired is None:
            fired = t
    check("drift detector fires after the certified skill breaks", fired is not None and fired < 260,
          f"fired at t={fired} (true change at 150)")
    fa = 0
    for s in range(100):
        rng = np.random.default_rng(1000 + s)
        d = DriftDetector(p0=0.95)
        ever = any(d.update(1.0 if rng.random() < 0.95 else 0.0) for _ in range(400))
        fa += ever
    check("drift detector false-alarm rate ≤ alpha under stationarity", fa / 100 <= 0.05,
          f"false-alarm rate={fa/100:.3f}")

    # ===== Q2d: behaviour-agnostic (passive) model-based OPE ================
    print("\n-- Q2: behaviour-agnostic OPE (certify from the buffer, no re-execution) --")
    w = DynamicalCausalWorld.default(np.random.default_rng(0))
    ag = ConstructorCausalAgent(w, seed=0); ag.explore(400)
    prog = tuple({0: 2.0} for _ in range(4))
    pa_reach = certify_passive(ag, prog, Box.from_dict({2: (1.5, 3.0)}))
    pa_unreach = certify_passive(ag, prog, Box.from_dict({3: (3.0, 3.5)}))
    check("passive certificate (buffer+model only) says POSSIBLE for a reachable skill",
          pa_reach["verdict"] == "POSSIBLE", f"p_model={pa_reach['p_hat']:.2f}, n={pa_reach['n']}")
    check("passive certificate says IMPOSSIBLE for an unreachable skill",
          pa_unreach["verdict"] == "IMPOSSIBLE", f"p_model={pa_unreach['p_hat']:.2f}")

    # calibration guard: a WRONG (linear) model is caught; the RFF model is trusted
    wn = DynamicalCausalWorld.nonlinear(np.random.default_rng(1))
    sat = wn.names.index("sat")
    prog_sat = tuple({1: 1.0} for _ in range(6))                 # intermediate setpoint
    satbox = Box.from_dict({sat: (2.3, 3.2)})                    # true sat(a1=1) ≈ 2.59
    a_lin = ConstructorCausalAgent(DynamicalCausalWorld.nonlinear(np.random.default_rng(1)),
                                   seed=1, rff=0); a_lin.explore_continuous(600)
    cal_lin = calibrate_passive(a_lin, DynamicalCausalWorld.nonlinear(np.random.default_rng(2)),
                                prog_sat, satbox, rng=np.random.default_rng(3))
    a_rff = ConstructorCausalAgent(DynamicalCausalWorld.nonlinear(np.random.default_rng(1)),
                                   seed=1, rff=20, rff_scale=2.0); a_rff.explore_continuous(600)
    cal_rff = calibrate_passive(a_rff, DynamicalCausalWorld.nonlinear(np.random.default_rng(2)),
                                prog_sat, satbox, rng=np.random.default_rng(3))
    check("calibration guard CATCHES a wrong (linear) model (not trustworthy)",
          not cal_lin["trustworthy"], f"gap={cal_lin['gap']:.2f} (p_model={cal_lin['p_model']:.2f} vs real {cal_lin['p_real']:.2f})")
    check("calibration guard TRUSTS the correct (RFF) model",
          cal_rff["trustworthy"], f"gap={cal_rff['gap']:.2f}")

    # ===== Q2e: MODEL-FREE OPE — right where a wrong model is wrong ==========
    print("\n-- Q2: model-free OPE (real transitions; correct where the model is wrong) --")
    SAT, A1, vstar, Rband = 3, 1, 1.0, (2.3, 3.2)            # nonlinear world indices

    def gt_stationary(v, steps=3000):                        # ground-truth sat mean
        w = DynamicalCausalWorld.nonlinear(np.random.default_rng(5)); x = w.reset()
        vals = [w.step({A1: v})[SAT] for _ in range(steps)][50:]
        return float(np.mean(vals))

    gt = gt_stationary(vstar)
    ag = ConstructorCausalAgent(DynamicalCausalWorld.nonlinear(np.random.default_rng(1)), seed=1)
    ag.explore_continuous(800)
    mf = certify_modelfree(ag, {A1: vstar}, Box.from_dict({SAT: Rband}), tau=0.9,
                           rng=np.random.default_rng(2))
    check("model-free OPE matches ground truth & certifies POSSIBLE (no model used)",
          mf["verdict"] == "POSSIBLE" and abs(mf["reliability"] - 1.0) < 0.15,
          f"reliability={mf['reliability']:.2f} (GT sat mean {gt:.2f}), n_onpolicy={mf['n_onpolicy']}")

    # the LINEAR model's stationary prediction is wrong on this nonlinear edge
    a_lin = ConstructorCausalAgent(DynamicalCausalWorld.nonlinear(np.random.default_rng(1)),
                                   seed=1, rff=0); a_lin.explore_continuous(800)
    x = np.zeros(a_lin.world.d)
    for _ in range(60):
        x, _ = a_lin.model.predict_next(x, {A1: vstar})
    check("model-FREE succeeds where the linear MODEL is wrong",
          abs(x[SAT] - gt) > 0.4, f"linear-model sat={x[SAT]:.2f} vs GT {gt:.2f} (model-free got it right)")

    # overlap guard: an action the behaviour never takes -> non-identifiable
    mf_no = certify_modelfree(ag, {A1: 5.0}, Box.from_dict({SAT: Rband}), tau=0.9,
                              rng=np.random.default_rng(2))
    check("overlap guard returns UNDECIDED with no coverage (provable wall)",
          mf_no["verdict"] == "UNDECIDED", f"n_onpolicy={mf_no['n_onpolicy']}")

    # ===== Q2f: CONTINUOUS-STATE model-free OPE (RBF+LSTD, no discretization) =
    print("\n-- Q2: continuous-state model-free OPE (RBF+LSTD; drops discretization) --")
    SAT2, A1b, vb, Rb, g = 3, 1, 1.0, (2.3, 3.2), 0.95

    def mc_disc(starts):                                  # ground-truth discounted reliability
        tot = []
        for x0 in starts[:120]:
            w = DynamicalCausalWorld.nonlinear(np.random.default_rng(7)); x = w.reset(np.array(x0))
            G = 0.0
            for t in range(200):
                x = w.step({A1b: vb}); G += (g ** t) * (Rb[0] <= x[SAT2] <= Rb[1])
            tot.append((1 - g) * G)
        return float(np.mean(tot))

    agc = ConstructorCausalAgent(DynamicalCausalWorld.nonlinear(np.random.default_rng(1)), seed=1)
    agc.explore_continuous(1500)
    cmd = {A1b: vb}; eff = Box.from_dict({SAT2: Rb})
    cont = certify_modelfree_continuous(agc, cmd, eff, gamma=g, rng=np.random.default_rng(2))
    starts = [xc for (xc, _) in agc.buffer if abs(xc[A1b] - vb) <= 0.4]
    mc = mc_disc(starts)
    # linear-SCM model-based discounted estimate from the same starts (the wrong baseline)
    a_lin = ConstructorCausalAgent(DynamicalCausalWorld.nonlinear(np.random.default_rng(1)), seed=1, rff=0)
    a_lin.explore_continuous(1500)
    mb = []
    for x0 in starts[:120]:
        x = np.array(x0, float); G = 0.0
        for t in range(200):
            x, _ = a_lin.model.predict_next(x, cmd); G += (g ** t) * (Rb[0] <= x[SAT2] <= Rb[1])
        mb.append((1 - g) * G)
    mb = float(np.mean(mb))
    check("continuous-state LSTD-RBF estimate is close to MC ground truth (|err|<0.15)",
          cont["value"] is not None and abs(cont["value"] - mc) < 0.15,
          f"LSTD={cont['value']:.2f} vs MC={mc:.2f}, n_onpolicy={cont['n_onpolicy']}")
    check("continuous model-free beats the wrong (linear-SCM) model-based estimate",
          abs(cont["value"] - mc) < abs(mb - mc) - 0.2,
          f"|LSTD-MC|={abs(cont['value']-mc):.2f} << |linear-model-MC|={abs(mb-mc):.2f} (model={mb:.2f})")
    no = certify_modelfree_continuous(agc, {A1b: 5.0}, eff, gamma=g, rng=np.random.default_rng(2))
    check("continuous OPE overlap guard returns UNDECIDED with no coverage",
          no["verdict"] == "UNDECIDED", f"n_onpolicy={no['n_onpolicy']}")

    # ===== Q2g: RESET-FREE whole-library self-certification (build-off of #1) ====
    print("\n-- Q2: reset-free library self-certification from one life-stream --")
    wlib = DynamicalCausalWorld.default(np.random.default_rng(1))
    aglib = ConstructorCausalAgent(wlib, seed=1)
    aglib.explore_continuous(800)              # learn the model from curiosity
    aglib.build_library()                      # mint+verify primitives (clone-based truth)
    aglib.practice()                           # rehearse SUSTAINED holds in one life (on-policy)
    # WELL-POSED stationary skills: the reset-free certificate measures long-run
    # occupancy, so it must be tested against a STATIONARY target (a settled-band hold),
    # NOT a from-rest primitive's transient/bundled box (which the slow chain overshoots —
    # the honest discounted-vs-stationary distinction). Measure chain1's settled band
    # under a sustained a0=+2 hold (clone, ground-truth band), then inject two skills that
    # CLAIM that band: a correct one (a0=+2) and a wrong one (a0=-2 can't reach it).
    wg = DynamicalCausalWorld.default(np.random.default_rng(9)); wg.reset()
    vals = [wg.step({0: 2.0})[2] for _ in range(300)][50:]
    band = (float(np.mean(vals)) - 0.5, float(np.mean(vals)) + 0.5)
    good = Constructor(name="hold_x0=+2(stationary)", precond=Box.any(),
                       effect=Box.from_dict({2: band}),
                       program=tuple({0: 2.0} for _ in range(8)),
                       provenance="primitive", reliability=1.0, n_trials=60)
    bad = Constructor(name="hold_x0=-2(wrong-claim)", precond=Box.any(),
                      effect=Box.from_dict({2: band}),
                      program=tuple({0: -2.0} for _ in range(8)),
                      provenance="primitive", reliability=1.0, n_trials=60)
    aglib.library.add(good); aglib.library.add(bad)
    rep = certify_library(aglib, rng=np.random.default_rng(3))
    g, b = rep["hold_x0=+2(stationary)"], rep["hold_x0=-2(wrong-claim)"]
    check("reset-free certificate DISCRIMINATES a good stationary skill from a wrong one",
          g["method"] == "model-free" and g["n"] >= 50 and (g["value"] or 0) >= 0.75
          and (b["value"] if b["value"] is not None else 0) < 0.5,
          f"good={g['value']} ({g['verdict']}) vs wrong-claim={b['value']} ({b['verdict']})")
    # the model-based passive branch screens a NON-hold (multi-knob) program, no clone
    multi = Constructor(name="multi-knob", precond=Box.any(),
                        effect=Box.from_dict({2: (1.5, 3.0)}),
                        program=({0: 2.0}, {1: 0.0}), provenance="primitive",
                        reliability=1.0, n_trials=60)
    aglib.library.add(multi)
    rep2 = certify_library(aglib, rng=np.random.default_rng(4))
    mb = rep2.get("multi-knob", {})
    check("non-hold program routed to model-based passive certificate (no clone)",
          mb.get("method") == "model-based"
          and mb.get("verdict") in ("POSSIBLE", "IMPOSSIBLE", "UNDECIDED"),
          f"multi-knob → {mb.get('method')}, verdict={mb.get('verdict')}, p={mb.get('value')}")

    # ===== Q2h: OVERLAP-DRIVEN CURIOSITY (act to make UNDECIDED skills certifiable) =
    print("\n-- Q2: overlap-driven curiosity (reward-free: resolve what I can DO) --")
    # Thin start: a short curiosity run leaves a well-posed stationary skill under-covered,
    # so its reset-free certificate is UNDECIDED for lack of on-policy coverage. The agent
    # then acts to RESOLVE that certificate — no reward, only the drive to decide possibility.
    def build_curio(seed):
        a = ConstructorCausalAgent(DynamicalCausalWorld.default(np.random.default_rng(seed)), seed=seed)
        a.explore(60); a.build_library()
        wgg = DynamicalCausalWorld.default(np.random.default_rng(9)); wgg.reset()
        bnd = float(np.mean([wgg.step({0: 2.0})[2] for _ in range(300)][50:]))
        a.library.add(Constructor(name="good", precond=Box.any(),
                                  effect=Box.from_dict({2: (bnd - 0.5, bnd + 0.5)}),
                                  program=tuple({0: 2.0} for _ in range(8)),
                                  provenance="primitive", reliability=1.0, n_trials=60))
        return a
    ad = build_curio(1)
    rep = ad.explore_to_certify(200, min_onpolicy=100, tau=0.8)   # directed (coverage + EIG)
    ab = build_curio(1); ab.explore(200)                          # equal budget, pure EIG
    base = certify_library(ab, min_onpolicy=100, tau=0.8, rng=np.random.default_rng(1))["good"]
    gb, ga = rep["before"]["good"], rep["after"]["good"]
    check("overlap-driven curiosity RESOLVES an UNDECIDED skill → POSSIBLE (reset-free)",
          gb["verdict"] == "UNDECIDED" and ga["verdict"] == "POSSIBLE",
          f"before={gb['verdict']} → after={ga['verdict']} (value={ga['value']:.2f})")
    check("equal-budget pure-EIG leaves it UNDECIDED (sustained-hold coverage is the win)",
          base["verdict"] == "UNDECIDED" and (ga["value"] or 0) > (base["value"] or 0),
          f"directed value={ga['value']:.2f} (POSSIBLE) vs EIG value={base['value']:.2f} ({base['verdict']})")

    # ===== Q2i: overlap-driven curiosity WIRED INTO the continual loop ===========
    print("\n-- Q2: live_round(certify_seek=True) — targeted recovery after drift --")
    A0, C1 = 0, 2
    HIGH = Box.from_dict({C1: (1.0, 4.5)})
    def high_knob(ag):
        for c in ag.library.possible():
            b = {v: (lo, hi) for (v, lo, hi) in c.effect.bounds}
            if C1 in b and b[C1][0] > 0:
                return c.program[0].get(A0)
        return None
    wdr = DynamicalCausalWorld.default(np.random.default_rng(0))
    adr = ConstructorCausalAgent(wdr, seed=0, forget=0.94)
    wdr.A[C1, A0] = 0.80                                  # regime A: a0 drives chain1 (+)
    adr.explore(160); adr.build_library(setpoints=(-2.0, 2.0))
    knob_a = high_knob(adr)
    wdr.A[C1, A0] = -0.80                                 # DRIFT: the edge flips sign
    rep = adr.live_round(steps=200, certify_seek=True, certify_min=80)
    knob_b = high_knob(adr)
    c_goal, r_goal = adr.achieve(HIGH)
    check("certify_seek detects drift and RE-COVERS the reopened skills with fresh data",
          rep["changed"] and rep["recovered"] >= 1 and rep["pruned"] >= 1,
          f"changed={rep['changed']}, recovered={rep['recovered']}, pruned={rep['pruned']}")
    check("targeted recovery rebuilds the correct new-regime library & stays capable",
          knob_a == 2.0 and knob_b == -2.0 and c_goal is not None and r_goal >= POSSIBLE_TAU,
          f"chain1-high knob {knob_a}→{knob_b}, achieve r={r_goal:.2f}")

    # ===== Q1: higher-lag latent detection (specificity strong, sens. majority) =
    print("\n-- Q1: higher-lag latent detection (honest: strong specificity) --")
    cf, df = 0, 0
    seeds = range(5)
    for s in seeds:
        wc = DynamicalCausalWorld.confounded(np.random.default_rng(s))
        ac = ConstructorCausalAgent(wc, seed=s); ac.explore(900)
        cf += (1 in [i for (i, _, _) in detect_latent_lag(ac)])     # S2 == idx 1
        wd = DynamicalCausalWorld.default(np.random.default_rng(s))
        ad = ConstructorCausalAgent(wd, seed=s); ad.explore(900)
        df += (len(detect_latent_lag(ad)) == 0)
    check("SPECIFICITY: no false latent on the fully-observed world (5/5 seeds)",
          df == 5, f"clean on {df}/5 seeds")
    check("SENSITIVITY: detects the hidden confounder S2 in the majority of seeds",
          cf >= 3, f"flagged S2 on {cf}/5 seeds (data-hungry signal, honest)")

    n_pass = sum(ok for _, ok, _ in CHECKS)
    print("\n" + "=" * 78 + f"\n{n_pass}/{len(CHECKS)} checks passed\n" + "=" * 78)
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
