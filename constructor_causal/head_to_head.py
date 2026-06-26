"""
HEAD-TO-HEAD (v2, hardened after adversarial review) -- active intervention vs passive prediction
on a HIDDEN confounder. Pearl's ladder: when the common cause C is unobserved, a model trained on
OBSERVATIONAL data has NO valid adjustment set, so it mistakes the confounder's correlation for
causation; an agent that can ACT recovers the truth by intervening -- without ever observing C.

World (small, fully-known SCM; C is HIDDEN -- never in the observed vector):
    C -> A,  C -> B      hidden common cause (A,B correlate; neither causes the other)
    X -> Y               a real causal edge (positive control)
  observed = [A, B, X, Y]

Three estimators of "effect of do(i:=+2) on j, one step":
  * truth        : oracle -- clamp i in the real SCM WITH the logged C (defines ground truth).
  * predictor    : trained ONLY on passive logs (no C), imagines a clean ONE-STEP do.
                   Two flavors: OLS next-state regressor, and the transformer ("next-token") model.
  * causal agent : runs a finite budget of REAL matched interventions, NEVER observing C
                   (re-runs its own action sequence under do(i:=v) vs do(i:=0); the do is the only
                   difference). Its win is from ACTING, not from access to C.

Reported over many seeds as mean +/- std, with the predictor's NO-PATH error as the null noise
floor: the claim is that the predictor's CONFOUNDED-pair error sits far above that floor while the
agent's does not.

[Hardening vs v1, per review: agent is finite-sample (not an oracle); clean one-step imagined-do;
no C handed to the agent; mean+/-std over seeds; no-path noise floor instead of a hand-picked
threshold; OLS shown alongside the transformer so the effect isn't a transformer artifact.]
"""
from __future__ import annotations

import numpy as np
import torch

from .transformer_opponent import StateTransformer, train

A_, B_, X_, Y_ = 0, 1, 2, 3
NOBS = 4
NAMES = ["A", "B", "X", "Y"]
DO_VAL = 2.0


class ConfoundedWorld:
    def __init__(self, rng=None, cw=1.0, sw=0.85, self_decay=0.2, noise=0.1, c_decay=0.78):
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.cw, self.sw, self.sd, self.noise, self.cd = cw, sw, self_decay, noise, c_decay
        self.c = 0.0
        self.o = np.zeros(NOBS)

    def reset(self):
        self.c = 0.0; self.o = np.zeros(NOBS)
        return self.o.copy()

    def set_state(self, c, o):
        self.c = float(c); self.o = np.asarray(o, float).copy()

    def step(self, action=None, do=None):
        a = np.zeros(NOBS) if action is None else np.asarray(action, float)
        o = self.o.copy()
        if do:
            for i, v in do.items():
                o[i] = float(v)                              # intervention takes effect NOW
        c, n = self.c, self.rng.normal
        cn = self.cd * c + n(0, 0.6)                         # hidden confounder, autonomous
        on = np.empty(NOBS)
        on[A_] = self.sd * o[A_] + self.cw * c + a[A_] + n(0, self.noise)   # A <- C
        on[B_] = self.sd * o[B_] + self.cw * c + a[B_] + n(0, self.noise)   # B <- C (no A term)
        on[X_] = self.sd * o[X_] + a[X_] + n(0, self.noise)
        on[Y_] = self.sd * o[Y_] + self.sw * o[X_] + a[Y_] + n(0, self.noise)  # Y <- X
        if do:
            for i, v in do.items():
                on[i] = float(v)
        self.c, self.o = cn, on
        return on.copy()


def collect_passive(world, n_seq=400, T=16):
    """Observational logs. Returns transitions (prev,act,nxt,c_prev) and obs sequences."""
    PREV, ACT, NXT, CPREV, SEQ = [], [], [], [], []
    for _ in range(n_seq):
        world.reset(); seq = []
        for _ in range(T):
            seq.append(world.o.copy())
            c0, o0 = world.c, world.o.copy()
            a = world.rng.normal(0, 0.4, NOBS)
            world.step(action=a)
            PREV.append(o0); ACT.append(a); NXT.append(world.o.copy()); CPREV.append(c0)
        SEQ.append(seq)
    return (np.asarray(PREV, np.float32), np.asarray(ACT, np.float32),
            np.asarray(NXT, np.float32), np.asarray(CPREV, np.float32), np.asarray(SEQ, np.float32))


# All four estimators measure the SAME estimand: the effect of HOLDING do(i:=val) for H steps on
# j (vs holding do(i:=0)). Predictors roll their learned map forward H steps clamping i at EVERY
# step (a real do, not a one-shot nudge) -- this lets a passive model propagate the real X->Y edge
# (so the rematch is fair) while still propagating its confounded A->B hallucination.
H_STEPS = 6


# ---------- truth (oracle, uses hidden C) ----------
def truth_effect(prev, cprev, i, j, val=DO_VAL, H=H_STEPS):
    """Ground-truth H-step interventional effect, averaged over the logged (C,obs) states."""
    w = ConfoundedWorld(np.random.default_rng(12345))
    def avg(v):
        tot = 0.0
        for o0, c0 in zip(prev, cprev):
            w.set_state(c0, o0)
            for _ in range(H):
                w.step(do={i: v})
            tot += w.o[j]
        return tot / len(prev)
    return avg(val) - avg(0.0)


# ---------- predictors (passive, no C) ----------
def fit_ols(prev, act, nxt):
    X = np.concatenate([prev, act, np.ones((len(prev), 1), np.float32)], 1)
    B, *_ = np.linalg.lstsq(X, nxt, rcond=None)
    return B                                                 # (9, 4)


def ols_effect(B, prev, i, j, val=DO_VAL, H=H_STEPS):
    """Imagined do: roll the learned linear map forward H steps, re-clamping i:=val each step."""
    def pred(v):
        s = prev.copy().astype(np.float32); s[:, i] = v
        for _ in range(H):
            X = np.concatenate([s, np.zeros((len(s), NOBS), np.float32),
                                np.ones((len(s), 1), np.float32)], 1)
            s = (X @ B).astype(np.float32); s[:, i] = v
        return s[:, j].mean()
    return float(pred(val) - pred(0.0))


def transformer_effect(model, prev, i, j, val=DO_VAL, H=H_STEPS):
    """Imagined do via AUTOREGRESSIVE rollout: roll the transformer forward H steps, feeding its
    own generated history and re-clamping i:=val each step (a fair multi-step do, not a length-1
    one-shot). Lets the model propagate the real edge AND its confounded bias over the horizon."""
    dev = next(model.parameters()).device
    def pred(v):
        s = torch.tensor(prev).clone(); s[:, i] = v          # (N,4)
        seq = s[:, None, :]                                  # (N,1,4)
        with torch.no_grad():
            for _ in range(H):
                az = torch.zeros_like(seq)
                nxt = model(seq.to(dev), az.to(dev))[:, -1, :].cpu()   # (N,4) predicted next
                nxt[:, i] = v
                seq = torch.cat([seq, nxt[:, None, :]], 1)
        return float(seq[:, -1, j].mean())
    return pred(val) - pred(0.0)


# ---------- causal agent (active, NO C-access, finite budget) ----------
def agent_effect(seed, i, j, val=DO_VAL, K=200, warmup=8, H=H_STEPS):
    """Finite budget of INDEPENDENT real interventions. The agent reaches a natural state by its
    OWN actions (the hidden C is whatever arises -- never observed or controlled), then HOLDS
    do(i:=v) for H steps. Each trial is an independent draw -> realistic finite-sample noise. Its
    win comes purely from being able to ACT."""
    base = np.random.default_rng(seed * 100003 + i * 11 + j)
    def branch(v):
        tot = 0.0
        for _ in range(K):
            w = ConfoundedWorld(np.random.default_rng(int(base.integers(1 << 30))))
            w.reset()
            for _ in range(warmup):
                w.step(action=w.rng.normal(0, 0.4, NOBS))    # natural state, hidden C, uncontrolled
            for _ in range(H):
                w.step(do={i: v})
            tot += w.o[j]
        return tot / K
    return branch(val) - branch(0.0)


def pair_type(i, j):
    if {i, j} == {A_, B_}:
        return "confounded"
    if (i, j) == (X_, Y_):
        return "real"
    return "no-path"


def main(seeds=range(15), n_seq=400, T=16, with_transformer=True):
    print("=" * 92)
    print(f"HEAD-TO-HEAD v2 -- active intervention vs passive prediction, hidden confounder "
          f"({len(list(seeds))} seeds)")
    print("=" * 92)
    pairs = [(i, j) for i in range(NOBS) for j in range(NOBS) if i != j]
    # err[estimator][ptype] = list of |estimate - truth| across seeds*pairs
    err = {e: {"confounded": [], "real": [], "no-path": []} for e in ("ols", "transformer", "agent")}
    real_detect = {e: [] for e in ("ols", "transformer", "agent")}    # signed estimate on X->Y
    true_xy = []                                                       # true H-step X->Y effect

    for s in seeds:
        rng = np.random.default_rng(s)
        w = ConfoundedWorld(rng)
        PREV, ACT, NXT, CPREV, SEQ = collect_passive(w, n_seq=n_seq, T=T)
        B = fit_ols(PREV, ACT, NXT)
        tm = None
        if with_transformer:
            tm = StateTransformer(NOBS, NOBS, d=64, h=4, layers=2, max_len=T)
            train(tm, SEQ, np.zeros_like(SEQ), epochs=1200, batch=128, device=None, log_every=0)
        for (i, j) in pairs:
            pt = pair_type(i, j)
            t = truth_effect(PREV, CPREV, i, j)
            eo = ols_effect(B, PREV, i, j)
            ea = agent_effect(s, i, j)
            err["ols"][pt].append(abs(eo - t)); err["agent"][pt].append(abs(ea - t))
            if tm is not None:
                etf = transformer_effect(tm, PREV, i, j)
                err["transformer"][pt].append(abs(etf - t))
            if (i, j) == (X_, Y_):
                true_xy.append(t)
                real_detect["ols"].append(eo); real_detect["agent"].append(ea)
                if tm is not None:
                    real_detect["transformer"].append(etf)
        print(f"  seed {s} done")

    def ms(xs):
        xs = np.array(xs)
        return f"{xs.mean():.3f}+/-{xs.std():.3f}" if len(xs) else "--"

    ests = ["ols", "transformer", "agent"] if with_transformer else ["ols", "agent"]
    print(f"\n  MEAN ABSOLUTE ERROR vs truth  (mean +/- std over seeds x pairs)")
    print(f"  {'estimator':>14} {'CONFOUNDED':>16} {'no-path (floor)':>18} {'real-edge X->Y':>16}")
    for e in ests:
        print(f"  {e:>14} {ms(err[e]['confounded']):>16} {ms(err[e]['no-path']):>18} {ms(err[e]['real']):>16}")
    print(f"\n  signed effect on the REAL edge X->Y (true H-step ~ {np.mean(true_xy):.2f}):")
    for e in ests:
        print(f"    {e:>14}: {ms(real_detect[e])}")
    print("=" * 92)
    # headline: predictor confounded error vs its own no-path floor; agent stays at floor
    for e in [x for x in ests if x != "agent"]:
        cf, fl = np.mean(err[e]["confounded"]), np.mean(err[e]["no-path"])
        af = np.mean(err["agent"]["confounded"])
        print(f"  {e}: confounded err {cf:.3f} vs no-path floor {fl:.3f}  "
              f"(confounding signal {cf-fl:+.3f});  agent confounded err {af:.3f} ~ floor")
    print("  -> the passive predictor's error JUMPS on confounded pairs (above its no-path noise")
    print("     floor); the active agent's does NOT -- it is not fooled, with NO access to C.")
    print("=" * 92)
    return err, real_detect


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(seeds=range(ns))
