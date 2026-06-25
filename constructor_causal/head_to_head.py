"""
HEAD-TO-HEAD -- active intervention vs a passively-trained predictor, on a HIDDEN CONFOUNDER.

The thesis (Pearl's ladder): a next-state predictor trained on OBSERVATIONAL data learns a
confounder's spurious correlation, so it MISPREDICTS the effect of an intervention. An agent that
ACTIVELY INTERVENES measures the true effect and is not fooled. Both get the SAME query; the only
difference is the predictor reasons from passive data while the causal agent is allowed to act.

World (small, fully-known SCM; C is HIDDEN -- never observed):
    C -> A,  C -> B      hidden common cause: A,B move together but neither causes the other
    X -> Y               a REAL causal edge (positive control the predictor should get right)
  observed = [A, B, X, Y]

Query: "effect of do(i := +2) on j, one step, averaged over realistic states."
  * truth        : clamp i in the REAL SCM (hidden C intact), measure j.            [ground truth]
  * predictor    : clamp i in the transformer's learned one-step prediction.        [IMAGINED do]
  * causal agent : clamp i in the REAL SCM but with a SMALL budget of samples.       [ACTIVE do]

Decisive cell: do(A) -> B. Truth ~ 0 (no edge). The predictor reports a large effect (it conflates
A's correlation-with-C for causation); the active agent reports ~ 0.
"""
from __future__ import annotations

import numpy as np
import torch

from .transformer_opponent import StateTransformer, train
from .device import get_device

A_, B_, X_, Y_ = 0, 1, 2, 3
NOBS = 4
NAMES = ["A", "B", "X", "Y"]


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
        """action: optional length-4 drive. do: optional {idx: value} clamp (an intervention).
        do(i):=v sets variable i to v BEFORE the dynamics run (so it propagates downstream this
        step) and holds it clamped in the output."""
        a = np.zeros(NOBS) if action is None else np.asarray(action, float)
        o = self.o.copy()
        if do:
            for i, v in do.items():
                o[i] = float(v)                              # intervention takes effect NOW
        c, n = self.c, self.rng.normal
        cn = self.cd * c + n(0, 0.6)                         # hidden confounder, autonomous
        on = np.empty(NOBS)
        on[A_] = self.sd * o[A_] + self.cw * c + a[A_] + n(0, self.noise)   # A <- C
        on[B_] = self.sd * o[B_] + self.cw * c + a[B_] + n(0, self.noise)   # B <- C  (NO A term)
        on[X_] = self.sd * o[X_] + a[X_] + n(0, self.noise)
        on[Y_] = self.sd * o[Y_] + self.sw * o[X_] + a[Y_] + n(0, self.noise)  # Y <- X
        if do:
            for i, v in do.items():
                on[i] = float(v)                             # stays clamped in the output
        self.c, self.o = cn, on
        return on.copy()


def collect_passive(world, n_seq=400, T=16):
    """Observational logs: small random drives, NO targeted interventions. Returns obs sequences
    (n_seq,T,4) for the transformer, plus the (C,obs) states (for real-world interventions)."""
    OB, ST = [], []
    for _ in range(n_seq):
        world.reset(); seq, st = [], []
        for _ in range(T):
            st.append((world.c, world.o.copy()))
            seq.append(world.o.copy())
            world.step(action=world.rng.normal(0, 0.4, NOBS))   # gentle excitation, no do()
        OB.append(seq); ST.append(st)
    return np.asarray(OB, np.float32), ST


def true_effect(states, i, val, j, baseline=0.0, noise_seed=1):
    """ACTIVE/true one-step interventional effect of do(i):=val on j, averaged over `states`
    (each a (c,obs)). With many states -> ground truth; with few -> the budgeted active agent."""
    rng = np.random.default_rng(noise_seed)
    w = ConfoundedWorld(rng)
    def avg(v):
        tot = 0.0
        for (c, o) in states:
            w.set_state(c, o); w.step(do={i: v}); tot += w.o[j]
        return tot / len(states)
    return avg(val) - avg(baseline)


def transformer_effect(model, windows, i, val, j, baseline=0.0):
    """IMAGINED one-step effect: feed a real window, clamp the LAST state's i, predict next, read
    j. The transformer can only reason from passive data -> it carries the confounded association."""
    dev = next(model.parameters()).device
    def avg(v):
        S = windows.clone()
        S[:, -1, i] = v                                       # clamp the intervened var (imagined do)
        Az = torch.zeros_like(S)
        with torch.no_grad():
            pred = model(S.to(dev), Az.to(dev))[:, -1, j]     # predicted next j
        return float(pred.mean())
    return avg(val) - avg(baseline)


def main(seed=0, n_seq=400, T=16):
    print("=" * 84)
    print("HEAD-TO-HEAD: imagined-do (passive transformer) vs active-do, hidden confounder C->A,B")
    print("=" * 84)
    rng = np.random.default_rng(seed)
    w = ConfoundedWorld(rng)
    OB, ST = collect_passive(w, n_seq=n_seq, T=T)

    # transformer (the "next-token" predictor), trained ONLY on passive logs
    Az = np.zeros_like(OB)
    m = StateTransformer(NOBS, NOBS, d=64, h=4, layers=2, max_len=T)
    train(m, OB, Az, epochs=1500, batch=128, device=None, log_every=0)

    flat = [s for seq in ST for s in seq]                     # all (c,obs) states
    truth_states = flat                                       # many -> ground truth
    agent_states = flat[::20]                                 # ~budgeted active agent (few do's)
    windows = torch.tensor(OB[:, :T])                         # real windows for the transformer

    def classify(t, p, c, eff=0.5, det=0.3):
        if abs(t) < eff:                                     # truth: NO real effect
            ph, ch = abs(p) > det, abs(c) > det
            if ph and not ch:
                return "PREDICTOR HALLUCINATES"
            return "both hallucinate" if ph and ch else "both ~0  (ok)"
        po = (p * t > 0) and abs(p) > 0.4 * abs(t)           # truth: real effect -> detect it?
        co = (c * t > 0) and abs(c) > 0.4 * abs(t)
        if po and co:
            return "both detect  (ok)"
        return "predictor misses" if co and not po else "causal misses"

    pairs = [(A_, B_, "confounded"), (B_, A_, "confounded"),
             (X_, Y_, "REAL causal"), (A_, Y_, "no path"), (X_, B_, "no path")]
    print(f"\n  effect of do(src:=+2) on tgt   (truth | predictor=imagined-do | causal=active-do)")
    print(f"  {'edge':>10} {'type':>12} {'truth':>8} {'predictor':>11} {'causal':>8}   {'verdict':>22}")
    rows, conf_perr, conf_cerr = [], [], []
    for i, j, kind in pairs:
        t = true_effect(truth_states, i, 2.0, j)
        p = transformer_effect(m, windows, i, 2.0, j)
        c = true_effect(agent_states, i, 2.0, j, noise_seed=7)
        print(f"  {NAMES[i]+'->'+NAMES[j]:>10} {kind:>12} {t:>8.2f} {p:>11.2f} {c:>8.2f}   {classify(t,p,c):>22}")
        rows.append((NAMES[i] + "->" + NAMES[j], kind, float(t), float(p), float(c)))
        if kind == "confounded":
            conf_perr.append(abs(p - t)); conf_cerr.append(abs(c - t))
    print("=" * 84)
    print(f"  CONFOUNDED-pair error (truth=0):  predictor {np.mean(conf_perr):.2f}   "
          f"active-causal {np.mean(conf_cerr):.2f}   "
          f"-> predictor is {np.mean(conf_perr)/max(np.mean(conf_cerr),1e-6):.0f}x more wrong")
    print("  The predictor, trained only on passive logs, mistakes the confounder's correlation for")
    print("  causation and hallucinates an A<->B effect. Acting (do) breaks the confounder -> truth.")
    print("=" * 84)
    return rows


if __name__ == "__main__":
    main()
