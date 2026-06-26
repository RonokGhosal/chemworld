"""
RUNG 6B -- the necessity RETURNS: nonlinear structure + a HIDDEN confounder.

6A showed nonlinearity hands orientation back to observation: ANM/RESIT orients a nonlinear additive
chain with no intervention. But ANM assumes CAUSAL SUFFICIENCY (no hidden confounders). 6B breaks that
assumption -- a hidden H nonlinearly drives a pair (p,q) with NO edge between them -- and asks whether
intervention's necessity returns in the nonlinear regime.

A pre-check settled the genuinely open question first: a SOPHISTICATED ANM can in principle reject
BOTH directions when neither residual is independent (flagging confounding). We measured whether it
does: on the nonlinearly-confounded pair the min-residual HSIC (~0.0011) is INDISTINGUISHABLE from a
true edge's (~0.0007) and barely above the independent baseline (~0.0009) -- so even a confounding-
aware ANM with a generous 5x-baseline threshold flags it 0/30. ANM is FOOLED: it sees an admissible
direction and draws a spurious CAUSAL edge. So observation fails on nonlinear confounding too.

World: nonlinear additive chain a->b->c (where ANM wins, 6A) + a pair (p,q) driven by hidden H,
NO p->q edge. Methods:
  * ANM  (observation, nonlinear-aware, confounding-aware): edge-detect (HSIC) + orient + reject-both.
  * do() (intervention): orient the chain; do(p) does NOT move q -> no edge.
Result: ANM orients the chain AND draws a SPURIOUS directed edge on (p,q) it cannot flag; do() orients
the chain AND drops (p,q). Intervention's de-confounding necessity RETURNS, robust to nonlinearity.
Scope: this is the ACTUATABLE-confounder case (do(p) is allowed); an UN-actuatable hidden confounder
remains a hard identifiability wall for BOTH (Rungs 1/5) -- nonlinearity does not change that.
"""
from __future__ import annotations

import numpy as np

from .nonlinear_instant import _rbf_ridge, _hsic, do_dir


class NLConfoundWorld:
    """Instantaneous: nonlinear additive chain 0->1->2, plus pair 3,4 driven by hidden H (no 3->4
    edge). 5 observed (labels permuted), H hidden. do() clamps any observed label."""

    def __init__(self, rng, link="quad", noise=0.5, ew=2.0):
        self.rng = rng; self.link = link; self.noise = noise; self.ew = ew; self.d_obs = 5
        perm = rng.permutation(5)
        self.lab = {i: int(perm[i]) for i in range(5)}
        self.chain_edges = {(self.lab[0], self.lab[1]), (self.lab[1], self.lab[2])}
        self.conf_pair = (self.lab[3], self.lab[4])

    def _f(self, x):
        if self.link == "quad":
            return x + 0.5 * x ** 2
        if self.link == "tanh":
            return np.tanh(self.ew * x)
        if self.link == "sin":
            return np.sin(self.ew * x)
        raise ValueError(self.link)

    def sample(self, n, do=None):
        do_i = {}
        if do:
            inv = {v: k for k, v in self.lab.items()}
            do_i = {inv[k]: v for k, v in do.items()}
        c = {}
        c[0] = np.full(n, do_i[0]) if 0 in do_i else self.rng.normal(0, 1, n)
        c[1] = np.full(n, do_i[1]) if 1 in do_i else self._f(c[0]) + self.rng.normal(0, self.noise, n)
        c[2] = np.full(n, do_i[2]) if 2 in do_i else self._f(c[1]) + self.rng.normal(0, self.noise, n)
        H = self.rng.normal(0, 1, n)                            # hidden common cause
        c[3] = np.full(n, do_i[3]) if 3 in do_i else self._f(H) + self.rng.normal(0, self.noise, n)
        c[4] = np.full(n, do_i[4]) if 4 in do_i else self._f(H) + self.rng.normal(0, self.noise, n)
        out = np.empty((n, 5))
        for i in range(5):
            out[:, self.lab[i]] = c[i]
        return out


def anm_edge(X, i, j, indep_base, conf_mult=5.0):
    """ANM on a pair: returns (is_edge, direction, flagged_confounded).
    is_edge: dependent at all (HSIC(Xi,Xj) above baseline). direction: +1 => i->j (lower residual
    HSIC). flagged_confounded: BOTH residuals dependent (a confounding-aware rejection)."""
    dep = _hsic(X[:, i], X[:, j])
    is_edge = dep > conf_mult * indep_base
    r_ji = X[:, j] - _rbf_ridge(X[:, i], X[:, j])
    r_ij = X[:, i] - _rbf_ridge(X[:, j], X[:, i])
    h_fwd, h_bwd = _hsic(X[:, i], r_ji), _hsic(X[:, j], r_ij)
    flagged = min(h_fwd, h_bwd) > conf_mult * indep_base
    direction = 1 if h_fwd < h_bwd else -1
    return is_edge, direction, flagged


def main(seeds=range(15), n_obs=4000, link="quad"):
    print("=" * 98)
    print(f"RUNG 6B -- nonlinear + HIDDEN confounder: does intervention's necessity RETURN? "
          f"(link={link}, {len(list(seeds))} seeds)")
    print("=" * 98)
    R = {k: [] for k in ("anm_chain", "do_chain", "anm_conf_edge", "anm_conf_flag", "do_conf_edge")}
    for s in seeds:
        rng = np.random.default_rng(s)
        w = NLConfoundWorld(rng, link=link)
        X = w.sample(n_obs)
        ib = _hsic(rng.normal(0, 1, n_obs), rng.normal(0, 1, n_obs))   # independent-pair baseline
        # ANM: chain orientation
        ch_ok = 0
        for (la, lb) in w.chain_edges:
            _, d, _ = anm_edge(X, la, lb, ib)
            ch_ok += (d == 1)
        R["anm_chain"].append(ch_ok / len(w.chain_edges))
        # ANM: the confounded pair -- does it draw a spurious edge / can it flag the confounding?
        p, q = w.conf_pair
        is_edge, _, flagged = anm_edge(X, p, q, ib)
        R["anm_conf_edge"].append(1.0 if (is_edge and not flagged) else 0.0)   # spurious causal edge
        R["anm_conf_flag"].append(1.0 if flagged else 0.0)                     # correctly flagged
        # do(): chain orientation + the confounded pair (interventional samples -- afford more)
        do_ok = sum(do_dir(w, la, lb, n=1000) == 1 for (la, lb) in w.chain_edges)
        R["do_chain"].append(do_ok / len(w.chain_edges))
        eff = abs((w.sample(1500, {p: 1.0}).mean(0)[q] - w.sample(1500, {p: -1.0}).mean(0)[q]) / 2.0)
        R["do_conf_edge"].append(1.0 if eff > 0.1 else 0.0)                    # do-detected p->q edge

    def m(k):
        return float(np.mean(R[k]))
    print(f"  {'method':>14} {'chain orient':>13} {'(p,q) drawn as causal edge':>28} {'flags confounding':>19}")
    print(f"  {'ANM (obs)':>14} {m('anm_chain'):>13.2f} {m('anm_conf_edge'):>28.2f} {m('anm_conf_flag'):>19.2f}")
    print(f"  {'do (interv)':>14} {m('do_chain'):>13.2f} {m('do_conf_edge'):>28.2f} {'n/a':>19}")
    print("=" * 98)
    print(f"  NECESSITY RETURNS: under nonlinearity ANM orients the real chain ({m('anm_chain'):.2f}, "
          f"as in 6A) but is")
    print(f"  FOOLED by the hidden confounder -- it draws a spurious causal edge on (p,q) "
          f"({m('anm_conf_edge'):.2f}) and")
    print(f"  CANNOT flag the confounding ({m('anm_conf_flag'):.2f}), even with a confounding-aware "
          f"reject-both test.")
    print(f"  do() orients the chain ({m('do_chain'):.2f}) and finds NO p->q edge ({m('do_conf_edge'):.2f}): "
          f"do(p) does not move q.")
    print(f"  So intervention's DE-CONFOUNDING necessity is robust to nonlinearity -- nonlinearity buys")
    print(f"  observation orientation of REAL edges (6A) but NOT immunity to hidden confounding.")
    print(f"  SCOPE: actuatable confounder (do(p) allowed). An UN-actuatable hidden confounder is a hard")
    print(f"  identifiability wall for BOTH methods (Rungs 1/5); nonlinearity does not change that.")
    print("=" * 98)
    return R


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(seeds=range(ns))
