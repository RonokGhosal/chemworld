"""
RUNG 5 -- where intervention is NECESSARY, not just better: INSTANTANEOUS (contemporaneous) causal
structure with hidden confounding.

Parts 1-4 used one-step-lagged worlds (x_{t+1}=A x_t), so TIME orients every edge and observation
alone recovers structure -- the reviews rightly noted intervention was never strictly required. Here
the structure is INSTANTANEOUS: within one experiment a linear system x = B x + e settles, with a
hidden confounder adding correlated noise to a pair. The world has three motifs:
  * CHAIN     a -> b -> c : no collider, so OBSERVATION cannot orient it -- a->b->c, a<-b->c, a<-b<-c
              are Markov-equivalent (a fundamental theorem, not a weak baseline).
  * COLLIDER  d -> e <- f : OBSERVATION *can* orient this (the v-structure rule) -- included so the
              passive baseline is demonstrably COMPETENT, not a strawman.
  * CONFOUND  (p,q) via hidden H : OBSERVATION draws a p-q edge it cannot tell from a real one.

PASSIVE is a real constraint-based learner (PC: partial-correlation skeleton + v-structure
orientation). ACTIVE intervenes: do(i=+/-delta), measure each j. We MEASURE (not assert) that passive
orients the collider but NOT the chain, and keeps the confounded pair; active orients everything and
drops the confounded pair. This is the regime where intervention is necessary.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


class InstantWorld:
    """x = B x + e settled in topological order; a hidden confounder adds correlated noise to a pair.
    B[i,j] = instantaneous effect of j on i. Variable LABELS are permuted so the topological order is
    hidden (no method may exploit index order)."""

    def __init__(self, edges, conf_pairs, d, rng, w=0.8, noise=1.0, conf_str=1.2):
        self.d = d; self.rng = rng; self.noise = noise
        perm = rng.permutation(d)
        self.lab = {i: int(perm[i]) for i in range(d)}        # internal i -> observed label
        self.B = np.zeros((d, d))
        self.order = list(range(d))                           # internal solve order (acyclic: src<tgt)
        self.true_dir = set()
        for (src, tgt) in edges:
            self.B[tgt, src] = w
            self.true_dir.add((self.lab[src], self.lab[tgt]))
        self.conf = [(a, b) for (a, b) in conf_pairs]
        self.conf_str = conf_str
        self.true_conf = set(frozenset((self.lab[a], self.lab[b])) for (a, b) in conf_pairs)

    def sample(self, do=None):
        e = self.rng.normal(0, self.noise, self.d)
        for (a, b) in self.conf:
            h = self.rng.normal() * self.conf_str
            e[a] += h; e[b] += h
        x = np.zeros(self.d)
        do_i = do or {}
        for i in self.order:
            x[i] = do_i[i] if i in do_i else self.B[i] @ x + e[i]
        out = np.empty(self.d)
        for i in range(self.d):
            out[self.lab[i]] = x[i]
        return out

    def collect(self, n, do=None):
        di = None
        if do:
            inv = {v: k for k, v in self.lab.items()}
            di = {inv[k]: v for k, v in do.items()}
        return np.array([self.sample(di) for _ in range(n)])


# ----------------------------- PASSIVE: constraint-based (PC) -----------------------------
def _pcorr(C, i, j, S):
    idx = [i, j] + list(S)
    sub = C[np.ix_(idx, idx)]
    P = np.linalg.pinv(sub)
    return -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1] + 1e-12)


def pc_skeleton(X, alpha=0.06, max_S=3):
    """PC skeleton: drop i-j if some conditioning set S (|S|<=max_S) makes partial corr < alpha.
    Records the separating set. Returns (edges:set[frozenset], sepset:dict)."""
    d = X.shape[1]; C = np.corrcoef(X.T)
    adj = {i: set(range(d)) - {i} for i in range(d)}
    sepset = {}
    for sz in range(0, max_S + 1):
        for i in range(d):
            for j in [v for v in adj[i] if v > i]:
                cand = sorted(adj[i] - {j})
                if len(cand) < sz:
                    continue
                for S in combinations(cand, sz):
                    if abs(_pcorr(C, i, j, S)) < alpha:
                        adj[i].discard(j); adj[j].discard(i)
                        sepset[frozenset((i, j))] = set(S)
                        break
    edges = set(frozenset((i, j)) for i in range(d) for j in adj[i] if i < j)
    return edges, sepset, adj


def pc_orient(edges, sepset, adj, d):
    """Orient unshielded colliders i->k<-j (k not in sepset(i,j)). Returns directed set."""
    directed = set()
    for k in range(d):
        nb = sorted(adj[k])
        for a, b in combinations(nb, 2):
            if frozenset((a, b)) not in edges:               # unshielded triple a-k-b
                if k not in sepset.get(frozenset((a, b)), set()):
                    directed.add((a, k)); directed.add((b, k))
    return directed


# ----------------------------- ACTIVE: interventional -----------------------------
def active_discover(world, n=400, delta=2.0, thresh=0.15):
    d = world.d
    eff = np.zeros((d, d))
    for i in range(d):
        xp = world.collect(n, do={i: delta}).mean(0)
        xm = world.collect(n, do={i: -delta}).mean(0)
        eff[:, i] = (xp - xm) / (2 * delta)                  # eff[j,i] = effect of do(i) on j
    directed = set()
    for i in range(d):
        for j in range(d):
            if i != j and abs(eff[j, i]) > thresh and abs(eff[i, j]) <= thresh:
                directed.add((i, j))                          # i moves j, j doesn't move i
    return directed, eff


def _orient_frac(directed, true_dir):
    if not true_dir:
        return 1.0, 0.0
    correct = sum(1 for e in true_dir if e in directed)
    rev = sum(1 for (a, b) in true_dir if (b, a) in directed)
    return correct / len(true_dir), rev / len(true_dir)


def main(seeds=range(15), n=500, n_obs=4000):
    print("=" * 96)
    print(f"RUNG 5 -- INSTANTANEOUS structure + hidden confounding: observe (PC) vs intervene "
          f"({len(list(seeds))} seeds)")
    print("=" * 96)
    # internal: chain 0->1->2 ; collider 3->5, 4->5 ; confounded pair (6,7) via hidden H. d=8.
    edges = [(0, 1), (1, 2), (3, 5), (4, 5)]
    confs = [(6, 7)]
    CH = lambda w: {(w.lab[0], w.lab[1]), (w.lab[1], w.lab[2])}       # chain true edges (labels)
    CO = lambda w: {(w.lab[3], w.lab[5]), (w.lab[4], w.lab[5])}       # collider true edges (labels)
    R = {k: [] for k in ("p_chain", "p_coll", "p_conf", "p_skelF1",
                          "a_chain", "a_coll", "a_conf", "a_chain_rev")}
    for s in seeds:
        rng = np.random.default_rng(s)
        w = InstantWorld(edges, confs, d=8, rng=rng)
        chain, coll = CH(w), CO(w)
        true_skel = set(frozenset(e) for e in (w.true_dir | {frozenset(p) for p in []})) \
            | set(frozenset(e) for e in w.true_dir) | w.true_conf
        # PASSIVE (PC)
        X = w.collect(n_obs)
        edges_p, sep, adj = pc_skeleton(X)
        dir_p = pc_orient(edges_p, sep, adj, w.d)
        R["p_chain"].append(_orient_frac(dir_p, chain)[0])
        R["p_coll"].append(_orient_frac(dir_p, coll)[0])
        R["p_conf"].append(1.0 if any(fs in edges_p for fs in w.true_conf) else 0.0)
        tp = len(edges_p & true_skel); fp = len(edges_p - true_skel); fn = len(true_skel - edges_p)
        R["p_skelF1"].append(tp / (tp + 0.5 * (fp + fn)) if tp else 0.0)
        # ACTIVE
        dir_a, eff = active_discover(w, n=n)
        cf, rev = _orient_frac(dir_a, chain)
        R["a_chain"].append(cf); R["a_chain_rev"].append(rev)
        R["a_coll"].append(_orient_frac(dir_a, coll)[0])
        R["a_conf"].append(1.0 if any((a, b) in dir_a or (b, a) in dir_a
                                       for fs in w.true_conf for (a, b) in [tuple(fs)]) else 0.0)
        print(f"  seed {s} done")

    def m(k):
        return float(np.mean(R[k]))
    print(f"\n  {'motif':>24} {'PASSIVE (PC)':>14} {'ACTIVE (do)':>13}")
    print(f"  {'CHAIN a->b->c orient':>24} {m('p_chain'):>14.2f} {m('a_chain'):>13.2f}")
    print(f"  {'COLLIDER d->e<-f orient':>24} {m('p_coll'):>14.2f} {m('a_coll'):>13.2f}")
    print(f"  {'CONFOUND p-q kept/drawn':>24} {m('p_conf'):>14.2f} {m('a_conf'):>13.2f}")
    print(f"  {'skeleton F1 (passive)':>24} {m('p_skelF1'):>14.2f} {'--':>13}")
    print("=" * 96)
    print(f"  FAIR baseline: passive (PC) is COMPETENT -- it orients the COLLIDER ({m('p_coll'):.2f}, a")
    print(f"  v-structure observation CAN identify) and recovers a clean skeleton (F1 {m('p_skelF1'):.2f}).")
    print(f"  But it CANNOT orient the CHAIN ({m('p_chain'):.2f}) -- a<-b<-c / a->b->c / a<-b->c are")
    print(f"  Markov-equivalent -- and keeps the confounded pair as an edge ({m('p_conf'):.2f}).")
    print(f"  ACTIVE orients the chain ({m('a_chain'):.2f}, reversed {m('a_chain_rev'):.2f}) AND the")
    print(f"  collider ({m('a_coll'):.2f}), and does NOT draw the confounded pair ({m('a_conf'):.2f}).")
    print(f"  This is the regime where intervention is NECESSARY: with no temporal order, observation is")
    print(f"  stuck at the Markov-equivalence class; only acting resolves direction and confounding.")
    print("=" * 96)
    return R


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(seeds=range(ns))
