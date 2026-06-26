"""
RUNG 5 -- where intervention is NECESSARY, not just better: INSTANTANEOUS (contemporaneous) causal
structure with hidden confounding -- and the HONEST scope of that "necessary".

Parts 1-4 used one-step-lagged worlds (x_{t+1}=A x_t), so TIME orients every edge and observation
alone recovers structure -- the reviews rightly noted intervention was never strictly required. Here
the structure is INSTANTANEOUS: within one experiment a linear system x = B x + e settles, with a
hidden confounder adding correlated noise to a pair. Three motifs (d=8, labels permuted to hide order):
  * CHAIN     a -> b -> c : no collider, so a->b->c, a<-b->c, a<-b<-c are Markov-equivalent.
  * COLLIDER  d -> e <- f : a v-structure OBSERVATION can orient (so passive is demonstrably COMPETENT).
  * CONFOUND  (p,q) via hidden H : observation draws a p-q edge it cannot tell from a real one.

PASSIVE is a real constraint-based learner (PC: partial-correlation skeleton + v-structure
orientation). ACTIVE intervenes: do(i=+/-delta), measure each j, then TRANSITIVE-REDUCE to the direct
DAG (do() recovers the transitive *closure*: do(a) moves c through b, so the raw rule emits a->c --
we remove it so ACTIVE is graded on direct-edge recovery, the same footing as PASSIVE). Both methods
get the SAME sample budget and BOTH report skeleton F1.

SCOPE (the honest caveat): "observation cannot orient the chain" is a theorem for LINEAR-GAUSSIAN
noise. Under linear NON-Gaussian noise, a LiNGAM/ICA measure orients the chain from observation alone.
We demonstrate this directly. So intervention is the DISTRIBUTION-ROBUST route to orientation -- it
works whatever the noise -- and it is strictly *necessary* only under Gaussianity (or if one restricts
to CI-based methods). That is the precise, defensible claim.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


class InstantWorld:
    """x = B x + e settled in topological order; a hidden confounder adds correlated noise to a pair.
    B[i,j] = instantaneous effect of j on i. Variable LABELS are permuted so the topological order is
    hidden. noise_dist in {'gaussian','laplace'} (laplace = linear non-Gaussian, for the scope arm)."""

    def __init__(self, edges, conf_pairs, d, rng, w=0.8, noise=1.0, conf_str=1.2, noise_dist="gaussian"):
        self.d = d; self.rng = rng; self.noise = noise; self.noise_dist = noise_dist
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

    def _noise(self, size):
        if self.noise_dist == "laplace":
            return self.rng.laplace(0.0, self.noise / np.sqrt(2.0), size)   # var = noise^2
        return self.rng.normal(0.0, self.noise, size)

    def sample(self, do=None):
        e = self._noise(self.d)
        for (a, b) in self.conf:
            h = float(self._noise(1)[0]) * self.conf_str
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
    """PC skeleton: drop i-j if some conditioning set S (|S|<=max_S) makes partial corr < alpha."""
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
    """Orient unshielded colliders i->k<-j (k not in sepset(i,j))."""
    directed = set()
    for k in range(d):
        for a, b in combinations(sorted(adj[k]), 2):
            if frozenset((a, b)) not in edges and k not in sepset.get(frozenset((a, b)), set()):
                directed.add((a, k)); directed.add((b, k))
    return directed


# ----------------------------- LiNGAM: orient under NON-Gaussianity (scope arm) -----------------------------
def lingam_dir(x, y):
    """Hyvarinen-Smith pairwise measure: +1 => x->y. ~0 (chance) for Gaussian; identifies for
    super-Gaussian/skewed noise. (CI-based PC ignores this higher-order info.)"""
    x = (x - x.mean()) / (x.std() + 1e-9); y = (y - y.mean()) / (y.std() + 1e-9)
    rho = float(np.mean(x * y))
    M = rho * (np.mean(x * np.tanh(y)) - np.mean(np.tanh(x) * y))
    return 1 if M > 0 else -1


# ----------------------------- ACTIVE: interventional -----------------------------
def active_discover(world, n=300, delta=2.0, thresh=0.15):
    d = world.d
    eff = np.zeros((d, d))
    for i in range(d):
        xp = world.collect(n, do={i: delta}).mean(0)
        xm = world.collect(n, do={i: -delta}).mean(0)
        eff[:, i] = (xp - xm) / (2 * delta)                  # eff[j,i] = effect of do(i) on j
    raw = set()
    for i in range(d):
        for j in range(d):
            if i != j and abs(eff[j, i]) > thresh and abs(eff[i, j]) <= thresh:
                raw.add((i, j))                               # i moves j, j doesn't move i (ancestral)
    return transitive_reduce(raw, d), eff                     # -> DIRECT DAG (drop a->c etc.)


def transitive_reduce(directed, d):
    """Remove i->j when a longer directed path i->...->k->...->j exists (do() yields the transitive
    closure; we recover the direct edges)."""
    reach = {i: set() for i in range(d)}
    for (i, j) in directed:
        reach[i].add(j)
    changed = True
    while changed:
        changed = False
        for i in range(d):
            add = set().union(*[reach[j] for j in reach[i]]) if reach[i] else set()
            if not add <= reach[i]:
                reach[i] |= add; changed = True
    reduced = set()
    for (i, j) in directed:
        if not any(k != i and k != j and k in reach[i] and j in reach[k] for k in range(d)):
            reduced.add((i, j))
    return reduced


# ----------------------------- scoring -----------------------------
def _orient_frac(directed, true_dir):
    if not true_dir:
        return 1.0, 0.0
    correct = sum(1 for e in true_dir if e in directed)
    rev = sum(1 for (a, b) in true_dir if (b, a) in directed)
    return correct / len(true_dir), rev / len(true_dir)


def _skel_f1(edge_set, true_skel):
    skel = set(frozenset(e) for e in edge_set)
    tp = len(skel & true_skel); fp = len(skel - true_skel); fn = len(true_skel - skel)
    return tp / (tp + 0.5 * (fp + fn)) if tp else 0.0


def main(seeds=range(15), n_active=300):
    print("=" * 98)
    print(f"RUNG 5 -- INSTANTANEOUS structure + hidden confounding: observe (PC) vs intervene (do) "
          f"({len(list(seeds))} seeds)")
    print("=" * 98)
    edges = [(0, 1), (1, 2), (3, 5), (4, 5)]                  # chain 0->1->2 ; collider 3->5,4->5
    confs = [(6, 7)]                                          # confounded pair via hidden H ; d=8
    d = 8
    n_obs = d * 2 * n_active                                  # SAMPLE-FAIR: passive gets active's budget
    CH = lambda w: {(w.lab[0], w.lab[1]), (w.lab[1], w.lab[2])}
    CO = lambda w: {(w.lab[3], w.lab[5]), (w.lab[4], w.lab[5])}
    R = {k: [] for k in ("p_chain", "p_coll", "p_conf", "p_f1",
                          "a_chain", "a_rev", "a_coll", "a_conf", "a_f1", "a_extra")}
    SCOPE = {"g_pc": [], "g_lin": [], "g_do": [], "l_pc": [], "l_lin": [], "l_do": []}
    for s in seeds:
        w = InstantWorld(edges, confs, d=d, rng=np.random.default_rng(s))
        chain, coll = CH(w), CO(w)
        true_skel = set(frozenset(e) for e in w.true_dir)    # 4 DIRECT edges; confounded p-q is NOT one
        # PASSIVE (PC) -- sample-fair
        X = w.collect(n_obs)
        edges_p, sep, adj = pc_skeleton(X)
        dir_p = pc_orient(edges_p, sep, adj, d)
        R["p_chain"].append(_orient_frac(dir_p, chain)[0])
        R["p_coll"].append(_orient_frac(dir_p, coll)[0])
        R["p_conf"].append(1.0 if any(fs in edges_p for fs in w.true_conf) else 0.0)
        R["p_f1"].append(_skel_f1(edges_p, true_skel))       # p-q counts as a FALSE positive
        # ACTIVE (do) -- transitive-reduced to direct DAG, graded the SAME way
        dir_a, eff = active_discover(w, n=n_active)
        cf, rev = _orient_frac(dir_a, chain)
        R["a_chain"].append(cf); R["a_rev"].append(rev)
        R["a_coll"].append(_orient_frac(dir_a, coll)[0])
        R["a_conf"].append(1.0 if any((a, b) in dir_a or (b, a) in dir_a
                                       for fs in w.true_conf for (a, b) in [tuple(fs)]) else 0.0)
        R["a_f1"].append(_skel_f1(dir_a, true_skel))
        R["a_extra"].append(len(dir_a - w.true_dir))         # transitive/spurious extras (should be ~0)
        # SCOPE arm: chain orientation by PC vs LiNGAM vs do, under gaussian vs laplace noise
        for tag, dist in (("g", "gaussian"), ("l", "laplace")):
            wd = InstantWorld(edges, confs, d=d, rng=np.random.default_rng(1000 + s), noise_dist=dist)
            Xd = wd.collect(n_obs)
            ep, spd, adp = pc_skeleton(Xd)
            SCOPE[f"{tag}_pc"].append(_orient_frac(pc_orient(ep, spd, adp, d), CH(wd))[0])
            lin = np.mean([1.0 if lingam_dir(Xd[:, wd.lab[a]], Xd[:, wd.lab[b]]) == 1 else 0.0
                           for (a, b) in [(0, 1), (1, 2)]])
            SCOPE[f"{tag}_lin"].append(lin)
            da, _ = active_discover(wd, n=n_active)
            SCOPE[f"{tag}_do"].append(_orient_frac(da, CH(wd))[0])
        print(f"  seed {s} done")

    def m(k, src=R):
        return float(np.mean(src[k]))
    print(f"\n  {'motif':>26} {'PASSIVE (PC)':>14} {'ACTIVE (do)':>13}   [both sample-fair: {n_obs} draws]")
    print(f"  {'CHAIN a->b->c orient':>26} {m('p_chain'):>14.2f} {m('a_chain'):>13.2f}")
    print(f"  {'COLLIDER d->e<-f orient':>26} {m('p_coll'):>14.2f} {m('a_coll'):>13.2f}")
    print(f"  {'CONFOUND p-q drawn as edge':>26} {m('p_conf'):>14.2f} {m('a_conf'):>13.2f}")
    print(f"  {'skeleton F1 (direct edges)':>26} {m('p_f1'):>14.2f} {m('a_f1'):>13.2f}")
    print("=" * 98)
    print(f"  Graded the SAME way: ACTIVE is transitive-reduced (extras {m('a_extra'):.2f}/seed) so a->c")
    print(f"  is dropped; PASSIVE (PC) is COMPETENT -- orients the COLLIDER ({m('p_coll'):.2f}) -- but")
    print(f"  CANNOT orient the CHAIN ({m('p_chain'):.2f}; Markov-equivalent) and is forced to draw the")
    print(f"  confounded pair ({m('p_conf'):.2f}), costing it skeleton F1 ({m('p_f1'):.2f} vs ACTIVE")
    print(f"  {m('a_f1'):.2f}). ACTIVE orients the chain ({m('a_chain'):.2f}, reversed {m('a_rev'):.2f})")
    print(f"  and drops the confounder ({m('a_conf'):.2f}).")
    print("-" * 98)
    print(f"  SCOPE -- is intervention NECESSARY, or only for CI-based methods under Gaussianity?")
    print(f"  {'noise':>10} {'PC-chain':>10} {'LiNGAM-chain':>14} {'do-chain':>10}")
    print(f"  {'gaussian':>10} {m('g_pc',SCOPE):>10.2f} {m('g_lin',SCOPE):>14.2f} {m('g_do',SCOPE):>10.2f}")
    print(f"  {'laplace':>10} {m('l_pc',SCOPE):>10.2f} {m('l_lin',SCOPE):>14.2f} {m('l_do',SCOPE):>10.2f}")
    print("=" * 98)
    print(f"  HONEST SCOPE: under linear NON-Gaussian (laplace) noise OBSERVATION can orient the chain")
    print(f"  (LiNGAM {m('l_lin',SCOPE):.2f}, exploiting non-Gaussianity PC ignores); under GAUSSIAN noise")
    print(f"  it provably cannot (LiNGAM {m('g_lin',SCOPE):.2f} ~ chance). do() orients REGARDLESS")
    print(f"  (gaussian {m('g_do',SCOPE):.2f}, laplace {m('l_do',SCOPE):.2f}). So intervention is the")
    print(f"  DISTRIBUTION-ROBUST route to orientation, and strictly *necessary* only under Gaussianity")
    print(f"  (or if one restricts to CI-based discovery). That is the precise claim.")
    print("=" * 98)
    return R, SCOPE


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(seeds=range(ns))
