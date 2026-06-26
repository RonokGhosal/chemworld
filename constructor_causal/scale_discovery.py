"""
PART 4 -- where does intervention-based discovery BREAK as the world grows? (local, find the wall)

Two sweeps, reusing the discovery stack (ConstructorCausalAgent.explore + recovered_dag):

  SWEEP A (size: compute + recall wall): DynamicalCausalWorld.wide(k) -> d=2k+3 variables (a real
    chain a0->chain1->chain2 plus k actuator-driven distractor sensors, NO confounder). We grow d
    and measure F1, exploration steps, and WALL-CLOCK runtime. Tells us the compute cost and
    whether recall holds as the graph widens.

  SWEEP B (confounding wall): a fixed actuator chain plus a growing number of HIDDEN confounders,
    each driving two un-actuatable sensors (the generic-confounder case from Part 1, where
    intervention cannot help). We measure spurious false edges vs #confounders -- the statistical/
    identifiability wall that compute cannot fix.

Honest framing: with default LINEAR features the model is O(p^2)/step, p~d, so compute scales
gently; the real walls are (i) exploration budget / statistical power and (ii) un-actuatable
confounding, not raw FLOPs -- exactly the "compute can't buy past these" point.
"""
from __future__ import annotations

import time

import numpy as np

from .world import DynamicalCausalWorld
from .agent import ConstructorCausalAgent


def sweep_size(ks=(5, 10, 20, 40), steps_per_d=60, seeds=range(3)):
    print("=" * 84)
    print("SWEEP A -- size (wide-k): F1, steps, runtime as the graph widens")
    print(f"  {'k':>4} {'d':>5} {'steps':>7} {'F1':>14} {'runtime_s':>12}")
    print("-" * 84)
    rows = []
    for k in ks:
        f1, rt, d = [], [], None
        for s in seeds:
            w = DynamicalCausalWorld.wide(k=k, rng=np.random.default_rng(s))
            d = w.d
            n_steps = steps_per_d * d
            t0 = time.time()
            ag = ConstructorCausalAgent(w, seed=s, experimenter="epistemic")
            ag.explore(n_steps=n_steps)
            rec = ag.recovered_dag()
            f1.append(rec["f1"]); rt.append(time.time() - t0)
        print(f"  {k:>4} {d:>5} {steps_per_d*d:>7} {np.mean(f1):>8.2f}+/-{np.std(f1):<4.2f} "
              f"{np.mean(rt):>12.1f}")
        rows.append((k, d, np.mean(f1), np.mean(rt)))
    return rows


def _confounded_world(n_conf, rng, chain_len=3, sensor_noise=1.0):
    """a0 -> m1 -> ... -> m_chain (actuator-driven, recoverable) + n_conf HIDDEN confounders, each
    driving a fresh pair of un-actuatable sensors."""
    n_act = 1
    chain = list(range(n_act, n_act + chain_len))            # m1..m_chain
    sensors = list(range(n_act + chain_len, n_act + chain_len + 2 * n_conf))
    hidden = list(range(n_act + chain_len + 2 * n_conf, n_act + chain_len + 2 * n_conf + n_conf))
    d = n_act + chain_len + 2 * n_conf + n_conf
    names = (["a0"] + [f"m{i}" for i in range(1, chain_len + 1)]
             + [f"S{i}" for i in range(2 * n_conf)] + [f"H{i}" for i in range(n_conf)])
    A = np.zeros((d, d))
    A[chain[0], 0] = 0.9                                      # a0 -> m1
    for a, b in zip(chain, chain[1:]):
        A[b, a] = 0.7; A[b, b] = 0.2
    A[chain[0], chain[0]] = 0.2
    noise = np.full(d, 0.1)
    for k in range(n_conf):                                   # H_k -> S_{2k}, S_{2k+1}
        h = hidden[k]
        A[h, h] = 0.9; noise[h] = 0.6
        A[sensors[2 * k], h] = 0.95; A[sensors[2 * k + 1], h] = 0.9
        noise[sensors[2 * k]] = 0.15; noise[sensors[2 * k + 1]] = sensor_noise
    return DynamicalCausalWorld(A=A, b=np.zeros(d), noise_std=noise, actuators=(0,),
                                names=tuple(names), hidden=tuple(hidden), rng=rng)


def sweep_confounders(n_confs=(0, 1, 2, 4, 8), n_steps=800, seeds=range(3)):
    print("\n" + "=" * 84)
    print("SWEEP B -- confounding wall: spurious false edges vs #hidden confounders (un-actuatable)")
    print(f"  {'#conf':>6} {'d':>5} {'chain-recall':>14} {'#false-edges':>14}")
    print("-" * 84)
    rows = []
    for nc in n_confs:
        rec_recall, false_e, d = [], [], None
        for s in seeds:
            w = _confounded_world(nc, np.random.default_rng(s))
            d = w.d
            ag = ConstructorCausalAgent(w, seed=s, experimenter="epistemic")
            ag.explore(n_steps=n_steps)
            r = ag.recovered_dag()
            rec_recall.append(r["recall"]); false_e.append(len(r["extra"]))
        print(f"  {nc:>6} {d:>5} {np.mean(rec_recall):>8.2f}+/-{np.std(rec_recall):<4.2f} "
              f"{np.mean(false_e):>8.2f}+/-{np.std(false_e):<4.2f}")
        rows.append((nc, d, np.mean(rec_recall), np.mean(false_e)))
    return rows


def main():
    a = sweep_size()
    b = sweep_confounders()
    print("\n" + "=" * 84)
    print("WALLS:")
    print(f"  compute -- runtime grew {a[0][3]:.1f}s (d={a[0][1]}) -> {a[-1][3]:.1f}s (d={a[-1][1]}); "
          f"F1 at largest d = {a[-1][2]:.2f} (recall holds / breaks).")
    print(f"  confounding -- false edges {b[0][3]:.1f} (0 conf) -> {b[-1][3]:.1f} ({b[-1][0]} conf): "
          f"intervention CANNOT de-confound un-actuatable sensors, and it gets WORSE with more "
          f"hidden common causes -- a wall compute cannot buy past.")
    print("=" * 84)


if __name__ == "__main__":
    main()
