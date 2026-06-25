"""
THE PRIZE: held-out control FROM THE LEARNED LATENT. After reward-free exploration in raw
nonlinear observation space, the split encoder gives z_c + a learned z_c dynamics model
(forward_c). Can the agent control a held-out goal (drive m3 into a band) using ONLY the
learned representation -- and does the causal/noise-aware encoder beat a prediction-first
encoder and approach the oracle (true latent + true dynamics)?

This also tests rollout stability: MPC on z_c works only if forward_c rolls out stably.
"""
from __future__ import annotations

import numpy as np
import torch

from .messy_world import MessyWorld, M3, A0, A1, AN
from .repr_encoder import collect, ReprModel, train_encoder
from .split_encoder import SplitModel, train_split, CTRL

CTRL_SETPOINTS = [(-2.0, -2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 2.0), (0.0, 0.0)]


def fit_readout(zc, m3):
    X = np.column_stack([zc, np.ones(len(zc))])
    b, *_ = np.linalg.lstsq(X, m3, rcond=None)
    return b                                            # m3 ~ [zc,1] @ b


def mpc_split(world, model, readout, budget, hold=6, band=4.0):
    """MPC on the LEARNED latent: roll forward_c under candidate sustained control holds,
    pick the one whose predicted m3-readout is highest, execute on the real world."""
    steps, reached = 0, False
    while steps < budget and not reached:
        o = torch.tensor(world.observe(), dtype=torch.float32)
        with torch.no_grad():
            zc0, _ = model.encode(o)
        best, best_v = None, -1e9
        for (c0, c1) in CTRL_SETPOINTS:
            zc = zc0.clone(); a_ctrl = torch.tensor([c0, c1], dtype=torch.float32)
            with torch.no_grad():
                for _ in range(hold):
                    zc = model.forward_c(zc, a_ctrl)
            v = float(np.dot(np.append(zc.numpy(), 1.0), readout))
            if v > best_v:
                best_v, best = v, (c0, c1)
        for _ in range(hold):
            if steps >= budget:
                break
            world.step(np.array([best[0], best[1], 0.0], np.float32)); steps += 1
            if world.true_m3() >= band:
                reached = True; break
    return reached, world.true_m3()


def oracle_control(world, budget, hold=6, band=4.0):
    """Upper bound: plan with the TRUE latent + true dynamics."""
    steps, reached = 0, False
    while steps < budget and not reached:
        best, best_v = None, -1e9
        for (c0, c1) in CTRL_SETPOINTS:
            w2 = world.clone(); w2.z = world.z.copy()
            for _ in range(hold):
                w2.step(np.array([c0, c1, 0.0], np.float32))
            if w2.true_m3() > best_v:
                best_v, best = w2.true_m3(), (c0, c1)
        for _ in range(hold):
            if steps >= budget:
                break
            world.step(np.array([best[0], best[1], 0.0], np.float32)); steps += 1
            if world.true_m3() >= band:
                reached = True; break
    return reached, world.true_m3()


def run(agent, seed, n=4000, budget=24):
    rng = np.random.default_rng(seed)
    ew = MessyWorld(rng, obs_dim=14, nonlinear=True); ew.reset()
    Ob, A, Oa, Zb, Za = collect(ew, n, rng)
    gw = ew.clone(np.random.default_rng(seed + 999))    # SAME sensor map, fresh episode
    if agent == "oracle":
        return oracle_control(gw, budget)
    if agent == "causal_split":
        m = train_split(Ob, A, Oa, seed=seed)
        with torch.no_grad():
            zc, _ = m.encode(torch.tensor(Oa))
        readout = fit_readout(zc.numpy(), Za[:, M3])
        return mpc_split(gw, m, readout, budget)
    if agent == "prediction_first":
        # generic predictive encoder (no split, no structured forward, no noise-awareness)
        pm = train_encoder(Ob, A, Oa, dz=6, hetero=False, inverse=False, seed=seed)
        with torch.no_grad():
            z = pm.encode(torch.tensor(Oa))
        readout = fit_readout(z.numpy(), Za[:, M3])
        # MPC via the generic forward model
        return _mpc_generic(gw, pm, readout, budget)


def _mpc_generic(world, pm, readout, budget, hold=6, band=4.0):
    steps, reached = 0, False
    while steps < budget and not reached:
        o = torch.tensor(world.observe(), dtype=torch.float32)
        with torch.no_grad():
            z0 = pm.encode(o)
        best, best_v = None, -1e9
        for (c0, c1) in CTRL_SETPOINTS:
            z = z0.clone(); a = torch.tensor([c0, c1, 0.0], dtype=torch.float32)
            with torch.no_grad():
                for _ in range(hold):
                    mu, _ = pm.forward_pred(z, a); z = mu
            v = float(np.dot(np.append(z.numpy(), 1.0), readout))
            if v > best_v:
                best_v, best = v, (c0, c1)
        for _ in range(hold):
            if steps >= budget:
                break
            world.step(np.array([best[0], best[1], 0.0], np.float32)); steps += 1
            if world.true_m3() >= band:
                reached = True; break
    return reached, world.true_m3()


def main(seeds=range(6)):
    print("=" * 76)
    print(f"MESSY-WORLD CONTROL from LEARNED latent -- drive m3>=4 ({len(list(seeds))} seeds)")
    print("=" * 76)
    agents = ["causal_split", "prediction_first", "oracle"]
    res = {a: [] for a in agents}
    for s in seeds:
        for a in agents:
            res[a].append(run(a, s)[0])
    print(f"  {'agent':>18} {'zero-shot control success':>26}")
    for a in agents:
        print(f"  {a:>18} {100*np.mean(res[a]):>23.0f}%")
    print("=" * 76)
    return res


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    main(seeds=range(ns))
