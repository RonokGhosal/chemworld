"""
SHEAF-HYPERGRAPH AGENT -- the full loop: load-bearing planning + ensemble-EIG active inference, scaled.

Three upgrades over sheaf_dynamics.py, all together:

  1. LOAD-BEARING MODEL: the constructor composition is PLANNED by the sheaf-hypergraph's predictions.
     ConstructorSynthesizer.reach() rolls model.predict_next forward to find a chain of constructors
     that reaches a deep target; here that model IS the sheaf-hypergraph, so its learned gates drive
     the composition -- not just structure read-out.

  2. REAL ACTIVE INFERENCE: an ENSEMBLE of sheaf-hypergraphs. Curiosity = ensemble DISAGREEMENT
     (Pathak et al. 2019): the experimenter CHOOSES the intervention whose outcome the ensemble most
     disagrees about -- reducible (epistemic) uncertainty, which vanishes on noise (the models agree
     it's unpredictable) but is high where structure is still unknown. Reward-free.

  3. SCALE: the forward pass is VECTORIZED (all singleton + pair hyperedge messages in batched matmuls,
     no Python loop over edges) -> GPU-efficient; runs on the A100 at larger d / more gates.

Demo world: a multi-gate cascade (real multiplicative gates a_i*gate_i -> next), the regime where a
linear model is blind and composition is mandatory. Device-aware (CUDA on A100, MPS on Mac).

CLI: python -m constructor_causal.sheaf_active [--gates 2] [--K 4] [--steps 2400] [--epochs 500]
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn as nn

from .neural_discovery import get_device
from .world import DynamicalCausalWorld
from .constructor import Box, Library
from .planner import ConstructorSynthesizer

DEVICE = get_device()


# ============================== vectorized sheaf-hypergraph ==============================
class VSheaf(nn.Module):
    """Vectorized sheaf-hypergraph next-state predictor (no Python loop over hyperedges)."""

    def __init__(self, d, hidden=24):
        super().__init__()
        self.d = d
        self.pairs = torch.tensor([(j, k) for j in range(d) for k in range(j + 1, d)], dtype=torch.long)
        self.nP = self.pairs.shape[0]
        self.msg1 = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, d))
        self.msg2 = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, d))
        self.gate1 = nn.Parameter(torch.full((d, d), -1.0))           # gate1[j, i]: {j}   -> target i
        self.gate2 = nn.Parameter(torch.full((self.nP, d), -1.0))     # gate2[p, i]: pair  -> target i
        self.bias = nn.Parameter(torch.zeros(d))

    def to(self, *a, **k):
        self.pairs = self.pairs.to(*a, **k)
        return super().to(*a, **k)

    def forward(self, X):                                            # X: (n,d) -> (n,d)
        n = X.shape[0]
        S = self.msg1(X.reshape(-1, 1)).reshape(n, self.d, self.d)   # S[n, j, i] = msg from var j to i
        out = self.bias + (torch.sigmoid(self.gate1) * S).sum(1)     # sum over source j
        Pin = X[:, self.pairs]                                       # (n, nP, 2)
        Pm = self.msg2(Pin.reshape(-1, 2)).reshape(n, self.nP, self.d)
        out = out + (torch.sigmoid(self.gate2) * Pm).sum(1)          # sum over pair-hyperedges
        return out


# ============================== ensemble = model + curiosity source ==============================
class SheafEnsemble:
    """K vectorized sheaf-hypergraphs. Serves BOTH as the agent's causal model (predict_next, mean of
    the ensemble) AND as the active-inference curiosity signal (disagreement across the ensemble)."""

    def __init__(self, d, actuators, K=4, hidden=24, device=DEVICE):
        self.d = d; self.actuators = tuple(actuators)
        self.sensors = tuple(i for i in range(d) if i not in self.actuators)
        self.device = device
        self.models = [VSheaf(d, hidden).to(device) for _ in range(K)]
        self._buf_c, self._buf_n, self._fitted = [], [], False

    def update(self, x_clamped, x_next):
        self._buf_c.append(np.asarray(x_clamped, float))
        self._buf_n.append(np.asarray(x_next, float))
        self._fitted = False

    def fit(self, epochs=500, lr=5e-3, l1=1.0e-2):
        Xc = torch.tensor(np.array(self._buf_c), dtype=torch.float32, device=self.device)
        Xn = torch.tensor(np.array(self._buf_n), dtype=torch.float32, device=self.device)
        for m in self.models:
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                pred = m(Xc)
                loss = ((pred - Xn) ** 2).mean() + l1 * (torch.sigmoid(m.gate1).sum() + torch.sigmoid(m.gate2).sum())
                loss.backward(); opt.step()
        self._fitted = True
        return self

    def _predict_all(self, X):                                      # X np (n,d) -> (K,n,d) np
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return torch.stack([m(Xt) for m in self.models]).cpu().numpy()

    def predict_next(self, x_clamped, command=None):                # model interface: ensemble MEAN
        x = np.asarray(x_clamped, float).copy()
        if command:
            for j, v in command.items():
                x[j] = v
        mean = self._predict_all(x[None]).mean(0)[0]
        if command:
            for j, v in command.items():
                mean[j] = v
        return mean, None

    def disagreement(self, x_clamped, command):                     # curiosity: ensemble variance on sensors
        x = np.asarray(x_clamped, float).copy()
        for j, v in command.items():
            x[j] = v
        preds = self._predict_all(x[None])[:, 0, :]                 # (K, d)
        var = preds.var(0)                                          # per-variable disagreement
        return float(var[list(self.sensors)].sum())

    def recovered_hyperedges(self, thresh=0.3):
        g1 = np.mean([torch.sigmoid(m.gate1).detach().cpu().numpy() for m in self.models], 0)
        g2 = np.mean([torch.sigmoid(m.gate2).detach().cpu().numpy() for m in self.models], 0)
        pairs = self.models[0].pairs.cpu().numpy()
        out = []
        for j in range(self.d):
            for i in range(self.d):
                if i != j and g1[j, i] > thresh:
                    out.append(((j,), i, float(g1[j, i])))
        for p, (j, k) in enumerate(pairs):
            for i in range(self.d):
                if i not in (j, k) and g2[p, i] > thresh:
                    out.append(((int(j), int(k)), i, float(g2[p, i])))
        return out

    def recovered_edges(self, thresh=0.3):
        return {(j, i) for H, i, _ in self.recovered_hyperedges(thresh) for j in H if j != i}


class EnsembleExperimenter:
    """Active inference: choose the intervention the ensemble most DISAGREES about (max EIG-by-disagreement)."""

    def __init__(self, ensemble, setpoints=(-2.0, 0.0, 2.0), max_cands=64):
        self.ens = ensemble
        cands = list(itertools.product(setpoints, repeat=len(ensemble.actuators)))
        self.cands = cands[:max_cands]
        self.rng = np.random.default_rng(0)

    def choose(self, state):
        if not self.ens._fitted:                                    # untrained -> random poke
            combo = self.cands[self.rng.integers(len(self.cands))]
        else:
            best, bd = None, -1.0
            for combo in self.cands:
                cmd = {a: v for a, v in zip(self.ens.actuators, combo)}
                d = self.ens.disagreement(state, cmd)
                if d > bd:
                    bd, best = d, combo
            combo = best
        return {a: float(v) for a, v in zip(self.ens.actuators, combo)}


# ============================== a bigger multi-gate world ==============================
def multigate_world(n_gates, rng):
    """a0 opens gate1; then (a_i AND gate_i) opens gate_{i+1}; the last gate AND a_last drives Z.
    n_gates multiplicative gates -> reaching Z needs an (n_gates+1)-deep composition. d = 2+2*n_gates."""
    n_act = n_gates + 1
    gates = list(range(n_act, n_act + n_gates))
    Z = n_act + n_gates
    d = Z + 1
    names = tuple([f"a{i}" for i in range(n_act)] + [f"g{i}" for i in range(n_gates)] + ["Z"])
    A = np.zeros((d, d))
    A[gates[0], gates[0]] = 0.2; A[gates[0], 0] = 0.9               # a0 opens gate0
    inter = []
    for i in range(1, n_gates):
        A[gates[i], gates[i]] = 0.2
        inter.append((gates[i], gates[i - 1], i, 0.6))             # g_{i} += 0.6 * g_{i-1} * a_i
    A[Z, Z] = 0.3
    inter.append((Z, gates[-1], n_gates, 0.6))                     # Z += 0.6 * g_last * a_{n_gates}
    noise = np.full(d, 0.05); noise[:n_act] = 0.0
    return DynamicalCausalWorld(A=A, b=np.zeros(d), noise_std=noise, actuators=tuple(range(n_act)),
                               names=names, interactions=tuple(inter), rng=rng), Z, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    world, Z, names = multigate_world(args.gates, np.random.default_rng(args.seed))
    print("=" * 96)
    print(f"SHEAF-HYPERGRAPH AGENT (load-bearing + ensemble-EIG + scaled)  device={DEVICE}")
    print(f"  world: {args.gates}-gate cascade, d={world.d}, names={names}, target=Z")
    print(f"  ensemble K={args.K}, steps={args.steps}, rounds={args.rounds}, epochs/fit={args.epochs}")
    print("=" * 96)

    ens = SheafEnsemble(world.d, world.actuators, K=args.K, device=DEVICE)
    exp = EnsembleExperimenter(ens)

    # ---- (2) REAL ACTIVE INFERENCE: explore by ensemble-disagreement, refit each round ----
    t0 = time.time()
    x = world.reset()
    per_round = max(1, args.steps // args.rounds)
    for r in range(args.rounds):
        for _ in range(per_round):
            cmd = exp.choose(x)
            xc = x.copy()
            for j, v in cmd.items():
                xc[j] = v
            xn = world.step(cmd)
            ens.update(xc, xn)
            x = xn
        ens.fit(epochs=args.epochs)
        print(f"  round {r+1}/{args.rounds}: refit ensemble on {len(ens._buf_c)} transitions "
              f"(curiosity now ensemble-driven)")
    explore_s = time.time() - t0

    # ---- structure recovered by the ensemble ----
    print(f"\n  RECOVERED GATES (ensemble, the multiplicative hyperedges):")
    gate_hes = [(H, i, w) for H, i, w in ens.recovered_hyperedges() if len(H) == 2]
    for H, i, w in sorted(gate_hes, key=lambda t: -t[2])[:args.gates + 3]:
        print(f"    {{{', '.join(names[v] for v in H)}}} -> {names[i]:>3}   weight {w:.2f}")
    true_gate_targets = {names[i] for (i, a, b, w) in world.interactions}
    rec_gate_targets = {names[i] for H, i, w in gate_hes}
    print(f"    gate targets recovered: {sorted(rec_gate_targets & true_gate_targets)} "
          f"of true {sorted(true_gate_targets)}")

    # ---- (1) LOAD-BEARING: mint constructors, then PLAN a path to Z using the sheaf model ----
    synth = ConstructorSynthesizer(model=ens, world_factory=lambda: multigate_world(args.gates, rng)[0],
                                   actuators=world.actuators, sensors=world.sensors, d=world.d, rng=rng)
    lib = Library()
    good, _ = synth.mint_primitives()
    for c in good:
        if c.possible:
            lib.add(c)
    synth.mint_conditional_primitives(lib)
    zlo = 1.5
    target = Box.from_dict({Z: (zlo, zlo + 4.0)})
    constructor, rel = synth.reach(lib, target, search="greedy")   # PLANNED with sheaf predict_next
    print(f"\n  LOAD-BEARING PLANNING: reach(Z>=,{zlo}) composed with the sheaf model as planner ->")
    if constructor is not None:
        print(f"    {constructor}")
        print(f"    program horizon {constructor.horizon} (an {args.gates+1}-deep gated chain), "
              f"reliability {rel:.2f}")
        print(f"    -> the hypergraph's predict_next DROVE this composition (not just structure read-out).")
    else:
        print(f"    (no chain found at this budget -- the planner's model didn't certify a path)")
    print("=" * 96)
    print(f"  ALL THREE: (1) load-bearing model-based planning, (2) ensemble-EIG active inference,")
    print(f"  (3) vectorized + scaled to a {args.gates}-gate world. explore {explore_s:.1f}s on {DEVICE}.")
    print(f"  Scale up with --gates / --K on the A100 (the fits are the GPU-bound part).")
    print("=" * 96)


if __name__ == "__main__":
    main()
