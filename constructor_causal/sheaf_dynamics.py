"""
SheafDynamicsModel -- a sheaf-hypergraph as the agent's CAUSAL MODEL (smallest integration).

The first time the hypergraph idea sits INSIDE the intervene -> learn -> build-constructors loop.
Instead of the linear BayesianDynamicsModel, the agent's "what happens if I poke this" model is a
sheaf-hypergraph next-state predictor:

  * each variable is a node with a continuous STALK;
  * candidate HYPEREDGES = singletons {j} and pairs {j,k} (the pairs are what a pairwise/linear model
    cannot represent -- a multiplicative GATE like Z += 0.5*gate*a1 IS a pair-hyperedge {gate,a1}->Z);
  * a shared RESTRICTION MLP per arity maps a hyperedge's member values into a per-target message;
  * a learned, L1-sparsified GATE weight per (hyperedge, target) selects which hyperedges drive each
    target; the readout sums the gated messages -> next state.
Trained by gradient descent (GPU-relevant). Exposes the agent's model interface: predict_next,
update, recovered_edges / recovered_hyperedges (read off the gate weights).

This is the SMALLEST version (per the scope): driven by a RANDOM experimenter (true EIG needs a
Bayesian posterior this neural model doesn't have -- deferred), on the `gated` world we already have.
Success = the loop runs end-to-end, the sheaf-hypergraph RECOVERS THE GATE (the {gate,a1}->Z
hyperedge a linear model misses), and the constructor library MINTS a constructor from it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .neural_discovery import get_device
from .world import DynamicalCausalWorld
from .constructor import Library
from .planner import ConstructorSynthesizer


class SheafDynamicsModel(nn.Module):
    def __init__(self, d, actuators, stalk=8, hidden=24, device=None):
        super().__init__()
        self.d = int(d)
        self.actuators = tuple(actuators)
        self.sensors = tuple(i for i in range(d) if i not in self.actuators)
        self.device = device or get_device()
        self.singles = [(j,) for j in range(d)]
        self.pairs = [(j, k) for j in range(d) for k in range(j + 1, d)]
        self.msg1 = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, d))   # {j}   -> per-target msg
        self.msg2 = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, d))   # {j,k} -> per-target msg
        self.gate1 = nn.Parameter(torch.full((len(self.singles), d), -1.0))                # softplus-ish select
        self.gate2 = nn.Parameter(torch.full((len(self.pairs), d), -1.0))
        self.bias = nn.Parameter(torch.zeros(d))
        self._buf_c, self._buf_n = [], []
        self._fitted = False
        self.to(self.device)

    # ---- the agent's model interface ----
    def update(self, x_clamped, x_next):
        self._buf_c.append(np.asarray(x_clamped, float))
        self._buf_n.append(np.asarray(x_next, float))
        self._fitted = False

    def _forward(self, X):                                   # X: (n,d) tensor -> pred (n,d)
        out = self.bias.expand(X.shape[0], -1).clone()
        for idx, (j,) in enumerate(self.singles):
            out = out + torch.sigmoid(self.gate1[idx]) * self.msg1(X[:, [j]])
        for idx, (j, k) in enumerate(self.pairs):
            out = out + torch.sigmoid(self.gate2[idx]) * self.msg2(X[:, [j, k]])
        return out

    def fit(self, epochs=1200, lr=5e-3, l1=1.2e-2, verbose=False):
        Xc = torch.tensor(np.array(self._buf_c), dtype=torch.float32, device=self.device)
        Xn = torch.tensor(np.array(self._buf_n), dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            pred = self._forward(Xc)
            mse = ((pred - Xn) ** 2).mean()
            sparse = torch.sigmoid(self.gate1).sum() + torch.sigmoid(self.gate2).sum()
            (mse + l1 * sparse).backward()
            opt.step()
        self._fitted = True
        if verbose:
            print(f"  sheaf fit: final MSE {float(mse):.4f}")
        return self

    def _ensure(self):
        if not self._fitted and self._buf_c:
            self.fit()

    def predict_next(self, x_clamped, command=None):
        self._ensure()
        x = np.asarray(x_clamped, float).copy()
        if command:
            for j, v in command.items():
                x[j] = v
        with torch.no_grad():
            pred = self._forward(torch.tensor(x[None], dtype=torch.float32, device=self.device))[0].cpu().numpy()
        if command:
            for j, v in command.items():
                pred[j] = v                                  # clamped knobs hold
        return pred, None

    # ---- causal-structure read-out (from the gate weights) ----
    def recovered_hyperedges(self, thresh=0.3):
        """List of (hyperedge:tuple, target:int, weight) the model kept (cross-variable, non-self)."""
        self._ensure()
        out = []
        g1 = torch.sigmoid(self.gate1).detach().cpu().numpy()
        g2 = torch.sigmoid(self.gate2).detach().cpu().numpy()
        for idx, (j,) in enumerate(self.singles):
            for i in range(self.d):
                if i != j and g1[idx, i] > thresh:
                    out.append(((j,), i, float(g1[idx, i])))
        for idx, (j, k) in enumerate(self.pairs):
            for i in range(self.d):
                if i not in (j, k) and g2[idx, i] > thresh:
                    out.append(((j, k), i, float(g2[idx, i])))
        return out

    def recovered_edges(self, thresh=0.3):
        """Flatten hyperedges to (source, target) cross-edges."""
        E = set()
        for H, i, _ in self.recovered_hyperedges(thresh):
            for j in H:
                if j != i:
                    E.add((j, i))
        return E


# ---------------------------------------------------------------------------
def collect_random(world, n, rng, lo=-2.0, hi=2.0):
    """The reward-free, RANDOM-experimenter exploration (active inference, Stage-1 acquisition)."""
    Xc, Xn = [], []
    x = world.reset()
    for _ in range(n):
        cmd = {j: float(rng.uniform(lo, hi)) for j in world.actuators}
        xc = x.copy()
        for j, v in cmd.items():
            xc[j] = v
        xn = world.step(cmd)
        Xc.append(xc); Xn.append(xn.copy())
        x = xn
    return np.array(Xc), np.array(Xn)


def demo(n_explore=4000, seed=0):
    print("=" * 90)
    print(f"SHEAF-HYPERGRAPH as the agent's causal model  (device={get_device()})  -- gated world")
    print("=" * 90)
    rng = np.random.default_rng(seed)
    world = DynamicalCausalWorld.gated(np.random.default_rng(seed))
    names = world.names
    true = world.true_edges()
    print(f"  world: names={names}, actuators={world.actuators}; TRUE cross-edges "
          f"{sorted((names[j], names[i]) for (j, i) in true)}")
    print(f"  the key structure is the multiplicative GATE  Z += 0.5*gate*a1  (a hyperedge {{gate,a1}}->Z)")

    # --- 1. reward-free random-intervention exploration, fit the sheaf-hypergraph as the model ---
    Xc, Xn = collect_random(world, n_explore, rng)
    model = SheafDynamicsModel(d=world.d, actuators=world.actuators)
    for xc, xn in zip(Xc, Xn):
        model.update(xc, xn)
    model.fit(verbose=True)

    # --- 2. read off the recovered causal structure ---
    print("\n  RECOVERED HYPEREDGES (sheaf gate weights):")
    for H, i, w in sorted(model.recovered_hyperedges(), key=lambda t: -t[2]):
        tag = "GATE (pair)" if len(H) == 2 else "edge"
        print(f"    {tag:>12}  {{{', '.join(names[v] for v in H)}}} -> {names[i]:>4}   weight {w:.2f}")
    # HONEST ground truth = linear cross-edges (a0->gate) PLUS the interaction hyperedges (gate,a1 -> Z).
    # (world.true_edges() only counts the LINEAR A matrix, so it omits the gate -- we add it back here.)
    true_full = set(true)
    for (i, a, b, w) in world.interactions:
        true_full |= {(a, i), (b, i)}                        # the gate contributes gate->Z and a1->Z
    rec = model.recovered_edges()
    tp = len(rec & true_full); fp = len(rec - true_full); fn = len(true_full - rec)
    f1 = tp / (tp + 0.5 * (fp + fn)) if tp else 0.0
    got_gate = any(len(H) == 2 and i == 3 and set(H) == {1, 2} for H, i, _ in model.recovered_hyperedges())
    print(f"  edge recovery vs FULL truth (incl. the gate): F1={f1:.2f} (tp={tp} fp={fp} fn={fn}); "
          f"GATE {{gate,a1}}->Z recovered as a hyperedge: {got_gate}")

    # --- 3. the constructor library, using the sheaf model as the agent's model ---
    synth = ConstructorSynthesizer(model=model, world_factory=lambda: DynamicalCausalWorld.gated(rng),
                                   actuators=world.actuators, sensors=world.sensors, d=world.d, rng=rng)
    lib = Library()
    good, idle = synth.mint_primitives()
    for c in good:
        if c.possible:
            lib.add(c)
    conds = synth.mint_conditional_primitives(lib)
    print(f"\n  CONSTRUCTOR LIBRARY (minted with the sheaf-hypergraph as the model):")
    print(f"    primitives kept: {len(lib.possible())}   conditional (gated) skills: {len(conds)}")
    print(lib)
    print("=" * 90)
    z_skill = [c for c in lib.constructors if 3 in {v for (v, _, _) in c.effect.bounds}]
    if z_skill:
        print(f"  THE CONSTRUCTOR FOR Z (the gated target a linear model can't reach with one knob):")
        print("   ", z_skill[0])
        print(f"    -> minted via the sheaf-hypergraph model + random-intervention loop + constructor library.")
    print(f"  SMALLEST INTEGRATION RUNS: hypergraph causal model + (random) intervention + constructors,")
    print(f"  one loop, on a world we already have. (True EIG active inference = the next stage.)")
    print("=" * 90)
    return model, lib


if __name__ == "__main__":
    demo()
