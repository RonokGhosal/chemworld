"""Acceptance test: the unified agent on NONLINEAR + PARTIALLY-OBSERVED worlds, reward-free.

Bars (all must go green):
  1a. world.nonlinear(): recover the EVEN edge a0^2 -> even as a discovered product feature
      (a linear model has NO a0->even edge by construction), via the Thompson agent.
  1b. world.nonlinear(): predict the SATURATING edge sat := 2*tanh(1.5*a1) with what-if RMSE < 0.1
      using a random-Fourier basis + continuous exploration (vs a linear model's RMSE > 0.3).
  2.  world.confounded(): the INTERVENING (Thompson) agent drives the spurious S1->S2 weight < 0.1,
      while a PASSIVE observer leaves it > 0.5 -- intervention beats correlation under a hidden cause.

Run:  ./.venv/bin/python -m constructor_causal.selftest_nonlinear_partial
"""
from __future__ import annotations

import sys

import numpy as np

from .agent import ConstructorCausalAgent
from .world import DynamicalCausalWorld as W

R = []
def check(name, cond, detail=""):
    R.append(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def test_nonlinear_even():
    """1a. Discover a0^2 -> even, which a linear learner cannot see as a linear edge."""
    print("\n[1a] NONLINEAR even edge (even := 1.2*a0^2; no linear a0->even):")
    world = W.nonlinear(np.random.default_rng(0))
    a0, even = world.names.index("a0"), world.names.index("even")
    # linear control: no interaction features -> it must NOT find a linear a0->even edge
    lin = ConstructorCausalAgent(W.nonlinear(np.random.default_rng(0)), seed=0, experimenter="thompson")
    lin.explore_continuous(800)
    lin_has_edge = (a0, even) in lin.model.recovered_edges()
    # unified agent: a nonlinear edge is invisible to the LINEAR model, so probe a0 across its
    # range (continuous exploration), then scan residuals for the product that explains 'even'.
    ag = ConstructorCausalAgent(world, seed=0, experimenter="thompson")
    ag.explore_continuous(800)
    found = ag.discover_interactions(z=3.5)
    inter = ag.model.recovered_interactions()
    w = abs(ag.model.interaction_weight(even, a0, a0)) if (a0, a0) in ag.model.interaction_pairs else 0.0
    print(f"      linear model finds a0->even edge: {lin_has_edge} (should be False)")
    print(f"      discovered products: {found};  recovered interactions: {sorted(inter)}")
    print(f"      interaction weight a0*a0 -> even = {w:.2f} (true 1.2)")
    check("linear learner misses a0->even (no linear edge)", not lin_has_edge)
    check("Thompson agent recovers a0^2 -> even as a product", ((a0, a0), even) in inter)


def test_nonlinear_saturating():
    """1b. Predict the tanh saturating edge with RFF + continuous exploration; RMSE < 0.1."""
    print("\n[1b] NONLINEAR saturating edge (sat := 2*tanh(1.5*a1)):")
    a1, sat = 1, 3
    grid = np.linspace(-2, 2, 21)
    true = 2.0 * np.tanh(1.5 * grid)                       # from rest (sat=0): sat_next = 2 tanh(1.5 a1)

    def whatif_rmse(agent):
        preds = []
        for v in grid:
            x0 = np.zeros(agent.world.d)
            mu, _ = agent.model.predict_next(x0, {a1: float(v)})
            preds.append(mu[sat])
        return float(np.sqrt(np.mean((np.array(preds) - true) ** 2)))

    rff = ConstructorCausalAgent(W.nonlinear(np.random.default_rng(1)), seed=1,
                                 experimenter="thompson", rff=80, rff_scale=1.5)
    rff.explore_continuous(1200)
    lin = ConstructorCausalAgent(W.nonlinear(np.random.default_rng(1)), seed=1, experimenter="thompson")
    lin.explore_continuous(1200)
    rmse_rff, rmse_lin = whatif_rmse(rff), whatif_rmse(lin)
    print(f"      what-if RMSE on sat:  RFF basis = {rmse_rff:.3f}   linear = {rmse_lin:.3f}")
    check("RFF predicts the saturating edge (RMSE < 0.1)", rmse_rff < 0.1, f"{rmse_rff:.3f}")
    check("linear model is much worse (RMSE > 0.3)", rmse_lin > 0.3, f"{rmse_lin:.3f}")


def test_confounder():
    """2. Intervention (Thompson) kills the spurious S1->S2 edge a passive observer accepts."""
    print("\n[2] HIDDEN CONFOUNDER (H drives S1 and S2; no real S1->S2 edge):")
    S1, S2 = 0, 1
    act = ConstructorCausalAgent(W.confounded(np.random.default_rng(0)), seed=0, experimenter="thompson")
    act.explore(500)
    pas = ConstructorCausalAgent(W.confounded(np.random.default_rng(0)), seed=0, experimenter="passive")
    pas.explore(500)
    w_act = abs(act.model.weight(S2, S1)); w_pas = abs(pas.model.weight(S2, S1))
    print(f"      spurious S1->S2 weight:  intervening(Thompson) = {w_act:.2f}   passive = {w_pas:.2f}")
    check("intervention drives spurious S1->S2 weight < 0.1", w_act < 0.1, f"{w_act:.2f}")
    check("passive observer is fooled (weight > 0.5)", w_pas > 0.5, f"{w_pas:.2f}")


if __name__ == "__main__":
    print("UNIFIED AGENT on nonlinear + partially-observed worlds (reward-free)")
    test_nonlinear_even()
    test_nonlinear_saturating()
    test_confounder()
    n = sum(R)
    print(f"\n{n}/{len(R)} checks passed")
    sys.exit(0 if n == len(R) else 1)
