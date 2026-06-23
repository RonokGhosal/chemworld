"""
Exactness test for the WeightedHeteroModel (Order 3 hardening): the ONLINE weighted
Sherman-Morrison precision-inverse must equal the BATCH weighted normal equations, per
sensor, to machine precision -- using the EXACT per-step weights w_{t,i}=1/sigma_i^2(x_t)
that the online update actually used (recorded as we go, since the head co-evolves).

    Lambda_i = alpha*I + sum_t w_{t,i} phi_t phi_t^T,   m_i = Lambda_i^{-1} sum_t w_{t,i} phi_t y_{t,i}
"""
from __future__ import annotations

import numpy as np

from .hetero import WeightedHeteroModel
from .noise_knob import NoiseKnobWorld

R = []
def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    print("=" * 72)
    print("WEIGHTED HETERO MODEL -- online weighted SM == batch weighted normal eqns")
    print("=" * 72)
    for bayes in (False, True):
        w = NoiseKnobWorld(3, np.random.default_rng(0)); w.reset()
        m = WeightedHeteroModel(w.d, w.actuators, hidden=w.hidden,
                                rng=np.random.default_rng(1), bayes_head=bayes)
        rng = np.random.default_rng(2)
        rec = []                                  # (phi, w_per_sensor, y_per_sensor)
        for _ in range(120):
            cmd = {j: float(rng.choice([-2, 0, 2])) for j in w.actuators}
            xc = w.x.copy()
            for j, v in cmd.items():
                xc[j] = v
            phi = m._phi(xc)
            wts = {i: 1.0 / m.sigma2(xc, i) for i in m.sensors}     # EXACT weights update will use
            xn = w.step(cmd)
            rec.append((phi.copy(), wts, {i: float(xn[i]) for i in m.sensors}))
            m.update(xc, xn)
            if not bayes:
                m.refit_head()                    # batch head refit changes future weights; recorded ones stay exact

        p = m.p
        dP = dM = 0.0
        for i in m.sensors:
            Lam = m.alpha * np.eye(p)
            r = np.zeros(p)
            for phi, wts, ys in rec:
                Lam += wts[i] * np.outer(phi, phi)
                r += wts[i] * phi * ys[i]
            Pi_batch = np.linalg.inv(Lam)
            dP = max(dP, float(np.max(np.abs(Pi_batch - m.P[i]))))
            dM = max(dM, float(np.max(np.abs(Pi_batch @ r - m._mean(i)))))
        tag = "bayes-head" if bayes else "batch-head"
        check(f"[{tag}] online weighted-SM P_i == batch inv(Lambda_i)", dP < 1e-7, f"max|dP|={dP:.2e}")
        check(f"[{tag}] online weighted mean_i == batch GLS solution", dM < 1e-7, f"max|dmean|={dM:.2e}")

    print("=" * 72)
    print(f"{sum(R)}/{len(R)} checks passed")
    print("=" * 72)
    return all(R)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
