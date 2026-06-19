"""
constructor_causal -- SHARED-STATISTICS exactness selftest (census flaw #9).

Because the conjugate precision Lambda_N = prior_coef*I + S carries NO sigma^2 and
depends only on the (shared) regressors, it is IDENTICAL for every sensor. The model
therefore stores ONE Gram S and ONE maintained inverse Pinv instead of d copies, cutting
memory O(d*p^2) -> O(p^2) and per-step cost d*O(p^2) -> O(p^2). This is an EXACT algebraic
identity, not an approximation -- so this test proves, against from-scratch linear algebra
(NOT the model's own internals):

  (1) the maintained shared Pinv equals inv(prior_coef*I + S) to machine precision, on
      both the pure rank-1 path (n < refresh cadence) and the periodic clean-recompute path;
  (2) each per-sensor posterior mean / scale equals the exact NIG solution from raw data;
  (3) fit_batch (vectorized one-pass refit) reproduces sequential update() replay EXACTLY,
      under both no forgetting and exponential forgetting;
  (4) the EIG simplification |sensors|*0.5*log1p(phi^T Pinv phi) equals the brute-force
      per-sensor sum over the materialized Student-t covariances; same for seq_info_gain;
  (5) the stored state really is shared (one ndarray Gram/inverse), not a per-sensor dict.
"""
from __future__ import annotations

import numpy as np

from .model import BayesianDynamicsModel
from .world import DynamicalCausalWorld as W

R = []
def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _gen(forget, N, seed, interaction_pairs=()):
    """Drive the default world with random commands; return (model_seq, Xc, Xn).
    Xc rows are the clamped current state (actuators forced), Xn the next state."""
    rng = np.random.default_rng(seed)
    w = W.default(np.random.default_rng(seed + 7))
    w.reset()
    acts = set(w.actuators)
    m = BayesianDynamicsModel(d=w.d, actuators=w.actuators, alpha=1e-2, sigma0=1.0,
                              forget=forget, interaction_pairs=interaction_pairs)
    Xc, Xn = [], []
    for _ in range(N):
        cmd = {j: float(rng.choice([-2.0, 0.0, 2.0])) for j in acts}
        xc = w.x.copy()
        for j, v in cmd.items():
            xc[j] = v
        xn = w.step(cmd)
        Xc.append(xc.copy()); Xn.append(xn.copy())
        m.update(xc, xn)
    return m, np.array(Xc), np.array(Xn)


def _exact_posterior(m, Xc, Xn, forget):
    """From-scratch NIG posterior for every sensor: weighted normal equations with the
    SAME exponential weights the sequential RLS uses (newest = weight 1)."""
    Phi = np.array([m._phi(x) for x in Xc])                       # (n, p)
    n = len(Phi)
    w = (forget ** np.arange(n - 1, -1, -1)) if forget < 1.0 else np.ones(n)
    prior_coef = m.alpha * (forget ** n if forget < 1.0 else 1.0)
    Lambda = prior_coef * np.eye(m.p) + (Phi * w[:, None]).T @ Phi
    Pinv = np.linalg.inv(Lambda)
    a_N = m._a0 + 0.5 * float(w.sum())
    out = {}
    for i in m.sensors:
        y = Xn[:, i]
        r = (Phi * w[:, None]).T @ y
        mean = Pinv @ r
        syy = float(w @ (y * y))
        b_N = m._b0 + 0.5 * max(syy - float(mean @ r), 0.0)
        out[i] = (mean, max(b_N / a_N, 1e-6))
    return Pinv, out


def main():
    print("=" * 78)
    print("constructor_causal -- SHARED-STATISTICS exactness (flaw #9)")
    print("=" * 78)

    # (5) the stored state is genuinely shared, not per-sensor
    m0, _, _ = _gen(1.0, 60, 0)
    check("S and Pinv are ONE shared ndarray (not a per-sensor dict)",
          isinstance(m0.S, np.ndarray) and isinstance(m0.Pinv, np.ndarray)
          and m0.S.shape == (m0.p, m0.p), f"S type={type(m0.S).__name__}")
    check("shared observation count n equals the number of steps", m0.n == 60.0, f"n={m0.n}")

    # (1)+(2) maintained Pinv == exact inverse; per-sensor means/scales exact.
    # n=160 < refresh_every(200): pure rank-1 RLS path (no mid-stream clean recompute).
    m, Xc, Xn = _gen(1.0, 160, 1)
    Pe, post = _exact_posterior(m, Xc, Xn, 1.0)
    check("[rank-1 path] maintained shared Pinv == inv(prior*I + S)",
          np.max(np.abs(m.Pinv - Pe)) < 1e-7, f"max|dPinv|={np.max(np.abs(m.Pinv - Pe)):.2e}")
    dmean = max(np.max(np.abs(m._post_light(i)[0] - post[i][0])) for i in m.sensors)
    ds2 = max(abs(m._post_light(i)[1] - post[i][1]) for i in m.sensors)
    check("[rank-1 path] every per-sensor posterior mean == exact NIG mean",
          dmean < 1e-7, f"max|dmean|={dmean:.2e}")
    check("[rank-1 path] every per-sensor noise scale s2 == exact b_N/a_N",
          ds2 < 1e-7, f"max|ds2|={ds2:.2e}")

    # (1) periodic clean-recompute path: n=450 triggers >=2 refreshes
    mr, Xcr, Xnr = _gen(1.0, 450, 2)
    Per, _ = _exact_posterior(mr, Xcr, Xnr, 1.0)
    check("[refresh path] maintained Pinv still == exact inverse after clean recomputes",
          np.max(np.abs(mr.Pinv - Per)) < 1e-7, f"max|dPinv|={np.max(np.abs(mr.Pinv - Per)):.2e}")

    # (3) fit_batch == sequential replay, no forgetting AND with forgetting
    for forget in (1.0, 0.95):
        ms, Xc2, Xn2 = _gen(forget, 150, 3)
        mb = BayesianDynamicsModel(d=ms.d, actuators=ms.actuators, alpha=1e-2,
                                   sigma0=1.0, forget=forget)
        mb.fit_batch(Xc2, Xn2)
        dS = np.max(np.abs(ms.S - mb.S)); dP = np.max(np.abs(ms.Pinv - mb.Pinv))
        dm = max(np.max(np.abs(ms._post_light(i)[0] - mb._post_light(i)[0]))
                 for i in ms.sensors)
        tag = "g=1" if forget == 1.0 else "g=0.95"
        check(f"[{tag}] fit_batch reproduces sequential update() exactly (S, Pinv, means)",
              dS < 1e-7 and dP < 1e-6 and dm < 1e-7,
              f"dS={dS:.2e} dPinv={dP:.2e} dmean={dm:.2e}")
        check(f"[{tag}] fit_batch and sequential recover the SAME DAG",
              ms.recovered_edges() == mb.recovered_edges(),
              f"seq={len(ms.recovered_edges())} batch={len(mb.recovered_edges())}")

    # (4) EIG simplification == brute-force per-sensor sum over materialized Student-t covs
    me, Xce, _ = _gen(1.0, 200, 4)
    phi = me._phi(Xce[123])
    q = float(phi @ me._ensure_pinv() @ phi)
    eig_fast = me.expected_info_gain(Xce[123])
    eig_brute = 0.0
    for i in me.sensors:
        _, Cov = me._posterior(i)                 # Cov = s2_i * Pinv (materialized)
        eig_brute += 0.5 * np.log1p(max(phi @ Cov @ phi, 0.0) / me.sigma2[i])
    check("expected_info_gain == |sensors|*0.5*log1p(phi^T Pinv phi)",
          abs(eig_fast - len(me.sensors) * 0.5 * np.log1p(q)) < 1e-9)
    check("...and == brute-force per-sensor sum over materialized Student-t covariances",
          abs(eig_fast - eig_brute) < 1e-9, f"|d|={abs(eig_fast - eig_brute):.2e}")

    # seq_info_gain == brute-force per-sensor chain-rule sum
    phis = [me._phi(Xce[100 + k]) for k in range(3)]
    seq_fast = me.seq_info_gain(phis)
    seq_brute = 0.0
    for i in me.sensors:
        P = me._ensure_pinv().copy()
        for ph in phis:
            Pp = P @ ph; qq = float(ph @ Pp)
            if qq <= 0:
                continue
            seq_brute += 0.5 * np.log1p(qq)
            P = P - np.outer(Pp, Pp) / (1.0 + qq)
    check("seq_info_gain == brute-force per-sensor chain-rule sum",
          abs(seq_fast - seq_brute) < 1e-9, f"|d|={abs(seq_fast - seq_brute):.2e}")

    print("=" * 78)
    print(f"{sum(R)}/{len(R)} checks passed")
    print("=" * 78)
    return all(R)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
