"""
INVARIANT CAUSAL PREDICTION (ICP) with an interaction basis -- the non-circular instrument.

Why this exists: every "win" so far HANDS the method `world.actuators` (which knobs exist, and which
variable each opens). That is a cheat real data does not grant. ICP (Peters, Buhlmann, Meinshausen 2016)
removes it: give it only the observed variables + an ENVIRONMENT label per row (which regime/batch the
row came from -- a label real perturbation data already has) and a target. It finds the target's causal
parents as the predictor set S whose conditional Y | X_S is INVARIANT across environments, and returns
the INTERSECTION of all accepted sets. Guarantee: with prob >= 1 - alpha the returned set is a SUBSET of
the true parents (Type-I control) -- it is high-precision by construction.

What it does NOT do: raise recall on faithful weak-coupling data. When the per-sample signal is below the
noise floor, sets that omit a true parent look as invariant as sets that include it, so the intersection
SHRINKS -- often to {}. It buys HONESTY (a subset-of-parents or {}), not recall.

CALIBRATION (audit pass 1 findings 1/3/4, now FIXED): the invariance test is a two-sample WELCH t on the
mean + a LEVENE test on the variance, per environment vs the rest, BONFERRONI-corrected over all
comparisons (Peters/Buhlmann/Meinshausen 2016; Levene for non-Gaussian robustness, cf. Heinze-Deml 2018).
Measured (selftest_icp): the true-set reject rate is ~alpha and FLAT in n_env (was 0.0065@2env ->
0.1885@32env under the old uncorrected normal-threshold z-test). `alpha` is now a real significance level.

REMAINING CAVEAT ON THE {} RETURN (finding 5): {} is still NOT proof "the wall is here". It also arises
from BASIS MISSPECIFICATION -- a mechanism outside span([1, X_i, X_i*X_j]) (a 3-way product, a saturating
nonlinearity) yields {} even at high SNR. So {} means "no invariant set found under THIS test AND THIS
basis". And proper correction REDUCES power, so at low SNR recovery degrades to a subset or {} rather than
a wrong edge -- honesty, not recall, is what it buys.

Interaction basis: the design for a set S is [1, X_i (i in S), X_i*X_j (i<j in S)], so a multiplicative
AND-gate C*a_X is representable -- without it, ICP cannot see product mechanisms at all.

Uses numpy + scipy.stats (Welch t, Levene) for calibrated reference distributions.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats


def _design(X, S):
    """[intercept | linear terms of S | pairwise products within S] -- interaction basis."""
    S = sorted(S)
    cols = [np.ones(len(X))]
    cols += [X[:, i] for i in S]
    cols += [X[:, a] * X[:, b] for a, b in combinations(S, 2)]
    return np.column_stack(cols)


def _invariant(resid, env, alpha, min_env=5):
    """Is the residual distribution the SAME across environments? For each environment e vs the rest, a
    two-sample WELCH t-test on the MEAN and a LEVENE test on the VARIANCE (Levene is robust to non-Gaussian
    residuals, unlike a Gaussian log-variance z), BONFERRONI-corrected over all mean+variance comparisons.
    Returns True if invariance is NOT rejected at level `alpha`. (Audit pass 1 findings 1/3/4: replaces the
    uncorrected normal-threshold z-test that inflated Type-I with n_env, over-rejected at small n, and broke
    on heavy tails.)"""
    pvals = []
    for e in np.unique(env):
        r_e = resid[env == e]; r_rest = resid[env != e]
        if len(r_e) < min_env or len(r_rest) < min_env:
            continue
        _, p_mean = stats.ttest_ind(r_e, r_rest, equal_var=False)
        try:
            _, p_var = stats.levene(r_e, r_rest)
        except ValueError:
            p_var = 1.0
        pvals += [p_mean if np.isfinite(p_mean) else 1.0, p_var if np.isfinite(p_var) else 1.0]
    if not pvals:
        return True
    return min(pvals) * len(pvals) >= alpha            # Bonferroni over all comparisons


def icp(X, env, target, *, max_set=None, alpha=0.05):
    """Estimate the causal parents of `target`, invariant across `env`. Consumes only data + env labels,
    never intervention/knob identity -- that is the non-circular point (checked in selftest_icp).

    Returns dict(parents=set, accepted=int, defined=bool): `parents` is the intersection of all accepted
    predictor sets (a subset of true parents in the CALIBRATED, in-basis regime); defined=False and
    parents=set() when NO set passes the invariance test -- which can mean the wall OR a miscalibrated test
    OR a mechanism outside the basis (see module docstring caveats), so do not read {} as proof alone."""
    X = np.asarray(X, float)
    n, d = X.shape
    env = np.asarray(env)
    others = [j for j in range(d) if j != target]
    max_set = len(others) if max_set is None else min(max_set, len(others))
    y = X[:, target]

    accepted = []
    for k in range(0, max_set + 1):
        for S in combinations(others, k):
            Phi = _design(X, S)
            beta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
            if _invariant(y - Phi @ beta, env, alpha):
                accepted.append(set(S))
    if not accepted:
        return dict(parents=set(), accepted=0, defined=False)
    parents = set(others)
    for S in accepted:
        parents &= S
    return dict(parents=parents, accepted=len(accepted), defined=True)


def gate_environments(n_per_env, envs, rng, *, w=0.6, sigma=0.3, decoy=True):
    """Structural AND-gate data with environment heterogeneity, for testing ICP without a dynamical world.
    Variables: C, a_X, [D decoy], X   where X = w * C * a_X + noise. Each env shifts the DISTRIBUTIONS of
    C and a_X (heterogeneity) but the mechanism X|C,a_X is INVARIANT -- exactly ICP's assumption. Returns
    (Xmat, env, names, target_idx, true_parents)."""
    rows, elab = [], []
    for e, (cmu, amu) in enumerate(envs):
        C = rng.normal(cmu, 1.0, n_per_env)
        aX = rng.normal(amu, 1.0, n_per_env)
        Xt = w * C * aX + rng.normal(0.0, sigma, n_per_env)
        cols = [C, aX] + ([rng.normal(0.0, 1.0, n_per_env)] if decoy else []) + [Xt]
        rows.append(np.column_stack(cols)); elab.append(np.full(n_per_env, e))
    Xmat = np.vstack(rows); env = np.concatenate(elab)
    names = ["C", "a_X"] + (["D"] if decoy else []) + ["X"]
    target = len(names) - 1
    true_parents = {0, 1}                          # {C, a_X}
    return Xmat, env, names, target, true_parents


def fork_environments(n_per_env, envs, rng, *, w=0.6, sigma=0.3):
    """Confounder fork C->X, C->Y (X=w*C*a_X, Y=w*C*a_Y, NO X->Y). Target Y; true parents {C, a_Y}. Used
    to test PRECISION: ICP must NOT put X (a proxy for C) into Y's parents. Returns (..., target=Y)."""
    rows, elab = [], []
    for e, (cmu, axmu, aymu) in enumerate(envs):
        C = rng.normal(cmu, 1.0, n_per_env)
        aX = rng.normal(axmu, 1.0, n_per_env)
        aY = rng.normal(aymu, 1.0, n_per_env)
        Xt = w * C * aX + rng.normal(0.0, sigma, n_per_env)
        Yt = w * C * aY + rng.normal(0.0, sigma, n_per_env)
        rows.append(np.column_stack([C, aX, aY, Xt, Yt])); elab.append(np.full(n_per_env, e))
    Xmat = np.vstack(rows); env = np.concatenate(elab)
    names = ["C", "a_X", "a_Y", "X", "Y"]
    return Xmat, env, names, 4, {0, 2}             # target Y=4, true parents {C, a_Y}
