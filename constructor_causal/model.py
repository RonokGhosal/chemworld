"""
The agent's belief about how the world works -- a *Bayesian* causal dynamics model.

For every observed sensor i we fit one Bayesian linear regression that predicts the
sensor's next value from a feature vector built from the (clamped) current state:

        x_{t+1,i} = w_i . phi_t + e_i,
        phi_t = [ observed vars , chosen products x_a*x_b , 1 ],
        w_i ~ N(0, sigma_i^2 / lambda0 I)  (prior),   e_i ~ N(0, sigma_i^2),
        sigma_i^2 ~ InverseGamma(a0, b0).

This is the CONJUGATE Normal-Inverse-Gamma model: the noise variance sigma_i^2 is
unknown and marginalised out, NOT plugged in. Two things follow, and they are the
whole point of using the conjugate form rather than a plugin-variance shortcut:

  * the precision Lambda_N = lambda0*I + S (S = sum phi phi^T) contains NO sigma^2.
    So each observation is an exact RANK-1 bump, and the posterior inverse is kept
    up to date in O(p^2) by Sherman-Morrison -- no O(p^3) re-inversion per step.
    Crucially, because Lambda_N has no sigma^2 it depends ONLY on the (shared)
    regressors, so it is IDENTICAL for every sensor: we store ONE Gram S and ONE
    maintained inverse Pinv = Lambda_N^{-1} (memory O(p^2), one rank-1 update per
    step), not d copies of the same matrix updated d times. Only the cross-statistics
    r_i = sum phi*y_i and syy_i = sum y_i^2, which depend on the TARGET, are per-sensor.

  * the weight posterior is a multivariate Student-t with nu = 2*a0 + n degrees of
    freedom. So the recovered causal DAG uses a t-test (a t-critical value), not a
    Gaussian z cutoff: edge j -> i exists iff weight w_i[j] is confidently non-zero
    AT THE RIGHT (finite-n) confidence. A fixed z=3 is materially over-confident at
    the small n where structure is first recovered; the t-test fixes that.
    Interventions are what let the agent confidently zero out a weight that
    observation alone could not (decoys AND hidden confounders).

  * expected information gain of a candidate experiment (active_inference):
    EIG(phi) = 1/2 * sum_i log(1 + phi^T Lambda_N_i^{-1} phi). The epistemic engine.
    It measures shrinkage of uncertainty about the *parameters*, not raw surprise,
    so a parent-free noise variable stops being interesting once learned. Since
    Lambda_N is shared, every sensor's term is identical and EIG collapses to
    |sensors| * 1/2 log(1 + phi^T Pinv phi) -- computed once, not d times.

Two extensions over a plain linear-Gaussian fit, both optional and both flowing
through one ``feature map``:

  * ``hidden``        -- indices the agent cannot observe. Dropped from the feature
                         vector and never modelled. A hidden common cause then
                         confounds two observed variables; only intervention
                         (forcing) breaks the confound. See world.confounded().
  * ``interaction_pairs`` -- products x_a*x_b added as features, so a multiplicative
                         "gate" becomes discoverable. See world.gated().

The posterior MEAN equals the plain ridge solution (alpha-scaled), so recovered
weights and EIG-based action selection are unchanged; only the UNCERTAINTY (test
calibration, predictive tails) and the per-step COMPUTE are improved.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _t_crit(z: float, nu: float) -> float:
    """Two-sided Student-t critical value matching the Gaussian level of ``z``, via the
    4-term Cornish-Fisher / Fisher expansion (numpy-only; no scipy). vs scipy at z=3:
    within ~0.1% for nu>=10 and ~0.7% for nu>=4 (e.g. ~3.96 at nu=10, ~3.42 at nu=20).
    Below nu~4 the asymptotic expansion is mildly ANTI-conservative (a few % too small,
    e.g. nu=3 ~9.0 vs exact ~9.2), i.e. slightly over-sensitive -- but edges are almost
    never recovered at nu<=3 anyway, so the practical effect is negligible."""
    nu = max(float(nu), 2.0)
    if nu > 1e7:
        return float(z)
    z2 = z * z
    c1 = z * (z2 + 1.0) / 4.0
    c2 = z * (5.0 * z2 * z2 + 16.0 * z2 + 3.0) / 96.0
    c3 = z * (3.0 * z2 ** 3 + 19.0 * z2 * z2 + 17.0 * z2 - 15.0) / 384.0
    c4 = z * (79.0 * z2 ** 4 + 776.0 * z2 ** 3 + 1482.0 * z2 ** 2
              - 1920.0 * z2 - 945.0) / 92160.0
    return float(z + c1 / nu + c2 / nu ** 2 + c3 / nu ** 3 + c4 / nu ** 4)


def _chi2_crit(z: float, k: int) -> float:
    """Upper-tail chi-square_k critical value at the one-sided Gaussian level Phi(z), via
    the Wilson-Hilferty cube-root approximation (numpy-only; no scipy). This is the
    threshold for a JOINT test that a BLOCK of k coefficients is non-zero (Mahalanobis
    statistic m^T Cov^{-1} m ~ chi2_k under the null). At k=1 it is exactly z^2 (the
    squared z-test); for k>1 Wilson-Hilferty is accurate to <1% in the upper tail."""
    k = max(int(k), 1)
    if k == 1:
        return z * z
    t = 1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))
    return float(k * t ** 3) if t > 0 else 0.0


class BayesianDynamicsModel:
    def __init__(self, d: int, actuators, alpha: float = 1e-2, sigma0: float = 1.0,
                 hidden=(), interaction_pairs=(), rff: int = 0, rff_scale: float = 1.0,
                 forget: float = 1.0, rng=None, a0: float = 1e-3,
                 rff_W=None, rff_b=None):
        self.d = int(d)
        self.actuators = tuple(actuators)
        self.hidden = tuple(hidden)
        # exponential forgetting (recursive least squares with a forgetting factor):
        # forget<1 decays old sufficient statistics each step, so the belief TRACKS a
        # changing world instead of averaging over its whole history. 1.0 = remember
        # everything (stationary); ~0.95 = an effective window of ~1/(1-forget) steps.
        self.forget = float(forget)
        # variables usable as regressors (observed = everything not hidden)
        self.cols = tuple(i for i in range(d) if i not in self.hidden)
        self._col_pos = {c: k for k, c in enumerate(self.cols)}
        # only consider interactions among observed variables
        self.interaction_pairs = tuple((int(a), int(b)) for (a, b) in interaction_pairs
                                       if a not in self.hidden and b not in self.hidden)
        # sensors we actually model: observed, non-actuator
        self.sensors = tuple(i for i in range(d)
                             if i not in self.actuators and i not in self.hidden)
        # optional random-Fourier-feature basis per variable (general smooth
        # nonlinearity): a block cos(W·x_j + b) for each observed variable j.
        self.rff = int(rff)
        self.rff_scale = float(rff_scale)
        if self.rff > 0:
            if rff_W is not None and rff_b is not None:    # reuse an existing basis
                self.rff_W, self.rff_b = dict(rff_W), dict(rff_b)
            else:
                r = rng if rng is not None else np.random.default_rng(0)
                self.rff_W = {j: r.normal(0.0, rff_scale, self.rff) for j in self.cols}
                self.rff_b = {j: r.uniform(0.0, 2 * np.pi, self.rff) for j in self.cols}
        self.p = (len(self.cols) + len(self.interaction_pairs)
                  + self.rff * len(self.cols) + 1)          # +1 bias
        # kwargs to REBUILD this model with a different feature set (e.g. when
        # discover_interactions adds product features) WITHOUT silently dropping the
        # RFF basis or the noise prior. interaction_pairs is intentionally excluded
        # (the rebuild overrides it); pass rff_W/rff_b to reuse the exact basis.
        self._initkw = dict(d=self.d, actuators=self.actuators, alpha=float(alpha),
                            sigma0=float(sigma0), hidden=self.hidden, rff=self.rff,
                            rff_scale=self.rff_scale, forget=self.forget, a0=float(a0))

        # ---- Normal-Inverse-Gamma hyperparameters -------------------------------
        # alpha = lambda0, the weight prior precision (ridge). a0/b0 = InvGamma prior
        # on the noise variance; b0 chosen so the prior SCALE b0/a0 == sigma0^2, so the
        # noise estimate STARTS at sigma0^2 before any data.
        # a0 is deliberately VAGUE (1e-3): with a strong prior, sigma2 = b_N/a_N gets
        # floored well above the true noise for a low-noise sensor under forgetting
        # (small effective n), which silently breaks the continual change-detector
        # (it standardises surprise by sqrt(sigma2)). Vague -> sigma2 -> RSS/n quickly
        # and tracks the real noise, while the t-dof nu = 2 a0 + n stays ~= n.
        self.alpha = float(alpha)
        self._a0 = float(a0)
        self._b0 = float(sigma0) ** 2 * self._a0
        p = self.p
        # ---- sufficient statistics ----------------------------------------------
        # SHARED across sensors (Lambda_N = prior_coef*I + S has no sigma^2 and depends
        # only on the regressors, so it is identical for every sensor): ONE Gram, ONE
        # maintained inverse, one observation count, one decaying prior coefficient.
        self.S = np.zeros((p, p))                          # shared Gram sum  sum phi phi^T
        self.Pinv = np.eye(p) / self.alpha                 # shared Lambda_N^{-1}
        self._prior_coef = self.alpha                      # g^t * alpha (decays w/ forgetting)
        self._since_refresh = 0
        self._Pinv_dirty = False
        self._refresh_every = 200            # clean-recompute cadence (amortized O(p^3/K))
        self.n = 0.0                                       # shared observation count
        # PER-SENSOR cross-statistics (depend on the target y_i):
        self.r = {i: np.zeros(p) for i in self.sensors}    # cross sum  sum phi*y_i
        self.syy = {i: 0.0 for i in self.sensors}          # sum y_i^2
        self.sigma2 = {i: float(sigma0) ** 2 for i in self.sensors}   # = b_N/a_N
        self._dof = {i: 2.0 * self._a0 for i in self.sensors}         # nu = 2 a_N
        self._cache = {}                                   # i -> (mean, s2)

    # ---- feature map --------------------------------------------------------
    def _phi(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        parts = [x[list(self.cols)]]
        if self.interaction_pairs:
            parts.append(np.array([x[a] * x[b] for (a, b) in self.interaction_pairs]))
        if self.rff > 0:
            parts.append(np.concatenate(
                [np.cos(self.rff_W[j] * x[j] + self.rff_b[j]) for j in self.cols]))
        parts.append(np.array([1.0]))
        return np.concatenate(parts)

    # feature-vector position of a linear variable / of an interaction pair
    def _lin_k(self, j: int) -> int:
        return self._col_pos[j]

    def _inter_k(self, a: int, b: int) -> int:
        return len(self.cols) + self.interaction_pairs.index((a, b))

    # ---- online update ------------------------------------------------------
    def _refresh_pinv(self) -> None:
        """Clean recompute of the shared posterior inverse (flush rank-1 drift/windup)."""
        ridge = self._prior_coef + 1e-9               # tiny floor keeps it invertible
        self.Pinv = np.linalg.inv(ridge * np.eye(self.p) + self.S)
        self._Pinv_dirty = False
        self._since_refresh = 0

    def _ensure_pinv(self) -> np.ndarray:
        if self._Pinv_dirty:
            self._refresh_pinv()
        return self.Pinv

    def _absorb_shared(self, phi: np.ndarray) -> None:
        """Advance the SHARED Gram and posterior inverse by one step's regressors --
        done ONCE per step regardless of how many sensors are modelled."""
        g = self.forget
        if g < 1.0:                                   # decay history (incl. the prior)
            self.S *= g
            self._prior_coef *= g
            self.n *= g
        self.S += np.outer(phi, phi)
        self.n += 1.0
        self._since_refresh += 1
        if self._Pinv_dirty or self._since_refresh >= self._refresh_every:
            self._refresh_pinv()                      # clean recompute: flush drift/windup
        else:
            # forgetting-RLS covariance update, O(p^2): handles the decay g AND the rank-1
            # add together. Lambda_t = g Lambda_{t-1} + phi phi^T  =>
            # Pinv_t = (1/g)(Pinv - (Pinv phi)(Pinv phi)^T / (g + phi^T Pinv phi)).
            # For g==1 this is exactly the Sherman-Morrison rank-1 update.
            P = self.Pinv
            Pphi = P @ phi
            self.Pinv = (P - np.outer(Pphi, Pphi) / (g + float(phi @ Pphi))) / g

    def update(self, x_clamped: np.ndarray, x_next: np.ndarray) -> None:
        phi = self._phi(x_clamped)
        self._absorb_shared(phi)                      # one O(p^2) shared update
        g = self.forget
        for i in self.sensors:                        # O(p) cross-stat update per sensor
            if g < 1.0:
                self.r[i] *= g
                self.syy[i] *= g
            y = float(x_next[i])
            self.r[i] += phi * y
            self.syy[i] += y * y
        self._cache.clear()

    def fit_batch(self, X_clamped, X_next) -> None:
        """Recompute ALL sufficient statistics from a batch in ONE vectorized pass
        (S = Phi^T diag(w) Phi, r_i = Phi^T diag(w) y_i) instead of n sequential rank-1
        updates -- the fast path for rebuilding the model with a new feature set
        (discover_interactions). The forgetting weights w_t = forget^(n-1-t) reproduce
        the sequential RLS decay EXACTLY (newest sample = weight 1), so the result is
        identical to replaying update() over the batch, up to one clean inversion."""
        Phi = np.array([self._phi(np.asarray(x, float)) for x in X_clamped])  # (n, p)
        n = len(Phi)
        if n == 0:
            return
        g = self.forget
        w = (g ** np.arange(n - 1, -1, -1)) if g < 1.0 else np.ones(n)
        Wp = Phi * w[:, None]
        self.S = Wp.T @ Phi
        self.n = float(w.sum())
        self._prior_coef = self.alpha * (g ** n if g < 1.0 else 1.0)
        for i in self.sensors:
            y = np.array([float(xn[i]) for xn in X_next])
            self.r[i] = Wp.T @ y
            self.syy[i] = float(w @ (y * y))
        self._refresh_pinv()
        self._cache.clear()

    # ---- posterior over weights (cached) -----------------------------------
    def _post_light(self, i: int):
        """(mean, s2) for sensor i using the SHARED posterior inverse. Side effect:
        refreshes self.sigma2[i] = b_N/a_N and self._dof[i] = 2 a_N. No per-sensor
        covariance matrix is formed -- callers needing a quadratic form use
        s2 * (phi @ Pinv @ phi) with the shared Pinv."""
        if i in self._cache:
            return self._cache[i]
        Pinv = self._ensure_pinv()
        mean = Pinv @ self.r[i]                        # m_N = Lambda_N^{-1} r
        a_N = self._a0 + 0.5 * self.n
        # b_N = b0 + 0.5 (sum y^2 - m_N^T Lambda_N m_N), and m_N^T Lambda_N m_N = m_N^T r
        b_N = self._b0 + 0.5 * max(self.syy[i] - float(mean @ self.r[i]), 0.0)
        s2 = max(b_N / a_N, 1e-6)
        self.sigma2[i] = s2
        self._dof[i] = 2.0 * a_N
        self._cache[i] = (mean, s2)
        return mean, s2

    def _posterior(self, i: int):
        """Public posterior: (m_N, Cov_t) where Cov_t = (b_N/a_N) * Lambda_N^{-1} is the
        SCALE matrix of the multivariate Student-t marginal. Cov is materialized on
        demand from the shared Pinv (external callers: prior.py, thompson.py)."""
        mean, s2 = self._post_light(i)
        return mean, s2 * self.Pinv

    def _mean(self, i: int) -> np.ndarray:
        return self._post_light(i)[0]

    # ---- prediction ("what if") --------------------------------------------
    def predict_next(self, x_clamped: np.ndarray, command: dict | None = None):
        x_clamped = np.asarray(x_clamped, float).copy()
        if command:
            for j, v in command.items():
                x_clamped[j] = v
        phi = self._phi(x_clamped)
        Pinv = self._ensure_pinv()
        q = max(float(phi @ Pinv @ phi), 0.0)   # phi^T Lambda_N^{-1} phi -- shared, no sigma^2
        mu = x_clamped.copy()             # hidden / unmodelled entries pass through
        sd = np.zeros(self.d)
        for i in self.sensors:
            mean, s2 = self._post_light(i)
            mu[i] = float(mean @ phi)
            # predictive sd in Gaussian form: var = s2*(1 + phi^T Lambda_N^{-1} phi). We
            # deliberately do NOT apply the Student-t variance inflation nu/(nu-2) here:
            # it is a minor calibration nicety, and inflating sd shrinks the standardized
            # one-step surprise the continual-change detector relies on. The t correction
            # stays where it matters -- the edge TEST (recovered_edges) -- not the sd.
            sd[i] = float(np.sqrt(max(s2 * (1.0 + q), 0.0)))
        for j in self.actuators:
            mu[j] = command.get(j, x_clamped[j]) if command else x_clamped[j]
            sd[j] = 0.0
        return mu, sd

    # ---- expected information gain (the reward-free objective) --------------
    def expected_info_gain(self, x_clamped: np.ndarray, command: dict | None = None) -> float:
        x_clamped = np.asarray(x_clamped, float).copy()
        if command:
            for j, v in command.items():
                x_clamped[j] = v
        phi = self._phi(x_clamped)
        Pinv = self._ensure_pinv()
        q = max(float(phi @ Pinv @ phi), 0.0)
        # Lambda_N is shared across sensors (no sigma^2), so every sensor's parameter
        # info gain 1/2 log(1 + phi^T Lambda_N^{-1} phi) is IDENTICAL: sum = count * term.
        return float(len(self.sensors) * 0.5 * np.log1p(q))

    def seq_info_gain(self, phis) -> float:
        """Total expected info gain about the PARAMETERS from a SEQUENCE of feature
        vectors, summed by the CHAIN RULE: each step's gain is conditioned on the
        previous steps (its precision-inverse rank-1 updated as if the step were
        observed). This avoids the double-counting that summing per-step EIG against
        the SAME posterior produces along a held-command rollout (consecutive phi are
        highly correlated, so the same parameter-axis information is otherwise counted
        up to H times). No y is needed -- parameter info gain is value-independent.
        Because Lambda_N is shared, the per-sensor chain is identical, so we run it ONCE
        and multiply by the sensor count. Reduces to expected_info_gain for a single phi."""
        if not phis:
            return 0.0
        P = self._ensure_pinv().copy()                # shared Lambda_N^{-1}
        g = 0.0
        for phi in phis:
            Pphi = P @ phi
            q = float(phi @ Pphi)
            if q <= 0.0:
                continue
            g += 0.5 * np.log1p(q)
            P = P - np.outer(Pphi, Pphi) / (1.0 + q)  # posterior inverse after this step
        return float(len(self.sensors) * g)

    def raw_surprise(self, x_clamped: np.ndarray, command: dict | None = None) -> float:
        _, sd = self.predict_next(x_clamped, command)
        var = np.array([sd[i] ** 2 for i in self.sensors])
        return float(0.5 * np.sum(np.log(2 * np.pi * np.e * np.clip(var, EPS, None))))

    # ---- per-sensor versions (for observation-gating: which channel to watch) --
    def eig_sensor(self, i: int, x_clamped, command=None) -> float:
        """Expected info gain about sensor i's parameters from watching it now. (Under
        the shared precision this is the same for every sensor -- it is purely a
        property of how much the regressors phi shrink the shared weight covariance.)"""
        x = np.asarray(x_clamped, float).copy()
        if command:
            for j, v in command.items():
                x[j] = v
        phi = self._phi(x)
        Pinv = self._ensure_pinv()
        return float(0.5 * np.log1p(max(float(phi @ Pinv @ phi), 0.0)))

    def surprise_sensor(self, i: int, x_clamped, command=None) -> float:
        """Predictive entropy of sensor i's next value (naive curiosity)."""
        x = np.asarray(x_clamped, float).copy()
        if command:
            for j, v in command.items():
                x[j] = v
        phi = self._phi(x)
        mean, s2 = self._post_light(i)
        var = s2 * (1.0 + max(float(phi @ self.Pinv @ phi), 0.0))
        return float(0.5 * np.log(2 * np.pi * np.e * max(var, EPS)))

    # ---- the recovered causal graph ----------------------------------------
    def recovered_edges(self, z: float = 3.0, eps: float = 0.05,
                        correct_multiplicity: bool = False, grouped: bool = True) -> set:
        """Confident cross-edges j->i (j observed, j != i): variables whose value
        CAUSALLY influences i's next value, through ANY channel (linear or nonlinear).

        The test runs on the whole BLOCK of features source j generates -- its linear
        term, every product it participates in, and its RFF block (``_source_block``) --
        so a state-dependent (multiplicative / rotational) effect whose LINEAR coefficient
        averages to ~0 but whose PRODUCT coefficient is strong is still recovered (e.g. a
        pendulum's sin(theta)->cos(theta_next), carried by sin*omega). See
        ``recovered_edges_grouped`` for the full rationale. The dispatch keeps backward
        compatibility EXACT:

          * block size 1 (pure-linear basis -- no products/RFF for j): the legacy
            STUDENT-t test on the single coefficient, |m[k]|/sqrt(Cov[k,k]) vs a t-critical
            value at nu = 2 a_N, AND |m[k]| > eps. Bit-identical to the historical
            behaviour, so every linear-world result is unchanged.
          * block size >1: the JOINT Mahalanobis test T = m_B^T Cov_BB^{-1} m_B vs a
            chi^2_k critical value (Wilson-Hilferty, finite-nu inflated by nu/(nu-2)),
            gated by an effect-size floor sqrt(m_B^T S_BB m_B / n) > eps.

        ``grouped=False`` forces the legacy linear-only test on every source (ignores any
        product/RFF features), for a caller that wants strictly-linear edges.
        ``correct_multiplicity=True`` raises the level by sqrt(z^2 + 2 ln m) over the m
        candidate edges (family-wise false-edge control; off by default)."""
        candidates = [(i, j) for i in self.sensors for j in self.cols if j != i]
        mfam = max(len(candidates), 1)
        z_nom = float(np.sqrt(z * z + 2.0 * np.log(mfam))) if correct_multiplicity else z
        Pinv = self._ensure_pinv()
        n_eff = max(float(self.n), 1.0)
        E = set()
        for i in self.sensors:
            mean, s2 = self._post_light(i)
            nu = float(self._dof[i])
            crit_t = _t_crit(z_nom, nu)
            infl = nu / (nu - 2.0) if nu > 2.0 else 1e9
            for j in self.cols:
                if j == i:
                    continue
                B = self._source_block(j) if grouped else [self._lin_k(j)]
                if not B:
                    continue
                if len(B) == 1:                              # legacy single-coefficient t-test
                    k = B[0]
                    std = float(np.sqrt(max(s2 * Pinv[k, k], EPS)))
                    if abs(mean[k]) > eps and abs(mean[k]) / std > crit_t:
                        E.add((j, i))
                else:                                        # joint block (Mahalanobis) test
                    mB = mean[B]
                    CovBB = s2 * Pinv[np.ix_(B, B)]
                    try:
                        T = float(mB @ np.linalg.solve(CovBB, mB))
                    except np.linalg.LinAlgError:
                        continue
                    SBB = self.S[np.ix_(B, B)]
                    sigB = float(np.sqrt(max(float(mB @ SBB @ mB) / n_eff, 0.0)))
                    if T > _chi2_crit(z_nom, len(B)) * infl and sigB > eps:
                        E.add((j, i))
        return E

    def recovered_interactions(self, z: float = 3.0, eps: float = 0.05) -> set:
        """Confident *interaction* edges (a,b)->i: i's response to a is gated by b.
        Same Student-t test as recovered_edges (t-critical at nu = 2 a_N)."""
        Pinv = self._ensure_pinv()
        E = set()
        for i in self.sensors:
            mean, s2 = self._post_light(i)
            crit = _t_crit(z, self._dof[i])
            for (a, b) in self.interaction_pairs:
                k = self._inter_k(a, b)
                std = float(np.sqrt(max(s2 * Pinv[k, k], EPS)))
                if abs(mean[k]) > eps and abs(mean[k]) / std > crit:
                    E.add(((a, b), i))
        return E

    def _source_block(self, j: int) -> list:
        """Feature positions through which SOURCE variable j can influence a target: its
        linear term, EVERY interaction it participates in (j*k for any k, incl. its
        square j*j), and its random-Fourier block. A product j*k is shared by BOTH j's
        and k's blocks -- correct, since a gate j*k is a real channel for each."""
        idx = []
        if j in self._col_pos:
            idx.append(self._col_pos[j])                     # linear term
        base_int = len(self.cols)
        for t, (a, b) in enumerate(self.interaction_pairs):
            if a == j or b == j:
                idx.append(base_int + t)                     # any product involving j
        if self.rff > 0 and j in self._col_pos:
            base_rff = len(self.cols) + len(self.interaction_pairs)
            start = base_rff + self._col_pos[j] * self.rff
            idx.extend(range(start, start + self.rff))       # j's nonlinear basis block
        return idx

    def recovered_edges_grouped(self, z: float = 3.0, eps: float = 0.05,
                                correct_multiplicity: bool = False) -> set:
        """Explicit nonlinear-aware (block) edge test. As of the default-wiring change this
        is simply ``recovered_edges`` with grouping ON (now the default); kept as a named
        entry point. An edge j->i exists iff the WHOLE BLOCK of features derived from j
        (linear + every product with j + j's RFF block) JOINTLY explains i's next value --
        not merely iff j's single linear coefficient is non-zero -- which is the fix for
        state-dependent (multiplicative / rotational) effects whose linear coefficient
        averages to ~0 (e.g. a pendulum's sin(theta)->cos(theta_next), carried by sin*omega).
        For a pure-linear basis (block size 1) it reduces EXACTLY to the legacy t-test."""
        return self.recovered_edges(z=z, eps=eps,
                                    correct_multiplicity=correct_multiplicity, grouped=True)

    def recovered_parents(self, i: int, z: float = 3.0, eps: float = 0.05) -> list:
        return sorted(j for (j, t) in self.recovered_edges(z, eps) if t == i)

    def recovered_marks(self, z: float = 3.0, eps: float = 0.05) -> dict:
        """The HONEST causal map: split recovered cross-edges into DIRECTED (genuinely
        do-identified) vs BIDIRECTED (an association the agent cannot interventionally
        orient or de-confound), instead of asserting every association as a confident
        directed edge. An edge j->i is do-identified iff:
          * j is an ACTUATOR -- the agent forces j, so do(j) identifies the effect; or
          * j is a sensor with an INSTRUMENT -- some actuator a with a->j recovered but
            a NOT a recovered direct parent of i, so forcing a moves j without otherwise
            moving i, which orients AND de-confounds j->i.
        Otherwise (e.g. two sensors driven by an UN-actuable hidden common cause -- the
        exact case where intervention has no reach) the j-i link is emitted as a
        BIDIRECTED mark j<->i (possibly confounded), NOT a confident directed edge.
        Returns {'directed': set((j,i)), 'bidirected': set((a,b)) with a<b}."""
        E = self.recovered_edges(z=z, eps=eps)
        acts = set(self.actuators)
        directed = set()
        for (j, i) in E:
            if j in acts or any((a, j) in E and (a, i) not in E for a in acts):
                directed.add((j, i))                  # do-identified (actuator or instrument)
        directed_pairs = {tuple(sorted(e)) for e in directed}
        bidirected = set()
        for (j, i) in E:
            pair = tuple(sorted((j, i)))
            if (j, i) not in directed and pair not in directed_pairs:
                bidirected.add(pair)                  # not orientable either way
        return {"directed": directed, "bidirected": bidirected}

    def weight(self, i: int, j: int) -> float:
        return float(self._mean(i)[self._lin_k(j)])

    def interaction_weight(self, i: int, a: int, b: int) -> float:
        return float(self._mean(i)[self._inter_k(a, b)])


def edge_scores(model: "BayesianDynamicsModel", true_edges: set):
    """Precision / recall / F1 of recovered cross-edges vs ground truth."""
    rec = model.recovered_edges()
    tp = len(rec & true_edges)
    prec = tp / len(rec) if rec else (1.0 if not true_edges else 0.0)
    recall = tp / len(true_edges) if true_edges else 1.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return {"precision": prec, "recall": recall, "f1": f1,
            "recovered": rec, "missing": true_edges - rec, "extra": rec - true_edges}


__all__ = ["BayesianDynamicsModel", "edge_scores"]
