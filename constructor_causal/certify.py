"""
Certificates: anytime-valid cloning-free reliability, and higher-lag latent detection.

Two of the agent's honest frontiers, addressed with results from the deep-research
pass (see RESEARCH.md). Both are reward-free.

Q2 — CLONING-FREE VERIFICATION.  ``AnytimeCS`` is a time-uniform (1-alpha) confidence
sequence for a [0,1]-bounded mean (the sub-Gaussian normal-mixture boundary of
Robbins / Howard-Ramdas): an interval valid SIMULTANEOUSLY at every t, so you may
peek after each observation and stop when satisfied. ``certify_reliability`` runs a
constructor over and over in ONE ongoing trajectory -- no clone(), no reset() -- and
accumulates a certificate that the success probability is >= tau (POSSIBLE), < tau
(IMPOSSIBLE), or not yet decided. This is the operational form of Constructor
Theory's "possible iff performable to arbitrarily high accuracy": the certificate
keeps tightening (hi - lo -> 0) as the agent lives.

Q1 — HIGHER-LAG LATENT DETECTION.  A slow hidden AR(1) common cause is observationally
equivalent to a self-loop at ONE lag (provably). With an EXTRA lag it leaves a second
timescale: ``detect_latent_lag`` flags a sensor whose model residual is still
predicted by its OWN PREVIOUS value -- a single-trajectory signature of an unmodelled
slow driver, sharper than the variance+self-loop heuristic in agent.detect_hidden.
"""
from __future__ import annotations

import numpy as np

from .model import _t_crit


# --------------------------------------------------------------------------- Q2
class AnytimeCS:
    """Time-uniform (1-alpha) confidence sequence for a [0,1]-bounded mean.

    Sub-Gaussian (sigma^2=1/4) normal-mixture boundary: with S_t = sum (X_i - mu),
    P(exists t: |S_t| >= B_t) <= alpha, B_t = sqrt(2 (V_t+rho) log( sqrt((V_t+rho)/rho)/a )),
    V_t = t/4, a = alpha/2 (two-sided). The mean CS is mean_t +/- B_t/t, valid at
    ALL t at once -- so you can stop on a data-dependent rule without inflating error.
    """

    def __init__(self, alpha: float = 0.05, rho: float = 1.0, sigma2: float = 0.25):
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.sigma2 = float(sigma2)
        self.t = 0
        self.s = 0.0

    def update(self, x) -> "AnytimeCS":
        self.t += 1
        self.s += float(x)
        return self

    def interval(self):
        if self.t == 0:
            return (0.0, 1.0)
        V = self.t * self.sigma2
        a = self.alpha / 2.0
        B = np.sqrt(2.0 * (V + self.rho) * np.log(np.sqrt((V + self.rho) / self.rho) / a))
        mu, r = self.s / self.t, B / self.t
        return (max(0.0, mu - r), min(1.0, mu + r))

    def verdict(self, tau: float) -> str:
        lo, hi = self.interval()
        if lo >= tau:
            return "POSSIBLE"
        if hi < tau:
            return "IMPOSSIBLE"
        return "UNDECIDED"


class BettingCS:
    """Waudby-Smith & Ramdas (JRSS-B 2024) BETTING confidence sequence for a [0,1]
    mean — the recommended construction (verified to strictly dominate the mixture CS
    above: variance-adaptive, LIL-optimal width ~sqrt(log log t / t), distribution-
    free up to boundedness). For a grid of candidate means m, the capital process
    K_t(m) = prod_i (1 + lambda_i(m)(X_i - m)) is a test martingale at the true mean
    p (Ville's inequality => validity), and the (1-alpha)-CS is {m : K_t(m) < 1/alpha}.
    Bets are the predictable variance-adaptive aGRAPA rule, truncated to keep the
    capital nonnegative. Same interface as AnytimeCS (drop-in)."""

    def __init__(self, alpha: float = 0.05, grid: int = 199, c: float = 0.5):
        self.alpha = float(alpha)
        self.c = float(c)
        self.ms = np.linspace(0.5 / (grid + 1), 1 - 0.5 / (grid + 1), grid)
        self.logK = np.zeros(grid)
        self.t = 0
        self._sum = 0.0
        self._ssd = 0.0
        self._mu = 0.5                                # predictable mean (1/2 prior)
        self._var = 0.25                             # predictable variance (1/4 prior)

    def update(self, x) -> "BettingCS":
        x = float(x)
        mu, var = self._mu, self._var               # predictable (uses data < t only)
        lam = (mu - self.ms) / (var + (mu - self.ms) ** 2)            # aGRAPA bet
        lam = np.clip(lam, -self.c / (1 - self.ms), self.c / self.ms)  # keep K >= 0
        self.logK += np.log1p(lam * (x - self.ms))
        self.t += 1
        self._sum += x
        self._ssd += (x - mu) ** 2
        self._mu = (0.5 + self._sum) / (self.t + 1)
        self._var = (0.25 + self._ssd) / (self.t + 1)
        return self

    def interval(self):
        if self.t == 0:
            return (0.0, 1.0)
        inside = self.logK < -np.log(self.alpha)
        if not inside.any():
            i = int(np.argmin(self.logK))
            return (float(self.ms[i]), float(self.ms[i]))
        idx = np.flatnonzero(inside)
        return (float(self.ms[idx[0]]), float(self.ms[idx[-1]]))

    def verdict(self, tau: float) -> str:
        lo, hi = self.interval()
        if lo >= tau:
            return "POSSIBLE"
        if hi < tau:
            return "IMPOSSIBLE"
        return "UNDECIDED"


class DriftDetector:
    """An e-process betting AGAINST a certified mean p0: a test martingale that grows
    when the live success rate moves away from p0. When its wealth crosses 1/alpha the
    stream is (anytime-validly) inconsistent with the prior certificate -> the world
    drifted and the constructor must be re-verified. Reuses the exact Ville's-
    inequality machinery; false-alarm probability <= alpha under stationarity."""

    def __init__(self, p0: float, alpha: float = 0.05, c: float = 0.5):
        self.p0 = float(p0)
        self.alpha = float(alpha)
        self.c = float(c)
        self.logW = 0.0
        self.t = 0
        self._sum = 0.0
        self._mu = 0.5

    def update(self, x) -> bool:
        x = float(x)
        lam = (self._mu - self.p0) / (0.25 + (self._mu - self.p0) ** 2)
        lam = np.clip(lam, -self.c / (1 - self.p0), self.c / self.p0)
        self.logW += np.log1p(lam * (x - self.p0))
        self.t += 1
        self._sum += x
        self._mu = (0.5 + self._sum) / (self.t + 1)
        return self.drift()

    def drift(self) -> bool:
        return self.logW >= -np.log(self.alpha)


def certify_reliability(env, program, effect, tau: float = 0.9, alpha: float = 0.05,
                        max_trials: int = 5000, drift_steps: int = 2, min_trials: int = 20,
                        cs_factory=BettingCS, rng=None, reset: bool = True):
    """CLONING-FREE certificate that ``program`` drives the substrate into ``effect``
    with probability >= tau. Runs the program repeatedly in ONE env -- never reset or
    cloned -- with a few random 'drift' steps between executions to vary the start
    (still a single trajectory). Stops as soon as the anytime-valid CS decides.

    Returns {verdict, interval, n, p_hat}. Honest caveat: anytime-validity assumes
    the per-execution success indicators are (conditionally) exchangeable; strong
    serial dependence widens effective error, which is why a change-detector should
    reopen the certificate when the world drifts."""
    rng = rng if rng is not None else np.random.default_rng()
    cs = cs_factory(alpha=alpha)                      # betting CS by default (tightest)
    if reset:                                         # embodied path passes reset=False
        env.reset()                                   # (continue from the live state)
    setpts = (-2.0, 0.0, 2.0)
    n, succ = 0, 0
    while n < max_trials:
        for _ in range(drift_steps):                 # vary the start, same trajectory
            env.step({j: float(rng.choice(setpts)) for j in env.actuators})
        x = None
        for cmd in program:
            x = env.step(cmd)
        hit = 1.0 if effect.contains(x) else 0.0
        cs.update(hit)
        succ += int(hit)
        n += 1
        if n >= min_trials and cs.verdict(tau) != "UNDECIDED":
            break
    lo, hi = cs.interval()
    p_hat = succ / max(n, 1)
    return {"verdict": cs.verdict(tau), "interval": (lo, hi), "n": n,
            "p_hat": p_hat, "estimate": p_hat, "quantity": "reach_from_rest"}


# ---- Q2 extension: behaviour-agnostic (passive) model-based OPE -------------
def certify_passive(agent, program, effect, tau: float = 0.9, alpha: float = 0.05,
                    precond=None, max_states: int = 3000):
    """BEHAVIOUR-AGNOSTIC certificate: estimate a constructor's reliability from the
    agent's PASSIVELY-collected buffer of visited states + its learned model, WITHOUT
    re-executing the constructor and WITHOUT knowing what behaviour produced the
    stream. For each visited state satisfying ``precond``, roll ``program`` forward
    under the learned model and check whether the predicted final state lands in
    ``effect``; wrap the per-state hits in a betting confidence sequence.

    This is model-based off-policy evaluation over the agent's own occupancy: it lets
    the agent SCREEN many candidate skills from one stream of experience for free,
    reserving costly real execution for the promising ones. HONEST LIMIT: the
    certificate is valid for the MODEL on the visited-state distribution — its
    real-world validity holds only insofar as the model is calibrated (use
    calibrate_passive / DriftDetector). Returns {verdict, interval, n, p_hat}."""
    states = [xc for (xc, _) in agent.buffer[-max_states:]]
    if precond is not None:
        states = [s for s in states if precond.contains(s)]
    if not states:
        return {"verdict": "UNDECIDED", "interval": (0.0, 1.0), "n": 0, "p_hat": 0.0,
                "estimate": None, "quantity": "model_rollout_reach"}
    cs = BettingCS(alpha=alpha)
    succ = 0
    for s in states:
        x = np.asarray(s, float).copy()
        for cmd in program:
            x, _ = agent.model.predict_next(x, cmd)
        hit = 1.0 if effect.contains(x) else 0.0
        cs.update(hit)
        succ += int(hit)
    lo, hi = cs.interval()
    p_hat = succ / len(states)
    return {"verdict": cs.verdict(tau), "interval": (lo, hi), "n": len(states),
            "p_hat": p_hat, "estimate": p_hat, "quantity": "model_rollout_reach"}


def calibrate_passive(agent, env, program, effect, n_real: int = 200, drift_steps: int = 2,
                      tol: float = 0.15, rng=None):
    """Validity guard for the passive (model-based) certificate: compare the model's
    predicted reliability (from the buffer) against the REAL in-stream success rate
    (a handful of actual executions, cloning-free). If |p_model - p_real| <= tol the
    model-based certificate is trustworthy; a large gap means the model is wrong on
    this skill's trajectory and the passive verdict must NOT be believed. Returns
    {p_model, p_real, gap, trustworthy}."""
    rng = rng if rng is not None else np.random.default_rng()
    p_model = certify_passive(agent, program, effect, alpha=0.05)["p_hat"]
    env.reset()
    setpts = (-2.0, 0.0, 2.0)
    succ = 0
    for _ in range(n_real):
        for _ in range(drift_steps):
            env.step({j: float(rng.choice(setpts)) for j in env.actuators})
        x = None
        for cmd in program:
            x = env.step(cmd)
        succ += int(effect.contains(x))
    p_real = succ / max(n_real, 1)
    gap = abs(p_model - p_real)
    return {"p_model": p_model, "p_real": p_real, "gap": gap, "trustworthy": gap <= tol}


# ---- Q2 extension: MODEL-FREE behaviour-agnostic OPE (density-ratio / DICE) ---
def _stationary_reliability(S, Sn, effect, svars, edges, bins):
    """Empirical stationary occupancy of `effect` for the Markov chain on the tracked
    sensors, built from real on-target transitions (S->Sn). The model-free core."""
    nd = len(svars)

    def flat(x):
        idx = 0
        for d in range(nd):
            b = int(np.clip(np.digitize(x[d], edges[d]) - 1, 0, bins - 1))
            idx = idx * bins + b
        return idx

    keys = sorted(set(flat(s) for s in S) | set(flat(s) for s in Sn))
    pos = {k: i for i, k in enumerate(keys)}
    m = len(keys)
    P = np.zeros((m, m))
    for s, sn in zip(S, Sn):
        P[pos[flat(s)], pos[flat(sn)]] += 1
    rs = P.sum(1)
    live = rs > 0
    P[live] /= rs[live][:, None]
    P[~live] = 1.0 / m                                 # absorbing-state guard
    pi = np.ones(m) / m
    for _ in range(2000):
        pi = pi @ P
    pi /= pi.sum()
    # which bins lie in the effect box (centre of each tracked var within its bounds)
    cent = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(nd)]
    bnd = {v: (lo, hi) for (v, lo, hi) in effect.bounds}
    rho = 0.0
    for k, i in pos.items():
        rem, ok = k, True
        for d in range(nd - 1, -1, -1):
            b = rem % bins; rem //= bins
            lo, hi = bnd[svars[d]]
            if not (lo <= cent[d][b] <= hi):
                ok = False
        if ok:
            rho += pi[i]
    return float(rho)


def certify_modelfree(agent, command, effect, tau: float = 0.9, action_tol: float = 0.4,
                      bins: int = 20, alpha: float = 0.05, min_onpolicy: int = 50,
                      n_boot: int = 200, rng=None):
    """MODEL-FREE, behaviour-agnostic OPE (a tabular instance of the DICE / stationary-
    distribution-correction idea). Estimate the long-run reliability of a STATIONARY
    intervention `command` (e.g. hold a knob at v) — the stationary fraction of time the
    tracked sensors occupy `effect` — from the agent's off-policy buffer, using ONLY
    REAL transitions in which the behaviour happened to match `command` (within
    action_tol). No dynamics model, no behaviour policy. Because it never rolls out a
    learned model, it is RIGHT where a wrong model is wrong (verified on the nonlinear
    world). A bootstrap CI gives the verdict.

    The binding limit is OVERLAP/positivity: if the behaviour rarely takes `command`,
    there is no coverage and the value is non-identifiable -> returns UNDECIDED (the
    provable wall from the research). CI is bootstrap (batch), not anytime-valid.
    Returns {verdict, reliability, interval, n_onpolicy}."""
    rng = rng if rng is not None else np.random.default_rng()
    svars = list(effect.vars())
    if not svars:
        return {"verdict": "UNDECIDED", "reliability": None, "estimate": None,
                "quantity": "stationary_fraction", "interval": (0.0, 1.0),
                "n_onpolicy": 0, "reason": "effect has no variables"}
    onS, onSn = [], []
    for (xc, xn) in agent.buffer:
        if all(abs(xc[a] - v) <= action_tol for a, v in command.items()):
            onS.append(np.array([xc[v] for v in svars]))
            onSn.append(np.array([xn[v] for v in svars]))
    n = len(onS)
    if n < min_onpolicy:                               # no coverage -> non-identifiable
        return {"verdict": "UNDECIDED", "reliability": None, "estimate": None,
                "quantity": "stationary_fraction", "interval": (0.0, 1.0),
                "n_onpolicy": n, "reason": "no overlap (behaviour rarely takes command)"}
    S, Sn = np.array(onS), np.array(onSn)
    edges = [np.linspace(min(S[:, d].min(), Sn[:, d].min()) - 1e-6,
                         max(S[:, d].max(), Sn[:, d].max()) + 1e-6, bins + 1)
             for d in range(len(svars))]
    rho = _stationary_reliability(S, Sn, effect, svars, edges, bins)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(_stationary_reliability(S[idx], Sn[idx], effect, svars, edges, bins))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    verdict = "POSSIBLE" if lo >= tau else ("IMPOSSIBLE" if hi < tau else "UNDECIDED")
    return {"verdict": verdict, "reliability": rho, "estimate": rho,
            "quantity": "stationary_fraction", "interval": (float(lo), float(hi)),
            "n_onpolicy": n}


def certify_modelfree_continuous(agent, command, effect, gamma: float = 0.95,
                                 action_tol: float = 0.4, n_centers: int = 20,
                                 ridge: float = 1e-4, alpha: float = 0.05, tau: float = 0.9,
                                 min_onpolicy: int = 50, n_boot: int = 150, rng=None):
    """CONTINUOUS-STATE model-free OPE (no discretization), via LSTD with an RBF basis
    — the linear/regularized-Lagrangian (DualDICE-family) closed form. For a stationary
    "hold command" target it estimates the gamma-DISCOUNTED occupancy reliability
    (gamma->1 approaches the long-run fraction), using only real on-target transitions:
    fit V on RBF state features by the LSTD normal equations A w = c with the ASYMMETRIC
    A = mean phi(s)(phi(s)-gamma phi(s'))^T + ridge*I (the verified-correct form; NOT
    the symmetric Gram), c = mean r(s) phi(s); reliability = (1-gamma)*mean V(s).

    Drops `certify_modelfree`'s binning, at the cost of FUNCTION-APPROXIMATION bias
    (a few %: RBF realizability + ridge + finite single-trajectory sample). The
    overlap/positivity wall is unchanged (UNDECIDED with no on-target coverage). CI is
    a BATCH bootstrap — NOT anytime-valid: an anytime-valid theorem for single-
    trajectory stationary-ratio OPE is an OPEN problem (the betting CS is provably
    anytime-valid only for single-step bandits). Returns {verdict, value, interval,
    n_onpolicy}."""
    rng = rng if rng is not None else np.random.default_rng()
    svars = list(effect.vars())
    if not svars:
        return {"verdict": "UNDECIDED", "value": None, "estimate": None,
                "quantity": "discounted_occupancy", "interval": (0.0, 1.0), "n_onpolicy": 0}
    onS, onSn = [], []
    for (xc, xn) in agent.buffer:
        if all(abs(xc[a] - v) <= action_tol for a, v in command.items()):
            onS.append(np.array([xc[v] for v in svars]))
            onSn.append(np.array([xn[v] for v in svars]))
    n = len(onS)
    if n < min_onpolicy:
        return {"verdict": "UNDECIDED", "value": None, "estimate": None,
                "quantity": "discounted_occupancy", "interval": (0.0, 1.0),
                "n_onpolicy": n, "reason": "no overlap"}
    S, Sn = np.array(onS), np.array(onSn)
    bnd = {v: (lo, hi) for (v, lo, hi) in effect.bounds}
    cen, bw = [], []
    for d in range(len(svars)):
        lo = min(S[:, d].min(), Sn[:, d].min()); hi = max(S[:, d].max(), Sn[:, d].max())
        c = np.linspace(lo, hi, n_centers); cen.append(c); bw.append(c[1] - c[0] if n_centers > 1 else 1.0)

    def feat(X):                                       # additive RBF features per tracked var
        cols = [np.exp(-0.5 * ((X[:, d:d + 1] - cen[d]) / bw[d]) ** 2) for d in range(len(svars))]
        cols.append(np.ones((len(X), 1)))
        return np.concatenate(cols, axis=1)

    def in_effect(X):
        ok = np.ones(len(X), bool)
        for d, v in enumerate(svars):
            lo, hi = bnd[v]; ok &= (X[:, d] >= lo) & (X[:, d] <= hi)
        return ok.astype(float)

    PS, PN, rew = feat(S), feat(Sn), in_effect(S)

    def value(idx):
        A = PS[idx].T @ (PS[idx] - gamma * PN[idx]) / len(idx) + ridge * np.eye(PS.shape[1])
        c = (rew[idx][:, None] * PS[idx]).mean(0)
        w = np.linalg.solve(A, c)
        return float(np.clip((1 - gamma) * (PS[idx] @ w).mean(), 0.0, 1.0))

    val = value(np.arange(n))
    boots = [value(rng.integers(0, n, n)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    verdict = "POSSIBLE" if lo >= tau else ("IMPOSSIBLE" if hi < tau else "UNDECIDED")
    return {"verdict": verdict, "value": val, "estimate": val,
            "quantity": "discounted_occupancy", "interval": (float(lo), float(hi)),
            "n_onpolicy": n}


# ---- Q2 extension: MODEL-FREE FINITE-HORIZON REACH (matches the effect box) ---
def _reach_windows(buffer, command, horizon: int, action_tol: float = 0.4):
    """States reached by holding ``command`` for ``horizon`` CONSECUTIVE steps, taken
    from the agent's buffer. Walk the (x_clamped, x_next) stream tracking the current
    run of on-target steps; whenever the trailing ``horizon`` steps were ALL on-target,
    emit that transition's ``x_next`` — the state after a ``horizon``-step hold, the same
    finite horizon ``characterize_effect`` used to define the box, but starting from the
    agent's OWN realistic visited states rather than from rest. Returns a list of reached
    state arrays. Windows within a run OVERLAP (stride 1), so they are serially dependent
    — the anytime-validity caveat of certify_reliability applies."""
    reached = []
    runlen = 0
    for (xc, xn) in buffer:
        on = all(abs(xc[a] - v) <= action_tol for a, v in command.items())
        runlen = runlen + 1 if on else 0
        if runlen >= horizon:
            reached.append(np.asarray(xn, float))
    return reached


def certify_modelfree_reach(agent, command, effect, horizon: int = 3, tau: float = 0.9,
                            action_tol: float = 0.4, alpha: float = 0.05,
                            min_windows: int = 15, cs_factory=BettingCS, rng=None):
    """MODEL-FREE FINITE-HORIZON REACH certificate that MATCHES a hold-skill's box.

    A hold-skill's effect box is characterized as the state after an ``h_prim``-step hold
    (a TRANSIENT, by design — short primitives must NOT reach deep/slow targets, so that
    composition is required). The right certificate therefore measures FINITE-HORIZON
    reach over that SAME horizon, NOT infinite-horizon occupancy (which a slow variable
    overshoots). This scans the buffer for every window of ``horizon`` consecutive
    on-target steps (via _reach_windows), tests whether the reached state lands in
    ``effect``, and wraps the 0/1 indicators in a betting confidence sequence.

    HONEST quantity: windows start from the agent's REALISTIC visited states, not from
    rest, so this answers 'when I actually hold this knob, do I land in the box `horizon`
    steps later?' — a more honest quantity than the synthetic from-rest reach. Overlapping
    windows are serially dependent, so anytime-validity is approximate (as in
    certify_reliability). < min_windows hold-windows -> UNDECIDED (no coverage; the
    positivity wall, now on consecutive holds). Returns {verdict, reach, estimate,
    quantity, interval, n_windows, n}."""
    reached = _reach_windows(agent.buffer, command, horizon, action_tol)
    m = len(reached)
    if m < min_windows:
        return {"verdict": "UNDECIDED", "reach": None, "estimate": None,
                "quantity": "reach_held_horizon", "interval": (0.0, 1.0),
                "n_windows": m, "n": m, "reason": "no hold-windows of length >= horizon"}
    cs = cs_factory(alpha=alpha)
    succ = 0
    for s in reached:
        hit = 1.0 if effect.contains(s) else 0.0
        cs.update(hit)
        succ += int(hit)
    lo, hi = cs.interval()
    reach = succ / m
    return {"verdict": cs.verdict(tau), "reach": reach, "estimate": reach,
            "quantity": "reach_held_horizon", "interval": (lo, hi),
            "n_windows": m, "n": m}


# ---- Q2 extension: RESET-FREE self-certification of the WHOLE library --------
def certify_library(agent, gamma: float = 0.95, tau: float = 0.9, alpha: float = 0.05,
                    action_tol: float = 0.4, min_onpolicy: int = 50, min_windows: int = 15,
                    rng=None):
    """Decide which library skills are still 'possible' (Constructor Theory's
    criterion) FROM THE AGENT'S OWN LIFE-STREAM — no clone(), no reset(), no
    re-execution. This is what the cloning-free certificates buy the main goal: the
    constructor library's POSSIBLE verdict, certified from one trajectory rather than by
    replaying the world. Each possible constructor is routed by program shape, and CRUCIALLY
    each is graded by a certificate whose QUANTITY MATCHES its effect box (a finite-horizon
    reach, by design transient so that composition is required — not infinite-horizon
    occupancy, which a slow variable overshoots):

      * a constant single-knob HOLD -> certify_modelfree_reach over the skill's OWN horizon
        (model-free, finite-horizon reach from realistic visited starts). If there are too
        few consecutive-hold windows it falls back to certify_passive (model-based screen).
      * any other program (multi-knob schedule / gated cascade / conditional) ->
        certify_passive: model-based H-step rollout from the visited-state occupancy.

    Returns {name: {verdict, value, quantity, n, method}}. A skill comes back UNDECIDED
    when the stream has no overlap with its command (the provable positivity wall, now on
    CONSECUTIVE holds) — that verdict is not a dead end but a POINTER to the command region
    the agent must visit to make the skill certifiable (the hook for overlap-driven
    curiosity; see RESEARCH.md). HONEST LIMITS: the reach quantity is 'reach from realistic
    starts' (not synthetic from-rest); the model-based fallback is only as trustworthy as
    the model (gate with calibrate_passive / DriftDetector); CIs are betting-CS over
    serially-dependent windows, so anytime-validity is approximate. (`gamma`/`min_onpolicy`
    are retained for call-signature compatibility; the router now uses `min_windows`. For an
    explicit stationary-occupancy query, call certify_modelfree_continuous directly.)"""
    rng = rng if rng is not None else np.random.default_rng()
    report = {}
    for c in agent.library.possible():
        prog = tuple(c.program)
        first = prog[0] if prog else {}
        is_hold = (len(first) == 1 and all(cmd == first for cmd in prog)
                   and next(iter(first)) in agent.actuators)
        if is_hold:
            res = certify_modelfree_reach(agent, dict(first), c.effect, horizon=len(prog),
                                          tau=tau, action_tol=action_tol, alpha=alpha,
                                          min_windows=min_windows, rng=rng)
            if res["n"] < min_windows:                 # too few hold-windows -> model-based screen
                res = certify_passive(agent, prog, c.effect, tau=tau, alpha=alpha,
                                      precond=c.precond)
                method = "model-based(passive-fallback)"
            else:
                method = "model-free-reach"
            report[c.name] = {"verdict": res["verdict"], "value": res.get("estimate"),
                              "quantity": res.get("quantity"),
                              "n": res.get("n_windows", res.get("n")), "method": method}
        else:
            res = certify_passive(agent, prog, c.effect, tau=tau, alpha=alpha,
                                  precond=c.precond)
            report[c.name] = {"verdict": res["verdict"], "value": res.get("estimate"),
                              "quantity": res.get("quantity"),
                              "n": res["n"], "method": "model-based"}
    return report


# --------------------------------------------------------------------------- Q1
def detect_latent_lag(agent, z: float = 3.0, min_extra: float = 0.1):
    """Flag sensors driven by a slow HIDDEN cause, from a single trajectory, using a
    SECOND lag. After the current model explains a sensor with its contemporaneous
    parents (incl. its own self-loop), test whether the residual is still predicted
    by the sensor's PREVIOUS value. A pure self-loop leaves white residuals (no extra
    lag); a hidden slow driver leaves a second timescale that the lagged value picks
    up. Returns [(sensor, lagged-coeff, |t|)] for flagged sensors."""
    buf = agent.buffer
    if len(buf) < 30:
        return []
    Xc = np.array([xc for (xc, _) in buf])
    Xn = np.array([xn for (_, xn) in buf])
    flagged = []
    for i in agent.model.sensors:
        mean = agent.model._mean(i)
        phi = np.array([agent.model._phi(x) for x in Xc])
        resid = Xn[:, i] - phi @ mean
        r = resid[1:]                                # residual predicting x_{t+1}
        lagged = Xc[:-1, i]                          # the sensor's value one step earlier
        r = r - r.mean()
        lagged = lagged - lagged.mean()
        denom = float(lagged @ lagged)
        if denom < 1e-9:
            continue
        slope = float(lagged @ r) / denom
        rr = r - slope * lagged
        # noise dof: the residual was already shaped by the model's p params, plus the
        # one lag slope -> n - p - 1 (n-2 over-states dof, inflating the t-stat), and
        # decide against the t-critical, not a fixed Gaussian z (matches recovered_edges).
        dof = max(len(r) - agent.model.p - 1, 1)
        se = float(np.sqrt(max(rr @ rr, 1e-12) / dof / denom))
        tstat = abs(slope) / max(se, 1e-12)
        if abs(slope) > min_extra and tstat > _t_crit(z, dof):
            flagged.append((i, float(slope), float(tstat)))
    return flagged


__all__ = ["AnytimeCS", "BettingCS", "DriftDetector", "certify_reliability",
           "certify_passive", "calibrate_passive", "certify_modelfree",
           "certify_modelfree_continuous", "certify_modelfree_reach",
           "certify_library", "detect_latent_lag"]
