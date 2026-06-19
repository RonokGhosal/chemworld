"""
The agent: causal DAGs + constructors + active inference, learning with no reward.

It ties the four ideas into one loop:

    DAGs              -- the form of its belief (which variable causes which).
    causal inference  -- it *intervenes* (forces knobs) to identify direction,
                         not just correlation; the decoy is rejected this way.
    active inference   -- it chooses those interventions to maximise expected
                         information gain (the reward-free EFE epistemic term).
    Constructor Theory -- it distils what it learns into composable constructors
                         and grows them, building an algebra of what it can do.

Lifecycle:
    explore(n)        reward-free curiosity loop; updates the causal belief.
    build_library()   mint & verify primitive constructors from the belief.
    achieve(target)   compose the library into a program that reaches a goal
                      (the goal appears only now, never during learning).
    whatif(state,cmd) counterfactual prediction from the learned model.
    recovered_dag()   the cause-effect map, scored against ground truth.
"""
from __future__ import annotations

import numpy as np

from .active_inference import (CertifyingExperimenter, EpistemicExperimenter,
                               NaiveSurpriseExperimenter, PassiveExperimenter,
                               RandomExperimenter)
from .constructor import Box, Library, MIN_TRIALS
from .model import BayesianDynamicsModel, edge_scores, _t_crit
from .planner import ConstructorSynthesizer
from .thompson import ThompsonExperimenter

_EXPERIMENTERS = {
    "epistemic": EpistemicExperimenter,      # reward-free info-gain (greedy, epsilon-explore)
    "thompson": ThompsonExperimenter,        # reward-free info-gain (posterior-sampled, no epsilon)
    "random": RandomExperimenter,
    "naive": NaiveSurpriseExperimenter,
    "passive": PassiveExperimenter,          # never intervenes (confounder foil)
}


def discover_actuators(world, probe: float = 5.0, steps: int = 2, tol: float = 0.5,
                       rng=None):
    """Discover which OBSERVED variables are actuators, by poking each one.

    Force a variable to a distinctive out-of-range value; if it holds there it is
    controllable (an actuator), otherwise it ignored the attempt and evolved (a
    sensor). The agent is NOT told the interface — it finds it. Returns the
    discovered actuator indices."""
    rng = rng if rng is not None else np.random.default_rng(0)
    found = []
    for j in world.observed:
        # require the variable to TRACK two distinct out-of-range setpoints. A genuine
        # actuator holds at both +probe and -probe; a sensor that merely sits near one
        # value (its own dynamics pin it) fails the other -> no false actuators.
        held_all = True
        for v in (probe, -probe):
            env = world.clone(rng)
            env.reset()
            ok = True
            for _ in range(steps):
                x = env.step({j: v})
                if abs(x[j] - v) > tol:
                    ok = False
                    break
            if not ok:
                held_all = False
                break
        if held_all:
            found.append(j)
    return tuple(found)


def make_experimenter(kind, model, actuators, horizon=2, epsilon=0.15, rng=None):
    cls = _EXPERIMENTERS[kind]
    if cls is EpistemicExperimenter:
        return cls(model, actuators, horizon=horizon, epsilon=epsilon, rng=rng)
    if cls is ThompsonExperimenter:
        return cls(model, actuators, horizon=horizon, rng=rng)   # no epsilon: Thompson explores
    if cls in (NaiveSurpriseExperimenter,):
        return cls(model, actuators, epsilon=epsilon, rng=rng)
    if cls is RandomExperimenter:
        return cls(model, actuators, rng=rng)
    return cls()                              # passive takes no config


class ConstructorCausalAgent:
    def __init__(self, world, seed: int = 0, horizon: int = 2, epsilon: float = 0.15,
                 interaction_pairs=(), experimenter="epistemic", rff: int = 0,
                 rff_scale: float = 1.0, forget: float = 1.0, actuators=None,
                 prior=None, embodied: bool = False):
        self.world = world
        self.rng = np.random.default_rng(seed)
        # the agent acts through the actuators it BELIEVES it has. Pass actuators=
        # discover_actuators(world) to make it discover them rather than be told
        # (the world still physically clamps only its true actuators).
        self.actuators = tuple(world.actuators if actuators is None else actuators)
        self.model = BayesianDynamicsModel(
            world.d, self.actuators, hidden=world.hidden,
            interaction_pairs=interaction_pairs, rff=rff, rff_scale=rff_scale,
            forget=forget, rng=np.random.default_rng(seed + 3))
        self._exp_kind, self._horizon, self._epsilon = experimenter, horizon, epsilon
        self.experimenter = make_experimenter(
            experimenter, self.model, self.actuators,
            horizon=horizon, epsilon=epsilon, rng=np.random.default_rng(seed + 1))
        # ---- knowledge-prior warm-start (causal_dag CausalPrior, duck-typed) ----
        # A prior concentrates the experiment budget on actuators feeding sensors the
        # prior is UNSURE about — confident-prior edges need not be re-tested (the
        # prior asserts them; interventions can still override). An empty/abstaining
        # prior -> fall back to all actuators, so a useless prior is harmless.
        self.prior = prior
        if prior is not None:
            focus = tuple(a for a in self.actuators
                          if a in set(prior.actuators_feeding_unsure_sensors()))
            if focus:
                self.experimenter = make_experimenter(
                    experimenter, self.model, focus, horizon=horizon,
                    epsilon=epsilon, rng=np.random.default_rng(seed + 1))
        # seeded clone stream: verification runs in fresh world copies, but each copy's
        # noise comes from a SEEDED master rng, so verification/planning is REPRODUCIBLE
        # at a fixed agent seed (independent noise streams, deterministic across runs).
        self._clone_rng = np.random.default_rng(seed + 4)
        self.synth = ConstructorSynthesizer(
            self.model,
            world_factory=lambda: world.clone(
                np.random.default_rng(int(self._clone_rng.integers(1 << 31)))),
            actuators=self.actuators, sensors=self.model.sensors, d=world.d,
            rng=np.random.default_rng(seed + 2),
            embodied=embodied, live_env=(world if embodied else None))
        # EMBODIED = one ongoing life: reset the world ONCE (birth), then NEVER again --
        # explore, verify, and the continual loop all run on this single trajectory, with
        # no clone() and no further reset(). The clone-based default is unchanged.
        self.embodied = bool(embodied)
        if self.embodied:
            world.reset()
        self.library = Library()
        self.idle_knobs: list = []
        self.conditionals: list = []
        self.history: list = []                    # (step, f1) recovery curve
        self.buffer: list = []                     # (x_clamped, x_next) transitions

    # ---- reward-free learning ----------------------------------------------
    def explore(self, n_steps: int, track_every: int = 0):
        """Curiosity loop. At each step: clamp the chosen command into the current
        state, observe the transition, update the causal belief. No reward."""
        # embodied: CONTINUE the one ongoing trajectory (no reset); else start fresh.
        x = (self.world.x.copy() if self.embodied and self.world.x is not None
             else self.world.reset())
        for t in range(n_steps):
            cmd = self.experimenter.choose(x)
            x_clamped = x.copy()
            for j, v in cmd.items():
                x_clamped[j] = v
            x_next = self.world.step(cmd)
            self.model.update(x_clamped, x_next)
            self.buffer.append((x_clamped.copy(), x_next.copy()))
            x = x_next
            if track_every and (t + 1) % track_every == 0:
                sc = edge_scores(self.model, self.world.true_edges())
                self.history.append((t + 1, sc["f1"]))
        return self

    def explore_continuous(self, n_steps: int, low: float = -2.0, high: float = 2.0):
        """Reward-free exploration with CONTINUOUS random knob values. A smooth
        nonlinearity (e.g. a saturating tanh edge) is only learnable if the agent
        sees the knob at intermediate values, not just a few discrete setpoints."""
        x = self.world.reset()
        for _ in range(n_steps):
            cmd = {j: float(self.rng.uniform(low, high)) for j in self.world.actuators}
            x_clamped = x.copy()
            for j, v in cmd.items():
                x_clamped[j] = v
            x_next = self.world.step(cmd)
            self.model.update(x_clamped, x_next)
            self.buffer.append((x_clamped.copy(), x_next.copy()))
            x = x_next
        return self

    # ---- localized re-exploration (cheaper recovery after a localized change) --
    def relevant_actuators(self, i: int, z: float = 3.0, eps: float = 0.05):
        """Actuators that are believed ANCESTORS of sensor i in the current model
        (backward reachability over model.recovered_edges). Falls back to ALL
        actuators if none are found, so recovery is never starved (e.g. when the
        change is upstream/hidden or a brand-new edge the model doesn't yet believe)."""
        edges = self.model.recovered_edges(z=z, eps=eps)         # set of (src j -> tgt t)
        children = {}
        for (j, t) in edges:
            children.setdefault(j, set()).add(t)
        # backward BFS from i: collect all ancestors
        anc, stack = set(), [i]
        while stack:
            t = stack.pop()
            for (j, tt) in edges:
                if tt == t and j not in anc:
                    anc.add(j); stack.append(j)
        acts = tuple(a for a in self.actuators if a in anc)
        return acts if acts else tuple(self.actuators)

    def explore_localized(self, n_steps: int, focus_sensor: int, **kw):
        """Re-explore using a TEMPORARY experimenter restricted to the actuators
        believed relevant to focus_sensor. The model still updates ALL sensors from
        each transition; only the intervention SEARCH SPACE shrinks (from
        |setpoints|^|all_actuators| to |setpoints|^|relevant|), which is the cost
        driver. Restores the full-actuator experimenter afterwards."""
        acts = self.relevant_actuators(focus_sensor)
        saved = self.experimenter
        self.experimenter = make_experimenter(
            self._exp_kind, self.model, acts, horizon=self._horizon,
            epsilon=self._epsilon, rng=np.random.default_rng())
        try:
            self.explore(n_steps, **kw)
        finally:
            self.experimenter = saved
        return acts

    def practice(self, setpoints=(-2.0, 0.0, 2.0), hold: int = 8, rounds: int = 40):
        """Rehearse SUSTAINED single-knob holds in the ONE ongoing life (no clone, no
        reset), recording transitions into the buffer and updating the belief.

        Why this exists: the reset-free certificate (certify.certify_library) reads a
        skill's reliability off the agent's OWN stream, but a SLOW/deep variable only
        builds up to its hold value under a sustained command — an i.i.d.-random stream
        of pokes never visits the skill's effect region, so its reward indicator is
        identically zero and the skill looks (falsely) un-achievable. Rehearsal is how
        an embodied agent generates the on-policy coverage its certificate needs: it
        actually performs each hold long enough for downstream variables to settle.
        Returns self."""
        x = (self.world.x.copy() if getattr(self.world, "x", None) is not None
             else self.world.reset())
        for r in range(rounds):
            for a in self.actuators:
                v = float(setpoints[r % len(setpoints)])
                for _ in range(hold):
                    xc = x.copy()
                    xc[a] = v
                    xn = self.world.step({a: v})
                    self.model.update(xc, xn)
                    self.buffer.append((xc.copy(), xn.copy()))
                    x = xn
        return self

    # ---- overlap-driven curiosity: act to make UNDECIDED skills certifiable ----
    def coverage_targets(self, min_onpolicy: int = 50, action_tol: float = 0.4,
                         since: int | None = None, priorities=None):
        """Under-COVERED skills: holds whose reset-free model-free certificate would be
        UNDECIDED purely for lack of on-policy coverage (the buffer rarely holds their
        command). Returns [(command, deficit)] with deficit = min_onpolicy - n_onpolicy > 0
        — the target set for overlap-driven curiosity: visiting the command resolves the
        open certificate. Only single-knob HOLD skills have an action-coverage wall;
        composite skills are screened model-based, so they are not coverage-targeted.

        ``since``: if given, count only transitions from buffer index ``since`` onward —
        i.e. FRESH coverage gathered in the current recovery phase. This is what makes
        drift-reopened skills re-coverable: after the world changes, the pre-drift buffer
        still 'covers' a command by count (so a whole-buffer count would see no deficit),
        but that coverage is STALE; counting only post-``since`` transitions forces the
        agent to re-cover each skill with NEW-regime data.

        ``priorities``: optional {sensor: weight} that SCALES a skill's deficit by
        (1 + max weight over its effect variables). Drift recovery passes the standardised
        per-sensor surprise here, so the skills whose EFFECT region the drift actually
        disturbed (the genuinely reopened ones) are covered FIRST — the agent re-explores
        toward what broke, not uniformly across every skill."""
        buf = self.buffer if since is None else self.buffer[since:]
        targets = []
        for c in self.library.possible():
            prog = tuple(c.program)
            first = prog[0] if prog else {}
            is_hold = (len(first) == 1 and all(cmd == first for cmd in prog)
                       and next(iter(first)) in self.actuators)
            if not is_hold:
                continue
            n = sum(1 for (xc, _) in buf
                    if all(abs(xc[a] - v) <= action_tol for a, v in first.items()))
            if n < min_onpolicy:
                deficit = float(min_onpolicy - n)
                if priorities:
                    boost = max((float(priorities.get(v, 0.0)) for v in c.effect.vars()),
                                default=0.0)
                    deficit *= 1.0 + max(boost, 0.0)
                targets.append((dict(first), deficit))
        return targets

    def explore_to_certify(self, n_steps: int, min_onpolicy: int = 50,
                           action_tol: float = 0.4, refresh: int = 5, horizon: int = 2,
                           eig_weight: float = 1.0, gamma: float = 0.95, tau: float = 0.9,
                           fresh: bool = False, report: bool = True, priorities=None):
        """OVERLAP-DRIVEN CURIOSITY (reward-free). Extend active inference's epistemic drive
        from 'reduce my parameter uncertainty' to 'resolve what I can DO': run a
        ``CertifyingExperimenter`` whose targets are refreshed every ``refresh`` steps from
        ``coverage_targets()``. It holds an under-covered skill's command until that skill
        has coverage (sustained holds emerge because the deficit persists), then moves to the
        next, and reduces to pure EIG once every skill is covered. Continues the ONE ongoing
        life (no reset for an embodied agent; otherwise continues from the current world
        state, accumulating coverage in the same trajectory) and updates the belief from
        every transition. No reward is defined — the only imperative is to make the
        constructor library's POSSIBLE verdict decidable.

        ``fresh``: count coverage only from this call's start (``since=len(buffer)``), so
        EVERY current skill is re-covered with new data — the drift-recovery mode used by
        ``live_round`` (a sign-flip leaves old coverage intact by count but stale in fact).
        ``report``: when True, bracket the run with ``certify_library`` and return
        {before, after}; when False (cheap path for the continual loop) skip it and return
        {covered}: the number of targeted skills that reached full FRESH coverage."""
        from .certify import certify_library
        since = len(self.buffer) if fresh else None
        before = (certify_library(self, gamma=gamma, tau=tau, action_tol=action_tol,
                                  min_onpolicy=min_onpolicy, rng=self.rng) if report else None)
        n_targets0 = len(self.coverage_targets(min_onpolicy=min_onpolicy,
                                               action_tol=action_tol, since=since,
                                               priorities=priorities))
        exp = CertifyingExperimenter(self.model, self.actuators, horizon=horizon,
                                     eig_weight=eig_weight, action_tol=action_tol,
                                     rng=self.rng)
        x = (self.world.x.copy() if getattr(self.world, "x", None) is not None
             else self.world.reset())
        for t in range(n_steps):
            if t % refresh == 0:
                exp.targets = self.coverage_targets(min_onpolicy=min_onpolicy,
                                                    action_tol=action_tol, since=since,
                                                  priorities=priorities)
            cmd = exp.choose(x)
            x_clamped = x.copy()
            for j, v in cmd.items():
                x_clamped[j] = v
            x_next = self.world.step(cmd)
            self.model.update(x_clamped, x_next)
            self.buffer.append((x_clamped.copy(), x_next.copy()))
            x = x_next
        if not report:
            remaining = len(self.coverage_targets(min_onpolicy=min_onpolicy,
                                                  action_tol=action_tol, since=since,
                                                  priorities=priorities))
            return {"covered": n_targets0 - remaining, "targets": n_targets0}
        after = certify_library(self, gamma=gamma, tau=tau, action_tol=action_tol,
                                min_onpolicy=min_onpolicy, rng=self.rng)
        return {"before": before, "after": after}

    def detect_hidden(self, sigma_floor: float = 0.5, min_selfloop: float = 0.2):
        """Tell a HIDDEN CAUSE from genuine noise.

        A slow unobserved common cause that feeds a variable with memory gets
        absorbed into an inflated SELF-LOOP, which whitens the residual — so naive
        residual-autocorrelation does NOT find it (verified). Its fingerprint is
        instead a variable that is BOTH hard to predict (large residual variance)
        AND strongly autoregressive (large self-loop): the self-loop tracks the slow
        driver, the leftover variance is the part it cannot explain. Pure noise is
        unpredictable but has NO self-loop (so it is correctly NOT flagged); a
        well-modelled variable has small residual variance. Returns
        [(sensor, sigma^2, self-loop)] for the flagged sensors.

        The variance+self-loop test alone false-positives on a GENUINE high-variance
        self-loop (no hidden cause), so we CONFIRM each candidate with the sharper
        second-lag residual test (certify.detect_latent_lag): a real self-loop whitens
        the residual at all lags, while a slow hidden driver leaves a second timescale
        the lagged value still predicts. A sensor is flagged only if BOTH agree.

        Honest caveat: this is still a heuristic for a genuinely hard identifiability
        problem — a hidden AR(1) cause and a real self-loop are not separable from
        one-step observational data alone; certifying it needs interventions on the
        affected variable, which are not available when it is only a sensor."""
        from .certify import detect_latent_lag
        lag_confirmed = {i for (i, _, _) in detect_latent_lag(self)}
        flagged = []
        for i in self.model.sensors:
            selfloop = abs(self.model.weight(i, i))
            if (self.model.sigma2[i] > sigma_floor and selfloop > min_selfloop
                    and i in lag_confirmed):
                flagged.append((i, float(self.model.sigma2[i]), float(self.model.weight(i, i))))
        return flagged

    # ---- discover interaction structure (no candidate pairs supplied) ------
    def discover_interactions(self, z: float = 4.0, eps: float = 0.05, window=None):
        """Find multiplicative 'gates' the agent was NOT told about.

        Fit the current (linear) model, then for every pair of observed variables
        test whether the product x_a·x_b explains the LEFTOVER structure in some
        sensor's residuals (a 1-D regression on the residual; t-stat > z). Any
        product that does is a hidden interaction. The model is then rebuilt WITH
        the discovered product features and refit on the buffered transitions, so
        the gate is in the belief from here on. ``window`` restricts the scan to
        the most recent transitions (use it after the world changes, so a gate that
        just appeared is found on current data, not averaged out by old regimes).
        Returns the discovered pairs."""
        if not self.buffer:
            return []
        buf = self.buffer if window is None else self.buffer[-window:]
        cols = self.model.cols
        Xc = np.array([xc for (xc, _) in buf])
        Xn = np.array([xn for (_, xn) in buf])
        pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i:]]   # incl a==b (squares)
        found = set()
        for i in self.model.sensors:
            mean = self.model._mean(i)
            phi = np.array([self.model._phi(x) for x in Xc])
            resid = Xn[:, i] - phi @ mean
            for (a, b) in pairs:
                p = Xc[:, a] * Xc[:, b]
                p = p - p.mean()
                denom = float(p @ p)
                if denom < 1e-9:
                    continue
                slope = float(p @ resid) / denom
                r2 = resid - slope * p
                # noise dof: subtract the p linear params that already shaped this
                # residual PLUS the one product slope -> n - p - 1. (n - 2 is biased
                # LOW, inflating the gate statistic; verified by Monte Carlo. The
                # centered product is orthogonal to the intercept, so n-p-1, not n-p-2.)
                dof = max(len(r2) - self.model.p - 1, 1)
                se = float(np.sqrt(max(r2 @ r2, 1e-12) / dof / denom))
                crit = _t_crit(z, dof)   # t-test on n-p-1 dof, matching recovered_edges
                if abs(slope) > eps and abs(slope) / max(se, 1e-12) > crit:
                    found.add((min(a, b), max(a, b)))
        found_pairs = sorted(found)
        if found_pairs:
            # rebuild WITH the discovered products, carrying the original config
            # (rff/rff_scale/sigma0/a0) and REUSING the existing RFF basis, so the refit
            # never silently drops nonlinear capacity or corrupts the noise prior.
            new = BayesianDynamicsModel(
                **self.model._initkw, interaction_pairs=found_pairs,
                rff_W=getattr(self.model, "rff_W", None),
                rff_b=getattr(self.model, "rff_b", None))
            # vectorized one-pass refit (S = Phi^T W Phi) -- exactly reproduces replaying
            # update() over the buffer, but O(n p^2) as a single matmul, not n Python steps.
            new.fit_batch(Xc, Xn)
            self.model = new
            self.experimenter.model = new
            self.synth.model = new
        return found_pairs

    # ---- distil constructors -----------------------------------------------
    def build_library(self, setpoints=(-2.0, 2.0), conditional=False):
        good, idle = self.synth.mint_primitives(setpoints=setpoints)
        self.idle_knobs = idle
        for c in good:
            if c.possible:
                self.library.add(c)
        if conditional:
            self.conditionals = self.synth.mint_conditional_primitives(
                self.library, setpoints=setpoints)
        return self.library

    def consolidate(self, n: int = 40):
        """Re-verify every constructor against the CURRENT world and prune the ones
        that no longer work (a skill built for a past regime). This is the agent's
        wake/sleep step: it keeps the library honest as the world drifts. Returns
        the pruned constructors."""
        # reverify sets n_trials=n, and `possible` requires n_trials>=MIN_TRIALS, so a
        # small n would prune every still-working skill purely on trial count. Clamp it.
        n = max(int(n), MIN_TRIALS)
        for c in self.library.constructors:
            self.synth.reverify(c, n=n)
        return self.library.keep(lambda c: c.possible)

    def live_round(self, steps: int = 130, z_change: float = 4.0,
                   setpoints=(-2.0, 2.0), rediscover: bool = False, window: int = 350,
                   certify_seek: bool = False, certify_min: int = 80, tau: float = 0.9):
        """One round of an AUTONOMOUS continual loop. The agent measures its own
        surprise — STANDARDISED by each sensor's learned noise, so an irreducibly
        noisy channel doesn't masquerade as a change — and if any sensor's error
        exceeds ``z_change`` sigmas it decides the world has moved and re-learns:
        explore, optionally re-run interaction discovery on recent data (structural
        drift), consolidate (prune broken skills), and rebuild. Nothing external
        tells it when. It also (re)builds if it has no skills yet. Returns a report.

        ``certify_seek``: when drift reopens the certificates of the CURRENT skills, spend
        the re-exploration budget RESOLVING them — `explore_to_certify(fresh=True)` re-covers
        each current skill's command with NEW-regime data (targeted recovery) instead of
        exploring blindly — then consolidate/rebuild as usual. With no library yet (first
        round) or on a stable round this is identical to the blind path, so the default
        (certify_seek=False) behaviour is unchanged."""
        pre = self.surprise()
        std = {i: pre[i] / (self.model.sigma2[i] ** 0.5 + 1e-9) for i in self.model.sensors}
        worst = max(std.values()) if std else 0.0
        changed = worst > z_change
        relearn = changed or not self.library.possible()
        covered = 0
        if certify_seek and relearn and self.library.possible():
            # re-explore toward what the drift DISTURBED: the standardised surprise prioritises
            # the skills whose effect region just spiked (the genuinely reopened certificates).
            rep = self.explore_to_certify(steps, min_onpolicy=certify_min, tau=tau,
                                          fresh=True, report=False, priorities=std)
            covered = rep["covered"]
        else:
            self.explore(steps)
        pruned: list = []
        if relearn:
            if rediscover:
                self.discover_interactions(window=window)
            pruned = self.consolidate()
            self.build_library(setpoints=setpoints, conditional=rediscover)
        out = {"changed": changed, "z_surprise": float(worst),
               "pruned": len(pruned), "skills": len(self.library.possible())}
        if certify_seek:
            out["recovered"] = covered
        return out

    def surprise(self, n_probe: int = 12):
        """Mean one-step prediction error per sensor under the CURRENT belief, on
        fresh interventions. A spike after the world changes is the change signal."""
        errs = {i: 0.0 for i in self.model.sensors}
        x = (self.world.x.copy() if self.embodied and self.world.x is not None
             else self.world.reset())
        for _ in range(n_probe):
            cmd = self.experimenter.choose(x)
            xc = x.copy()
            for j, v in cmd.items():
                xc[j] = v
            mu, _ = self.model.predict_next(xc, cmd)
            x = self.world.step(cmd)
            for i in self.model.sensors:
                errs[i] += abs(mu[i] - x[i]) / n_probe
        return errs

    # ---- use what was learned (a goal appears only now) --------------------
    def achieve(self, target: Box, start=None, search="bfs"):
        """Compose the library into a constructor that reaches ``target``.
        ``search`` is "bfs" (uninformed) or "greedy" (informed best-first)."""
        return self.synth.reach(self.library, target, start=start, search=search)

    def whatif(self, state, command, steps: int = 1):
        """Predict the trajectory of forcing ``command`` for ``steps`` ticks,
        from the learned model alone (no peeking at the world)."""
        x = np.asarray(state, float).copy()
        traj = [x.copy()]
        for _ in range(steps):
            x, _ = self.model.predict_next(x, command)
            traj.append(x.copy())
        return np.array(traj)

    # ---- the recovered cause-effect map ------------------------------------
    def recovered_dag(self):
        return edge_scores(self.model, self.world.true_edges())

    def named_edges(self, edges):
        return sorted(f"{self.world.names[j]}→{self.world.names[i]}" for (j, i) in edges)

    def dag_marks(self, z: float = 3.0, eps: float = 0.05):
        """Honest causal map: DIRECTED (do-identified) edges + BIDIRECTED
        (possibly-confounded) marks for associations the agent cannot orient by
        intervention. See model.recovered_marks."""
        return self.model.recovered_marks(z=z, eps=eps)

    def named_marks(self, marks=None):
        m = marks if marks is not None else self.dag_marks()
        nm = self.world.names
        return {"directed": sorted(f"{nm[j]}→{nm[i]}" for (j, i) in m["directed"]),
                "bidirected": sorted(f"{nm[a]}↔{nm[b]}" for (a, b) in m["bidirected"])}


__all__ = ["ConstructorCausalAgent"]
