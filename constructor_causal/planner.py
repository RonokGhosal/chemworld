"""
From understanding to capability: synthesising and composing constructors.

The agent learns the causal model with NO reward (active_inference.py). This module
is how that understanding is cashed out as *constructors* -- verified, repeatable
transformations -- and how small ones are grown into big ones.

Three jobs:

  1. mint_primitives  -- for each knob/setpoint that the learned model says *does
     something*, build the primitive constructor "hold this knob there for a few
     steps", DISCOVER its effect region by running it, and verify it really is
     reliable. Knobs that move nothing (e.g. an inert actuator) are reported as
     causally idle -- the agent finds out which levers matter.

  2. characterize_effect / verify -- a constructor's effect box is *measured*, not
     declared: run the program many times and take the region the substrate
     reliably lands in. Reliability against any target box is likewise measured by
     actually running the program in the world (a legitimate repeated experiment).

  3. reach(target) -- model-based planning that turns a goal into a program by
     COMPOSING library constructors (here, chaining a primitive with itself via the
     compose() algebra). A deep, slow variable that no single primitive can move is
     reached by the composite -- and the composite's *new* capability is found by
     re-characterising it. Composition manufactures abilities its parts lacked.

Note the honest separation: LEARNING is reward-free and goal-free. A *target* only
appears at planning time, and it is a target to reach, not a reward to maximise --
the model that makes reaching it possible was already built by curiosity alone.
"""
from __future__ import annotations

import numpy as np

from .constructor import Box, Constructor, compose, prog_key, POSSIBLE_TAU


def default_init_sampler(sensors, scale=0.3):
    """Sample a 'resting' start state: knobs at 0, sensors jittered near 0."""
    def sampler(rng, d):
        x = np.zeros(d)
        for i in sensors:
            x[i] = rng.normal(0.0, scale)
        return x
    return sampler


class ConstructorSynthesizer:
    def __init__(self, model, world_factory, actuators, sensors, d,
                 h_prim: int = 3, max_compose: int = 4, rng=None,
                 embodied: bool = False, live_env=None):
        self.model = model
        self.world_factory = world_factory      # () -> fresh world, same params
        # EMBODIED single-trajectory mode: measure/verify skills in ONE ongoing env
        # (never cloned, never reset) instead of in fresh parallel copies. live_env is
        # that one world; the cost is real time spent performing the skill, not free
        # parallel rollouts -- the honest price an embodied agent actually pays.
        self.embodied = bool(embodied)
        self.live_env = live_env
        self.actuators = tuple(actuators)
        self.sensors = tuple(sensors)
        self.d = int(d)
        self.h_prim = int(h_prim)
        self.max_compose = int(max_compose)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._init = default_init_sampler(self.sensors)
        # learned heuristic memory: (model-distance-to-target, true remaining #constructors)
        # harvested from every solved plan. A k-NN over it calibrates "how far is this state,
        # really, in STEPS" -- replacing greedy's raw state-unit distance with a learned,
        # step-consistent estimate, so best-first (A*) expands fewer nodes as experience grows.
        self._h_data: list = []

    # ---- run a program in fresh real-world instances ------------------------
    def _finals(self, program, n, init=True, prefix=None):
        """Final states of running ``program`` n times. If ``prefix`` is given it is
        run first in the SAME env (commands persist), so the program is measured
        from the state -- and held knobs -- the prefix establishes. This is how a
        context-dependent (conditional) constructor is characterized and verified."""
        if self.embodied:
            return self._finals_live(program, n, prefix=prefix)
        finals = []
        for _ in range(n):
            env = self.world_factory()
            x0 = self._init(self.rng, self.d) if init else None
            x = env.reset(x0)
            if prefix:
                for cmd in prefix:
                    x = env.step(cmd, noise=True)
            for cmd in program:
                x = env.step(cmd, noise=True)
            finals.append(x.copy())
        return np.array(finals)

    def _finals_live(self, program, n, prefix=None, drift_steps: int = 2):
        """SINGLE-TRAJECTORY finals (embodied): run ``program`` n times in the ONE
        ongoing env -- never cloned, never reset -- with a few random drift steps
        between runs to vary the start. The substrate carries over between runs, so a
        skill is measured exactly as the agent would actually perform it in one life."""
        env = self.live_env
        setpts = (-2.0, 0.0, 2.0)
        finals = []
        for _ in range(n):
            for _ in range(drift_steps):                 # vary the start, same trajectory
                env.step({j: float(self.rng.choice(setpts)) for j in self.actuators})
            if prefix:
                for cmd in prefix:
                    env.step(cmd, noise=True)
            x = None
            for cmd in program:
                x = env.step(cmd, noise=True)
            finals.append(x.copy())
        return np.array(finals)

    def certify(self, c: Constructor, target: Box, tau: float = POSSIBLE_TAU,
                alpha: float = 0.05, max_trials: int = 200, prefix=None):
        """Anytime-valid CLONING-FREE certificate that ``c`` reaches ``target``, run in
        the ONE ongoing live env (no clone, no reset). Accumulates a betting confidence
        sequence and stops as soon as it decides POSSIBLE/IMPOSSIBLE -- the operational
        form of Constructor Theory's 'possible iff performable to arbitrarily high
        accuracy'. Writes the measured reliability/n onto ``c`` and returns the report."""
        from .certify import certify_reliability
        prog = (tuple(prefix) + tuple(c.program)) if prefix else c.program
        res = certify_reliability(self.live_env, prog, target, tau=tau, alpha=alpha,
                                  max_trials=max_trials, reset=False, rng=self.rng)
        c.reliability, c.n_trials = res["p_hat"], res["n"]
        c.cs_verdict = res["verdict"]
        return res

    # ---- measure the region a program reliably reaches ----------------------
    def characterize_effect(self, program, n=60, max_std=0.5, min_move=0.3, prefix=None,
                            settle_tol=0.25, n_settle=30):
        """Effect box = a generous band around each sensor the program reliably AND
        STABLY controls.

        A sensor counts as controlled if, across trials, its final value is consistent
        (low std), meaningfully displaced from rest (|mean|>min_move), AND SETTLED by the
        program's horizon -- holding the program's last command ONE more step barely moves
        its mean (|Δmean| <= settle_tol). The settling test EXCLUDES slow downstream
        variables still IN FLIGHT at the horizon (e.g. a deep chain that hasn't converged):
        a short primitive therefore does NOT claim a deep/slow variable in its box, so
        reaching that variable genuinely requires COMPOSITION -- and, crucially, the box now
        matches what a SUSTAINED hold actually maintains, so the reset-free reach certificate
        (certify.certify_modelfree_reach) agrees with it instead of seeing the slow variable
        overshoot its transient from-rest value. Its box is mean ± max(4·std, 0.2). High-
        variance sensors (e.g. ``static``) fail the consistency test and are excluded.
        ``settle_tol=None`` disables the filter (legacy: include still-transient values)."""
        finals = self._finals(program, n, prefix=prefix)
        ext = (self._finals(tuple(program) + (program[-1],), n_settle, prefix=prefix)
               if settle_tol is not None and len(program) > 0 else None)
        bounds = {}
        for i in self.sensors:
            col = finals[:, i]
            mu, sd = float(np.mean(col)), float(np.std(col))
            if sd <= max_std and abs(mu) > min_move:
                if ext is not None and abs(float(np.mean(ext[:, i])) - mu) > settle_tol:
                    continue                       # still in flight at horizon -> not stable
                w = max(4.0 * sd, 0.2)
                bounds[i] = (mu - w, mu + w)
        return Box.from_dict(bounds)

    def verify(self, c: Constructor, target: Box, n=60, prefix=None) -> float:
        """Reliability of c against ``target``, by running it in the world (after
        an optional ``prefix`` that establishes c's precondition)."""
        finals = self._finals(c.program, n, prefix=prefix)
        hits = int(sum(target.contains(x) for x in finals))
        c.reliability = hits / n
        c.n_trials = n
        return c.reliability

    def reverify(self, c: Constructor, n=40) -> float:
        """Re-measure whether a constructor STILL works in the CURRENT world, by
        running its from-rest program and checking it lands in its own effect box.
        Used by consolidation: a skill built for an old regime fails this after the
        world changes, and gets pruned. (full_program is from rest for any skill.)"""
        finals = self._finals(c.full_program, n, init=True)
        hits = int(sum(c.effect.contains(x) for x in finals))
        c.reliability = hits / n
        c.n_trials = n
        return c.reliability

    # ---- 1. primitives ------------------------------------------------------
    def mint_primitives(self, setpoints=(-2.0, 2.0)):
        """Returns (good, idle): verified primitive constructors, and a report of
        knobs/setpoints that turned out causally idle."""
        good, idle = [], []
        for j in self.actuators:
            moved_any = False
            for v in setpoints:
                program = tuple({j: float(v)} for _ in range(self.h_prim))
                effect = self.characterize_effect(program)
                if not effect.bounds:                 # this knob moved nothing
                    idle.append((j, v))
                    continue
                moved_any = True
                name = f"hold_x{j}={v:+.0f}×{self.h_prim}"
                c = Constructor(name=name, precond=Box.any(), effect=effect,
                                program=program, provenance="primitive")
                c.controls = set(effect.vars())
                self.verify(c, effect)
                good.append(c)
            if not moved_any:
                pass
        return good, idle

    # ---- 2b. conditional (context-dependent) primitives --------------------
    def mint_conditional_primitives(self, library, setpoints=(-2.0, 2.0), max_rounds=4):
        """Discover constructors that only work AFTER another one has run -- i.e.
        new transformations that become possible once you already have a skill --
        and ITERATE, so skills stack into arbitrarily deep chains.

        For each library constructor ``prereq`` and each knob, run the knob while
        ``prereq``'s from-rest program is held first (commands persist, its effect
        holds). If that controls a NEW variable the chain couldn't reach, mint a
        conditional constructor with precond = prereq.effect and full_program =
        prereq.full_program ++ this hold. Repeating rounds conditions on the new
        conditionals too: a0 opens gate1, then (a1 | gate1) opens gate2, then
        (a2 | gate2) drives Z -- a three-deep ladder, discovered bottom-up. Returns
        all new conditional constructors."""
        for c in library.constructors:                 # ensure controls set
            if not hasattr(c, "controls"):
                c.controls = set(c.effect.vars())
        new = []
        for _ in range(max_rounds):
            added = []
            for prereq in list(library.possible()):
                already = getattr(prereq, "controls", set(prereq.effect.vars()))
                for j in self.actuators:
                    for v in setpoints:
                        prog = tuple({j: float(v)} for _ in range(self.h_prim))
                        eff_full = self.characterize_effect(prog, prefix=prereq.full_program)
                        fresh = set(eff_full.vars()) - already
                        if not fresh:
                            continue
                        eff = Box.from_dict({var: (lo, hi) for (var, lo, hi)
                                             in eff_full.bounds if var in fresh})
                        cond = Constructor(
                            name=f"hold_x{j}={v:+.0f}×{self.h_prim}|{prereq.name}",
                            precond=prereq.effect, effect=eff, program=prog,
                            provenance=f"conditional({prereq.name})",
                            full_program=tuple(prereq.full_program) + prog)
                        self.verify(cond, eff, prefix=prereq.full_program)
                        cond.controls = set(already) | set(eff.vars())
                        if cond.possible and library.add(cond):
                            added.append(cond)
            new += added
            if not added:
                break
        return new

    # ---- helpers for planning ----------------------------------------------
    def _model_final(self, program, start):
        x = np.asarray(start, float).copy()
        for cmd in program:
            x, _ = self.model.predict_next(x, cmd)
        return x

    @staticmethod
    def _box_distance(x, target: Box) -> float:
        """L1 distance from x to the target box (0 inside) -- the raw (greedy) heuristic."""
        d = 0.0
        for (v, lo, hi) in target.bounds:
            if x[v] < lo:
                d += lo - x[v]
            elif x[v] > hi:
                d += x[v] - hi
        return float(d)

    def _h_learned(self, x, target: Box, min_pts: int = 4) -> float:
        """LEARNED heuristic: an ADMISSIBLE estimate of remaining #constructors to target.

        Calibrates remaining ~ a * distance from harvested (model-distance, true-remaining)
        pairs, using the LOWER-ENVELOPE slope a = min_i(remaining_i / distance_i) -- the
        steepest line through the origin that stays BELOW every harvested point -- so a*d
        never OVER-estimates the observed remaining and the heuristic stays admissible (A*
        remains optimal). Least-squares-through-origin (the previous calibration) fits the
        cloud centre and over-estimates ~a third of states, which is NOT admissible. Falls
        back to the raw model distance until a few points are harvested."""
        d = self._box_distance(x, target)
        if len(self._h_data) < min_pts:
            return d
        ds = np.array([dd for dd, _ in self._h_data])
        rs = np.array([rr for _, rr in self._h_data])
        a = float(np.min(rs / np.maximum(ds, 1e-9)))   # lower-envelope slope (admissible)
        return a * d

    def _harvest(self, program, target, start):
        """Record (distance, remaining #constructors) at each block of a SOLVED plan -> _h_data."""
        x = np.asarray(start, float).copy()
        nblocks = max(1, len(program) // self.h_prim)
        for bi in range(nblocks):
            self._h_data.append((self._box_distance(x, target), nblocks - bi))
            for cmd in program[bi * self.h_prim:(bi + 1) * self.h_prim]:
                x, _ = self.model.predict_next(x, cmd)

    # ---- 2c. continuous control: solve for a setpoint that hits a target ----
    def solve_setpoint(self, target: Box, start=None, horizon=5, n_grid=61,
                       lo=-3.0, hi=3.0, n_verify=60):
        """Find a CONTINUOUS knob value (not just the ±2 grid) that drives the
        substrate into ``target``. For each actuator, sweep candidate setpoints,
        predict the outcome with the learned model, and pick the one the model lands
        most deeply INSIDE the target (max margin from the box edges, so the verified
        reliability is robust, not edge-of-box marginal). Verify it in the world.
        Lets the agent hit a narrow target the coarse primitive library overshoots."""
        start = np.zeros(self.d) if start is None else np.asarray(start, float)

        def margin(x):  # how far inside the box (min distance to any edge); <0 = outside
            m = np.inf
            for (v, a, b) in target.bounds:
                m = min(m, x[v] - a, b - x[v])
            return m

        best, best_margin = None, 0.0
        for j in self.actuators:
            cand_j, cm = None, 0.0
            for v in np.linspace(lo, hi, n_grid):
                program = tuple({j: float(v)} for _ in range(horizon))
                x = start.copy()
                for cmd in program:
                    x, _ = self.model.predict_next(x, cmd)
                mg = margin(x)
                if mg > cm:                          # deepest-inside setpoint for this knob
                    cm, cand_j = mg, (float(v), program)
            if cand_j is not None and cm > best_margin:
                best_margin, best = cm, (j, cand_j)
        if best is None:
            return None
        j, (v, program) = best
        c = Constructor(name=f"hold_x{j}={v:+.2f}×{horizon}", precond=Box.any(),
                        effect=target, program=program, provenance="solved-setpoint")
        self.verify(c, target, n=n_verify)
        return c if c.possible else None

    # ---- 3. reach a target by composing the library (BFS or informed) ------
    def reach(self, library, target: Box, start=None, n_verify=60, max_depth=5,
              node_cap=4000, search="bfs"):
        """Find/compose a constructor that reliably drives the substrate from
        ``start`` into ``target``, by searching over chains of library constructors.

        A chain c1 ≫ c2 ≫ ... is valid when each step's guaranteed effect satisfies
        the next step's precondition (compose()'s box algebra), and the first step is
        applicable at the start. One operator covers every case: a deep/slow variable
        (same primitive chained with itself), a gate (two distinct constructors), or a
        multi-gate cascade (three-plus). The learned model proposes chains; the world
        verifies them.

        ``search``:
          "bfs"    -- uninformed breadth-first; shortest chain first. Fine for small
                      libraries, but branches by the whole library at every node.
          "greedy" -- best-first by a model-based heuristic: the L1 distance from the
                      chain's predicted final state to the target box. Expands toward
                      the goal, so a few distractor-laden libraries don't blow it up.

        Sets ``self.last_nodes`` (nodes expanded). Returns (constructor, reliability)
        or (None, 0.0)."""
        import heapq
        from collections import deque

        start = np.zeros(self.d) if start is None else np.asarray(start, float)
        pool = library.possible()
        self.last_nodes = 0

        # already have a directly-applicable cached constructor? RE-VERIFY it against
        # the CURRENT world (a stored reliability can be STALE after drift/relearning)
        # and REUSE it on its re-measured POINT reliability >= tau. Reuse scans the
        # WHOLE library and uses the point estimate -- not `possible`'s strict
        # certification bound -- because reuse is re-verified live every time, so a
        # borderline-but-still-reliable solution to a hard goal must stay reusable
        # (certification rigor belongs to claims, not to operational caching).
        for c in library.constructors:
            if c.effect.subseteq(target) and c.precond.contains(start):
                rel = self.verify(c, target, n=n_verify)
                if rel >= POSSIBLE_TAU:
                    return c, rel

        seeds = [c for c in pool if c.precond.contains(start)]
        seen = {prog_key(c.program) for c in seeds}

        def try_goal(comp):
            x = self._model_final(comp.program, start)
            if not target.contains(x):
                return None
            cand = Constructor(name=comp.name, precond=comp.precond, effect=target,
                               program=comp.program, provenance=comp.provenance,
                               full_program=comp.full_program)
            self.verify(cand, target, n=n_verify)
            return cand if cand.possible else None

        def extensions(comp):
            for c in pool:
                nxt = compose(comp, c)
                if nxt is None:
                    continue
                k = prog_key(nxt.program)
                if k in seen:
                    continue
                seen.add(k)
                yield nxt

        if search in ("greedy", "learned"):
            learned = (search == "learned")

            def prio(comp):
                x = self._model_final(comp.program, start)
                if learned:                                   # A*: g (blocks so far) + learned h
                    g = comp.horizon / self.h_prim
                    return g + self._h_learned(x, target)
                return self._box_distance(x, target)          # greedy: raw model distance

            heap: list = []
            counter = 0
            for c in seeds:
                heapq.heappush(heap, (prio(c), counter, c)); counter += 1
            while heap and self.last_nodes < node_cap:
                _, _, comp = heapq.heappop(heap)
                self.last_nodes += 1
                got = try_goal(comp)
                if got is not None:
                    if learned:
                        self._harvest(got.program, target, start)
                    library.add(got); return got, got.reliability
                if comp.horizon >= max_depth * self.h_prim:
                    continue
                for nxt in extensions(comp):
                    heapq.heappush(heap, (prio(nxt), counter, nxt)); counter += 1
        else:  # bfs
            frontier = deque(seeds)
            while frontier and self.last_nodes < node_cap:
                comp = frontier.popleft()
                self.last_nodes += 1
                got = try_goal(comp)
                if got is not None:
                    library.add(got); return got, got.reliability
                if comp.horizon >= max_depth * self.h_prim:
                    continue
                frontier.extend(extensions(comp))

        # fallback: no library chain reached it — solve for a continuous setpoint
        solved = self.solve_setpoint(target, start=start, n_verify=n_verify)
        if solved is not None:
            library.add(solved)
            return solved, solved.reliability
        return None, 0.0


__all__ = ["ConstructorSynthesizer", "default_init_sampler"]
