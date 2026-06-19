# constructor_causal

**Learning the causal *algebra* of an unknown world, with no reward.**

This package fuses four ideas into one working agent:

| Idea | Role here |
|---|---|
| **Causal DAGs / SCMs** | The form of the world *and* of the agent's belief: who causes whom. |
| **Causal inference** | The agent **intervenes** (forces knobs), so it identifies cause vs. mere correlation — and rejects a decoy that correlation alone could not. |
| **Active inference** | Action selection minimises **expected free energy** — but with the *pragmatic / reward* term deleted. Only the **epistemic** term (expected information gain) remains. That is what "learning without reward" means here. |
| **Constructor Theory** | An intervention `do(x=v)` is a **constructor**: a repeatable transformation. A *program* of interventions is a **composite** constructor. Chaining small constructors builds bigger ones — an **algebra of what the agent can reliably do**. |

The thesis: **delete the reward term and an agent still has a complete drive** —
resolve uncertainty about the world's causal mechanism. Capability is not the
objective; it is a *by-product* of understanding. A goal is introduced only at the
very end, to *use* the library — never to train it.

## The loop

```
        ┌──────────────────────────── reward-free ────────────────────────────┐
  watch │  pick the intervention with highest expected info gain  (Epistemic)  │
        │            │                                                          │
        │            ▼                                                          │
        │   force a knob (a primitive constructor)  ──►  observe transition     │
        │            │                                                          │
        │            ▼                                                          │
        │   update the Bayesian causal DAG belief   ◄──────────────────────────┘
        └────────────┘
                     │   (later, still no reward)
                     ▼
   distil verified primitive constructors  ──►  COMPOSE them into bigger ones
                     │
                     ▼   (a goal appears only now — to use, not to learn)
   compose the library to reach an arbitrary target;  answer "what if I do X?"
```

## Why each classic trap is handled

- **Decoy (correlation ≠ causation).** `decoy` is driven by `chain1`, so it
  correlates with `chain2` — but is not its cause. Interventions break the
  confound; the posterior weight `decoy→chain2` stays at zero with high
  confidence, so the edge is rejected.
- **Noisy-TV trap.** `static` is pure high-variance noise. *Naive* curiosity
  (maximise surprise) is hypnotised by it. Expected **information gain about the
  parameters** is ~0 once static's parent-free law is learned, so the epistemic
  agent walks past it. (See `discovery.py` in the repo root for the minimal
  version of this same point.)
- **Depth / slowness.** `chain2` is slow and two hops downstream; no single short
  pulse moves it into the target region. Only a **composed** constructor (the
  primitive chained with itself) reaches it — a capability *neither parent had
  alone*, discovered by re-characterising the composite.

## Layout

| File | What it is |
|---|---|
| `world.py` | `DynamicalCausalWorld` — the hidden dynamical SCM (actuators vs. sensors). Factories: `default`, `hard`, `confounded`, `gated`, `cascade` (two gates), `wide` (distractors), `nonlinear` (even + saturating edges). Supports interaction, nonlinear terms, and hidden variables. Ground truth. |
| `model.py` | `BayesianDynamicsModel` — conjugate **Normal-Inverse-Gamma** Bayesian regression over a feature map (linear + product + **random-Fourier** features, hidden vars excluded), with **exponential forgetting** for non-stationary worlds. The precision carries no σ², so it is **shared across all sensors** (one Gram + one maintained inverse, memory O(p²) not O(d·p²), one rank-1 update per step not d); a **Student-t** edge test reads off the recovered DAG **and interaction edges**, and `recovered_marks` splits them into **directed** (do-identified) vs **bidirected** (possibly-confounded). |
| `constructor.py` | `Box`, `Constructor`, `Library`, `compose()` — Constructor Theory made operational; precondition/effect algebra; `full_program` carries each skill's from-rest program for stacking. |
| `active_inference.py` | `EpistemicExperimenter` (reward-free EFE; **samples candidates** when knobs are many), `CertifyingExperimenter` (**extended epistemic term: info gain about the model + about POSSIBILITY** — act to make UNDECIDED skills certifiable; reduces to pure EIG when none are) + `Random` / `NaiveSurprise` / `Passive` foils. |
| `planner.py` | `ConstructorSynthesizer` — mint/verify primitives, **iterated conditional** primitives, and `reach(target)` as a planner (`search="bfs"` uninformed or `"greedy"` **informed best-first** over a model-based heuristic) that chains constructors to any depth. |
| `agent.py` | `ConstructorCausalAgent` (+ `discover_actuators`) — `explore` / `explore_continuous` / `practice` (**rehearse sustained holds in one life → on-policy coverage**) / `explore_to_certify` (**overlap-driven curiosity: act to make UNDECIDED skills certifiable**) + `coverage_targets` / `discover_interactions` / `detect_hidden` / `build_library` / `achieve` / `whatif` / `consolidate` / `surprise` / `live_round` (autonomous loop; `certify_seek=True` → **targeted recovery: after drift, re-cover the reopened skills with fresh data instead of exploring blindly**). |
| `observation_gating.py` | The noisy-TV trap *with actions that gate observation*: info-gain vs. naive surprise when you choose what to watch. |
| `certify.py` | Cloning-free reliability certificates: `BettingCS`/`AnytimeCS` (anytime-valid), `certify_reliability` (in-stream), `certify_passive` (behaviour-agnostic **model-based** OPE) + `calibrate_passive`, `certify_modelfree` (model-free, tabular stationary), `certify_modelfree_continuous` (model-free, **RBF+LSTD, no discretization**), `certify_library` (**reset-free POSSIBLE-verdict over the whole library, from one life-stream**), `DriftDetector`, `detect_latent_lag`. See `RESEARCH.md`. |
| `demo*.py` | Narrated runs: `demo`, `demo_advanced`, `demo_frontier`, `demo_frontier2`, `demo_continual`, `demo_autonomous`. |
| `selftest*.py` | Every headline claim, asserted (15 suites, **163 checks**; the certify suite is 28). Includes `selftest_shared_stats` (shared-Gram exactness vs from-scratch linear algebra) and `selftest_marks` (directed vs bidirected honesty). |
| `make_paper.py` | Generates `PAPER.pdf` / `PAPER.md` (pure-matplotlib, no LaTeX). |

## Paper

A full write-up — motivation, implementation, results (65 checks), and a roadmap to a
self-continual learning agent — is in **[PAPER.pdf](PAPER.pdf)** (source: `PAPER.md`,
regenerate with `python -m constructor_causal.make_paper`).

## Run

```bash
# from the ChemicalWorld directory (uses the project .venv; numpy-only)
./.venv/bin/python -m constructor_causal.demo            # basic: DAG → library → compose
./.venv/bin/python -m constructor_causal.selftest        # 12 checks
./.venv/bin/python -m constructor_causal.demo_advanced   # the three hard cases
./.venv/bin/python -m constructor_causal.selftest_advanced  # 14 checks
./.venv/bin/python -m constructor_causal.demo_frontier   # discovery + deep cascade
./.venv/bin/python -m constructor_causal.selftest_frontier  # 10 checks
./.venv/bin/python -m constructor_causal.demo_frontier2  # informed planner + nonlinear
./.venv/bin/python -m constructor_causal.selftest_frontier2  # 9 checks
./.venv/bin/python -m constructor_causal.demo_continual  # a world that changes underneath
./.venv/bin/python -m constructor_causal.selftest_continual  # 9 checks
./.venv/bin/python -m constructor_causal.demo_autonomous  # discovers interface; self-driven loop
./.venv/bin/python -m constructor_causal.selftest_autonomous  # 11 checks

# long-horizon non-stationary deployment + knowledge-prior fusion (see below)
./.venv/bin/python -m constructor_causal.demo_deploy     # many regimes; continual vs discover-once
./.venv/bin/python -m constructor_causal.selftest_deploy # 10 checks
./.venv/bin/python -m constructor_causal.demo_prior      # RAG/LLM prior: F1 vs budget by accuracy
./.venv/bin/python -m constructor_causal.selftest_prior  # 15 checks (CI-safe)
./.venv/bin/python -m constructor_causal.demo_prior prepare           # real-LLM: print questions
./.venv/bin/python -m constructor_causal.demo_prior score '<answers>' # ...a Claude subagent answers
```

## Long-horizon deployment & knowledge-prior fusion (`deploy.py`, `prior.py`, `semantic_worlds.py`)

Two additions turn the single-episode agent into a deployed, continually-learning one,
and fuse it with the sibling `causal_dag` RAG/LLM prior.

**Long-horizon non-stationary deployment.** `RegimeSchedule` declares a sequence of world
mutations (`flip_edge`, `scale_edge`, `add_edge`, `remove_edge`, `inject_gate`,
`noise_burst`); `deploy()` runs the agent's existing `live_round` across them with a
per-round metrics timeline, against a discover-once `deploy_baseline()`. On a 5-regime run
(sign flip · strengthen · structural add · noise+remove), the continual agent holds
**belief-error ~0.01 and F1 = 1.0 in every regime** with **detection latency 0–1 round**,
while the frozen baseline's belief diverges (error → **0.74**, surprise pinned at **~59σ**,
F1 drops to 0.86 on the structural change). `explore_localized()` re-checks only the
changed edge's believed ancestors — on a 6-actuator world it recovers a flipped edge **2×
more accurately** for the same budget, shrinking the intervention grid **729→3** (and
correctly falling back to all actuators when the change is a brand-new edge). Verified by
`selftest_deploy.py` (10/10).

**Knowledge-prior fusion.** `CausalPrior` adapts a `causal_dag.rag` oracle (simulated, or a
real Claude model on the semantic worlds) into a belief seed: it **asserts** edges it is
confident about, which interventions then **verify and override** (a wrong prior edge is
dropped once the reverse is recovered). A real Claude model orients `heater_world` and
`tank_world` (`heater_power→room_temp→thermostat`, `inflow→level→outflow`) at **100%** from
variable names alone, handing the dynamical agent a **correct causal graph at F1 = 1.0 with
zero experiments** (vs 0.0 without, needing 8–20 interventions); an accuracy-0 prior starts
wrong but is corrected back to F1 1.0 by interventions (soft-prior safety). Verified by
`selftest_prior.py` (15/15, CI-safe). Honest scope: the agent already identifies edge
*direction* by intervening, so the prior fills no Markov-equivalence gap here — its value is
the zero-experiment head start, override safety, and seeding localized re-exploration, not
"fewer experiments" (active inference is already efficient).

## The three hard cases (`*_advanced`)

These are the open problems the basic system flagged — now demonstrated, all reward-free:

1. **Hidden confounder.** `world.confounded()`: an unobserved `H` drives both `S1`
   and `S2` with no edge between them. A **passive** observer infers a strong
   spurious `S1→S2` (weight ≈ 0.85); the **intervening** agent forces `S1`,
   decorrelating it from `H`, and the weight collapses to ≈ 0. *Only intervention
   tells a shared hidden cause from a real edge.*
2. **Distinct-constructor composition.** `world.gated()`: `Z := 0.3·Z + 0.5·gate·a1`,
   `gate ← a0`. No single knob — and no *self-chaining* of one knob — can move `Z`
   (a 9-step `a0` hold reaches `Z` with reliability 0.00). The agent discovers the
   **interaction** `a1·gate→Z`, finds `a1` idle from rest, mints a **conditional**
   constructor (`a1 | gate-open`), and reaches `Z` by composing *two different*
   constructors in order: `open-the-gate(a0) ≫ drive-the-gated(a1|gate)`,
   reliability 1.00. This is "combine sequences of constructors into bigger
   constructors" in the literal sense — capability that exists only in the chain.
3. **Observation-gating (noisy-TV).** When the action chooses *what to observe*,
   naive surprise stares at the noise channel (83% of looks) and learns the
   informative law only 75% of the time; the info-gain objective watches noise
   only 29% and learns it 100%. The trap that *doesn't* bite under full
   observation does bite here — see also the minimal `discovery.py` at the repo root.

## The frontier cases (`*_frontier`)

The two caveats the advanced cases still carried — *interactions are supplied* and
*composition is shallow* — are removed here, still reward-free:

- **Interaction discovery.** The agent starts with a **linear** belief and is given
  **no** candidate products. After exploring it scans each sensor's residuals for
  leftover structure (`discover_interactions()`), proposes the product feature that
  explains it (a t-test on the residual), rebuilds the model with that feature, and
  refits. On `gated()` it recovers exactly `a1·gate→Z`; on `cascade()` it recovers
  both `a1·gate1→gate2` and `a2·gate2→Z` — no spurious extras.
- **Deep composition (a two-gate cascade).** `world.cascade()`: `gate1←a0`,
  `gate2 := gate1·a1`, `Z := gate2·a2`. Each knob is idle until the previous gate is
  open. Iterated conditional minting stacks skills bottom-up (`a1|gate1`, then
  `a2|gate2`), and a **BFS planner** chains **three distinct** constructors in order:
  `a0 ≫ (a1|gate1) ≫ (a2|gate2)`, reliability 1.00, Z reached. This is "combine
  sequences of constructors into bigger constructors, and so on" — literally.

## The frontier-2 cases (`*_frontier2`)

Scaling the planner, and taking the belief nonlinear:

- **Informed planner.** `world.wide(k)` has one real chain `a0→chain1→chain2` plus
  `k` useless knobs each driving a dead-end sensor — so the library is large and
  every primitive composes with every other. Uninformed BFS expands **314** nodes
  to find the answer; `reach(..., search="greedy")`, ordering the frontier by the
  model-predicted distance to the target, expands **13** — same chain, ~24× cheaper.
- **Nonlinear structure, two ways.** `world.nonlinear()`:
  - `even := 0.3·even + 1.2·a0²` is **even** in `a0`, so the linear correlation is
    exactly zero — a linear learner finds **no** `a0→even` edge. `discover_interactions()`
    recovers it as the product `a0·a0→even`.
  - `sat := 0.3·sat + 2·tanh(1.5·a1)` saturates. With **continuous** exploration
    (`explore_continuous`) and a **random-Fourier-feature** basis (`rff=…`), the
    model predicts the curve — "what if I hold `a1=v`" RMSE **0.03** vs the linear
    model's **0.65** (~22× more accurate counterfactuals).

## Continual learning (`*_continual`)

The loop also runs on a **non-stationary** world. We flip and re-scale the edge
`a0→chain1` (`+0.8 → −0.8 → +1.3`) without telling the agent. Three mechanisms keep
it current, all reward-free:

- **Forgetting** — the belief is recursive least squares with a forgetting factor
  (`forget<1`), so it tracks recent dynamics. The recovered weight follows the true
  edge across all three regimes.
- **Change detection** — `agent.surprise()` is one-step prediction error; on `chain1`
  it spikes to ~3–4 the moment the world flips, then falls below 0.05 once re-learnt.
- **Consolidation** — `agent.consolidate()` re-verifies every constructor against the
  current world and prunes the ones that broke. The skill that drove `chain1` high by
  holding `a0=+2` is pruned on the sign flip and rebuilt to hold `a0=−2`, then flipped
  back. The library stays true to the world, not its past.

## Autonomy (`*_autonomous`) — closing the failure audit

A self-audit found the agent was *handed* its interface, *couldn't tell a hidden
cause from noise*, *couldn't do fine control*, and *needed an external schedule to
re-learn*. All four are now fixed, reward-free:

- **Discovers its interface** — `discover_actuators(world)` pokes each variable to an
  out-of-range value; the ones that hold are its knobs. Correct on default/gated/cascade.
- **Hidden cause vs. noise** — `agent.detect_hidden()` flags a variable that is
  poorly predicted *yet strongly autoregressive* (the fingerprint of a slow hidden
  driver). It flags the confounded `S2` and does **not** flag the pure-noise `static`.
  (Honest: this is a heuristic for a genuinely hard identifiability problem — a hidden
  AR cause and a real self-loop aren't separable from one-step observational data.)
- **Continuous control** — `synth.solve_setpoint()` solves for an intermediate knob
  value (centered in the target) when the coarse ±2 library overshoots; hits the
  narrow target the audit showed failing, reliability 1.00.
- **Autonomous continual loop** — `agent.live_round()` watches its *own* standardised
  surprise (so an irreducibly noisy channel doesn't false-alarm) and decides when to
  re-learn. With no external schedule it detects a **parametric** flip (z≈56σ, prunes
  stale skills, relearns the edge) and a **structural** change — a gate that newly
  appears (z≈27σ, re-discovers `a1·gate`).

## Honest scope

This is a **method demonstrator**, not a generality claim. In the spirit of the
project's `PLAN.md`:

- The model is **closed-form Bayesian** over a feature map: linear + discovered
  pairwise products + (optional) random-Fourier features. That covers gates and
  smooth univariate nonlinearity, but RFF is reliable only **on the training
  manifold** (it extrapolates poorly off it), and structure read-off (recovered
  edges) is still done from the linear/product blocks, not the RFF block. General,
  off-manifold, high-order structure learning remains open.
- The agent now **discovers its actuators** (controllability) and **flags hidden
  state** (detect_hidden) — but flagging is not *certifying*: a hidden AR(1) cause
  and a genuine self-loop are not separable from one-step observational data, so
  identifying the hidden variable itself remains open. It is still told which
  variables it can *observe* (hidden ones are simply absent from its inputs).
- Composition covers **self-chaining, distinct, and iterated conditional**
  constructors, planned to arbitrary depth by either uninformed BFS or an
  **informed best-first** search (which scales to distractor-heavy libraries). The
  heuristic is greedy on a model-predicted distance — not admissible, so the chain
  it returns is short but not provably optimal.
- Continual learning is now **autonomous** and handles both **parametric** drift
  (forgetting + standardised-surprise change detection + consolidation) and
  **structural** drift (a newly-appearing gate triggers re-discovery). Still open:
  skill merging/abstraction, a competence/empowerment drive, and the fact that
  verification still relies on a **cloneable, resettable** world (an unrealistic
  privilege a true embodied agent lacks).
- Reliability ("a task is *possible*") is an **empirical threshold over finite
  trials**, the operational stand-in for Constructor Theory's "arbitrarily good
  approximation" — not a proof of possibility.

What it does show, cleanly and falsifiably (`selftest.py`): a reward-free agent
recovers a causal DAG, rejects a decoy, ignores irreducible noise, builds a
verified library of composable constructors, and composes them to achieve goals
it was never trained on.

## Independent audit (2026-06)

The DAG claims were independently verified by adversarial agents that **ran probes
against the code**, not just read it. Verdict: **the DAG is real, not decoration.**

- `recovered_edges()` is a genuine per-sensor Bayesian-posterior z-test; recovery is
  exact (F1=1.0) and **stable from z=3 to z=10** — not a threshold knife-edge.
- The **hidden-confounder** result holds and is decisive: a passive observer infers a
  spurious `S1→S2` edge (~0.82, every seed); an intervening agent rejects it (~0.025,
  no seed); a **random-forcing control debiases identically**, proving it is the
  `do()`-intervention — not the curiosity objective — that breaks the confound. An
  added *real* `S1→S2` edge is correctly retained.
- The DAG is **load-bearing**: wiping the posterior collapses what-if to 0 and makes
  planning honestly return "impossible."

Three issues were found and addressed:

1. **(fixed) Stale cached reliability** — `planner.reach()` returned a directly-
   applicable constructor's *stored* reliability without re-checking. It now
   **re-verifies against the current world** before trusting it (confirmed: a skill
   reads 1.00, then 0.00 after the world is broken).
2. **(added, off by default) Multiplicity correction** — `recovered_edges(
   correct_multiplicity=True)` raises the threshold to `sqrt(z² + 2 ln m)` to control
   the family-wise false-edge rate at large `n`. Off by default because it costs
   recall on deliberately weak edges (the `hard` world); enable it when many variables
   + large samples make false positives the bigger risk.
3. **(verified honest) EIG-vs-naive framing** — the package already discloses that on
   fully-observed worlds curiosity *ties* naive surprise, and locates the genuine
   separation in the observation-gating test; no over-claim to fix.

All 54 selftests pass after the fixes.
