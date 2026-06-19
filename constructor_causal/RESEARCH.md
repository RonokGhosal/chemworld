# Research: certifying latents & cloning-free verification

*Deep-research synthesis (108 agents, 26 sources fetched, 120 claims extracted, 25
adversarially verified at 3 votes/claim → 21 confirmed / 4 killed) plus an
implementation roadmap for `constructor_causal`. Run `wf_726cac12-8ff`, 2026-06-17.*

The agent's two honest, unsolved frontiers (from the failure audit):

1. **Certify latents** — detect *and identify* hidden confounders; specifically
   tell a slow hidden common cause (an AR(1) latent driving two observed variables)
   from a genuine self-loop.
2. **Cloning-free verification** — certify a constructor's reliability from a single
   ongoing trajectory, without resetting/cloning the world to fresh i.i.d. starts.

> **Verification coverage.** Q1 below is from the first run (21/25 claims verified).
> Q2 was empty in that run, so a **focused Q2 re-run** (`wf_badfeaee-72b`, 105 agents,
> 23 sources, **25/25 claims verified, 0 killed**) now backs it — see "Q2 (verified
> re-run)". The betting-CS / e-process core is fully cited; reset-free RL, DICE-OPE,
> conformal-under-drift, and the Constructor-Theory *primary text* were fetched but
> not independently verified, so those are flagged below.

---

## Q1 — Certifying latents (verified)

The headline, which our own probe rediscovered empirically: **a slow hidden AR(1)
common cause is observationally equivalent to an inflated self-loop; you cannot
separate them from one-step observational data alone.** Escaping that needs *added
structure*. Four routes, each with a hard limit.

### (a) Non-Gaussianity — certify confounder-*free*, label the rest
`lvLiNGAM` / **ParceLiNGAM** / **RCD** exploit non-Gaussian innovations. ParceLiNGAM
finds "parcels" of variables *provably unaffected* by any latent confounder (and
returns "unknown" elsewhere); RCD emits a **bidirected** arrow for a pair sharing a
latent confounder and a **directed** arrow for a pair that doesn't.
- *Limit (verified):* in latent-variable LiNGAM only an **observational-equivalence
  class** is identifiable (Hoyer et al. 2008) — not all causal orders.
- *Killed claims (don't overclaim):* ParceLiNGAM is **not** an exact faithfulness-free
  exogeneity test (refuted 0–3); RCD does **not** identify latents from purely
  observational data alone (refuted 1–2). Treat as population/identifiability
  results with finite-sample (HSIC) test error.
- *Code:* official `lingam` library — `BottomUpParceLiNGAM`, `rcd`.
- Sources: arXiv:1303.7410 (ParceLiNGAM), arXiv:2001.04197 (RCD).

### (b) Rank / tetrad constraints via trek-separation — the latent-factor detector
Vanishing sub-covariance minors (rank/tetrad constraints) encode **strictly more**
structure than conditional-independence tests, and **trek (t-)separation** gives
their exact graphical meaning: `rank Σ_{A,B} ≤ r` for all consistent parameters iff
sets with `#C_A+#C_B ≤ r` t-separate A from B. CI is the special case
`rank Σ_{A∪C,B∪C}=#C`. This is the math behind the TETRAD program and latent-factor
discovery (FOFC, GIN/triad).
- *Implication for us:* this only becomes available with **≥4 observed variables** —
  our 2-variable confounder (`S1`,`S2`) is below the threshold.
- Source: Sullivant, Talaska & Draisma, *Ann. Statist.* 38(3):1665 (2010), arXiv:0812.1938.

### (c) Restricted mixed-graph classes — BAPs as a tractable middle ground
**Bow-Free Acyclic Path Diagrams** (a subclass of ADMGs): directed edges = direct
effects, bidirected = hidden confounders, with no pair having both. BAPs sit between
latent-free DAGs and fully-general MAGs/ADMGs, giving a **smaller equivalence class
and more accurate estimation** than FCI/RFCI/FCI+ *if not heavily misspecified*.
- Source: Nowzohour, Maathuis, Evans, Bühlmann, *EJS* 11(2) (2017), arXiv:1508.01717.

### (d) Proximal causal inference — actually *correct* a latent (needs 2 proxies)
The strongest verified route to **identify the effect** of an unmeasured confounder
(not just flag it). With **two proxies** of the hidden `U` — a negative-control
exposure `Z` (`Y ⊥ Z | A,U,X`) and a negative-control outcome `W`
(`W ⊥ (A,Z) | U,X`) — the effect is **nonparametrically identified without ever
modeling `U`**, via an outcome **bridge function** `h` solving a Fredholm integral
equation of the first kind. Generalizes Robins' g-formula → proximal g-computation;
extends to time-varying confounding (longitudinal PCI under a semiparametric MSMM,
doubly-robust).
- *Identification needs completeness*; categorical case reduces to the checkable
  `min(d_z, d_w) ≥ d_u` — each proxy needs ≥ as many states as the latent.
- *Limit (verified):* completeness/bridge-existence are **strong and largely
  untestable**; absent them, only **partial identification**. A single negative
  control is **detection-only** — it cannot recover bias magnitude (Lipsitch et al.
  2010), and a non-null NC association is **not** a clean existence test (refuted 1–2).
- Sources: Tchetgen Tchetgen et al., *Statistical Science* 39(3) (2024), arXiv:2009.10982;
  Ying et al., *JRSS-B* 85(3):684 (2023); Cui et al., *JASA* 119(546) (2024);
  Nature Rev. Methods Primers s43586-023-00249-4.

### The provable frontier (verified synthesis)
- **Provably out of reach without extra structure:** separating a slow AR(1) latent
  common cause from a genuine self-loop using one-step observational data alone;
  full causal-order recovery under latent confounding; recovering bias magnitude
  from a *single* negative control.
- **Escape hatches:** non-Gaussianity (certifies a *subset*), rank/tetrad over ≥4
  vars, a BAP restriction, or ≥2 proxies (PCI) — each buying identification only
  under its own assumptions.

### What this means for *our* agent (Q1)
- **(i) Do now — we can intervene.** This is our real edge, and it is the one piece
  the literature says interventions uniquely unlock. The constructive test:
  **a time-constant mismatch.** For a sensor with a suspiciously high self-loop that
  *also* has an actuator feeding it, inject a one-step pulse through that actuator
  and measure the **intervention-response decay rate** τ_int; compare to the
  **observational autocorrelation rate** τ_obs. A genuine self-loop gives
  τ_obs ≈ τ_int; a hidden *slow* driver makes τ_obs ≫ τ_int (the baseline memory is
  slower than the true dynamics). This *certifies* (not just flags) a latent, using
  exactly the actuator-forcing the agent already has. Implemented in `certify.py`.
- **(ii) Needs added assumptions.** To go from "a latent is here" to *correcting* it:
  designate two proxy sensors and apply proximal g-computation (PCI); or rely on
  non-Gaussian innovations and run ParceLiNGAM/RCD over **≥4** observed variables to
  label bidirected (shared-latent) vs directed pairs.
- **(iii) Out of reach.** For our 1-latent / 2-affected-variable confounder where one
  of the two is the actuator and *no* actuator feeds the affected sensor, neither the
  mismatch test nor PCI applies — identification is provably impossible without a
  second proxy. Our `detect_hidden` flag is the correct ceiling there. **Say so.**

---

## Q2 — Cloning-free (reset-free) verification (VERIFIED re-run, 25/25 claims)

**Verified core (built into `certify.py`):**
- A **confidence sequence** is the right primitive — time-uniform coverage
  `P(∀t: p∈C_t) ≥ 1-α`, valid at every stopping time, so you may peek continuously
  and stop when an endpoint crosses τ. Mechanically guaranteed by nonnegative
  (super)martingales + **Ville's inequality** / e-processes — proven the *only*
  admissible route to anytime-valid inference (Ramdas et al.).
- **Adopt the betting CS** (Waudby-Smith & Ramdas): `K_t(m)=∏(1+λ_i(m)(X_i−m))`,
  `C_t={m: K_t(m)<1/α}`. Distribution-free up to boundedness, variance-adaptive,
  LIL-optimal `√(log log t / t)` width, and **empirically dominates** Hoeffding /
  empirical-Bernstein / the mixture CS. *We confirmed this ourselves:* width **0.05
  vs 0.13** at p=0.97 (both cover ≥0.95); `certify_reliability` now certifies
  **~4.6× faster** (n=64 vs 297).
- **Change detector = a second e-process** betting against the certified `p₀`; wealth
  crossing `1/α` ⇒ drift ⇒ re-verify. We confirmed: fires ~33 steps after a 0.95→0.5
  shift, false-alarm rate 0.000. Reuses the same Ville machinery.
- **Verified hard limits:** classical HCOPE/OPE concentration bounds (Thomas et al.)
  need **independent** importance-weighted returns and a **known behaviour policy** —
  they do **not** validate a single dependent trajectory. The betting CS does, *iff*
  the per-step conditional structure holds; serial dependence still inflates the
  effective sample size. And there is **no distribution-free guarantee under
  adversarial drift** — the change detector is mandatory, not optional.
- *Fetched but not independently verified this pass:* reset-free RL (EARL, Leave-No-
  Trace, MEDAL), stationary-distribution-correction OPE (DualDICE/GenDICE/marginalized
  IS), conformal-under-drift (Gibbs-Candès), and the Constructor-Theory primary text.

**Behaviour-agnostic OPE — built (`certify_passive` / `calibrate_passive`).** The
agent can now certify a skill from its PASSIVELY-collected buffer of visited states +
its learned model, with **no re-execution and no knowledge of the behaviour that
produced the stream**: roll the constructor's program forward under the model from
each visited state, wrap the per-state hits in a betting CS. This is model-based OPE
over the agent's own occupancy — it lets the agent *screen many candidate skills from
one stream for free*. We confirmed it agrees with active execution on a well-modelled
world (reachable→POSSIBLE, unreachable→IMPOSSIBLE, from the buffer alone). The hard
limit is explicit and *guarded*: a model-based certificate is only as good as the
model, so `calibrate_passive` compares the model's predicted reliability against a
handful of real in-stream executions — on the nonlinear world it CATCHES the wrong
(linear) model (gap 1.00, untrustworthy) and TRUSTS the correct (RFF) one (gap 0.00).
**Model-free OPE — built (`certify_modelfree`).** The DICE family estimates the
stationary occupancy ratio ζ\*=d^π/d^D from off-policy data via a Fenchel/Lagrangian
minimax and recovers value as `E_{d^D}[ζ·r]` — **behaviour-agnostic** (no π_b, no
per-step ratios). We implement the tabular/low-dim instance: for a stationary "hold
a=v" intervention, estimate the long-run reliability (stationary fraction of time in
R) from the agent's off-policy buffer using ONLY real transitions where the behaviour
matched the command — no dynamics model at all. Verified payoff: on the nonlinear
world it recovers the true stationary level (2.55 vs ground-truth 2.59, reliability
1.00) **where the linear model-based estimate is wrong (1.87)** — model-free is right
exactly where a wrong model fails (a fixed command linearizes the value function, so
the tabular estimate from real transitions is unbiased). The **binding limit is
overlap/positivity** (verified): a deterministic target is non-identifiable if the
behaviour rarely takes it — `certify_modelfree` returns **UNDECIDED** with no
coverage (the provable wall). CI is bootstrap (batch), not anytime-valid; CoinDICE
(Dai et al. 2020) is the cited route to its own (asymptotic χ²/finite-sample) CI.
Sources (verified): Nachum et al. (DualDICE, NeurIPS 2019); Yang et al. (BestDICE /
regularized Lagrangian, NeurIPS 2020); Liu et al. (marginalized IS, NeurIPS 2018);
Dai et al. (CoinDICE, NeurIPS 2020). **Continuous-state model-free OPE — built (`certify_modelfree_continuous`).** A focused
verification pass (102 agents, 24/25 claims) confirmed the linear DICE form but
**refuted** the symmetric-Gram closed form I had sketched: the correct estimator is the
**asymmetric LSTD(Q)** normal equation `A = E[φ(s)(φ(s)−γφ(s′))ᵀ] + λI` (not `E[ggᵀ]`),
`reliability = (1−γ)·E[V̂]`. Implemented with an **RBF state basis** over the on-target
sub-data — discretization-free. Verified: estimate **0.88 vs MC ground truth 0.93**
(the residual is the function-approximation bias the theory predicts), and it crushes
the wrong-model baseline (`|err| 0.06` vs the linear-SCM's `0.93`). Notes: it targets
the γ-DISCOUNTED occupancy reliability (the tabular `certify_modelfree` targets the
exact stationary fraction); same overlap wall (UNDECIDED with no coverage); deterministic
targets need ε-smoothing in principle (we condition on the a≈v sub-data instead).

**Anytime-valid model-free OPE — OPEN, not faked.** The same pass established (well-
supported) that there is **no published anytime-valid theorem** for single-trajectory
stationary-ratio (DICE) OPE: CoinDICE's CI is fixed-n (asymptotic χ² / finite-sample
VC), and feeding an *estimated*, serially-dependent `ζ̂(s,a)·r` into a betting CS
**breaks** the martingale-difference structure the CS validity proof requires.
Anytime-valid OPE is provable only for **single-step contextual bandits** (Waudby-Smith
et al. betting CS) or asymptotically (AsympCS). So `certify_modelfree_continuous` ships
with a **batch bootstrap CI**, explicitly NOT anytime-valid — the honest state of the
art. Sources: Nachum et al. (DualDICE, NeurIPS 2019); Yang et al. (BestDICE, NeurIPS
2020); Liu et al. (RKHS marginalized-IS, NeurIPS 2018); Dai et al. (CoinDICE, NeurIPS
2020). Remaining open: a genuine RKHS/kernel DualDICE, and any anytime-valid guarantee
for this single-trajectory setting (a real research gap).

**Building off #1 → the main goal: reset-free self-certification of the WHOLE library
(`certify_library`, `agent.practice`).** The payoff of the continuous model-free OPE is
not the estimator in isolation — it is that the constructor library's POSSIBLE verdict
can now be decided from the agent's *one ongoing life-stream*, with no `clone()`/`reset()`
(the resettable-rollout crutch the minting path still uses). `certify_library` routes each
possible skill by program shape — constant single-knob holds → `certify_modelfree_continuous`
(assumption-light, model-free), other schedules → `certify_passive` (model-based) — and
returns a from-buffer verdict per skill. Demonstrated reset-free: a correct stationary
hold certifies **POSSIBLE (value 0.96)** while a deliberately false `reliability=1.0`
claim on an unreachable target is **overturned → IMPOSSIBLE (value 0.002)**.

Two concrete lessons #1 taught us, both surfaced live while wiring this:
1. **On-policy coverage is the binding constraint, even when the *action* is covered.**
   A slow/deep variable only enters its effect region under a *sustained* command; an
   i.i.d.-random buffer holds `a0≈2` at isolated steps, so chain1 never builds up
   (mean −0.11, never reaching its 2.0+ band) and the reward indicator is identically
   zero. The estimator is correct; the data lacks the relevant *state* occupancy. Fix:
   `agent.practice()` — the agent rehearses sustained holds in its one life, which is
   exactly what an embodied agent does by *performing* its skills. This makes the
   overlap wall an **exploration signal**: UNDECIDED-on-no-coverage names where to go.
2. **Reach-from-rest ≠ stationary occupancy.** A from-rest primitive's effect box is a
   *transient* snapshot (h_prim steps); the slow chain *overshoots* it under a sustained
   hold (chain2 → 3.3 vs the box top 2.58). The clone-based `verify` measures reach
   reliability; `certify_modelfree_continuous` measures γ-discounted occupancy — genuinely
   different quantities for slow vars. They agree only for skills whose effect *is* a
   stationary set; `certify_library` must therefore be read as a stationary-occupancy
   certificate, not a drop-in replacement for reach reliability.

**Overlap-driven curiosity — BUILT (`CertifyingExperimenter`, `agent.explore_to_certify`).**
The fusion of #1 into the reward-free objective: extend the EFE epistemic term from
*information gain about the model* to *also* information gain about *possibility* —
`score(c) = coverage_gain(c) + EIG(c)`, where `coverage_gain` sums the coverage deficit of
every UNDECIDED-for-lack-of-coverage skill whose command `c` satisfies. The integer deficit
dwarfs the EIG bits, so open certificates are resolved first and EIG only breaks ties; with
no open certificates it reduces *exactly* to pure-EIG `EpistemicExperimenter`, so initial
reward-free learning is untouched. A target's deficit persists until its skill is covered,
so **sustained holds emerge on their own** (no hard-coded commit) — exactly what slow
downstream vars need. Still reward-free: the only imperative is to make the library's
POSSIBLE verdict *decidable*. **Verified (5/5 seeds, reset-free):** from a thin-coverage
start where a well-posed stationary skill is UNDECIDED, `explore_to_certify` resolves it to
**POSSIBLE (value ≈ 0.95)**, while an *equal budget* of pure-EIG exploration leaves it
**UNDECIDED (value ≈ 0.87)**. The two reach near-identical action-coverage *counts* — the
win is coverage *quality*: directed curiosity produces sustained holds (states build into
the effect region), EIG produces scattered isolated hits (no buildup), so only the directed
estimate's CI clears τ. This closes the certifier→explorer loop: `certify_library`'s
UNDECIDED verdict is the target signal, and the agent acts to extinguish it.

**Wired into the continual loop — `live_round(certify_seek=True)`.** After drift the agent
spends its re-exploration budget RESOLVING reopened certificates instead of exploring
blindly: `explore_to_certify(fresh=True)` re-covers each current skill's command with
NEW-regime data (a sign-flip leaves old coverage intact by *count* but stale in *fact*, so
coverage is counted only from the recovery phase's start), and the standardised per-sensor
surprise is passed as `priorities` so the skills whose effect region just spiked — the
genuinely reopened ones — are covered first. Then the usual consolidate/rebuild runs.
Verified (4/4 seeds) on a sign-flip world: drift detected, reopened skills re-covered with
fresh data, stale skills pruned, the new-regime library rebuilt correctly (chain1-high skill
swaps a0 = +2 → −2), and the agent stays capable (achieve r = 1.00). The default
`certify_seek=False` path is byte-for-byte unchanged (continual + autonomous suites green).
HONEST scope: the win is coverage *quality* (sustained holds → buildup, which resolves
certificates pure-EIG leaves UNDECIDED — see the standalone result above), not raw coverage
*count* — in small / joint-command worlds blind exploration already sets every knob each
step, so count barely differs; the concentration benefit grows with command scarcity.

**Still open (research-grade, not engineering).** Anytime-valid single-trajectory
stationary-ratio OPE; a genuine RKHS/kernel DualDICE; conformal prediction under drift.

Sources (verified): Howard/Ramdas/McAuliffe/Sekhon (*Ann. Statist.* 2021; *Prob.
Surveys* 2020); Ramdas/Grünwald/Vovk/Shafer (*Stat. Sci.* 2023); Waudby-Smith &
Ramdas (*JRSS-B* 2024); Johari/Pekelis/Walsh (*Oper. Res.* 2021); Thomas et al. (AAAI/
ICML 2015). Full grounded discussion below.

## Q2 — Cloning-free (reset-free) verification — extended discussion

Goal: replace `verify()`'s 60 fresh `world.clone()` rollouts with a certificate
accumulated from the **live stream**. Three literatures converge.

### Anytime-valid inference — the right object
A **confidence sequence** is an interval `[lo_t, hi_t]` valid *simultaneously at all
t* (and all data-dependent stopping times): `P(∀t: p ∈ [lo_t,hi_t]) ≥ 1-α`. Unlike a
fixed-N confidence interval, you can **peek after every observation and stop when
satisfied** — exactly the in-stream regime. Two constructions:
- **Method-of-mixtures / sub-Gaussian boundary** (Howard, Ramdas, McAuliffe, Sekhon,
  *Ann. Statist.* 2021; Robbins) — closed-form radius for a bounded mean.
- **Betting confidence sequences / e-processes** (Waudby-Smith & Ramdas, *JRSS-B*
  2024, "estimating means by betting") — a capital process `K_t(m)=∏(1+λ_s(X_s−m))`;
  the CS is `{m : K_t(m) < 1/α}`. Tightest known for bounded means; `λ_s` is any
  predictable bet (GRAPA/ONS). The matching one-sided **e-process** is the natural
  test of "is reliability ≥ τ?".
- Sources fetched: arXiv:1810.08240 (Howard & Ramdas), arXiv:2010.09686
  (Waudby-Smith & Ramdas). *(Not adversarially verified in this pass.)*

### No fresh i.i.d. starts — OPE and reset-free RL
- **Off-policy evaluation:** per-decision / weighted importance sampling,
  **doubly-robust** estimators (Jiang & Li, arXiv:1511.03722), and
  **marginalized IS / stationary-distribution correction** (Xie et al., NeurIPS 2019;
  Liu et al., arXiv:1906.04733) estimate a target policy's value from off-policy
  data — relevant when constructor executions are opportunistic, not on-policy.
- **Reset-free / autonomous RL:** the **EARL** benchmark and reset-free formulations
  (Sharma et al., arXiv:2112.09605; Leave-No-Trace; forward-backward reset
  controllers) study learning/evaluating *without* environment resets — our exact
  constraint. *(Fetched; not verified this pass.)*

### Change as the re-verification trigger
A constructor certified "possible" can later break (the world drifts). **CUSUM** /
**Bayesian online change-point detection** on the in-stream success indicator (or on
predictive surprise, which we already compute) is the trigger to *reopen* the
certificate — tying Q2 to the autonomous loop we already have (`live_round`).

### Operationalizing Constructor Theory's "possible"
Constructor Theory: a task is **possible iff a constructor can perform it to
*arbitrarily high accuracy***. A fixed-N reliability (our current `≥τ over 60
trials`) is a crude stand-in. The faithful operationalization is an **anytime-valid
confidence sequence on the success probability `p`**, maintained in-stream:
- declare **POSSIBLE** when `lo_t ≥ τ` (reliability certified above threshold, at
  level `1-α`, valid despite continuous peeking);
- declare **IMPOSSIBLE-so-far** when `hi_t < τ`;
- otherwise **UNDECIDED — keep gathering** (the certificate keeps tightening, the
  "arbitrarily high accuracy" limit made operational: `hi_t − lo_t → 0`).

This *is* "approximable to arbitrary accuracy" turned into a running certificate,
and it needs no cloning: every time the agent executes a constructor from a
precondition-satisfying state during normal life, that is one more Bernoulli draw.

### What this means for *our* agent (Q2)
- **(i) Do now.** Add an **anytime-valid `AnytimeCS`** (sub-Gaussian mixture
  boundary — closed form, numpy-only) and a `certify_reliability` that consumes the
  in-stream success/failure of opportunistic constructor executions and returns
  POSSIBLE / IMPOSSIBLE / UNDECIDED with a valid `[lo,hi]`. Replaces fixed-N cloning.
  Implemented + coverage-tested in `certify.py`. Wire CUSUM/BOCPD on the indicator
  to reopen a certificate (we already detect change via standardized surprise).
- **(ii) Needs added assumptions.** Full OPE (importance weights / DR) requires a
  known behaviour policy and overlap; betting-CS reliability needs the executions to
  be (conditionally) exchangeable given the precondition — mild, but worth stating.
- **(iii) Out of reach.** A *distribution-free, assumption-free* guarantee over an
  adversarially drifting world is impossible (no-free-lunch); anytime-valid validity
  holds under stationarity-between-changepoints, which is why change detection is the
  partner, not optional.

---

## Prioritized implementation roadmap

| # | Item | Question | Lift | Status |
|---|---|---|---|---|
| 1 | `AnytimeCS` (mixture) + `BettingCS` + in-stream `certify_reliability` | Q2 | low | **built** (`certify.py`) |
| 2 | Coverage + betting-dominance + drift tests | Q2 | low | **built** (`selftest_certify.py`, 12/12) |
| 3 | `detect_latent_lag` higher-lag latent detector | Q1 | low | **built** (specificity 5/5, sensitivity 4/5) |
| 4 | `DriftDetector` e-process to reopen a certificate on drift | Q2 | med | **built** (`certify.py`) |
| 7 | Focused **Q2 deep-research re-run** (verify anytime-valid / OPE) | Q2 | — | **done** (25/25 verified) |
| 5 | Rank/tetrad (t-separation) latent-factor detector over ≥4 vars | Q1(ii) | med | next |
| 6 | Proxy designation + proximal g-computation (correct a latent) | Q1(ii) | high | research-grade |
| 8 | Behaviour-agnostic **model-based** OPE (`certify_passive`) wrapped in a CS + `calibrate_passive` guard | Q2 | med | **built** (`certify.py`) |
| 9 | Behaviour-agnostic **model-free** OPE (`certify_modelfree`, tabular stationary-occupancy / DICE) | Q2 | high | **built** (`certify.py`) |

## Sources (verified Q1 unless noted)
- Tashiro, Shimizu, Hyvärinen, Washio — ParceLiNGAM — arXiv:1303.7410
- Maeda & Shimizu — RCD — arXiv:2001.04197
- Sullivant, Talaska, Draisma — trek separation — arXiv:0812.1938
- Nowzohour, Maathuis, Evans, Bühlmann — BAPs — arXiv:1508.01717
- Tchetgen Tchetgen, Ying, Cui, Shi, Miao — proximal causal inference — arXiv:2009.10982; *Stat. Sci.* 39(3) 2024; *JRSS-B* 85(3) 2023; *JASA* 119(546) 2024; Nat. Rev. Methods Primers s43586-023-00249-4
- Lipsitch, Tchetgen Tchetgen, Cohen — negative controls — *Epidemiology* 21(3) 2010
- *(Q2, fetched, unverified this pass)* Howard & Ramdas arXiv:1810.08240; Waudby-Smith & Ramdas arXiv:2010.09686; Jiang & Li arXiv:1511.03722; Liu et al. arXiv:1906.04733; Sharma et al. (EARL) arXiv:2112.09605
