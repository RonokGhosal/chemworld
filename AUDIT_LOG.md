# Continuous architecture audit — running log

Autonomous deep-audit of the `constructor_causal` program: read the code, hunt real flaws, cross-check the
literature. Each pass targets a slice of the architecture, verifies findings adversarially (CONFIRMED /
PLAUSIBLE / REJECTED), and cycles to new ground. House rule: real defects only, no manufactured findings;
match confidence to evidence. Newest passes appended at the bottom.

Cycle plan:
- Pass 1 — newest modules: `sheaf_confirm` (DiD stats), `icp` (invariance test + multiple testing), `frontier_map` (methodology) ✅
- Pass 2 — neural + core: `neural_dag`/`neural_scale` (RESIT order vs edges; Gaussian LSNM objective), `sheaf_active` (ensemble-EIG≟BOED), `constructor`
- Pass 3 — machinery: `planner` (reach/mint_conditional), `certify` (BettingCS / e-processes / anytime-valid)
- Pass 4 — evaluation: `dag_adapter`/SID, `gates.py` pre-registration, Sachs benchmark methodology
- Pass 5+ — re-cycle with fresh literature

---

## Pass 1 (2026-07-05) — newest modules. 14 findings; 4 undermine reported claims. Status: fixing load-bearing subset.

**HIGH**
1. `icp.py:50` — invariance test unions 2·n_env subtests vs a fixed `z_crit=3.0`, NO Bonferroni → per-set Type-I grows ~linearly with n_env (measured true-set reject 0.0065@2env → 0.1885@32env). The "P(⊆ true parents) ≥ 1−α" docstring guarantee has no operative α. **[FIXING: Bonferroni over env]**
2. `frontier_map.py:103` — `interventional_recall` scores target NODES not EDGES: a wrong-source edge counts as recall; a spurious edge into a true target can't be a false+. Makes "never invents spurious X→Y" unmeasurable. **[FIXING: edge-level metric + re-run]**
3. `icp.py:47` — normal `z_crit` on a df≈4 t-stat: P(|t₄|>3)=0.040 vs assumed 0.0027 (14.8× tail understatement); measured true-set reject 20%/9%/3% at n=5/10/100. Violates "high precision by construction" at small env. **[FIXING: t-based mean test via model._t_crit]**
4. `icp.py:49` — log-variance z-approx assumes Gaussian (kurtosis 3); under t₃ residuals true-set reject RISES with n (0.44→0.85 at n=30→2000). ICP returns {} from misspecification on non-Gaussian (biological) noise. **[DEFER: needs distribution-free residual test (Heinze-Deml 2018)]**

**MEDIUM**
5. `icp.py:60` — "{} = the wall is here" conflates weak signal with miscalibration + basis misspecification (3-way AND returns {} in 7/10 high-SNR seeds; tanh 10/10). **[FIXING: honest docstring + caveat]**
6. `frontier_map.py:11` — "0 false+" near-guaranteed: no multiplicative distractor + gate_sig rejects linear edges, so the low-threshold arm has nothing to wrongly fire on; engineering-vs-information verdict not actually discriminated. **[DEFER: add linear distractor sensor]**
7. `frontier_map.py:22` — "acting buys ~3-4× SNR headroom" not reconstructable (obs flat at 1/3 everywhere) and mislabels a do-magnitude effect (did∝w, linear) as SNR ((w/σ)²). **[FIXING: retract the claim]**
8. `sheaf_confirm.py:96` — no FDR/Bonferroni across ≤80 candidate pairs; shared rest_cache makes tests dependent; max-|did| target = winner's curse. (Mitigated today by z=4+min_effect+gate_sig.) **[DEFER: BH-FDR / permutation null + null-world harness]**
9. `frontier_map.py:76` — Wall A arms differ in 3 things (actuator, A[C,C] 0.2→0.5, noise 0.05→1.0), not just actuatability. **[FIXING: match A[C,C]; note driver diff is intrinsic]**
10. `frontier_map.py` — threshold-lowering applied only to intervention, never observation → acting-vs-watching gap not shown threshold-independent. **[DEFER: add obs_low column]**
11. `frontier_map.py` — n=3 seeds, no CI, integer-fragile means ("≤+0.7 gate" is a mean of ~3 integers). **[DEFER: seeds≥10 + CI + gates=5 + non-Gaussian noise]**
12. `sheaf_confirm.py` — levels-DiD rests on untested parallel-trends/additive-separability; fails under saturating direct effects (not fired in tested linear worlds). **[DEFER: changes-in-changes / placebo reach]**
13. `sheaf_confirm.py` — empty-prefix DiD is ~0 (unconfirmable source); reachable(0.3) vs find_prereq(0.5) threshold mismatch. **[DEFER: minor, not triggered in shipped world]**
14. `sheaf_confirm.py` — dof label 2n−2 vs 4-group Welch (~4n−4); np.var ddof=0 understates SE. Negligible at n=60. **[DEFER: ddof=1 + Welch dof]**

Literature gap: Peters/Bühlmann/Meinshausen 2016 Bonferroni-correct per-env tests + use F-test with correct df; Heinze-Deml/Peters 2018 switch to nonparametric residual tests for exactly the non-Gaussian fragility we have. Same Gaussian-likelihood fragility (Schultheiss & Bühlmann 2022) likely undercuts the neural LSNM objective — next pass.

**Pass-1 fixes applied (commit 8c4ca12 + follow-up):** edge-level metric (2), retract 3-4x SNR (7), match A[C,C] (9), honest ICP {} docstring (5). **ICP calibration cluster now FIXED + tested:** Welch-t + Levene + Bonferroni over environments (findings 1/3/4/13/18); selftest_icp adds a calibration check — invariant-null reject flat 0.035/0.040 across n_env=2/32 (was 0.006→0.19). rest_cache setpoint-key bug (12) fixed.

---

## Pass 2 (2026-07-05) — INTENDED neural+active; ACTUALLY re-ran pass-1 modules (args-string bug in the audit script, since fixed). Became a 2nd independent adversarial audit → 19 findings, several NEW and claim-undermining.

**NEW, claim-undermining (surfaced immediately):**
- **rank 2 (frontier_map):** "acting beats watching" is a THRESHOLD asymmetry, not a recovery gap — observation at the same low (0.05) threshold reportedly recovers the deep gates too; the real win is ~2x PRECISION (~0.8 vs ~0.38). "Observation never recovers them" is unsupported. **[caveat added; obs_low re-measure deferred]**
- **rank 10 + 16 (frontier_map):** Wall A recall=0 is TAUTOLOGICAL — recover_structure_interventional only proposes sources from library.possible(), so an un-reachable C can never be a candidate; 0/2 is a design invariant, not a measurement. Arms also differ in C's scale/noise (2.25/0.05 vs 1.0/1.0). Only the reachable-arm precision (~0.8) is genuinely measured. **[caveat added; reframed as design property]**
- **rank 8 (sheaf_active):** "ensemble-EIG active inference (Pathak 2019)" is DECORATIVE — every gate-recovery number uses random_collect + frontier_collect (random exploration); EIG/disagreement lives only in sheaf_active.py, off the result path, and the ensemble shares init/data so disagreement can collapse. **[claim-attribution audit deferred]**
- **rank 5 (sheaf_confirm):** selftest_sheaf_confirm scores "0 false gates" at TARGET level — the exact defect frontier_map just retracted; edge-level it's ~0.5 spurious/run. **[rescore selftest_sheaf_confirm — deferred, needs a real run]**

**FIXED this pass:** ICP calibration cluster (1/3/4/13/14/18) via Welch+Levene+Bonferroni; rest_cache setpoint-key (12).
**Deferred (logged):** rank 6 sheaf_confirm source never interventionally identified (reach opens whole chain — needs a leave-one-out do(source:=0) control); rank 7 gate_sig necessary-not-sufficient (regress t~a+source+a:source, test interaction term); rank 11 no FWER/FDR over ≤80 candidates; rank 19 envelope mapped on ONE noise family (additive homoscedastic) — the multiplicative/heteroscedastic protein regime CLAUDE.md targets is UNTESTED.

**Process fix:** architecture_audit.js now parses args if delivered as a JSON string (was `typeof args === 'object'` → fell through to DEFAULT). Neural modules (neural_dag/neural_scale/neural_discovery) still UN-audited — re-queued.

Pass-2 next targets: neural GPU path (RESIT order-vs-edges; LSNM Gaussian-NLL wrong-direction, Schultheiss-Bühlmann 2022); planner._finals/reach (the reachability set that makes Wall A tautological; embodied single-trajectory autocorrelation, rank 17); sheaf_active EnsembleExperimenter disagreement-collapse; cross-module "active inference / observation never recovers" attribution audit.

---

## Pass 3 (2026-07-05) — planner + certify + neural. Completed 6/10 agents before SESSION LIMIT (resets 5:50pm PT); synthesize/2 verifies/lit failed. Findings salvaged from journal.

**certify.py (anytime-valid inference — CONFIRMED, serious):**
- **[HIGH] Overlapping-window martingale break (PRIMARY path).** `_reach_windows` emits a window every step (stride 1) once a hold-run reaches `horizon`, so a K-step hold yields K−horizon+1 windows sharing horizon−1 steps → breaks the test-martingale independence BettingCS relies on → intervals UNDERCOVER. `certify_library` + the live agent's recovery metric route every single-knob hold through this. **The anytime-valid certificate is invalid on its main path.**
- [MEDIUM] Validity caveat says "exchangeability" but the real requirement is the capital-process martingale / constant-conditional-mean; single-trajectory streams satisfy neither; positive serial dependence can inflate miscoverage well past alpha with no stated bound.
- [MEDIUM] `BettingCS.interval` fabricates a razor-thin interval when the CS is empty (martingale rejected everywhere — a dependence/misspecification signal) → spuriously confident POSSIBLE/IMPOSSIBLE verdict.
- [LOW] model-free stationary-reach: percentile bootstrap of a nonsmooth functional undercovers; 1/m absorbing-row fill biases pi.

**neural_dag.py (CONFIRMED) — docstring CORRECTED this pass:** "connected-DAG recovery" recovers only a topological ORDER (no adjacency); order_accuracy iterates over ground-truth edges only → cannot see false positives. "Break the multiplicative-noise wall classical ANM can't" is NEVER measured (baseline = random permutation only). Graded vs chance, not varsortability/sortnregress (Reisach 2021).

**neural_scale.py (CONFIRMED) — docstring CORRECTED this pass:** all noise Gaussian → multiplicative regime is EXACTLY the conditionally-Gaussian LSNM best case; never tested under misspecification where Gaussian-LSNM fails. `orient_flow` uses likelihood-ratio orientation with a fixed Gaussian base — the fragile ML rule (Immer 2023). Single-direction benchmark (always x→y) can't distinguish skill from directional bias.

**planner.py (HIGH):**
- **[HIGH] verify()/reliability is near-tautological** — each effect box is mean ±4σ of the SAME skill's own output, then "reliability" = fraction of fresh trials landing in that same ±4σ box. "possible" is decided by characterize_effect's own filters, not an independent test. EVERY gate-recovery number rests on this library. **[needs an independent reliability criterion — deferred]**
- [HIGH] Wall A recall=0 is structural (CONFIRMS pass-2 rank 10 — caveat already added).
- [MEDIUM] confirm_gate max|DiD|-over-sensors with no multiplicity correction = winner's curse (CONFIRMS pass-2 rank 11).
- [LOW] _finals_live dependent single-trajectory samples treated as i.i.d. by Wilson/naive-SE (anti-conservative for slow-mixing worlds).

**FIXED this pass:** neural_dag + neural_scale docstring overclaims corrected.
**Deferred (serious, need careful code + re-verification, session-limited):** certify overlapping-window fix (stride=horizon non-overlapping windows) + empty-CS interval guard; planner independent-reliability criterion; selftest_sheaf_confirm edge-rescore (finding 5).

**LOOP STATUS: paused by session limit (resets 5:50pm PT).** Multi-agent audit passes cannot spawn until reset. Continuing with direct main-agent fixes meanwhile.

