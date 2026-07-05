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

