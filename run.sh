#!/usr/bin/env bash
# Convenience runner for the ChemicalWorld causal-AI stacks.
# ALWAYS run from this directory:  cd /Users/macuser/Downloads/ChemicalWorld && ./run.sh <target>
#   chmod +x run.sh   (once)   then   ./run.sh <target>
# Examples:  ./run.sh deploy   ./run.sh fusion   ./run.sh tests   ./run.sh all
set -e
cd "$(dirname "$0")"            # so it works no matter where you call it from
PY=.venv/bin/python

case "${1:-help}" in

  # ---- constructor_causal: long-horizon deployment + knowledge-prior fusion ----
  deploy)   $PY -m constructor_causal.demo_deploy ;;              # continual vs discover-once + localized
  fusion)   $PY -m constructor_causal.demo_prior ;;               # RAG/LLM prior: F1 vs budget by accuracy
  llm-prep) $PY -m constructor_causal.demo_prior prepare ;;       # print orientation questions for a real LLM
  llm-score) shift; $PY -m constructor_causal.demo_prior score "$1" ;;  # ./run.sh llm-score '<answers json>'
  llm-stress) PYTHONPATH=. $PY -m causal_dag.llm_stress ;;        # stress test: anonymized + confounded decoys
  llm-stress-score) shift; PYTHONPATH=. $PY -m causal_dag.llm_stress score "$1" ;;

  # ---- constructor_causal: the original reward-free agent ----
  demo)     $PY -m constructor_causal.demo ;;
  autonomous) $PY -m constructor_causal.demo_autonomous ;;
  aleatoric) PYTHONPATH=. $PY -m constructor_causal.demo_aleatoric ;;  # heteroscedastic head vs action-dependent noisy TV
  gated)    PYTHONPATH=. $PY -m constructor_causal.demo_gated ;;        # gated edge + budget: harder active inference
  deep-chain) PYTHONPATH=. $PY -m constructor_causal.demo_deep_chain ;; # deep delayed chain: sustained/sequenced interventions

  # ---- causal_dag: DAG discovery / RAG / the interventional agent ----
  validate)   PYTHONPATH=. $PY causal_dag/validate.py ;;          # 6-point correctness gate
  experiment) PYTHONPATH=. $PY causal_dag/experiment.py ;;        # scaling studies A-E
  agent)      PYTHONPATH=. $PY -m causal_dag.demo_agent ;;        # interventional agent vs PC
  paper)      PYTHONPATH=. $PY -m causal_dag.make_paper ;;        # regenerate causal_dag/PAPER.pdf

  # ---- Wave 1: caveat-fixing credibility studies ----
  hard-regime) PYTHONPATH=. $PY -m causal_dag.hard_regime ;;       # recovery falloff: weak edges/noise/few n
  confounders) PYTHONPATH=. $PY -m causal_dag.confounders ;;       # multi-confounder de-confounding
  benchmarks)  PYTHONPATH=. $PY -m causal_dag.benchmarks ;;        # Sachs/asia/... real topologies

  # ---- Wave 2: real data, external tool, real retrieval ----
  realdata)    PYTHONPATH=. $PY -m causal_dag.realdata ;;          # PC/NOTEARS on REAL Sachs cytometry
  compare)     PYTHONPATH=. $PY -m causal_dag.compare_causal_learn ;;  # vs causal-learn PC/GES, same data
  rag)         PYTHONPATH=. $PY -m causal_dag.llm_rag ;;           # retrieval-grounded orientation (prepare)
  rag-score)   shift; PYTHONPATH=. $PY -m causal_dag.llm_rag score "$1" ;;

  # ---- Wave 3: nonlinear, scale, unification ----
  nonlinear)   PYTHONPATH=. $PY -m causal_dag.nonlinear_discovery ;;  # kernel(HSIC) PC vs Fisher-z PC
  scaling)     PYTHONPATH=. $PY -m causal_dag.scaling ;;          # DAGMA to d=200
  unified)     PYTHONPATH=. $PY -m causal_dag.demo_unified ;;     # one API over the whole toolkit

  # ---- Wave 4: knowledge beyond the training cutoff (temporal holdout) ----
  temporal)    PYTHONPATH=. $PY -m causal_dag.temporal ;;        # 2026 post-cutoff causal facts (prepare)
  temporal-score) shift; PYTHONPATH=. $PY -m causal_dag.temporal score "$1" ;;
  fiction)     PYTHONPATH=. $PY -m causal_dag.fiction ;;          # post-cutoff FICTIONAL world (bias-free)
  fiction-score) shift; PYTHONPATH=. $PY -m causal_dag.fiction score "$1" ;;

  # ---- verification: every selftest in both packages ----
  tests)
    for s in selftest selftest_advanced selftest_frontier selftest_frontier2 \
             selftest_continual selftest_autonomous selftest_deploy selftest_prior \
             selftest_aleatoric selftest_gated selftest_deep_chain selftest_dag_adapter \
             selftest_actuator_target selftest_sheaf_confirm selftest_frontier_map selftest_icp; do
      printf "  %-26s " "$s"; $PY -m constructor_causal.$s 2>&1 | grep -oE "[0-9]+/[0-9]+ checks passed" | tail -1
    done
    for s in selftest_agent selftest_hard_regime selftest_confounders selftest_benchmarks \
             selftest_retrieval selftest_realdata selftest_nonlinear selftest_scaling \
             selftest_unified selftest_temporal selftest_fiction selftest_sid; do
      printf "  %-26s " "causal_dag.$s"; PYTHONPATH=. $PY -m causal_dag.$s 2>&1 | grep -oE "[0-9]+/?[0-9]* checks passed" | tail -1
    done
    for s in selftest_hidden_cause_sachs selftest_edge_eprocess; do   # root-level: hidden-cause + Phase 0 e-process
      printf "  %-26s " "$s"; PYTHONPATH=. $PY $s.py 2>&1 | grep -oE "[0-9]+/[0-9]+ checks passed" | tail -1
    done
    ;;

  all) "$0" deploy; "$0" fusion; "$0" tests ;;

  *) echo "targets: deploy | fusion | llm-prep | llm-score '<json>' | llm-stress | demo | autonomous"
     echo "         validate | experiment | agent | paper | hard-regime | confounders | benchmarks | tests | all" ;;
esac
