"""
constructor_causal -- learning the causal algebra of a world, without reward.

A small, self-contained system that fuses four ideas:

  * Causal DAGs / SCMs   (the form of the world and of the agent's belief)
  * Causal inference     (interventions identify cause vs. mere correlation)
  * Active inference      (act to minimise expected free energy -- here, only its
                          epistemic term: maximise expected information gain)
  * Constructor Theory    (interventions are repeatable constructors; chaining
                          them composes bigger constructors; the agent grows an
                          algebra of reliably-achievable transformations)

The headline: the agent is dropped into a world it knows nothing about, driven
purely by curiosity (no reward, no goals), recovers the causal graph, distils a
library of composable constructors, and only *afterwards* -- when handed an
arbitrary goal -- composes that library to achieve it.

Entry points:
    python -m constructor_causal.demo        # narrated end-to-end run
    python -m constructor_causal.selftest    # assertions that each claim holds
"""
import os as _os
# Pin BLAS to one thread BEFORE numpy is imported: this workload is thousands of
# tiny (≤7×7) matrix inversions, where multi-threaded BLAS spends all its time on
# thread overhead. Pinning gives a ~15× speedup here. (No effect if numpy was
# already imported by the host process.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

# The full research stack is imported tolerantly: the lightweight, portable modules
# (device, bigworld, transformer_opponent) must import with ONLY numpy+torch, so a
# fresh GPU box without the research deps (causal_dag, scipy, sklearn, ...) can still
# run `python -m constructor_causal.transformer_opponent`. When the heavy deps ARE
# present (local dev), this block runs exactly as before.
try:
    from .world import DynamicalCausalWorld
    from .model import BayesianDynamicsModel, edge_scores
    from .constructor import (Box, Constructor, Library, compose, estimate_reliability,
                              POSSIBLE_TAU, MIN_TRIALS)
    from .active_inference import (EpistemicExperimenter, CertifyingExperimenter,
                                   RandomExperimenter, NaiveSurpriseExperimenter,
                                   PassiveExperimenter)
    from .planner import ConstructorSynthesizer
    from .agent import ConstructorCausalAgent, discover_actuators
    from .causal_graph import CausalGraph, build_causal_graph
    from .certify import (AnytimeCS, BettingCS, DriftDetector, certify_reliability,
                          certify_passive, calibrate_passive, certify_modelfree,
                          certify_modelfree_continuous, certify_modelfree_reach,
                          certify_library, detect_latent_lag)
    from .deploy import (RegimeSchedule, RoundMetrics, deploy, deploy_baseline,
                         measure_localized_saving)
    from .prior import CausalPrior, world_to_B
    from .semantic_worlds import heater_world, tank_world
except ImportError as _e:  # portable path: research deps absent (e.g. fresh GPU box)
    import warnings as _w
    _w.warn(f"constructor_causal: research stack not fully importable ({_e}); "
            "lightweight modules (device, bigworld, transformer_opponent) still work.")

__all__ = [
    "DynamicalCausalWorld", "BayesianDynamicsModel", "edge_scores",
    "Box", "Constructor", "Library", "compose", "estimate_reliability",
    "POSSIBLE_TAU", "MIN_TRIALS",
    "EpistemicExperimenter", "CertifyingExperimenter", "RandomExperimenter",
    "NaiveSurpriseExperimenter", "PassiveExperimenter",
    "ConstructorSynthesizer", "ConstructorCausalAgent",
    "discover_actuators", "CausalGraph", "build_causal_graph",
    "AnytimeCS", "BettingCS", "DriftDetector",
    "certify_reliability", "certify_passive", "calibrate_passive",
    "certify_modelfree", "certify_modelfree_continuous", "certify_modelfree_reach",
    "certify_library", "detect_latent_lag",
    # long-horizon deployment + knowledge-prior fusion
    "RegimeSchedule", "RoundMetrics", "deploy", "deploy_baseline",
    "measure_localized_saving", "CausalPrior", "world_to_B",
    "heater_world", "tank_world",
]
