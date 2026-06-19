"""
Knowledge-prior fusion: a causal_dag RAG/LLM oracle -> a belief seed for the
constructor_causal agent.

The honest value proposition (constructor_causal already gets edge *direction* for
free, because it intervenes — there is no Markov-equivalence gap to fill here):

  * The prior ASSERTS edges it is confident about. Those edges count toward the
    recovered graph immediately, with zero experiments — so the fused belief reaches
    a good F1 far sooner than experiments alone.
  * Interventions are GROUND TRUTH and OVERRIDE the prior: an asserted edge whose
    reverse the model confidently recovers is dropped. A wrong prior costs a little
    early precision, never a permanent error (the soft-prior safety property,
    mirroring causal_dag's NOTEARS reverse-edge penalty and the interventional
    override in causal_dag.agent).
  * The prior also concentrates the experiment budget on knobs whose causal role it
    could NOT pin down (see actuators_feeding_unsure_sensors), and seeds localized
    re-exploration after drift.

Convention bridge (critical): constructor_causal uses (source, target) edge tuples
and `world.A[target, source]`; causal_dag's KnowledgeOracle uses `B[i,j]=1 => i->j`
(i.e. B[source, target]). `world_to_B` performs the transpose so both agree that
B[source, target] = 1.
"""
from __future__ import annotations

import numpy as np

from causal_dag.rag import KnowledgeOracle
from .model import _t_crit


def world_to_B(world) -> np.ndarray:
    """constructor_causal world -> oracle adjacency B[source, target] = 1.
    world.true_edges() yields (j, i) meaning j->i (source j, target i)."""
    d = world.d
    B = np.zeros((d, d), int)
    for (j, i) in world.true_edges():
        B[j, i] = 1
    return B


class CausalPrior:
    """A knowledge prior over a constructor_causal world's edges.

    Edges are stored in constructor_causal (source, target) convention to match
    model.recovered_edges() and world.true_edges(). `_asserted` are the edges the
    prior is confident about; `_unsure_acts` are actuators whose role it could not pin.
    """

    def __init__(self, asserted: set, observed, actuators, sensors):
        self._asserted = set(asserted)
        self.observed = tuple(observed)
        self.actuators = tuple(actuators)
        self.sensors = tuple(sensors)
        # an actuator is "unsure" if the prior never asserts it as a cause (source)
        confident_sources = {s for (s, t) in self._asserted if s in self.actuators}
        self._unsure_acts = tuple(a for a in self.actuators if a not in confident_sources)

    # ---- construction paths -------------------------------------------------
    @classmethod
    def from_oracle(cls, world, accuracy=0.9, abstain=0.1, rng=None) -> "CausalPrior":
        """SIMULATED path (mechanism demo): a calibrated KnowledgeOracle over the
        world's true edges. accuracy/abstain are the knobs."""
        B = world_to_B(world)
        oracle = KnowledgeOracle(B, accuracy=accuracy, abstain=abstain, rng=rng)
        obs = world.observed
        asserted = set()
        for ia in range(len(obs)):
            for ib in range(ia + 1, len(obs)):
                a, b = obs[ia], obs[ib]
                ans = oracle.orient_edge(a, b)          # +1: a->b, -1: b->a, 0: none
                if ans > 0:
                    asserted.add((a, b))
                elif ans < 0:
                    asserted.add((b, a))
        return cls(asserted, world.observed, world.actuators, world.sensors)

    @classmethod
    def from_llm_answers(cls, world, answers: dict) -> "CausalPrior":
        """REAL-LLM path. `answers` maps 'NameA -- NameB' -> the chosen cause NAME
        (the exact format causal_dag/llm_experiment.py uses). Only true adjacencies
        are asked; a chosen cause name orients that edge."""
        idx = {nm: i for i, nm in enumerate(world.names)}
        asserted = set()
        for key, cause in answers.items():
            a_name, b_name = [s.strip() for s in key.split("--")]
            if cause not in (a_name, b_name):
                continue
            c = idx[cause]
            e = idx[b_name] if cause == a_name else idx[a_name]
            asserted.add((c, e))                        # (source cause -> target effect)
        return cls(asserted, world.observed, world.actuators, world.sensors)

    @classmethod
    def from_true_edges(cls, world) -> "CausalPrior":
        """A PERFECT prior (ground truth) — stands in for a flawless expert; used to
        upper-bound the prior's value and to CI-test the real-LLM path offline."""
        return cls(set(world.true_edges()), world.observed, world.actuators, world.sensors)

    # ---- what the agent / scorer consume ------------------------------------
    def asserted_edges(self) -> set:
        return set(self._asserted)

    def actuators_feeding_unsure_sensors(self):
        """Actuators whose causal role the prior could not pin down -> where the
        experimenter should spend its budget. Empty -> caller falls back to all."""
        return self._unsure_acts

    def _confidently_absent(self, model, src, tgt, z, min_effect) -> bool:
        """True if the model has enough POWER on src->tgt to say the edge is absent:
        the upper confidence bound on |coefficient| is below the minimum meaningful
        effect. (|mean| < eps alone is not enough -- an UNEXCITED pair also has small
        mean; we require small mean AND small std, i.e. genuine evidence of absence.)"""
        if tgt not in model.sensors or src not in model.cols or src == tgt:
            return False
        mean, Cov = model._posterior(tgt)
        k = model._lin_k(src)
        std = float(np.sqrt(max(Cov[k, k], 1e-12)))
        crit = _t_crit(z, model._dof[tgt])
        return abs(float(mean[k])) + crit * std < min_effect

    def fused_edges(self, model, z: float = 3.0, eps: float = 0.05,
                    min_effect: float = 0.2) -> set:
        """The agent's effective belief = prior assertions (minus those the agent
        overrode) UNION the model's confidently-recovered edges. Two overrides:
          - wrong DIRECTION: drop asserted (s,t) if the model recovers (t,s);
          - ABSENT: drop asserted (s,t) once the agent has POWER on that pair and
            confidently rejects it (so a spurious NON-adjacent prior edge cannot
            persist forever -- the previous version only caught wrong-direction)."""
        rec = model.recovered_edges(z=z, eps=eps)
        overridden = set()
        for (s, t) in self._asserted:
            if (t, s) in rec:                                          # real reverse edge
                overridden.add((s, t))
            elif self._confidently_absent(model, s, t, z, min_effect):  # confidently no edge
                overridden.add((s, t))
        return (self._asserted - overridden) | rec

    def fused_scores(self, model, true_edges: set, z: float = 3.0, eps: float = 0.05) -> dict:
        rec = self.fused_edges(model, z=z, eps=eps)
        tp = len(rec & true_edges)
        prec = tp / len(rec) if rec else (1.0 if not true_edges else 0.0)
        recall = tp / len(true_edges) if true_edges else 1.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        return {"precision": prec, "recall": recall, "f1": f1,
                "recovered": rec, "missing": true_edges - rec, "extra": rec - true_edges}


def raw_orientation_accuracy(world, answers: dict) -> dict:
    """Score an LLM answer set's raw pairwise orientation accuracy on the world's
    TRUE edges (the Kıcıman-style measurement)."""
    idx = {nm: i for i, nm in enumerate(world.names)}
    true_e = world.true_edges()
    correct = total = 0
    for key, cause in answers.items():
        a_name, b_name = [s.strip() for s in key.split("--")]
        a, b = idx[a_name], idx[b_name]
        if (a, b) not in true_e and (b, a) not in true_e:
            continue                                    # not a real adjacency
        total += 1
        c = idx.get(cause)
        e = b if c == a else a
        if (c, e) in true_e:
            correct += 1
    return {"correct": correct, "total": total,
            "accuracy": correct / total if total else float("nan")}
