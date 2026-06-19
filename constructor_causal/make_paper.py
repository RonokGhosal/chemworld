"""
Generate PAPER.pdf (and PAPER.md) for the constructor_causal project.

Pure-matplotlib paginated layout engine (no LaTeX / pandoc needed). Authoring is
done once as a list of structured blocks; we render to a multi-page PDF with inline
figures + mathtext equations, and serialize the same blocks to Markdown.

Run:  ./.venv/bin/python -m constructor_causal.make_paper
"""
from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
PW, PH = 8.5, 11.0
ML, MR, MT, MB = 0.92, 0.92, 0.95, 0.85
TW = PW - ML - MR

INK = "#1a1a1a"
ACCENT = "#26456e"
MUTED = "#5b5b5b"
BOX = "#eef2f7"
BOXE = "#9bb3d1"


# ----------------------------------------------------------------------------- figures
def fig_loop(fig, rect):
    l, b, w, h = rect
    ax = fig.add_axes([l, b, w, h]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    nodes = [
        (1.5, 4.4, "WATCH\nread state"),
        (4.0, 4.4, "MODEL\nBayesian causal\nbelief + memory"),
        (6.5, 4.4, "CHOOSE\nmax expected\ninfo gain"),
        (8.6, 4.4, "ACT\ndo(x=v)\nintervene"),
    ]
    for (x, y, t) in nodes:
        ax.add_patch(FancyBboxPatch((x - 0.95, y - 0.7), 1.9, 1.4,
                     boxstyle="round,pad=0.06,rounding_size=0.12",
                     fc=BOX, ec=BOXE, lw=1.2))
        ax.text(x, y, t, ha="center", va="center", fontsize=8, color=INK)
    for x0, x1 in [(2.45, 3.05), (4.95, 5.55), (7.45, 7.65)]:
        ax.add_patch(FancyArrowPatch((x0, 4.4), (x1, 4.4), arrowstyle="-|>",
                     mutation_scale=12, color=ACCENT, lw=1.4))
    # feedback arrow act -> watch (the loop)
    ax.add_patch(FancyArrowPatch((8.6, 3.6), (1.5, 3.6), arrowstyle="-|>",
                 mutation_scale=12, color=ACCENT, lw=1.4,
                 connectionstyle="arc3,rad=0.28"))
    ax.text(5.0, 2.35, "repeat — no reward, no goal: only resolve uncertainty",
            ha="center", va="center", fontsize=8.5, style="italic", color=MUTED)
    # downstream products
    ax.add_patch(FancyBboxPatch((2.0, 0.5), 6.0, 1.0,
                 boxstyle="round,pad=0.06,rounding_size=0.12", fc="#f3efe7", ec="#cbb98f", lw=1.1))
    ax.text(5.0, 1.0, "by-products:  causal map   ·   constructor library   ·   \"what if I do X?\"",
            ha="center", va="center", fontsize=8.5, color=INK)


def fig_results(fig, rect):
    l, b, w, h = rect
    pad = 0.012
    cw = w / 3.0
    ax1 = fig.add_axes([l + pad, b + 0.10 * h, cw - 2 * pad, 0.74 * h])
    ax2 = fig.add_axes([l + cw + pad, b + 0.10 * h, cw - 2 * pad, 0.74 * h])
    ax3 = fig.add_axes([l + 2 * cw + pad, b + 0.10 * h, cw - 2 * pad, 0.74 * h])

    ax1.bar(["curiosity", "random", "naive"], [0.88, 0.84, 0.82],
            color=[ACCENT, "#b0b0b0", "#c8956b"])
    ax1.set_ylim(0, 1.0); ax1.set_title("DAG recovery F1\n(hard world, budget 16)", fontsize=8)
    ax1.tick_params(labelsize=7); ax1.axhline(0.84, color="#bbb", lw=0.6, ls="--")

    from matplotlib.ticker import NullFormatter, ScalarFormatter
    bars = ax2.bar(["BFS", "informed"], [314, 13], color=["#b0b0b0", ACCENT])
    ax2.set_title("planner nodes expanded\n(wide world)", fontsize=8)
    ax2.set_yscale("log"); ax2.set_ylim(8, 500); ax2.tick_params(labelsize=7)
    ax2.set_yticks([10, 100]); ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.yaxis.set_minor_formatter(NullFormatter())
    for r, v in zip(bars, [314, 13]):
        ax2.text(r.get_x() + r.get_width() / 2, v * 1.05, str(v), ha="center", va="bottom", fontsize=7)

    bars = ax3.bar(["linear", "RFF"], [0.65, 0.03], color=["#b0b0b0", ACCENT])
    ax3.set_title("'what-if' error on a\nsaturating edge (RMSE)", fontsize=8)
    ax3.set_ylim(0, 0.75); ax3.tick_params(labelsize=7)
    for r, v in zip(bars, [0.65, 0.03]):
        ax3.text(r.get_x() + r.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)


def fig_continual(fig, rect):
    l, b, w, h = rect
    pad = 0.015
    ax1 = fig.add_axes([l + pad, b + 0.13 * h, w / 2 - 2 * pad, 0.70 * h])
    ax2 = fig.add_axes([l + w / 2 + pad, b + 0.13 * h, w / 2 - 2 * pad, 0.70 * h])
    regimes = ["R1", "R2", "R3"]
    true_w = [0.80, -0.80, 1.30]
    rec_w = [0.80, -0.80, 1.30]
    x = range(3)
    ax1.axhline(0, color="#bbb", lw=0.6)
    ax1.plot(x, true_w, "o--", color="#b0b0b0", label="true", ms=5)
    ax1.plot(x, rec_w, "o-", color=ACCENT, label="recovered", ms=4)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(regimes, fontsize=7)
    ax1.set_title("edge a0->chain1 tracked\nacross regimes", fontsize=8)
    ax1.tick_params(labelsize=7); ax1.legend(fontsize=6, loc="lower right")
    spike = [2.93, 4.17]; settled = [0.05, 0.02]
    xb = [0, 1]
    ax2.bar([i - 0.18 for i in xb], spike, width=0.36, color="#c8956b", label="at change")
    ax2.bar([i + 0.18 for i in xb], settled, width=0.36, color=ACCENT, label="re-settled")
    ax2.set_xticks(xb); ax2.set_xticklabels(["flip", "change"], fontsize=7)
    ax2.set_title("chain1 prediction error\n(change detection)", fontsize=8)
    ax2.tick_params(labelsize=7); ax2.legend(fontsize=6)


def fig_ladder(fig, rect):
    l, b, w, h = rect
    ax = fig.add_axes([l, b, w, h]); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    steps = [(1.6, "C1  primitive\nhold a0  →  gate1"),
             (5.0, "C2  conditional\n(a1 | gate1) → gate2"),
             (8.4, "C3  conditional\n(a2 | gate2) →  Z")]
    for (x, t) in steps:
        ax.add_patch(FancyBboxPatch((x - 1.25, 1.8), 2.5, 1.3,
                     boxstyle="round,pad=0.05,rounding_size=0.1", fc=BOX, ec=BOXE, lw=1.2))
        ax.text(x, 2.45, t, ha="center", va="center", fontsize=8, color=INK)
    for x0, x1 in [(2.95, 3.7), (6.35, 7.1)]:
        ax.add_patch(FancyArrowPatch((x0, 2.45), (x1, 2.45), arrowstyle="-|>",
                     mutation_scale=13, color=ACCENT, lw=1.5))
    ax.text(5.0, 0.85, r"composite:  C1 $\gg$ C2 $\gg$ C3   reaches Z   (reliability 1.00)",
            ha="center", va="center", fontsize=9, color=ACCENT)
    ax.text(5.0, 0.25, "neither knob alone — and no single knob repeated — can move Z",
            ha="center", va="center", fontsize=8, style="italic", color=MUTED)


FIGS = {"loop": (fig_loop, 2.7, "Figure 1. The reward-free perception–action loop. "
                 "Prediction lives inside the loop as the tool for choosing experiments; "
                 "the products are the causal map and the constructor library."),
        "results": (fig_results, 2.5, "Figure 2. Three headline results, each reward-free: "
                    "the objective matters under scarcity (left); the informed planner scales "
                    "(center); a nonlinear basis fixes counterfactuals (right)."),
        "ladder": (fig_ladder, 2.3, "Figure 3. Constructors compose into bigger constructors. "
                   "Conditional skills stack bottom-up; the planner chains three distinct ones "
                   "to crack a two-gate cascade."),
        "continual": (fig_continual, 2.4, "Figure 4. Continual learning. The belief tracks a "
                      "changing causal edge across three regimes (left); one-step prediction error "
                      "spikes at each change and re-settles after the agent re-learns (right). "
                      "Broken skills are pruned and rebuilt each time.")}


# ----------------------------------------------------------------------------- PDF engine
class Doc:
    def __init__(self, path):
        self.pdf = PdfPages(path)
        self.page = 0
        self._new()

    def _new(self):
        self.fig = plt.figure(figsize=(PW, PH))
        self.y = PH - MT
        self.page += 1

    def _save(self):
        self.fig.text(PW / 2 / PW, 0.42 / PH, f"Constructor-Causal  ·  p.{self.page}",
                      ha="center", va="center", fontsize=7.5, color=MUTED)
        self.pdf.savefig(self.fig); plt.close(self.fig)

    def _brk(self, need):
        if self.y - need < MB:
            self._save(); self._new()

    def _text(self, x, s, size, color=INK, weight="normal", style="normal", ha="left"):
        self.fig.text(x / PW, self.y / PH, s, fontsize=size, color=color,
                      ha=ha, va="top", weight=weight, style=style)

    def gap(self, dh):
        self.y -= dh

    def lines(self, s, size, wrap, color=INK, weight="normal", style="normal",
              x=ML, lead=1.5, ha="left"):
        lh = size / 72.0 * lead
        for raw in s.split("\n"):
            for w in (textwrap.wrap(raw, wrap) or [""]):
                self._brk(lh)
                self._text(x, w, size, color, weight, style, ha)
                self.y -= lh

    def title(self, s):
        self.lines(s, 19, 52, color=INK, weight="bold"); self.gap(0.05)

    def byline(self, s):
        self.lines(s, 10, 100, color=MUTED, style="italic"); self.gap(0.18)

    def h1(self, s):
        self._brk(0.5); self.gap(0.12)
        self.lines(s, 13.5, 70, color=ACCENT, weight="bold"); self.gap(0.05)

    def h2(self, s):
        self._brk(0.4); self.gap(0.07)
        self.lines(s, 11, 80, color=INK, weight="bold"); self.gap(0.03)

    def p(self, s):
        self.lines(s, 9.4, 104, color=INK); self.gap(0.085)

    def bullet(self, items):
        for it in items:
            lh = 9.4 / 72.0 * 1.5
            wrapped = textwrap.wrap(it, 98) or [""]
            self._brk(lh)
            self._text(ML, "•", 9.4, ACCENT)
            self._text(ML + 0.18, wrapped[0], 9.4, INK)
            self.y -= lh
            for cont in wrapped[1:]:
                self._brk(lh); self._text(ML + 0.18, cont, 9.4, INK); self.y -= lh
        self.gap(0.07)

    def code(self, s):
        self.gap(0.02)
        for raw in s.split("\n"):
            for w in (textwrap.wrap(raw, 96, subsequent_indent="    ") or [""]):
                lh = 8.0 / 72.0 * 1.5
                self._brk(lh)
                self._text(ML + 0.05, w, 8.0, "#243b4a", ha="left")
                self.fig.patches  # noqa
                self.y -= lh
        self.gap(0.08)

    def eq(self, latex):
        size = 11.5
        lh = size / 72.0 * 2.0
        self._brk(lh + 0.05)
        self.gap(0.04)
        self._text(PW / 2, latex, size, color=INK, ha="center")
        self.y -= lh

    def table(self, headers, rows, widths):
        size = 8.4
        lh = size / 72.0 * 1.7
        xcols = [ML]
        for wfrac in widths[:-1]:
            xcols.append(xcols[-1] + wfrac * TW)
        self._brk(lh * (len(rows) + 1) + 0.1)
        for c, head in enumerate(headers):
            self._text(xcols[c], head, size, ACCENT, weight="bold")
        self.y -= lh
        for row in rows:
            # wrap each cell, advance by max lines
            cellwraps = []
            for c, cell in enumerate(row):
                charw = max(6, int(widths[c] * TW / (size / 72.0 * 0.6)))
                cellwraps.append(textwrap.wrap(cell, charw) or [""])
            nlines = max(len(cw) for cw in cellwraps)
            self._brk(lh * nlines)
            for ln in range(nlines):
                for c, cw in enumerate(cellwraps):
                    if ln < len(cw):
                        self._text(xcols[c], cw[ln], size, INK)
                self.y -= lh
        self.gap(0.12)

    def figure(self, name):
        draw, hin, caption = FIGS[name]
        need = hin + 0.5
        self._brk(need)
        self.gap(0.05)
        rect = [ML / PW, (self.y - hin) / PH, TW / PW, hin / PH]
        draw(self.fig, rect)
        self.y -= hin
        self.gap(0.04)
        self.lines(caption, 8.2, 118, color=MUTED, style="italic")
        self.gap(0.12)

    def close(self):
        self._save(); self.pdf.close()


# ----------------------------------------------------------------------------- content
def build(doc: Doc, md: list):
    def P(s): doc.p(s); md.append(s + "\n")
    def H1(s): doc.h1(s); md.append("\n## " + s + "\n")
    def H2(s): doc.h2(s); md.append("\n### " + s + "\n")
    def B(items): doc.bullet(items); md.extend("- " + i for i in items); md.append("")
    def EQ(latex, mdtext): doc.eq(latex); md.append("\n> " + mdtext + "\n")
    def CODE(s): doc.code(s); md.append("```\n" + s + "\n```")
    def TAB(h, r, w): doc.table(h, r, w); md.append(_md_table(h, r))
    def FIG(n): doc.figure(n); md.append(f"\n*{FIGS[n][2]}*\n")

    doc.title("Learning the Causal Algebra of a World, Without Reward")
    doc.byline("Constructors  ·  Causal DAGs  ·  Active Inference  —  a method demonstrator and a "
               "roadmap toward a self-continual learning agent")
    doc.byline("R. Ghosal  ·  ChemicalWorld / constructor_causal  ·  June 2026")
    md.append("# Learning the Causal Algebra of a World, Without Reward")
    md.append("*Constructors · Causal DAGs · Active Inference. R. Ghosal, June 2026.*\n")

    H1("Abstract")
    P("We build an agent that is dropped into a world it knows nothing about and figures out the "
      "cause-and-effect rules by running its own experiments — with no reward, no goals, and no "
      "labels. Its only drive is to reduce uncertainty about how the world works. The system fuses "
      "four ideas: causal DAGs (the form of its belief), causal inference (it intervenes, so it "
      "learns causation not just correlation), active inference (it picks experiments that teach it "
      "the most), and Constructor Theory (it distils what it learns into repeatable, composable "
      "\"constructors\" — transformations it can reliably perform — and grows them into bigger ones). "
      "The thesis, demonstrated across eight test worlds and 65 falsifiable checks, is simple: "
      "delete the reward term from active inference and the agent still has a complete objective. "
      "Capability — reaching goals it was never trained on, and answering \"what if I do X?\" — falls "
      "out of understanding as a by-product. Driven by a self-audit, we then make the agent autonomous: "
      "it discovers its own interface (which variables are its knobs), flags a hidden cause vs. noise, "
      "does fine continuous control, and runs a self-driven continual loop that detects when the world "
      "changes -- both a flipped edge and a newly-appearing gate -- and re-learns with no external "
      "schedule. We describe the implementation, the results (65 falsifiable checks), what they tell "
      "us, what we deliberately did NOT fake-fix, and the remaining path to a fully self-continual agent.")

    doc.figure("loop")
    md.append(f"\n*{FIGS['loop'][2]}*\n")

    H1("1.  The idea, in one paragraph")
    P("Most learning agents chase a reward. But a child poking at a new toy is not maximising a "
      "score — it is building a model of what the toy does. That is the regime we target. Formally, "
      "active inference says an agent should minimise its Expected Free Energy, which splits into two "
      "parts:")
    EQ(r"$G(\pi)\;\;=\;\;-\,\mathrm{(expected\ info\ gain)}\;\;-\;\;\mathrm{(expected\ reward)}$",
       "G(pi) = -(info gain) - (reward).   Left term = epistemic (we keep it); "
       "right term = pragmatic (we delete it).")
    P("Our claim is that the second term is optional. Strike it, and the agent is left with a pure "
      "epistemic drive: act so as to learn the most about the world's causal mechanism. Everything "
      "else in the system is machinery in service of computing that one quantity well and turning "
      "what it learns into things the agent can do.")

    H1("2.  Background, briefly")
    H2("Causal DAGs and interventions")
    P("A structural causal model (SCM) describes a world as variables on a directed acyclic graph, "
      "where each variable is a function of its parents plus noise. The crucial operation is the "
      "intervention do(X=v): you reach in and force a variable to a value, severing the arrows that "
      "normally set it. Interventions are what separate causation from correlation — two variables "
      "can move together because one causes the other, or because a third thing drives both; only "
      "forcing one and watching the other tells them apart.")
    H2("Constructor Theory")
    P("Constructor Theory (Deutsch & Marletto) reframes physics around which transformations are "
      "possible, rather than around states evolving under laws. A task is a transformation specified "
      "by its endpoints (an input region of state space and an output region). A constructor is a "
      "thing that, presented with the input, reliably produces the output and is left able to do it "
      "again — it is repeatable, like a catalyst. A task is possible if a constructor for it can be "
      "built to arbitrarily high reliability. The bridge to causal inference is the heart of this "
      "project: an intervention do(X=v) IS the simplest constructor, and a sequence of interventions "
      "is a composite one.")

    H1("3.  Architecture")
    P("The system is six cooperating parts. Probabilistic reasoning handles inference and choice; "
      "deterministic code handles execution and verification.")
    B(["World (world.py) — the hidden ground-truth dynamical SCM. Some variables are actuators the "
       "agent may force; the rest are sensors it can only move through the dynamics.",
       "Model (model.py) — the agent's Bayesian belief about the dynamics; its posterior IS the "
       "recovered causal graph, and it yields the information-gain signal in closed form.",
       "Experimenter (active_inference.py) — reward-free action selection by expected information "
       "gain (the epistemic term above).",
       "Constructors (constructor.py) — Constructor Theory made operational: tasks as boxes, "
       "constructors as verified programs, and a composition algebra.",
       "Synthesiser/Planner (planner.py) — distils constructors from the belief and composes them "
       "to reach goals.",
       "Agent (agent.py) — the loop that ties it together: explore, discover, distil, compose, "
       "predict."])

    H1("4.  How it works (implementation)")
    H2("4.1  The world: a dynamical causal model")
    P("State evolves one tick at a time. Forcing an actuator clamps it (overriding its dynamics) and "
      "the clamp persists, exactly like dialling a setpoint; sensors then evolve through the graph. "
      "This makes time matter: a deep, slow sensor can only be moved by holding upstream knobs over "
      "several steps — which is precisely where composing constructors earns its keep.")
    EQ(r"$x_{t+1,i} \;=\; \sum_j A_{ij}\,x_{t,j} \;+\; b_i \;+\; e_{t,i},"
       r"\qquad e_{t,i}\sim\mathcal{N}(0,\sigma_i^2)$",
       "x_{t+1,i} = sum_j A_ij x_{t,j} + b_i + noise.")
    H2("4.2  The belief: Bayesian regression whose posterior is the graph")
    P("For each sensor we fit one Bayesian linear regression predicting its next value from a "
      "feature vector built from the current state. Conjugacy gives a closed-form Gaussian posterior "
      "over the weights — not just a best guess but a covariance, i.e. how unsure we still are about "
      "each causal link. Two things drop straight out. First, the recovered DAG: an edge j->i exists "
      "when weight w_i[j] is confidently non-zero (|mean|/std beyond a threshold). Interventions are "
      "what let the agent drive that std down and zero out spurious links. Second, the experiment "
      "value — the expected information gain of observing a transition out of a state with feature "
      "vector phi:")
    EQ(r"$\mathrm{EIG}(\phi)\;=\;\frac{1}{2}\sum_i \log\left(1+\phi^{\top}\Sigma_i\,\phi\,/\,\sigma_i^2\right)$",
       "EIG(phi) = (1/2) sum_i log(1 + phi^T Sigma_i phi / sigma_i^2).")
    P("This measures shrinkage of uncertainty about the parameters — not raw surprise. That "
      "distinction is the whole game: a pure-noise variable is endlessly surprising but teaches "
      "nothing once its (parent-free) law is known, so its EIG collapses and the agent stops "
      "chasing it. That is the principled cure for the classic \"noisy-TV\" trap.")
    H2("4.3  Constructors and composition")
    P("A region of state space is an axis-aligned box (intervals per variable), so membership and "
      "subset tests are exact and cheap. A constructor bundles a task (precondition box -> effect "
      "box), a program (a schedule of interventions), and a measured reliability. Composition is the "
      "engine that grows them:")
    EQ(r"$\frac{C_1:\,P\to Q \quad\;\; C_2:\,Q'\to R \quad\;\; Q\subseteq Q'}"
       r"{C_2\circ C_1:\;\;P\to R}$",
       "If C1 takes P to Q and C2 takes Q' to R with Q a subset of Q', then C2 after C1 takes P to R.")
    P("Composability is a causal statement: C2 may assume its precondition because C1 guarantees it "
      "as a postcondition. Chaining small, individually-verified constructors yields big ones that "
      "reach variables no single primitive can move. Reliability is estimated by actually running "
      "the program many times in fresh copies of the world — a legitimate repeated experiment — and "
      "a task counts as \"possible\" once reliability clears a threshold over enough trials, the "
      "operational stand-in for Constructor Theory's \"arbitrarily good approximation\".")
    H2("4.4  Reward-free experiment selection")
    P("At each step the experimenter scores candidate interventions by their expected information "
      "gain over a short model-rollout and acts on the best (with a little dithering for coverage). "
      "When a world has many knobs the joint grid is exponential, so above a cap the experimenter "
      "samples candidate settings instead of enumerating them. Two foils — uniform-random and "
      "\"maximise raw surprise\" — are kept to show that the choice of objective, not the machinery, "
      "is what matters.")
    H2("4.5  Distilling and composing constructors")
    P("From the learned model the agent mints primitive constructors (\"hold this knob here\"), "
      "discovering each one's effect by running it. Constructors that move nothing reveal which "
      "knobs are causally idle. It then mints conditional (context-dependent) constructors — a knob "
      "that is useless from rest but does something once another constructor has set the stage — and "
      "iterates, stacking skills bottom-up. To reach a goal it searches over chains of library "
      "constructors, verifying promising ones in the real world.")
    H2("4.6  Two extensions that remove early limitations")
    B(["Interaction discovery: rather than being told which products matter, the agent fits a linear "
       "model, scans each sensor's residuals for leftover structure, and proposes the product "
       "feature (a \"gate\") that explains it — recovering multiplicative causes it was never handed.",
       "Nonlinear basis: random Fourier features let the belief fit smooth nonlinear edges (e.g. a "
       "saturating tanh), so counterfactual predictions track the curve instead of a straight line."])

    H1("5.  Experiments and results")
    P("Eight worlds, each built to exercise one defence, scored by 65 falsifiable checks across seven "
      "test suites. Everything below is reward-free.")
    TAB(["World", "What it tests", "Headline result"],
        [["default", "chain + decoy + noise", "recover DAG (F1 1.0); reject decoy; ignore noise"],
         ["hard", "weak edge, idle knob", "curiosity F1 0.88 vs random 0.84"],
         ["confounded", "hidden common cause", "passive infers spurious edge (w 0.85); intervention kills it (0.04)"],
         ["gated", "multiplicative gate", "discover gate; compose 2 distinct skills to reach Z"],
         ["cascade", "two-gate cascade", "compose 3 distinct skills; reliability 1.0"],
         ["wide", "many distractor knobs", "informed planner 13 nodes vs BFS 314"],
         ["nonlinear", "even + saturating edges", "linear blind to a0^2; RFF what-if 22x better"],
         ["default (drifting)", "non-stationary world", "track flipping edge; spike->settle; prune & rebuild skills"]],
        [0.16, 0.34, 0.50])

    doc.figure("results")
    md.append(f"\n*{FIGS['results'][2]}*\n")

    H2("5.1  It learns the causal map with no reward")
    P("On the default world the agent recovers exactly the true edges (a0->chain1, chain1->chain2, "
      "chain1->decoy) with F1 = 1.0. It rejects the decoy — a variable correlated with the target "
      "but not its cause — because its interventions break the confound. And it recognises the "
      "pure-noise channel as parent-free and stops exciting it, exactly as the information-gain "
      "objective predicts.")
    H2("5.2  The objective matters — but only when it must")
    P("On a harder world (a weak edge that must be probed, plus a useless distractor knob and noisy "
      "sensors), curiosity beats random F1 0.88 to 0.84 — a modest margin, because the noise-variance "
      "recalibration lifted random's floor by suppressing its spurious edges — by concentrating its "
      "scarce budget on the informative knob. Honesty cuts both ways: on the easy fully-observed world all objectives tie, "
      "and naive surprise only falls into the noisy-TV trap when the agent must also choose WHAT to "
      "observe — which we show in a dedicated observation-gating test (naive watches noise 83% of the "
      "time and learns the real law only 75% of the time; the info-gain agent watches noise 29% and "
      "learns it 100%).")
    H2("5.3  Composition manufactures capability")
    P("A deep, slow variable that a single short intervention cannot move (reliability 0.00) is "
      "reached by the same primitive chained with itself (reliability ~1.0). A multiplicative gate, "
      "where a variable rises only when two inputs are both high, is cracked by composing two "
      "DIFFERENT constructors in order — open the gate, then drive the gated variable. A two-gate "
      "cascade needs three distinct context-dependent constructors composed in sequence; the agent "
      "discovers and stacks them and reaches the target with reliability 1.0.")

    doc.figure("ladder")
    md.append(f"\n*{FIGS['ladder'][2]}*\n")

    H2("5.4  Intervention defeats a hidden confounder")
    P("When an unobserved common cause drives two visible variables, a passive observer infers a "
      "strong but spurious link between them (weight ~0.85). The intervening agent forces one of "
      "them, decorrelating it from the hidden cause, and the spurious weight collapses to ~0. This "
      "is the textbook reason interventions matter, shown end-to-end.")
    H2("5.5  The planner scales, and the belief goes nonlinear")
    P("In a world padded with useless knobs, uninformed breadth-first search over the constructor "
      "library expands 314 nodes to find the answer; an informed best-first search ordered by the "
      "model's predicted distance to the goal expands 13 — the same answer, ~24x cheaper. And two "
      "nonlinear edges that defeat a purely linear learner are handled: an even edge (output "
      "proportional to a0-squared) has zero linear correlation, so a linear model finds no edge at "
      "all, yet the agent recovers it as the product a0*a0; a saturating tanh edge is predicted with "
      "RMSE 0.03 by the random-Fourier model versus 0.65 by the linear one.")

    H2("5.6  It keeps learning when the world changes")
    P("The loop also runs continually. We flip and re-scale a causal edge (a0->chain1: +0.8, then "
      "-0.8, then +1.3) without telling the agent. Three mechanisms keep it current: the belief uses "
      "recursive least squares with a forgetting factor, so it tracks recent dynamics rather than "
      "averaging over all history; one-step prediction error is monitored, and it spikes the instant "
      "the world moves (chain1 error jumps to ~3-4 standard units, then falls back below 0.05 as the "
      "belief re-settles); and a consolidation step re-verifies every constructor against the current "
      "world, pruning the ones that broke. The recovered weight tracks the true edge across all three "
      "regimes; each sign flip prunes the stale skills and the library is rebuilt — the constructor "
      "that drove chain1 high by holding a0=+2 is replaced by one holding a0=-2, then flipped back. "
      "The library stays true to the world, not to its own past.")

    doc.figure("continual")
    md.append(f"\n*{FIGS['continual'][2]}*\n")

    H2("5.7  Closing a self-audit: discovering the interface and running itself")
    P("We asked where the system was failing and fixed four gaps, all reward-free. (i) DISCOVER THE "
      "INTERFACE: rather than being told which variables are knobs, the agent pokes each one to an "
      "out-of-range value and keeps those that hold there (controllability); correct on every world. "
      "(ii) HIDDEN CAUSE vs NOISE: a slow unobserved common cause is absorbed into an inflated "
      "self-loop, so residual autocorrelation does NOT find it (we checked) -- but its fingerprint is "
      "being poorly predicted yet strongly autoregressive, which flags the confounded variable while "
      "leaving a pure-noise channel alone. (iii) CONTINUOUS CONTROL: when the coarse +/-2 library "
      "overshoots a narrow target, the planner solves for an intermediate setpoint that lands centred "
      "in it. (iv) AUTONOMOUS LOOP: the agent watches its own surprise -- standardised by each "
      "sensor's noise, so an irreducibly noisy channel cannot raise a false alarm -- and decides for "
      "itself when to re-learn. With no external schedule it catches a parametric flip (~56 sigma, "
      "prunes the broken skills, relearns the edge) and a structural change, a gate that newly appears "
      "(~27 sigma, re-discovers the interaction), and stays quiet when nothing changed.")
    P("We were equally careful about what NOT to claim. Two suspected failures, tested, did not "
      "actually bite (an inadmissible-heuristic blow-up on the gated world; false-positive interactions "
      "under multiple testing). And several limits are genuine, not fixed: certifying a hidden variable "
      "(an AR cause is indistinguishable from a self-loop in one-step data), verification that still "
      "assumes a cloneable/resettable world, and untested scale.")

    H1("6.  What this tells us")
    B(["Reward is not necessary for competence. A complete, well-defined drive — reduce uncertainty "
       "about the causal mechanism — produces an agent that can later be handed arbitrary goals and "
       "meet them. Understanding first; capability falls out.",
       "Curiosity must be about learning, not surprise. Information gain over parameters, not raw "
       "predictive surprise, is what avoids the noisy-TV trap and what makes scarce experiments count.",
       "Causal structure needs intervention. Decoys and hidden confounders both fool passive "
       "observation; forcing variables is what makes the recovered graph trustworthy.",
       "Constructor composition is a genuine engine of capability. Stacking verified transformations "
       "produces abilities none of the parts had alone — the operational meaning of \"combine "
       "sequences of constructors into bigger constructors\".",
       "Honesty about regimes. The objective's advantage, the planner's advantage, and the nonlinear "
       "basis's advantage each appear only under the conditions that call for them; we measure where "
       "they help rather than asserting they always do."])

    H1("7.  Toward a self-continual learning agent")
    P("Section 5.6 shows the loop already surviving a changing world. A fully self-continual agent "
      "would keep it turning forever, accumulating and consolidating skills while discovering its own "
      "interface and structure. The pieces below are the concrete next steps; the first four are "
      "near-term and build directly on what exists.")
    H2("7.1  Discover the interface (done; certifying the latent is still open)")
    P("This is now implemented (5.7): the agent discovers its actuators by controllability probing, "
      "and flags hidden state by the poorly-predicted-yet-autoregressive fingerprint. What remains is "
      "CERTIFICATION — from one-step observational data a hidden AR(1) cause is indistinguishable from "
      "a real self-loop, so the agent can say 'something unobserved drives this' but not identify the "
      "latent. That needs interventions on the affected variable (unavailable when it is only a "
      "sensor) or an explicit latent-variable model fit by EM / variational inference — the machinery "
      "the sibling pi_ct_nsdm project already uses.")
    H2("7.2  A posterior over structure, not a point estimate")
    P("Our per-edge significance test is a thresholded point estimate. The principled object is a "
      "posterior over graphs, and the principled acquisition is the mutual information between the "
      "next experiment and that posterior (BALD). An ensemble of structures with Bernoulli edge "
      "probabilities makes this tractable and turns experiment selection into honest D-optimal "
      "design — the same machinery the sibling pi_ct_nsdm project already prototypes.")
    H2("7.3  An informed, learned planner")
    P("The best-first heuristic is greedy and not admissible. Two upgrades: learn a value function "
      "over the constructor library (how close does owning skill S leave me to a family of goals?) "
      "to guide search, and cache successful chains as new named constructors so the library deepens "
      "rather than re-deriving. This is the options framework of hierarchical RL, but with verified, "
      "reusable, reward-free skills.")
    H2("7.4  Off-manifold-robust nonlinear discovery")
    P("Random Fourier features predict well on the training manifold but extrapolate poorly off it, "
      "and structure is still read from the linear/product blocks. The fix is interventional group-"
      "relevance: force a candidate parent across its range and test whether a whole nonlinear "
      "feature block for it carries causal weight — a nonlinear analogue of the linear edge test — "
      "with Gaussian-process or deep-kernel beliefs for calibrated off-manifold uncertainty.")
    H2("7.5  Non-stationarity and consolidation (mostly done)")
    P("Sections 5.6-5.7 implement this: forgetting, AUTONOMOUS change detection by standardised "
      "surprise, library consolidation, and structural re-discovery when a gate appears. What remains "
      "turns it from \"survives scripted changes\" into \"survives an open-ended life\":")
    B(["Skill merging and abstraction: consolidation currently prunes; it should also merge near-"
       "duplicate skills and promote frequently-used chains to named macro-constructors.",
       "Competence progress as a second drive: alongside information gain, reward learnable progress "
       "(am I getting better at predicting / achieving?) to allocate effort and to avoid the "
       "\"dark-room\" failure where an agent minimises surprise by doing nothing.",
       "Empowerment: prefer states from which many transformations remain possible, a constructor-"
       "theoretic objective that keeps options open without any external reward."])
    H2("7.6  The continual loop")
    P("Putting it together, the agent we are building toward runs one loop indefinitely: explore the "
      "most uncertain frontier; discover new variables, edges, and interactions; distil and verify "
      "constructors; compose them into deeper skills; consolidate the library; and detect when the "
      "world has changed and the cycle must reopen. No reward is defined at any point. Goals, when "
      "they arrive, are solved by composing what the agent already understands — which is, we argue, "
      "the right shape for an agent that must keep learning in a world it was never told the rules to.")

    H1("8.  Reproducibility")
    P("The system is ~3.4k lines of numpy-only Python. Every claim in Section 5 is asserted by an "
      "executable check that exits non-zero on failure.")
    CODE("# from the ChemicalWorld directory (project .venv; numpy only)\n"
         "python -m constructor_causal.demo            # basic: DAG -> library -> compose\n"
         "python -m constructor_causal.demo_advanced   # confounder, gate, observation-gating\n"
         "python -m constructor_causal.demo_frontier   # interaction discovery + deep cascade\n"
         "python -m constructor_causal.demo_frontier2  # informed planner + nonlinear\n"
         "python -m constructor_causal.demo_continual  # a world that changes underneath the agent\n"
         "python -m constructor_causal.demo_autonomous # discovers interface; self-driven loop\n"
         "python -m constructor_causal.selftest{,_advanced,_frontier,_frontier2,_continual,_autonomous}  # 65 checks")

    H1("References (selected)")
    B(["J. Pearl. Causality (2009). Structural causal models and do-calculus.",
       "K. Friston et al. Active inference and epistemic value. The free-energy principle.",
       "D. Deutsch & C. Marletto. Constructor Theory (2013-).",
       "P. Hoyer et al. Nonlinear causal discovery with additive noise models (2009).",
       "N. Houlsby et al. Bayesian Active Learning by Disagreement / BALD (2011).",
       "A. Rahimi & B. Recht. Random features for large-scale kernel machines (2007).",
       "J. Schmidhuber. Formal theory of creativity, fun, and intrinsic motivation."])


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    pdf_path = os.path.join(HERE, "PAPER.pdf")
    md_path = os.path.join(HERE, "PAPER.md")
    doc = Doc(pdf_path)
    md: list = []
    build(doc, md)
    doc.close()
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", pdf_path)
    print("wrote", md_path)


if __name__ == "__main__":
    main()
