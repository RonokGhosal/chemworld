"""
HARDER sensor world, one notch (commander's Order 4): same deep gated chain + noise knob, but
now the 20 sensors ALSO mix in 6 DISTRACTOR latents -- autonomous AR(0.85) nuisance factors that
are uncontrolled and irrelevant to the goal. The encoder must keep its controllable z_c clean of
both the irreducible noise AND these autocorrelated distractors. No pixels yet.

Same protocol as the headline (no-leak 80 binary labels + diagnostic + prediction-first + oracle,
per seed, bands 8/12 budget 18).
"""
from __future__ import annotations

from . import messy_noleak

WORLD = dict(obs_dim=20, nonlinear=True, n_distract=6)

if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    messy_noleak.main(seeds=range(ns), world_kw=WORLD, label="[HARDER: +6 distractor sensors]")
