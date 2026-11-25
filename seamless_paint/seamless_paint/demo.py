"""Headless demonstration: build acceptance patterns without a browser.

Reproduces the three required acceptance scenarios and writes deliverables
into ``output/`` (or a directory passed as the first argument):

1. diagonal thin line crossing two period boundaries;
2. semi-transparent brush larger than the texture unit;
3. corner dab whose print appears identically in all four corners.

Run::

    python3 -m seamless_paint.demo [output_dir]
"""

from __future__ import annotations

import os
import sys

from .kernel import SeamlessTile, HAVE_NUMPY
from .history import save_project


def build_demo(output_dir: str = "output") -> dict:
    W = H = 128
    tile = SeamlessTile(W, H, background=(1.0, 1.0, 1.0, 1.0))

    # 1. Diagonal thin line: enters through the top-left, exits through the
    #    bottom-right, and must line up again in the tiled montage.
    diagonal = [(-24 + t * 4.0, -24 + t * 4.0) for t in range(44)]
    tile.add_stroke(diagonal, 1.8, (0.10, 0.25, 0.85, 1.0),
                    soft=0.15, policy="neighbor", spacing=0.9)

    # 2. Semi-transparent soft brush with radius larger than the unit: many
    #    periodic copies overlap, combined as a coverage union.  Recorded as a
    #    single-point stroke so it is part of the replayable stroke history
    #    (the kernel treats one point as one fully-parameterised dab).
    tile.add_stroke([(W * 0.75, H * 0.3)], W * 1.2,
                    (0.9, 0.25, 0.45, 0.35), soft=0.5,
                    policy="neighbor", spacing=1.0)

    # 3. Corner dab: one deposit, identical prints in all four corners.
    tile.add_stroke([(0.0, 0.0)], 9.0, (0.15, 0.65, 0.25, 1.0),
                    soft=0.2, spacing=1.0)

    paths = save_project(tile, output_dir, base_name="acceptance_demo")
    return paths


def main(argv=None):
    out = argv[1] if argv and len(argv) > 1 else "output"
    print(f"numpy accelerated kernel: {HAVE_NUMPY}")
    paths = build_demo(out)
    for kind, path in paths.items():
        print(f"{kind:8s} -> {os.path.abspath(path)}")


if __name__ == "__main__":
    main(sys.argv)
