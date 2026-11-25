"""Auxiliary CLI: lattice painting sample and tiled whole-image diff acceptance.

The graphical entry point is the painting window::

    python3 -m seamless_paint.webapp --port 8000

This module is the *command-line auxiliary*: it paints a non-orthogonal 60
degree lattice sample with a rotatable flat nib, saves the unit chart, the
basis-recording history JSON and the true world-space lattice tiling, then
runs an independent plain-loop reference renderer over the **whole 3x3 tiled
image** and writes the amplified difference image used for acceptance.

Run::

    python3 -m seamless_paint.lattice_demo

Outputs (by default in the repository-level ``output/`` directory):

* ``lattice_sample_unit.png``         - W x H parameter cell chart;
* ``lattice_sample_history.json``     - strokes + explicit ``repeat_basis``;
* ``lattice_sample_lattice_tiled.png`` - true oblique 3x3 world tiling;
* ``lattice_accept_reference3x3.png`` - independent reference tiling;
* ``lattice_accept_kernel3x3.png``    - kernel tiling at acceptance size;
* ``lattice_accept_diff3x3.png``      - amplified whole-image diff (black=0);
* ``lattice_accept_compare3x3.png``   - reference | kernel | diff strip;
* ``lattice_basis_rewrap.png`` / ``lattice_basis_deform.png`` - the two
  change-basis semantics rendered as non-mutating previews;
* ``lattice_sample_acute15_tiled.png`` / ``lattice_sample_bignib_tiled.png`` -
  extra edge cases required by the test brief.

Exit code is 0 only when the maximum tiled-image channel difference is within
the rounding tolerance.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Sequence, Tuple

from .kernel import SeamlessTile, HAVE_NUMPY
from .lattice import Lattice
from .history import save_project, export_lattice_tiling
from .pngio import write_png

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "output")

Pixel = List[int]


# --------------------------------------------------------------------------- #
# Independent plain-loop reference (dab/copy loops, premultiplied accumulation)
# --------------------------------------------------------------------------- #
def _coverage(e: float, soft: float) -> float:
    if soft <= 0.0:
        return 1.0 if e <= 1.0 + 1e-9 else 0.0
    if e <= 1.0 - soft + 1e-9:
        return 1.0
    if e >= 1.0:
        return 0.0
    return (1.0 - e) / soft


def _nib_metric(rx: float, ry: float, angle: float):
    c, sn = math.cos(angle), math.sin(angle)
    ix2, iy2 = 1.0 / (rx * rx), 1.0 / (ry * ry)
    q00 = c * c * ix2 + sn * sn * iy2
    q11 = sn * sn * ix2 + c * c * iy2
    q01 = c * sn * (ix2 - iy2)
    return q00, q01, q11


def reference_chart(width: int, height: int, L: Lattice,
                    dabs: Sequence[dict],
                    background=(1.0, 1.0, 1.0, 1.0)) -> List[List[Pixel]]:
    """Independent W x H chart reference.

    Structured dab-first: every lattice copy of every dab contributes a
    coverage, periodic copies of the *same* dab combine through the union
    rule ``1 - prod(1-c_k)`` and are composited exactly once.  Candidate pixel
    boxes come from a conservative inverse-row norm support bound, a different
    derivation from the kernel's eigendecomposition.
    """
    buf = [[[background[0] * background[3],
             background[1] * background[3],
             background[2] * background[3],
             background[3]] for _ in range(width)]
           for _ in range(height)]
    ax, ay, bx, by = L.ax, L.ay, L.bx, L.by

    for dab in dabs:
        x, y = float(dab["x"]), float(dab["y"])
        rx = float(dab.get("rx", dab.get("radius", 1.0)))
        ry = float(dab.get("ry", dab.get("radius", 1.0)))
        angle = float(dab.get("angle", 0.0))
        soft = float(dab.get("soft", 0.25))
        cr, cg, cb, ca = (float(v) for v in dab["color"])
        q00, q01, q11 = _nib_metric(rx, ry, angle)
        rmax = max(rx, ry)
        es = rmax * math.hypot(L.m00, L.m01) + 1e-9
        et = rmax * math.hypot(L.m10, L.m11) + 1e-9
        sc, tc = L.world_to_cell(x, y)
        i0 = math.floor(-es - sc) - 1
        i1 = math.ceil(1.0 + es - sc) + 1
        j0 = math.floor(-et - tc) - 1
        j1 = math.ceil(1.0 + et - tc) + 1

        npx = width * height
        union_one = [1.0] * npx
        mark = bytearray(npx)
        touched: List[int] = []

        for j in range(j0, j1 + 1):
            tj = tc + j
            r0 = max(0, math.ceil(height * (tj - et) - 0.5 - 1e-7))
            r1 = min(height - 1,
                     math.floor(height * (tj + et) - 0.5 + 1e-7))
            if r0 > r1:
                continue
            for i in range(i0, i1 + 1):
                si = sc + i
                c0 = max(0, math.ceil(width * (si - es) - 0.5 - 1e-7))
                c1 = min(width - 1,
                         math.floor(width * (si + es) - 0.5 + 1e-7))
                if c0 > c1:
                    continue
                for row in range(r0, r1 + 1):
                    dt = (row + 0.5) / height - tj
                    base = row * width
                    for col in range(c0, c1 + 1):
                        ds = (col + 0.5) / width - si
                        dx = ds * ax + dt * bx
                        dy = ds * ay + dt * by
                        e2 = (q00 * dx * dx + 2.0 * q01 * dx * dy
                              + q11 * dy * dy)
                        if e2 > 1.0 + 1e-7:
                            continue
                        cov = _coverage(math.sqrt(max(0.0, e2)), soft)
                        if cov <= 0.0:
                            continue
                        idx = base + col
                        if not mark[idx]:
                            mark[idx] = 1
                            touched.append(idx)
                            union_one[idx] = 1.0 - cov
                        else:
                            union_one[idx] *= 1.0 - cov

        for idx in touched:
            coverage = 1.0 - union_one[idx]
            a = ca * coverage
            if a <= 0.0:
                continue
            row, col = divmod(idx, width)
            px = buf[row][col]
            ia = 1.0 - a
            px[0] = cr * a + px[0] * ia
            px[1] = cg * a + px[1] * ia
            px[2] = cb * a + px[2] * ia
            px[3] = a + px[3] * ia

    out = [[[0, 0, 0, 0] for _ in range(width)] for _ in range(height)]
    for row in range(height):
        for col in range(width):
            p = buf[row][col]
            if p[3] > 1e-9:
                scale = 255.0 / p[3]
                out[row][col] = [
                    min(255, max(0, int(p[0] * scale + 0.5))),
                    min(255, max(0, int(p[1] * scale + 0.5))),
                    min(255, max(0, int(p[2] * scale + 0.5))),
                    min(255, max(0, int(p[3] * 255.0 + 0.5))),
                ]
    return out


def samples_to_dabs(tile: SeamlessTile) -> List[dict]:
    """Independent stroke reconstruction from raw points.

    Deliberately does NOT read ``rec["samples"]``: the acceptance must catch
    unwrap/resampling bugs, not validate the kernel's own dab list.
    """
    dabs: List[dict] = []
    for rec in tile.history:
        L = tile.lattice
        raw = [(float(x), float(y)) for x, y in rec["raw_points"]]
        folded = []
        for x, y in raw:
            s, t = L.world_to_cell(x, y)
            folded.append((s - math.floor(s), t - math.floor(t)))

        unwrapped = [folded[0]]
        os_ = ot = 0.0
        for k in range(1, len(folded)):
            ds = folded[k][0] - folded[k - 1][0]
            dt = folded[k][1] - folded[k - 1][1]
            best_shift, best_len = (0, 0), math.inf
            for js in (-1, 0, 1):
                for jt in (-1, 0, 1):
                    wx, wy = L.cell_to_world(ds + js, dt + jt)
                    ln = wx * wx + wy * wy
                    if ln < best_len:
                        best_len, best_shift = ln, (js, jt)
            ds += best_shift[0]
            dt += best_shift[1]
            os_ += ds
            ot += dt
            unwrapped.append((folded[0][0] + os_, folded[0][1] + ot))

        world_pts = [L.cell_to_world(s, t) for s, t in unwrapped]
        radius = float(rec["radius"])
        spacing = float(rec.get("spacing")
                        or max(0.5, min(radius * 0.25, 4.0)))
        samples = [world_pts[0]]
        remainder = spacing
        for (x0, y0), (x1, y1) in zip(world_pts[:-1], world_pts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg == 0.0:
                continue
            ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
            travel = remainder
            while travel <= seg:
                samples.append((x0 + ux * travel, y0 + uy * travel))
                travel += spacing
            remainder = travel - seg
        if samples[-1] != world_pts[-1]:
            samples.append(world_pts[-1])

        rx = rec.get("rx")
        ry = rec.get("ry")
        if rx is None:
            rx = radius
        if ry is None:
            ry = radius * float(rec.get("aspect", 1.0))
        for x, y in samples:
            dabs.append({
                "x": float(x), "y": float(y),
                "rx": float(rx), "ry": float(ry),
                "angle": float(rec.get("angle", 0.0)),
                "color": tuple(float(c) for c in rec["color"]),
                "soft": float(rec.get("soft", 0.25)),
            })
    return dabs


def reference_world_block(chart: List[List[Pixel]], W: int, H: int,
                          L: Lattice, bounds: Tuple[float, float, float, float],
                          out_w: int, out_h: int) -> List[List[Pixel]]:
    """Independent inverse-map whole-block tiler sampling the ref chart."""
    x0, y0, x1, y1 = bounds
    img = [[[255, 255, 255, 255] for _ in range(out_w)]
           for _ in range(out_h)]
    for row in range(out_h):
        fy = (row + 0.5) / out_h
        wy = y0 + fy * (y1 - y0)
        for col in range(out_w):
            fx = (col + 0.5) / out_w
            wx = x0 + fx * (x1 - x0)
            s = L.m00 * wx + L.m01 * wy
            t = L.m10 * wx + L.m11 * wy
            sf, tf = s - math.floor(s), t - math.floor(t)
            ci = min(W - 1, int(sf * W))
            ri = min(H - 1, int(tf * H))
            img[row][col] = list(chart[ri][ci])
    return img


def as_lists(img):
    if HAVE_NUMPY and not isinstance(img, list):
        return img.tolist()
    return img


def diff_stats(a, b) -> Tuple[int, float, List[List[Pixel]]]:
    h = len(a)
    w = len(a[0])
    worst = 0
    total = 0
    n = 0
    heat = [[[0, 0, 0, 255] for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            dmax = 0
            for k in range(4):
                d = abs(int(a[r][c][k]) - int(b[r][c][k]))
                dmax = max(dmax, d)
                total += d
                n += 1
            worst = max(worst, dmax)
            v = min(255, dmax * 80)
            heat[r][c] = [v, max(0, v - 90), max(0, v - 170), 255]
    return worst, total / max(1, n), heat


def hstack(panels: Sequence[List[List[Pixel]]], gap: int = 8
           ) -> Tuple[List[List[Pixel]], int, int]:
    h = len(panels[0])
    w = len(panels[0][0])
    total_w = w * len(panels) + gap * (len(panels) - 1)
    strip = [[[235, 238, 245, 255] for _ in range(total_w)]
             for _ in range(h)]
    for p, panel in enumerate(panels):
        xoff = p * (w + gap)
        for r in range(h):
            for c in range(w):
                strip[r][xoff + c] = list(panel[r][c])
    return strip, total_w, h


# --------------------------------------------------------------------------- #
# Sample artwork
# --------------------------------------------------------------------------- #
def _wave(x0: float, x1: float, base_y: float, amp: float,
          cycles: float, phase: float = 0.0, n: int = 48):
    pts = []
    for k in range(n + 1):
        x = x0 + (x1 - x0) * k / n
        y = base_y + amp * math.sin(
            phase + 2.0 * math.pi * cycles * k / n)
        pts.append((x, y))
    return pts


def build_sample() -> SeamlessTile:
    W = H = 160
    L = Lattice(float(W), 0.0, W / 2.0, round(W * math.sqrt(3.0) / 2.0, 3))
    tile = SeamlessTile(W, H, background=(1.0, 1.0, 1.0, 1.0), lattice=L)

    nib = dict(spacing=2.0, policy="neighbor")
    # Magenta calligraphic sweep: dense smooth wave crossing cell edges.
    tile.add_stroke(
        _wave(-40, 200, 52, 15, 1.35, phase=0.4),
        9.0, (0.84, 0.12, 0.42, 0.96), soft=0.18,
        angle=math.radians(32), aspect=0.32, **nib)
    # Teal counter-sweep with the nib rotated the other way.
    tile.add_stroke(
        _wave(-40, 200, 112, 14, 1.35, phase=2.2),
        8.0, (0.04, 0.52, 0.60, 0.92), soft=0.2,
        angle=math.radians(-52), aspect=0.38, **nib)
    # Amber upper accent crossing the a-axis edge.
    tile.add_stroke(
        _wave(-24, 184, 22, 7, 1.1, phase=1.0),
        6.0, (0.95, 0.62, 0.08, 0.95), soft=0.15,
        angle=math.radians(72), aspect=0.5, **nib)
    # Dark nib dab exactly on the lattice vertex: must print identically in
    # all four chart corners after tiling.
    tile.add_stroke(
        [(0.0, 0.0)], 8.0, (0.10, 0.10, 0.16, 1.0), soft=0.1,
        angle=math.radians(-24), aspect=0.42)
    # Purple nib on the midpoint of the a edge (top/bottom seam).
    tile.add_stroke(
        [(W / 2.0, 0.0)], 6.5, (0.36, 0.20, 0.72, 0.95), soft=0.2,
        angle=math.radians(18), aspect=0.45)
    return tile


def build_acute_sample() -> SeamlessTile:
    W = H = 128
    ang = math.radians(15.0)
    L = Lattice(float(W), 0.0,
                round(150.0 * math.cos(ang), 2),
                round(150.0 * math.sin(ang), 2))
    tile = SeamlessTile(W, H, background=(1, 1, 1, 1), lattice=L)
    tile.add_stroke([(-20, 40), (40, 55), (110, 38), (160, 52)],
                    7.0, (0.75, 0.1, 0.25, 0.95), soft=0.2,
                    angle=0.7, aspect=0.4, spacing=4.0)
    tile.add_stroke([(0, 0)], 6.0, (0.08, 0.08, 0.12, 1),
                    soft=0.1, angle=-0.5, aspect=0.4)
    return tile


def build_bignib_sample() -> SeamlessTile:
    W = H = 64
    L = Lattice(float(W), 0.0, W / 2.0, W * 0.8660254)
    tile = SeamlessTile(W, H, background=(1, 1, 1, 1), lattice=L)
    # Radius comparable to the whole cell: support-driven enumeration spans
    # many neighbours in both lattice directions.
    tile.add_stroke([(32, 32)], 52.0, (0.15, 0.35, 0.85, 0.5),
                    soft=0.55, angle=0.6, aspect=0.72)
    tile.add_stroke([(0, 0)], 10.0, (0.85, 0.2, 0.3, 0.95),
                    soft=0.15, angle=-0.4, aspect=0.4)
    return tile


def run(out_dir: str, copies: int = 3, accept_px: int = 460,
        tol: int = 2) -> int:
    os.makedirs(out_dir, exist_ok=True)
    tile = build_sample()
    W, H = tile.width, tile.height
    L = tile.lattice

    paths = save_project(tile, out_dir, "lattice_sample",
                         write_world_tiling=True)
    print("sample saved:")
    for key, p in paths.items():
        print(f"  {key:14s} {p}")

    # -- Basis-change semantic previews (non-mutating) --------------------- #
    new_basis = Lattice(172.0, -6.0, 66.0, 148.0)
    for mode, fname in [("rewrap", "lattice_basis_rewrap.png"),
                        ("deform", "lattice_basis_deform.png")]:
        png = tile.preview_basis_png(new_basis, mode=mode, view_px=384)
        p = os.path.join(out_dir, fname)
        with open(p, "wb") as fh:
            fh.write(png)
        print(f"  basis preview {mode:6s} {p}")

    # -- Extra edge-case tilings ------------------------------------------ #
    acute = build_acute_sample()
    p, w, h = export_lattice_tiling(
        acute, os.path.join(out_dir, "lattice_sample_acute15_tiled.png"),
        copies=3, max_px=700)
    print(f"acute 15deg tiling {p} ({w}x{h})")

    big = build_bignib_sample()
    report = big.history[0].get("last_cells")
    p, w, h = export_lattice_tiling(
        big, os.path.join(out_dir, "lattice_sample_bignib_tiled.png"),
        copies=3, max_px=700)
    print(f"huge-nib tiling   {p} ({w}x{h}), "
          f"central dab unique cell identities={len(report or [])}")

    # -- Whole tiled image acceptance diff -------------------------------- #
    print("\nbuilding independent reference ...")
    dabs = samples_to_dabs(tile)
    ref_chart = reference_chart(W, H, L, dabs)

    img_k, ow, oh, bounds = tile.render_cell_block_uint8(
        copies, max_px=accept_px)
    kernel_block = as_lists(img_k)
    ref_block = reference_world_block(
        ref_chart, W, H, L, bounds, ow, oh)

    worst, mean, heat = diff_stats(kernel_block, ref_block)
    strip, sw, sh = hstack([ref_block, kernel_block, heat])

    ref_path = os.path.join(out_dir, "lattice_accept_reference3x3.png")
    ker_path = os.path.join(out_dir, "lattice_accept_kernel3x3.png")
    diff_path = os.path.join(out_dir, "lattice_accept_diff3x3.png")
    strip_path = os.path.join(out_dir, "lattice_accept_compare3x3.png")
    write_png(ref_path, ref_block, ow, oh, 4)
    write_png(ker_path, kernel_block, ow, oh, 4)
    write_png(diff_path, heat, ow, oh, 4)
    write_png(strip_path, strip, sw, sh, 4)

    # Numerical seam check: world renders shifted by a/b must be identical.
    r0 = (-40.0, -30.0, 220.0, 190.0)
    base = as_lists(tile.render_world_uint8(*r0, 180, 150))
    seam_ok = True
    for dx, dy in [(L.ax, L.ay), (L.bx, L.by)]:
        shifted = as_lists(tile.render_world_uint8(
            r0[0] + dx, r0[1] + dy, r0[2] + dx, r0[3] + dy, 180, 150))
        if shifted != base:
            seam_ok = False

    print(f"\nacceptance block: {ow}x{oh} px, {copies}x{copies} cells")
    print(f"dabs checked: {len(dabs)}")
    print(f"max channel diff : {worst} (tolerance {tol})")
    print(f"mean channel diff: {mean:.5f}")
    print(f"basis-shift seam invariance: {'PASS' if seam_ok else 'FAIL'}")
    print(f"diff image: {diff_path}")
    print(f"compare strip (reference | kernel | amplified diff): {strip_path}")

    ok = worst <= tol and seam_ok
    print("ACCEPTANCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the lattice painting sample and tiled diff "
                    "acceptance images. The graphical entry point is "
                    "`python3 -m seamless_paint.webapp`.")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output directory (default: repo output/)")
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--accept-px", type=int, default=460)
    parser.add_argument("--tol", type=int, default=2)
    args = parser.parse_args(argv)
    return run(args.out, copies=args.copies, accept_px=args.accept_px,
               tol=args.tol)


if __name__ == "__main__":
    sys.exit(main())
