"""Tests for the two-basis-vector repeat lattice.

Coverage:

* screen/world -> cell coordinate transforms and the default orthogonal basis;
* support-driven neighbour enumeration and cell-identity de-duplication
  (exactly four identities at a lattice vertex);
* an *independent* plain-loop reference renderer for oblique, acute-angle and
  near-degenerate lattices, compared on the full tiled image;
* seam / corner symmetry, periodic world-render translation invariance,
  coverage multiplicity (union, never double-darkened) and full cover by a
  very large nib spanning several neighbours;
* rejection of collinear / zero / non-finite bases and bad brush radii;
* both basis-change semantics (deform vs physical-stroke rewrap);
* export JSON recording the repeat basis and save/load image equality;
* HTTP endpoints driving the real lattice implementation.
"""

import json
import math
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seamless_paint.kernel import SeamlessTile, SeamlessInputError, HAVE_NUMPY
from seamless_paint.lattice import (
    Lattice,
    brush_metric,
    support_extents,
    enumerate_cell_ids,
    cell_ranges_for_stamp,
)
from seamless_paint.history import save_project, load_project


def as_lists(img):
    if HAVE_NUMPY and not isinstance(img, list):
        return img.tolist()
    return img


def alpha_grid(tile):
    img = as_lists(tile.to_uint8())
    return [[px[3] for px in row] for row in img]


# --------------------------------------------------------------------------- #
# Independent reference: direct ellipse/copy loops, no kernel geometry calls.
# --------------------------------------------------------------------------- #
def _ref_coverage(e, soft):
    if soft <= 0.0:
        return 1.0 if e <= 1.0 + 1e-9 else 0.0
    if e <= 1.0 - soft + 1e-9:
        return 1.0
    if e >= 1.0:
        return 0.0
    return (1.0 - e) / soft


def reference_lattice_chart(width, height, lattice, dabs,
                            background=(0.0, 0.0, 0.0, 0.0)):
    """Plain-loop independent RGBA reference for the W x H chart.

    Each dab is ``dict(x, y, rx, ry, angle, color=(r,g,b,a), soft, policy)``.
    Copy ranges come from a deliberately conservative row-norm bound (a
    different derivation than the kernel's eigendecomposition), and periodic
    coverages are combined with the same union rule the specification calls
    for.  Buffer is premultiplied so quantization matches ``to_uint8``.
    """
    L = lattice
    buf = [[[background[k] * background[3] if k < 3 else background[3]
             for k in range(4)] for _ in range(width)]
           for _ in range(height)]

    for dab in dabs:
        x, y = dab["x"], dab["y"]
        rx, ry = dab.get("rx", dab.get("radius", 1.0)), \
                 dab.get("ry", dab.get("radius", 1.0))
        ang = dab.get("angle", 0.0)
        soft = dab.get("soft", 0.25)
        policy = dab.get("policy", "neighbor")
        cr, cg, cb, ca = dab["color"]
        rmax = max(rx, ry)
        # Conservative cell-coordinate support extents via inverse row norms.
        es = rmax * math.hypot(L.m00, L.m01) + 1e-9
        et = rmax * math.hypot(L.m10, L.m11) + 1e-9
        sc, tc = L.world_to_cell(x, y)
        i0 = math.floor(-es - sc) - 1
        i1 = math.ceil(1.0 + es - sc) + 1
        j0 = math.floor(-et - tc) - 1
        j1 = math.ceil(1.0 + et - tc) + 1
        c, sn = math.cos(ang), math.sin(ang)

        for row in range(height):
            t = (row + 0.5) / height
            for col in range(width):
                s = (col + 0.5) / width
                px, py = L.cell_to_world(s, t)
                union_one = 1.0
                best_e2 = math.inf
                for jj in range(j0, j1 + 1):
                    for ii in range(i0, i1 + 1):
                        cx = x + ii * L.ax + jj * L.bx
                        cy = y + ii * L.ay + jj * L.by
                        dx, dy = px - cx, py - cy
                        qx = c * dx + sn * dy
                        qy = -sn * dx + c * dy
                        e2 = (qx / rx) ** 2 + (qy / ry) ** 2
                        if e2 > 1.0 + 1e-7:
                            continue
                        e = math.sqrt(max(0.0, e2))
                        cov = _ref_coverage(e, soft)
                        if cov <= 0.0:
                            continue
                        if policy == "lattice":
                            if e2 < best_e2:
                                best_e2 = e2
                        else:
                            union_one *= 1.0 - cov
                if policy == "lattice":
                    if not math.isfinite(best_e2):
                        continue
                    coverage = _ref_coverage(math.sqrt(best_e2), soft)
                else:
                    coverage = 1.0 - union_one
                if coverage <= 0.0:
                    continue
                a = ca * coverage
                pxbuf = buf[row][col]
                ia = 1.0 - a
                pxbuf[0] = cr * a + pxbuf[0] * ia
                pxbuf[1] = cg * a + pxbuf[1] * ia
                pxbuf[2] = cb * a + pxbuf[2] * ia
                pxbuf[3] = a + pxbuf[3] * ia

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


def samples_to_dabs(tile, color_override=None):
    dabs = []
    for rec in tile.history:
        col = color_override or tuple(rec["color"])
        r = float(rec["radius"])
        ang = float(rec.get("angle", 0.0))
        asp = float(rec.get("aspect", 1.0))
        rx = rec.get("rx")
        ry = rec.get("ry")
        if rx is None:
            rx = r
        if ry is None:
            ry = r * asp
        for x, y in rec["samples"]:
            dabs.append({
                "x": float(x), "y": float(y),
                "rx": float(rx), "ry": float(ry),
                "angle": ang, "color": col,
                "soft": float(rec.get("soft", 0.25)),
                "policy": rec.get("policy", "neighbor"),
            })
    return dabs


def independent_stroke_dabs(lattice, points, *, radius, color,
                            soft=0.25, angle=0.0, aspect=1.0,
                            spacing=None, policy="neighbor"):
    """Re-derive periodic unwrap + arc-length samples independently."""
    raw = [(float(x), float(y)) for x, y in points]
    folded = []
    for x, y in raw:
        s, t = lattice.world_to_cell(x, y)
        folded.append((s - math.floor(s), t - math.floor(t)))

    unwrapped = [folded[0]]
    os_ = ot = 0.0
    for k in range(1, len(folded)):
        ds = folded[k][0] - folded[k - 1][0]
        dt = folded[k][1] - folded[k - 1][1]
        best_shift, best_len = (0, 0), math.inf
        for js in (-1, 0, 1):
            for jt in (-1, 0, 1):
                wx, wy = lattice.cell_to_world(ds + js, dt + jt)
                ln = wx * wx + wy * wy
                if ln < best_len:
                    best_len, best_shift = ln, (js, jt)
        ds += best_shift[0]
        dt += best_shift[1]
        os_ += ds
        ot += dt
        unwrapped.append((folded[0][0] + os_, folded[0][1] + ot))

    world_pts = [lattice.cell_to_world(s, t) for s, t in unwrapped]
    if spacing is None:
        spacing = max(0.5, min(float(radius) * 0.25, 4.0))
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

    rx = float(radius)
    ry = float(radius) * float(aspect)
    return [{
        "x": x, "y": y, "rx": rx, "ry": ry, "angle": angle,
        "color": tuple(color), "soft": soft, "policy": policy,
    } for x, y in samples]


def history_to_independent_dabs(tile):
    dabs = []
    for rec in tile.history:
        dabs.extend(independent_stroke_dabs(
            tile.lattice, rec["raw_points"],
            radius=rec["radius"], color=rec["color"],
            soft=rec["soft"], angle=rec.get("angle", 0.0),
            aspect=rec.get("aspect", 1.0),
            spacing=rec.get("spacing"), policy=rec.get("policy", "neighbor")))
    return dabs


def max_image_diff(a, b):
    worst = 0
    mean_sum = 0
    count = 0
    for row_a, row_b in zip(a, b):
        for pa, pb in zip(row_a, row_b):
            for ka in range(4):
                d = abs(int(pa[ka]) - int(pb[ka]))
                worst = max(worst, d)
                mean_sum += d
                count += 1
    return worst, mean_sum / max(1, count)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class LatticeGeometryTests(unittest.TestCase):
    def test_world_cell_roundtrip_and_default_basis(self):
        L = Lattice(64, 0, 0, 48)
        self.assertEqual((L.ax, L.ay, L.bx, L.by), (64.0, 0.0, 0.0, 48.0))
        for x, y in [(0, 0), (13, -27), (200, 91), (-40, 33)]:
            s, t = L.world_to_cell(x, y)
            wx, wy = L.cell_to_world(s, t)
            self.assertAlmostEqual(wx, x, places=9)
            self.assertAlmostEqual(wy, y, places=9)

        L2 = Lattice(40, 0, 12, 36)
        for x, y in [(5, 8), (38, 30), (-70, 120)]:
            s, t = L2.world_to_cell(x, y)
            wx, wy = L2.cell_to_world(s, t)
            self.assertAlmostEqual(wx, x, places=8)
            self.assertAlmostEqual(wy, y, places=8)

        tile = SeamlessTile(72, 56)
        self.assertTrue(tile._legacy_ortho)
        self.assertEqual(tile.lattice, Lattice.orthogonal(72, 56))

    def test_reject_collinear_zero_and_nonfinite(self):
        for kwargs in [
            dict(ax=64, ay=0, bx=32, by=0),       # collinear horizontal
            dict(ax=10, ay=10, bx=20, by=20),     # collinear diagonal
            dict(ax=0, ay=0, bx=0, by=40),        # zero vector
            dict(ax=1, ay=2, bx=float("nan"), by=4),
            dict(ax=float("inf"), ay=0, bx=0, by=4),
        ]:
            with self.assertRaises(SeamlessInputError):
                Lattice(**kwargs)

        # Near-degenerate lattice accepted by default (geometry still valid),
        # rejected only when the caller opts into the sine floor.
        acute = Lattice(32, 0, 31, 3)
        self.assertLess(acute.relative_det, 0.1)
        with self.assertRaises(SeamlessInputError):
            Lattice(32, 0, 31, 3, reject_near_degenerate=True, min_sine=0.2)
        Lattice(32, 0, 31, 3, reject_near_degenerate=True, min_sine=0.05)

    def test_bad_brush_radii_rejected(self):
        with self.assertRaises(SeamlessInputError):
            brush_metric(0.0)
        with self.assertRaises(SeamlessInputError):
            brush_metric(10.0, aspect=0.0)
        tile = SeamlessTile(16, 16)
        with self.assertRaises(SeamlessInputError):
            tile.stamp(4, 4, -2, (0, 0, 0, 1))
        with self.assertRaises(SeamlessInputError):
            tile.stamp(4, 4, 3, (0, 0, 0, 1), policy="bogus")

    def test_vertex_dab_enumerates_exactly_four_unique_identities(self):
        L = Lattice(40, 0, 20, 35)
        ids = enumerate_cell_ids(L, 0.0, 0.0, radius=3.0, aspect=0.6, angle=0.4)
        self.assertEqual(sorted(ids), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(len(ids), len(set(ids)))

        tile = SeamlessTile(32, 32, lattice=Lattice(32, 0, 10, 30))
        report = tile.stamp(0, 0, 4.0, (0, 0, 0, 1), soft=0.0)
        self.assertEqual(report["cell_count"], 4)
        self.assertEqual(len(set(report["cells"])), 4)

    def test_neighbour_range_is_support_driven(self):
        L = Lattice(24, 0, 8, 20)
        q_small = brush_metric(5.0)[0:3]
        es_s, et_s = support_extents(L, *q_small)
        _, _, il0, ih0, jl0, jh0 = cell_ranges_for_stamp(L, 12, 10, es_s, et_s)
        n_small = (ih0 - il0 + 1) * (jh0 - jl0 + 1)

        q_big = brush_metric(60.0)[0:3]
        es_b, et_b = support_extents(L, *q_big)
        _, _, il1, ih1, jl1, jh1 = cell_ranges_for_stamp(L, 12, 10, es_b, et_b)
        ni = ih1 - il1 + 1
        nj = jh1 - jl1 + 1
        self.assertGreater(ni, 3)
        self.assertGreater(nj, 3)
        self.assertGreater((ih1 - il1 + 1) * (jh1 - jl1 + 1), n_small * 4)

        ids = enumerate_cell_ids(L, 12, 10, radius=60.0)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len(ids), 9)  # never a fixed 3x3 neighbourhood

    def test_support_extents_match_ellipse_axis_extrema(self):
        # A unit-radius circle has cell extents equal to inverse-row norms.
        L = Lattice(40, 0, 12, 36)
        q00, q01, q11, _, _ = brush_metric(1.0)
        es, et = support_extents(L, q00, q01, q11)
        self.assertAlmostEqual(es, math.hypot(L.m00, L.m01), places=8)
        self.assertAlmostEqual(et, math.hypot(L.m10, L.m11), places=8)


# --------------------------------------------------------------------------- #
# Independent reference comparisons on oblique / acute lattices
# --------------------------------------------------------------------------- #
class LatticeReferenceRenderTests(unittest.TestCase):
    def _compare(self, W, H, L, dabs, tol=2):
        tile = SeamlessTile(W, H, background=(0, 0, 0, 0), lattice=L)
        for d in dabs:
            tile.stamp(
                d["x"], d["y"], d.get("radius", d["rx"]),
                d["color"], soft=d.get("soft", 0.25),
                policy=d.get("policy", "neighbor"),
                angle=d.get("angle", 0.0),
                rx=d.get("rx"), ry=d.get("ry"),
            )
        got = as_lists(tile.to_uint8())
        ref = reference_lattice_chart(W, H, L, dabs)
        worst, mean = max_image_diff(got, ref)
        self.assertLessEqual(worst, tol,
                             f"worst diff {worst} (mean {mean:.4f})")
        return tile, got, ref, worst

    def test_oblique_60deg_rotated_nibs_match_reference(self):
        L = Lattice(40, 0, 20, int(round(40 * math.sqrt(3) / 2)))
        dabs = [
            dict(x=5, y=8, rx=6.0, ry=2.2, angle=math.radians(35),
                 color=(0.9, 0.2, 0.5, 1.0), soft=0.0),
            dict(x=38, y=30, rx=5.0, ry=5.0, angle=0.0,
                 color=(0.1, 0.6, 0.7, 0.7), soft=0.3),
            dict(x=0, y=0, rx=4.0, ry=1.5, angle=math.radians(60),
                 color=(0, 0, 0, 1.0), soft=0.2),
            dict(x=40, y=0, rx=5.5, ry=2.5, angle=math.radians(-20),
                 color=(0.9, 0.8, 0.1, 0.85), soft=0.15),
        ]
        self._compare(40, 40, L, dabs)

    def test_acute_angle_cell_matches_reference(self):
        L = Lattice(34, 0, 33, 6)  # ~10.3 degree acute cells
        dabs = [
            dict(x=12, y=10, rx=5.0, ry=1.6, angle=0.5,
                 color=(0.8, 0.1, 0.2, 0.9), soft=0.25),
            dict(x=33, y=6, rx=4.0, ry=4.0, angle=0.0,
                 color=(0.1, 0.3, 0.9, 0.8), soft=0.1),
            dict(x=0, y=0, rx=3.5, ry=1.2, angle=1.1,
                 color=(0, 0, 0, 1.0), soft=0.0),
        ]
        self._compare(34, 26, L, dabs)

    def test_near_degenerate_lattice_matches_reference(self):
        L = Lattice(32, 0, 31, 3)  # ~5.7 degrees
        dabs = [
            dict(x=16, y=16, rx=5.0, ry=2.0, angle=-0.6,
                 color=(0.2, 0.7, 0.3, 0.9), soft=0.3),
            dict(x=0, y=0, rx=3.0, ry=1.1, angle=0.9,
                 color=(0, 0, 0, 1.0), soft=0.2),
        ]
        self._compare(32, 20, L, dabs, tol=2)

    def test_lattice_nearest_copy_policy_matches_reference(self):
        L = Lattice(36, 0, 14, 32)
        dabs = [
            dict(x=10, y=12, rx=7.0, ry=3.0, angle=0.7,
                 color=(0.6, 0.2, 0.8, 0.8), soft=0.35, policy="lattice"),
        ]
        self._compare(36, 32, L, dabs)

    def test_full_tiled_3x3_diff_against_reference(self):
        """The acceptance check: whole 3x3 tiled image vs independent ref."""
        W = H = 36
        L = Lattice(W, 0, W / 2, W * 0.8660254)
        tile = SeamlessTile(W, H, background=(1, 1, 1, 1), lattice=L)
        tile.add_stroke([(-12, 10), (18, 20), (44, 8)], 4.5,
                        (0.84, 0.13, 0.42, 0.95), soft=0.2,
                        angle=0.6, aspect=0.4, spacing=2.0)
        tile.add_stroke([(0, 0)], 3.5, (0.1, 0.1, 0.15, 1), soft=0.15,
                        angle=-0.4, aspect=0.5)

        # Independent reference re-derives unwrap/resampling from raw points,
        # so a kernel sampling bug cannot validate itself via stored samples.
        dabs = history_to_independent_dabs(tile)
        ref_chart = reference_lattice_chart(W, H, L, dabs,
                                            background=(1, 1, 1, 1))
        # Build the 3x3 chart montage independently.
        ref_tiled = [[None] * (W * 3) for _ in range(H * 3)]
        for ry in range(3):
            for rx in range(3):
                for r in range(H):
                    for c in range(W):
                        ref_tiled[ry * H + r][rx * W + c] = ref_chart[r][c]
        got = as_lists(tile.to_uint8(copies=3))
        worst, mean = max_image_diff(got, ref_tiled)
        self.assertLessEqual(worst, 2, f"tiled worst diff {worst}")
        self.assertLess(mean, 0.05)


# --------------------------------------------------------------------------- #
# Seams, symmetry and coverage multiplicity
# --------------------------------------------------------------------------- #
class SeamAndCoverageTests(unittest.TestCase):
    def test_lattice_vertex_four_corner_opposite_symmetry(self):
        L = Lattice(32, 0, 10, 30)
        tile = SeamlessTile(32, 32, lattice=L)
        tile.stamp(0, 0, 7.0, (0, 0, 0, 1), soft=0.0,
                   angle=0.7, aspect=0.4)
        img = as_lists(tile.to_uint8())

        def px(r, c):
            return tuple(img[r][c][:3])

        for k in range(8):
            # +s+t versus -s-t, and -s+t versus +s-t.
            self.assertEqual(px(k, k), px(31 - k, 31 - k))
            self.assertEqual(px(k, 31 - k), px(31 - k, k))

    def test_world_render_is_invariant_under_basis_translations(self):
        L = Lattice(44, 0, 15, 38)
        tile = SeamlessTile(44, 38, lattice=L)
        tile.add_stroke([(-20, 8), (18, 22), (50, 30)], 5.0,
                        (0.9, 0.2, 0.4, 0.9), soft=0.3,
                        angle=0.5, aspect=0.6, spacing=2.0)
        tile.stamp(0, 0, 4, (0, 0, 0, 1), soft=0.2, angle=0.9, aspect=0.5)

        rect = (-30.0, -25.0, 90.0, 85.0)
        base = as_lists(tile.render_world_uint8(*rect, 120, 110))
        for dx, dy, name in [(L.ax, L.ay, "a"), (L.bx, L.by, "b")]:
            shifted = as_lists(tile.render_world_uint8(
                rect[0] + dx, rect[1] + dy,
                rect[2] + dx, rect[3] + dy, 120, 110))
            self.assertEqual(shifted, base,
                             f"world render not periodic under basis {name}")

    def test_huge_nib_spans_neighbours_and_fully_covers(self):
        L = Lattice(28, 0, 9, 24)
        tile = SeamlessTile(28, 22, background=(0, 0, 0, 0), lattice=L)
        report = tile.stamp(14, 11, 60.0, (0, 0, 0, 1.0), soft=0.0)
        self.assertGreater(report["cell_count"], 9)
        alpha = alpha_grid(tile)
        self.assertEqual(min(min(row) for row in alpha), 255)

    def test_overlapping_copies_union_without_double_darkening(self):
        L = Lattice(24, 0, 8, 20)
        tile = SeamlessTile(24, 20, background=(0, 0, 0, 0), lattice=L)
        # A very large soft nib guarantees several copies overlap each pixel;
        # union coverage <= 1 means deposited alpha never exceeds color alpha.
        tile.stamp(12, 10, 40.0, (0, 0, 0, 0.5), soft=0.6)
        alpha = alpha_grid(tile)
        worst = max(max(row) for row in alpha)
        self.assertLessEqual(worst, 129)
        self.assertGreater(worst, 100)  # and the dab clearly landed

        # Two stamps still composite normally (source-over), so paint is not
        # incorrectly unioned across separate dabs.
        tile2 = SeamlessTile(8, 8, background=(0, 0, 0, 0))
        tile2.stamp(4, 4, 1.5, (0, 0, 0, 0.5), soft=0.0)
        a1 = alpha_grid(tile2)[4][4]
        tile2.stamp(4, 4, 1.5, (0, 0, 0, 0.5), soft=0.0)
        a2 = alpha_grid(tile2)[4][4]
        self.assertGreater(a2, a1)

    def test_stamp_report_matches_enumeration(self):
        L = Lattice(40, 0, 12, 36)
        tile = SeamlessTile(40, 36, lattice=L)
        report = tile.stamp(17, 19, 9.0, (0.2, 0.4, 0.8, 0.8),
                            soft=0.3, angle=0.4, aspect=0.5)
        ids = enumerate_cell_ids(L, 17, 19, radius=9.0,
                                 aspect=0.5, angle=0.4)
        self.assertEqual(sorted(report["cells"]), sorted(ids))
        self.assertEqual(report["cell_count"], len(ids))


# --------------------------------------------------------------------------- #
# Stroke unwrap / resampling fidelity
# --------------------------------------------------------------------------- #
class LatticeStrokePathTests(unittest.TestCase):
    def test_samples_trace_input_length_and_spacing(self):
        L = Lattice(60, 0, 25, 52)
        tile = SeamlessTile(60, 52, lattice=L)
        pts = [(x, 22 + 7 * math.sin(x / 8.0))
               for x in range(-18, 82, 3)]
        rec = tile.add_stroke(pts, 5.0, (0, 0, 0, 1),
                              spacing=1.5, angle=0.4, aspect=0.6)
        sp = rec["samples"]
        gaps = [math.hypot(sp[i + 1][0] - sp[i][0],
                           sp[i + 1][1] - sp[i][1])
                for i in range(len(sp) - 1)]
        # Dabs sit on the requested arc-length spacing; cross-joint gaps may
        # be slightly shorter because the chord cuts a polyline corner.
        self.assertLessEqual(max(gaps[:-1]), 1.5 + 1e-6)
        self.assertGreater(min(gaps[:-1]), 1.3)
        raw_len = sum(math.hypot(pts[i + 1][0] - pts[i][0],
                                 pts[i + 1][1] - pts[i][1])
                      for i in range(len(pts) - 1))
        sample_len = sum(gaps)
        self.assertLess(abs(sample_len - raw_len), 3.0)  # <= ~2 spacings

    def test_stroke_crossing_vertex_keeps_short_continuous_path(self):
        L = Lattice(48, 0, 24, int(round(48 * 0.8660254)))
        tile = SeamlessTile(48, 42, lattice=L)
        # Dense straight line driven straight through the origin vertex.
        pts = [(-30 + k * 2.0, 12 - 0.4 * (-30 + k * 2.0))
               for k in range(31)]
        rec = tile.add_stroke(pts, 3.0, (0.1, 0.2, 0.8, 0.9),
                              spacing=1.0, aspect=0.7, angle=0.3)
        sp = rec["samples"]
        gaps = [math.hypot(sp[i + 1][0] - sp[i][0],
                           sp[i + 1][1] - sp[i][1])
                for i in range(len(sp) - 1)]
        self.assertLess(max(gaps), 1.0 + 1e-6)
        raw_len = sum(math.hypot(pts[i + 1][0] - pts[i][0],
                                 pts[i + 1][1] - pts[i][1])
                      for i in range(len(pts) - 1))
        self.assertLess(abs(sum(gaps) - raw_len), 2.5)

    def test_stroke_render_matches_independent_raw_reference(self):
        L = Lattice(40, 0, 14, 36)
        W = H = 40
        tile = SeamlessTile(W, H, background=(0, 0, 0, 0), lattice=L)
        pts = [(-12, 8), (-2, 14), (12, 9), (26, 18), (46, 12)]
        tile.add_stroke(pts, 4.5, (0.8, 0.1, 0.4, 0.95), soft=0.25,
                        angle=0.7, aspect=0.45, spacing=1.6)
        dabs = history_to_independent_dabs(tile)
        ref = reference_lattice_chart(W, H, L, dabs)
        worst, mean = max_image_diff(as_lists(tile.to_uint8()), ref)
        self.assertLessEqual(worst, 2, f"worst {worst} mean {mean}")


# --------------------------------------------------------------------------- #
# Basis-change semantics
# --------------------------------------------------------------------------- #
class BasisChangeTests(unittest.TestCase):
    def _populated(self, lattice=None):
        tile = SeamlessTile(48, 40, lattice=lattice)
        tile.add_stroke([(-5, 10), (20, 22), (50, 30)], 5.0,
                        (1, 0.2, 0.3, 0.8), soft=0.3,
                        angle=0.5, aspect=0.5, spacing=2.0)
        tile.add_stroke([(2, 2)], 4, (0.1, 0.1, 0.2, 1), soft=0.2,
                        angle=0.8, aspect=0.6)
        return tile

    def test_deform_preserves_chart_and_replays_identically(self):
        tile = self._populated(Lattice.orthogonal(48, 40))
        before = as_lists(tile.to_uint8())
        report = tile.change_basis(Lattice(50, -4, 10, 42), mode="deform")
        self.assertTrue(report["preserved_buffer"])
        after = as_lists(tile.to_uint8())
        self.assertEqual(after, before)  # retained buffer, literally

        # Rewritten history must replay to the exact same preserved chart.
        tile.replay()
        replayed = as_lists(tile.to_uint8())
        self.assertEqual(replayed, before)

    def test_deform_transforms_samples_and_metric(self):
        oldL = Lattice.orthogonal(48, 40)
        tile = self._populated(oldL)
        rec = tile.history[0]
        old_sample = rec["samples"][1][:]
        q00, q01, q11, _, _ = brush_metric(
            5.0, aspect=0.5, angle=0.5)
        newL = Lattice(50, -4, 10, 42)
        tile.change_basis(newL, mode="deform")
        rec2 = tile.history[0]
        # T = B_new B_old^{-1}; for the ortho old basis a world point maps
        # through p = (x/W)*a_new + (y/H)*b_new.
        x, y = old_sample
        expect = newL.cell_to_world(x / 48.0, y / 40.0)
        self.assertAlmostEqual(rec2["samples"][1][0], expect[0], places=6)
        self.assertAlmostEqual(rec2["samples"][1][1], expect[1], places=6)

        # Q' = T^{-T} Q T^{-1}; T for the ortho case is diag(ax'/W, by'/H)
        # plus the skew terms, so verify numerically with the full map.
        t00 = newL.ax / 48.0
        t01 = newL.bx / 40.0
        t10 = newL.ay / 48.0
        t11 = newL.by / 40.0
        det_t = t00 * t11 - t01 * t10
        i00, i01 = t11 / det_t, -t01 / det_t
        i10, i11 = -t10 / det_t, t00 / det_t
        a00 = q00 * i00 + q01 * i10
        a01 = q00 * i01 + q01 * i11
        a10 = q01 * i00 + q11 * i10
        a11 = q01 * i01 + q11 * i11
        e00 = i00 * a00 + i10 * a10
        e01 = i00 * a01 + i10 * a11
        e11 = i01 * a01 + i11 * a11
        self.assertAlmostEqual(rec2["metric"][0], e00, places=9)
        self.assertAlmostEqual(rec2["metric"][1], e01, places=9)
        self.assertAlmostEqual(rec2["metric"][2], e11, places=9)

    def test_rewrap_keeps_physical_samples_and_matches_fresh_replay(self):
        L1 = Lattice(48, 0, 12, 40)
        tile = self._populated(L1)
        samples_before = [p[:] for p in tile.history[0]["samples"]]
        L2 = Lattice(46, 3, 8, 42)
        tile.change_basis(L2, mode="rewrap")
        samples_after = tile.history[0]["samples"]
        for a, b in zip(samples_before, samples_after):
            self.assertAlmostEqual(a[0], b[0], places=9)
            self.assertAlmostEqual(a[1], b[1], places=9)

        rewrapped = as_lists(tile.to_uint8())
        fresh = SeamlessTile(48, 40, lattice=L2)
        for rec in tile.history:
            fresh._render_record(rec)
        self.assertEqual(as_lists(fresh.to_uint8()), rewrapped)

    def test_invalid_mode_rejected(self):
        tile = SeamlessTile(16, 16)
        with self.assertRaises(SeamlessInputError):
            tile.change_basis(Lattice(16, 0, 8, 16), mode="explode")


# --------------------------------------------------------------------------- #
# Persistence / export format
# --------------------------------------------------------------------------- #
class LatticePersistenceTests(unittest.TestCase):
    def test_history_records_repeat_basis_and_reloads_equal(self):
        tmp = tempfile.mkdtemp()
        L = Lattice(48, 0, 24, int(round(48 * 0.8660254)))
        tile = SeamlessTile(48, 48, lattice=L)
        tile.add_stroke([(-8, 12), (20, 26), (52, 18)], 4.0,
                        (0.2, 0.5, 0.9, 0.9), soft=0.25,
                        angle=0.7, aspect=0.45, spacing=2.0)
        paths = save_project(tile, tmp, "lat", write_world_tiling=True)
        self.assertTrue(os.path.exists(paths["lattice_tiled"]))

        with open(paths["history"], encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(payload["version"], 2)
        rb = payload["repeat_basis"]
        self.assertAlmostEqual(rb["a"][0], L.ax, places=6)
        self.assertAlmostEqual(rb["b"][1], L.by, places=6)
        self.assertNotEqual(rb["kind"], "orthogonal")
        self.assertIn("lattice", payload)

        loaded = load_project(paths["history"])
        self.assertEqual(loaded.lattice, L)
        worst, _ = max_image_diff(as_lists(tile.to_uint8()),
                                  as_lists(loaded.to_uint8()))
        self.assertLessEqual(worst, 1)

    def test_undo_keeps_history_bookkeeping_consistent(self):
        tile = SeamlessTile(24, 24)
        tile.add_stroke([(2, 2), (10, 10)], 3.0, (0, 0, 0, 0.8))
        tile.add_stroke([(20, 2), (12, 12)], 3.0, (0.8, 0, 0, 0.8))
        self.assertEqual(len(tile.history), 2)
        self.assertIsNotNone(tile.undo())
        self.assertEqual(len(tile.history), 1)
        self.assertIsNotNone(tile.undo())
        self.assertEqual(len(tile.history), 0)
        self.assertIsNone(tile.undo())

    def test_legacy_history_without_basis_loads_orthogonal(self):
        tmp = tempfile.mkdtemp()
        hp = os.path.join(tmp, "old.json")
        with open(hp, "w", encoding="utf-8") as fh:
            json.dump({"width": 16, "height": 12,
                       "background": [1, 1, 1, 1], "strokes": []}, fh)
        t = load_project(hp)
        self.assertEqual(t.lattice, Lattice.orthogonal(16, 12))
        self.assertTrue(t._legacy_ortho)


# --------------------------------------------------------------------------- #
# HTTP / UI backend endpoints
# --------------------------------------------------------------------------- #
class LatticeWebEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from seamless_paint.webapp import create_server
        cls.tmp = tempfile.mkdtemp()
        cls.httpd = create_server("127.0.0.1", 0, 64, 64, cls.tmp)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path):
        with urllib.request.urlopen(self._url(path), timeout=20) as r:
            return r.status, r.read(), r.headers.get("Content-Type")

    def _post(self, path, payload):
        req = urllib.request.Request(
            self._url(path), data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_lattice_endpoints_end_to_end(self):
        status, st = self._post("/api/lattice", {
            "ax": 64, "ay": 0, "bx": 32, "by": 55.4, "mode": "rewrap"})
        self.assertEqual(status, 200)
        self.assertTrue(st["ok"])
        self.assertAlmostEqual(st["lattice"]["angle_deg"], 60.0, delta=0.2)
        self.assertFalse(st["legacy_ortho"])
        self.assertIn("view", st)

        status, resp = self._post("/api/stroke", {
            "points": [[-10, 8], [20, 20], [70, 40]],
            "radius": 5, "color": [0.8, 0.1, 0.4, 0.95],
            "soft": 0.2, "angle_deg": 35, "aspect": 0.45})
        self.assertEqual(status, 200)
        self.assertGreater(len(resp["stroke"]["last_cells"]), 0)

        _, png, ctype = self._get("/api/world.png?w=128&h=128")
        self.assertEqual(ctype, "image/png")
        self.assertTrue(png.startswith(b"\x89PNG"))

        _, png2, _ = self._get("/api/lattice_preview.png?copies=3&max=256")
        self.assertTrue(png2.startswith(b"\x89PNG"))

        _, png3, _ = self._get("/api/download/lattice_tile?copies=3")
        self.assertTrue(png3.startswith(b"\x89PNG"))

        status, prev = self._post("/api/basis/preview", {
            "ax": 60, "ay": 2, "bx": 28, "by": 58, "mode": "deform"})
        self.assertEqual(status, 200)
        self.assertTrue(prev["world_png"].startswith("data:image/png;base64,"))

        code, bad = self._post("/api/lattice",
                               {"ax": 64, "ay": 0, "bx": 32, "by": 0})
        self.assertEqual(code, 400)
        self.assertFalse(bad["ok"])
        self.assertIn("error", bad)

        # Session must remain usable after the rejected basis.
        _, state_bytes, _ = self._get("/api/state")
        self.assertTrue(json.loads(state_bytes)["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
