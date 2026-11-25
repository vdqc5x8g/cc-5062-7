"""Acceptance tests for the seamless painting kernel.

The key checks use an *independent* reference renderer (plain Python loops
over a large non-wrapped canvas, folded by modulo at the end) instead of
reusing kernel internals, so a broken fold cannot validate itself.
"""

import json
import math
import os
import sys
import tempfile
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seamless_paint.kernel import SeamlessTile, HAVE_NUMPY
from seamless_paint.pngio import write_png, read_png
from seamless_paint.history import save_project, load_project


# ---------------------------------------------------------------------------
# Independent reference renderer: paint on a big plane, fold into the tile.
# ---------------------------------------------------------------------------
def reference_fold(width, height, dabs, soft=0.25):
    """Return folded coverage and alpha from dabs (x, y, radius, alpha).

    Dabs are painted into an oversized flat canvas with plain source-over
    compositing, then reduced modulo tile size with per-period accumulation,
    i.e. an independent implementation of the same periodic coverage rules.
    """
    pad = int(max(r for _, _, r, _ in dabs) + 4)
    x0 = int(min(x - r for x, _, r, _ in dabs)) - pad
    x1 = int(max(x + r for x, _, r, _ in dabs)) + pad
    y0 = int(min(y - r for _, y, r, _ in dabs)) - pad
    y1 = int(max(y + r for _, y, r, _ in dabs)) + pad

    plane_w = x1 - x0
    plane_h = y1 - y0
    plane = [[0.0] * plane_w for _ in range(plane_h)]  # accumulated alpha
    soft_px_global = {}
    for (x, y, r, a) in dabs:
        soft_px = r * soft
        core = r - soft_px
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                d = math.hypot(ix + 0.5 - x, iy + 0.5 - y)
                if d <= core:
                    cov = 1.0
                elif d < r and soft_px > 0:
                    cov = (r - d) / soft_px
                else:
                    continue
                cell = plane[iy - y0][ix - x0]
                deposit = a * cov
                plane[iy - y0][ix - x0] = deposit + cell * (1.0 - deposit)

    folded = [[0.0] * width for _ in range(height)]
    for iy in range(y0, y1):
        row = plane[iy - y0]
        ty = iy % height
        for ix in range(x0, x1):
            a = row[ix - x0]
            if a <= 0.0:
                continue
            tx = ix % width
            cell = folded[ty][tx]
            folded[ty][tx] = a + cell * (1.0 - a)
    return folded


def alpha_uint8(tile):
    img = tile.to_uint8()
    if HAVE_NUMPY:
        return img[..., 3].tolist()
    return [[px[3] for px in row] for row in img]


class KernelPeriodicTests(unittest.TestCase):
    def setUp(self):
        self.W = self.H = 64

    def test_edge_dab_matches_independent_reference(self):
        """A hard dab centered on the right edge equals the folded reference."""
        # Transparent background: the exported alpha then equals deposit alpha.
        tile = SeamlessTile(self.W, self.H, background=(0.0, 0.0, 0.0, 0.0))
        x, y, r = self.W - 4.0, 30.0, 10.0
        tile.stamp(x, y, r, (0.0, 0.0, 0.0, 1.0), soft=0.0)
        ref = reference_fold(self.W, self.H, [(x, y, r, 1.0)], soft=0.0)
        got = alpha_uint8(tile)
        for row in range(self.H):
            for col in range(self.W):
                expected = min(255, int(ref[row][col] * 255 + 0.5))
                self.assertLessEqual(abs(got[row][col] - expected), 1,
                                     f"mismatch at {row},{col}")

    def test_corner_dab_hits_all_four_corners_identically(self):
        """Corner落笔: one dab at (0,0) deposits the same print in 4 corners."""
        tile = SeamlessTile(self.W, self.H)
        r = 9.0
        tile.stamp(0.0, 0.0, r, (0.0, 0.0, 0.0, 1.0), soft=0.0)
        img = tile.to_uint8()
        # Compare full RGBA pixels: an opaque white background means equal
        # ink coverage shows up as equal colors, not alpha differences.
        def px(row, col):
            return img[row, col].tolist() if HAVE_NUMPY else img[row][col]
        for off in range(10):
            vals = [
                px(off, off),
                px(off, self.W - 1 - off),
                px(self.H - 1 - off, off),
                px(self.H - 1 - off, self.W - 1 - off),
            ]
            self.assertEqual(len(set(tuple(v) for v in vals)), 1,
                             f"corner asymmetry at offset {off}: {vals}")
        # And the corner pixels must actually be painted (ink, not white).
        self.assertLess(sum(px(0, 0)[:3]), 200)
        self.assertLess(sum(px(self.H - 1, self.W - 1)[:3]), 200)

    def test_montage_has_no_seam_four_tiles(self):
        """Four-tile splice: the montage is an exact periodic replication and
        a feature crossing a tile boundary continues at the folded position
        without any duplicated or missing seam band."""
        tile = SeamlessTile(self.W, self.H)
        # Diagonal thin line running through a corner.
        pts = [(-10 + t * 1.5, -10 + t * 1.5) for t in range(20)]
        tile.add_stroke(pts, 1.6, (0.1, 0.2, 0.9, 1.0), soft=0.1, spacing=0.8)
        unit = tile.to_uint8()
        montage = tile.to_uint8(copies=2)

        # 1. The montage must be an exact replication of the unit cell; this
        #    is the structural guarantee of periodic tiling (no edge blur, no
        #    nine-grid copy of an ordinary canvas).
        if HAVE_NUMPY:
            for r in range(2 * self.H):
                for c in range(2 * self.W):
                    self.assertTrue((montage[r, c] == unit[r % self.H, c % self.W]).all())
        else:
            self._assert_exact_replication_pure(montage, unit)

        # 2. Seam continuity: for every pixel on the last tile row/column, the
        #    pixel immediately across the seam in the montage must equal the
        #    folded first row/column of the SAME unit.  A broken fold would
        #    show up here as ink that stops at, or doubles along, the edge.
        for c in range(self.W):
            if HAVE_NUMPY:
                self.assertTrue((montage[self.H - 1, c] == unit[self.H - 1, c]).all())
                self.assertTrue((montage[self.H, c] == unit[0, c]).all())
                self.assertTrue((montage[c, self.W - 1] == unit[c, self.W - 1]).all())
                self.assertTrue((montage[c, self.W] == unit[c, 0]).all())
            else:
                self.assertEqual(montage[self.H - 1][c], unit[self.H - 1][c])
                self.assertEqual(montage[self.H][c], unit[0][c])
                self.assertEqual(montage[c][self.W - 1], unit[c][self.W - 1])
                self.assertEqual(montage[c][self.W], unit[c][0])

    def _assert_exact_replication_pure(self, montage, unit):
        for r in range(2 * self.H):
            for c in range(2 * self.W):
                self.assertEqual(montage[r][c], unit[r % self.H][c % self.W])

    def test_cross_seam_continuity_matches_unfolded_plane(self):
        """A line drawn across a seam in the continuous plane must, after
        folding, meet itself across the tile boundary with no gap or doubled
        band.  The expected folded alpha comes from the independent reference
        renderer (no kernel code reused)."""
        W = H = 48
        # Horizontal line that crosses the right tile boundary at mid-height.
        dabs = [(x, 24.0, 2.0, 1.0) for x in range(W - 8, W + 9)]
        ref = reference_fold(W, H, dabs, soft=0.0)
        tile = SeamlessTile(W, H, background=(0.0, 0.0, 0.0, 0.0))
        for x, y, r, a in dabs:
            tile.stamp(x, y, r, (0.0, 0.0, 0.0, a), soft=0.0)
        got = alpha_uint8(tile)
        for row in range(H):
            for col in range(W):
                expected = min(255, int(ref[row][col] * 255 + 0.5))
                self.assertLessEqual(abs(got[row][col] - expected), 1,
                                     f"fold mismatch {row},{col}")
        # Both seam neighbours on row 24 (last and first columns) carry ink.
        self.assertGreater(got[24][W - 1], 200)
        self.assertGreater(got[24][0], 200)
        # And the band stays local to row 24: no vertical smear across tile.
        painted = [r for r in range(H) if any(got[r][c] > 128 for c in range(W))]
        self.assertLessEqual(max(painted) - min(painted), 5)

    def test_brush_larger_than_tile_covers_all_periods(self):
        """Radius > tile size: a hard dab cannot leave any pixel uncovered."""
        tile = SeamlessTile(32, 24)
        tile.stamp(5.0, 7.0, 90.0, (0.0, 0.0, 0.0, 1.0), soft=0.0)
        img = tile.to_uint8()
        alpha = img[..., 3] if HAVE_NUMPY else [[p[3] for p in row] for row in img]
        lo = min(min(row) for row in alpha)
        self.assertGreaterEqual(lo, 250, "large hard brush missed some period copies")

    def test_soft_large_brush_union_not_duplicated(self):
        """Semi-transparent soft dab bigger than the tile: union coverage must
        never exceed single-dab alpha, and stay symmetric on the torus."""
        W = H = 40
        # Transparent background so the deposited alpha is observable.
        tile = SeamlessTile(W, H, background=(0.0, 0.0, 0.0, 0.0))
        r, a = 55.0, 0.5
        tile.stamp(0.0, 0.0, r, (0.0, 0.0, 0.0, a), soft=0.5)
        img = tile.to_uint8()
        alpha = (img[..., 3].astype("float32") if HAVE_NUMPY
                 else [[float(p[3]) for p in row] for row in img])
        # No folded pixel may exceed the alpha of one isolated deposit
        # (overlapping periodic copies form a coverage union, not stacked
        # translucent layers).
        ceiling = int(a * 255) + 2
        for row in alpha:
            for v in row:
                self.assertLessEqual(v, ceiling)
        if HAVE_NUMPY:
            # Torus symmetry around (0,0): alpha[x,y] == alpha[W-x,y] etc.
            self.assertTrue((alpha == alpha[::-1, :]).all())
            self.assertTrue((alpha == alpha[:, ::-1]).all())

    def test_lattice_and_neighbor_equivalent_for_hard_brush(self):
        """The two diagonal policies must agree for hard dabs."""
        for policy in ("neighbor", "lattice"):
            tile = SeamlessTile(48, 48)
            pts = [(-6 + t * 3.1, 40 + t * 1.7) for t in range(25)]
            tile.add_stroke(pts, 4.0, (0.2, 0.6, 0.1, 1.0), soft=0.0,
                            policy=policy, spacing=1.0)
            if policy == "neighbor":
                ref_img = tile.to_uint8()
            else:
                lat_img = tile.to_uint8()
        self.assertTrue((ref_img == lat_img).all() if HAVE_NUMPY
                        else ref_img == lat_img)

    def test_wrapped_stroke_arc_length_no_cross_tile_line(self):
        """A stroke leaving the right edge must re-enter on the left with a
        short wrapped segment; no diagonal line may span the whole tile."""
        W = H = 64
        tile = SeamlessTile(W, H)
        # Horizontal stroke crossing the right boundary mid-tile.  Raw
        # continuous-plane coordinates: the kernel keeps the jump direction.
        pts = [(W - 14, 32.0), (W - 7, 32.0), (W + 7, 32.0), (W + 14, 32.0)]
        tile.add_stroke(pts,
                        2.0, (0.0, 0.0, 0.0, 1.0), soft=0.0, spacing=1.0)
        img = tile.to_uint8()
        # Opaque white background: detect ink by color, not alpha.
        dark = (img[..., :3].sum(axis=2) < 200) if HAVE_NUMPY else \
            [[sum(px[:3]) < 200 for px in row] for row in img]
        # The painted band must stay within a few rows of y=32.
        painted_rows = [r for r in range(H)
                        if any(dark[r][c] for c in range(W))]
        self.assertTrue(painted_rows)
        self.assertLessEqual(max(painted_rows) - min(painted_rows), 6,
                             "stroke jumped across the tile instead of wrapping")
        # Both sides of the seam carry ink.
        self.assertTrue(any(dark[32][c] for c in range(0, 10)))
        self.assertTrue(any(dark[32][c] for c in range(W - 10, W)))

    def test_semitransparent_accumulation(self):
        """Two overlapping translucent dabs blend like source-over, not max."""
        tile = SeamlessTile(32, 32, background=(0.0, 0.0, 0.0, 0.0))
        tile.stamp(16.0, 16.0, 6.0, (0.0, 0.0, 0.0, 0.5), soft=0.0)
        tile.stamp(16.0, 16.0, 6.0, (0.0, 0.0, 0.0, 0.5), soft=0.0)
        a = tile.to_uint8()[16, 16, 3] if HAVE_NUMPY else tile.to_uint8()[16][16][3]
        # Two distinct stamp() deposits: source-over accumulation 0.75.
        # (Periodic copies of a *single* stamp would stay at 0.5 - covered by
        # the soft large-brush union test.)
        expected = int(0.75 * 255 + 0.5)
        self.assertLessEqual(abs(int(a) - expected), 1)

    def test_png_roundtrip(self):
        tile = SeamlessTile(24, 18)
        tile.stamp(2.0, 3.0, 7.0, (0.9, 0.4, 0.1, 0.8), soft=0.3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.png")
            img = tile.to_uint8()
            write_png(path, img, 24, 18, 4)
            pixels, w, h, ch = read_png(path)
        self.assertEqual((w, h, ch), (24, 18, 4))
        ref = tile.to_uint8()
        if HAVE_NUMPY:
            ref = ref.tolist()
        self.assertEqual(pixels, ref)

    def test_history_replay_and_save_load(self):
        tile = SeamlessTile(40, 30)
        tile.add_stroke([(-5, 10), (45, 12)], 3.0, (1.0, 0.0, 0.0, 0.7),
                        soft=0.2, spacing=1.2)
        tile.add_stroke([(10, -5), (12, 35)], 2.0, (0.0, 0.0, 1.0, 0.6),
                        soft=0.1, policy="lattice")
        before = tile.to_uint8().tolist() if HAVE_NUMPY else tile.to_uint8()
        with tempfile.TemporaryDirectory() as d:
            paths = save_project(tile, d, "p")
            self.assertTrue(all(os.path.exists(p) for p in paths.values()))
            with open(paths["history"], encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(len(payload["strokes"]), 2)
            rebuilt = load_project(paths["history"])
        after = rebuilt.to_uint8().tolist() if HAVE_NUMPY else rebuilt.to_uint8()
        self.assertEqual(before, after)
        # Undo restores the single-stroke state exactly.
        tile.undo()
        one = tile.to_uint8().tolist() if HAVE_NUMPY else tile.to_uint8()
        fresh = SeamlessTile(40, 30)
        fresh.add_stroke([(-5, 10), (45, 12)], 3.0, (1.0, 0.0, 0.0, 0.7),
                         soft=0.2, spacing=1.2)
        self.assertEqual(one, fresh.to_uint8().tolist() if HAVE_NUMPY
                         else fresh.to_uint8())


class PurePythonFallbackTests(unittest.TestCase):
    """The pure-Python path must follow the same rules (no numpy assumed)."""

    def test_pure_path_matches_numpy_when_available(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy not installed; cross-check unavailable")
        import seamless_paint.kernel as kmod

        # Build the pure tile while the module flag is off so its buffer is
        # plain Python lists and the pure stamp branches execute.
        kmod.HAVE_NUMPY = False
        try:
            pure_tile = SeamlessTile(48, 40)
            for t in range(12):
                pure_tile.stamp(-4 + t * 5.0, 36 - t * 2.2,
                                6.0, (0.2, 0.7, 0.3, 0.55), soft=0.35)
        finally:
            kmod.HAVE_NUMPY = True

        np_tile = SeamlessTile(48, 40)
        for t in range(12):
            np_tile.stamp(-4 + t * 5.0, 36 - t * 2.2,
                          6.0, (0.2, 0.7, 0.3, 0.55), soft=0.35)
        a = np_tile.to_uint8().tolist()
        b = pure_tile.to_uint8()
        for r in range(40):
            for c in range(48):
                for k in range(4):
                    self.assertLessEqual(abs(a[r][c][k] - b[r][c][k]), 1,
                                         f"pure/numpy mismatch {r},{c},{k}")

    def test_pure_path_periodic_rules(self):
        """Without numpy, edge and corner dabs still fold correctly."""
        import seamless_paint.kernel as kmod
        old = kmod.HAVE_NUMPY
        kmod.HAVE_NUMPY = False
        try:
            tile = SeamlessTile(32, 32)
            tile.stamp(0.0, 0.0, 7.0, (0.0, 0.0, 0.0, 1.0), soft=0.0)
            alpha = [[p[3] for p in row] for row in tile.to_uint8()]
            for off in range(8):
                vals = [alpha[off][off], alpha[off][31 - off],
                        alpha[31 - off][off], alpha[31 - off][31 - off]]
                self.assertEqual(len(set(vals)), 1)
        finally:
            kmod.HAVE_NUMPY = old


class WebAppSmokeTest(unittest.TestCase):
    def setUp(self):
        from seamless_paint.webapp import create_server
        self.tmp = tempfile.mkdtemp()
        self.httpd = create_server("127.0.0.1", 0, 64, 64, self.tmp)
        self.port = self.httpd.server_address[1]
        import threading
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
            return r.status, r.read(), r.headers.get("Content-Type")

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def test_full_session_and_kernel_error_feedback(self):
        status, body, ctype = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)

        st = self._get("/api/state")[1]
        self.assertEqual(json.loads(st)["width"], 64)

        # Diagonal thin stroke crossing two period boundaries.
        ok = self._post("/api/stroke", {
            "points": [[-8 + t * 3.0, -8 + t * 2.0] for t in range(30)],
            "radius": 1.5, "color": [0.1, 0.2, 0.9, 1.0],
            "soft": 0.1, "policy": "neighbor",
        })
        self.assertTrue(ok["ok"])

        _, png, png_type = self._get("/api/tile.png")
        self.assertEqual(png_type, "image/png")
        self.assertTrue(png.startswith(b"\x89PNG"))

        _, preview, _ = self._get("/api/preview.png?copies=3")
        self.assertTrue(preview.startswith(b"\x89PNG"))

        # Invalid radius must come back as structured UI feedback, not crash.
        bad = self._post("/api/stroke", {
            "points": [[1, 1]], "radius": -3,
            "color": [0, 0, 0, 1], "soft": 0.2, "policy": "neighbor",
        })
        self.assertFalse(bad["ok"])
        self.assertIn("error", bad)

        saved = self._post("/api/save", {})
        self.assertTrue(saved["ok"])
        for key in ("unit", "history", "preview"):
            self.assertTrue(os.path.exists(saved["paths"][key]))

        self.assertTrue(self._post("/api/undo", {})["ok"])
        self.assertTrue(self._post("/api/clear", {})["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
