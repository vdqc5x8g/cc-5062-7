"""Toroidal painting kernel.

The editable tile lives on a flat torus R^2 / (W Z x H Z).  Every brush
deposit is evaluated in the *unwrapped* continuous plane and folded into the
tile with modular coordinates, so a stroke that leaves one edge re-enters
through the opposite edge without ever producing a straight line across the
tile.

Two period policies are supported for diagonal stamps, as required:

* ``neighbor`` (default): neighbour-block continuation.  Every periodic copy
  of the stamp that intersects the tile is enumerated explicitly.  Copies may
  overlap (large brushes cross several periods) and their soft ramps are
  combined as a true coverage union
  ``cov_union = 1 - prod(1 - cov_k)`` instead of being duplicated.
* ``lattice``: lattice-basis sampling.  Every tile pixel takes the value of
  the single nearest lattice copy.  For a hard brush the two policies are
  mathematically identical; they only differ in how soft edges of different
  copies interact.

Pixel buffers use premultiplied straight-premultiplied RGBA floats so that
semi-transparent deposits blend with source-over compositing.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .errors import SeamlessInputError  # re-exported for callers
from . import lattice as lat

try:
    import numpy as _np
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - exercised only on stripped interpreters
    _np = None
    HAVE_NUMPY = False

RGBA = Tuple[float, float, float, float]

__all__ = ["SeamlessTile", "SeamlessInputError", "HAVE_NUMPY"]


class SeamlessTile:
    """An RGBA texture tile with periodic brush stamping."""

    POLICIES = ("neighbor", "lattice")

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        background: RGBA = (1.0, 1.0, 1.0, 1.0),
        lattice: "Optional[lat.Lattice | Tuple[float, float, float, float]]" = None,
    ):
        if width <= 0 or height <= 0:
            raise SeamlessInputError("tile dimensions must be positive")
        self.width = int(width)
        self.height = int(height)
        self.background = tuple(float(c) for c in background)
        if lattice is None:
            self.lattice = lat.Lattice.orthogonal(self.width, self.height)
        elif isinstance(lattice, lat.Lattice):
            self.lattice = lattice
        else:
            self.lattice = lat.Lattice(*[float(v) for v in lattice])
        # The exact axis-aligned W x H lattice keeps the original code paths
        # bit-compatible (and the old orthogonal tests unchanged).
        L = self.lattice
        self._legacy_ortho = (
            L.ax == float(self.width) and L.ay == 0.0
            and L.bx == 0.0 and L.by == float(self.height)
        )
        self._world_grid_cache = None
        if HAVE_NUMPY:
            self._buf = _np.zeros((self.height, self.width, 4), dtype=_np.float64)
            self._buf[..., 0] = background[0] * background[3]
            self._buf[..., 1] = background[1] * background[3]
            self._buf[..., 2] = background[2] * background[3]
            self._buf[..., 3] = background[3]
        else:
            pb = (
                background[0] * background[3],
                background[1] * background[3],
                background[2] * background[3],
                background[3],
            )
            self._buf = [[list(pb) for _ in range(self.width)] for _ in range(self.height)]
        # Stroke history: list of serializable dicts, the authoritative record
        # used for undo, replay and persistence.
        self.history: List[dict] = []

    # ------------------------------------------------------------------ #
    # Core primitives
    # ------------------------------------------------------------------ #
    def stamp(
        self,
        x: float,
        y: float,
        radius: float,
        color: RGBA,
        soft: float = 0.25,
        policy: str = "neighbor",
        angle: float = 0.0,
        aspect: float = 1.0,
        rx: Optional[float] = None,
        ry: Optional[float] = None,
        metric: Optional[Sequence[float]] = None,
    ) -> dict:
        """Deposit one brush dab at world position ``(x, y)``.

        Coordinates are NOT required to lie inside the cell: the dab is placed
        on the continuous world plane and every periodic lattice copy
        intersecting the stored chart is written.  ``radius`` may exceed the
        cell size.  The brush is a rotatable world-space ellipse (circle by
        default); ``metric=(q00,q01,q11)`` may define a transformed ellipse.

        Returns a report with the de-duplicated lattice cell identities.
        """
        if policy not in self.POLICIES:
            raise SeamlessInputError(f"policy must be one of {self.POLICIES}")
        if radius is not None and float(radius) <= 0.0 and metric is None:
            raise SeamlessInputError("radius must be positive")

        plain_circle = (
            metric is None and rx is None and ry is None
            and float(aspect) == 1.0 and float(angle) == 0.0
        )
        if self._legacy_ortho and plain_circle:
            return self._stamp_legacy(
                x, y, radius, color, soft=soft, policy=policy)
        return self._stamp_lattice(
            x, y, radius, color, soft=soft, policy=policy,
            angle=angle, aspect=aspect, rx=rx, ry=ry, metric=metric,
        )

    def _stamp_legacy(
        self,
        x: float,
        y: float,
        radius: float,
        color: RGBA,
        soft: float = 0.25,
        policy: str = "neighbor",
    ) -> dict:
        """Original orthogonal W x H torus path (bit-compatible)."""
        if radius <= 0:
            raise SeamlessInputError("radius must be positive")
        soft = min(max(float(soft), 0.0), 1.0)
        r = float(radius)
        soft_px = r * soft
        core = r - soft_px
        alpha = min(max(float(color[3]), 0.0), 1.0)
        if alpha == 0.0:
            return
        premul = (color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha)

        # Period cells whose radius box intersects the unit cell [0,T).
        # A copy j is centered at c_j = xm + j*T (xm = folded center) and
        # reaches the tile whenever its radius box [c_j-r, c_j+r] overlaps
        # [0, T].  Solving those two inequalities directly gives the integer
        # bounds below; counting with floor/ceil on the raw quotient (no
        # algebraic reshuffling) keeps exact period boundaries stable.  Copy
        # centers are built around the ORIGINAL center by shifting the
        # indices by the period distance between original and folded
        # positions, so dabs far outside the cell use the correct lattice
        # copies; when r >= T the range spans many overlapping copies and no
        # period is ever missed.
        xm = x % self.width
        ym = y % self.height
        jx_lo = math.ceil((0.0 - r - xm) / self.width)
        jx_hi = math.floor((self.width + r - xm) / self.width)
        jy_lo = math.ceil((0.0 - r - ym) / self.height)
        jy_hi = math.floor((self.height + r - ym) / self.height)
        jx_lo, jx_hi = int(jx_lo), int(jx_hi)
        jy_lo, jy_hi = int(jy_lo), int(jy_hi)
        kx0 = math.floor(x / self.width + 1e-9) - math.floor(xm / self.width + 1e-9)
        ky0 = math.floor(y / self.height + 1e-9) - math.floor(ym / self.height + 1e-9)
        nx_lo, nx_hi = jx_lo - kx0, jx_hi - kx0
        ny_lo, ny_hi = jy_lo - ky0, jy_hi - ky0

        if HAVE_NUMPY:
            self._stamp_numpy(
                x, y, r, core, soft_px, premul,
                nx_lo, nx_hi, ny_lo, ny_hi, policy,
            )
        else:
            self._stamp_pure(
                x, y, r, core, soft_px, premul,
                nx_lo, nx_hi, ny_lo, ny_hi, policy,
            )
        cells = [(nx, ny)
                 for ny in range(ny_lo, ny_hi + 1)
                 for nx in range(nx_lo, nx_hi + 1)]
        return {"cells": cells, "cell_count": len(cells)}

    # ------------------------------------------------------------------ #
    # General lattice path: world-space rotatable ellipse, cell identities
    # ------------------------------------------------------------------ #
    def _prepare_deposit(self, color, alpha_override=None):
        alpha = min(max(float(color[3]), 0.0), 1.0)
        if alpha_override is not None:
            alpha = min(max(float(alpha_override), 0.0), 1.0)
        if alpha == 0.0:
            return None
        return (color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha)

    def _stamp_lattice(self, x, y, radius, color, *, soft=0.25,
                       policy="neighbor", angle=0.0, aspect=1.0,
                       rx=None, ry=None, metric=None):
        L = self.lattice
        soft = min(max(float(soft), 0.0), 1.0)
        if metric is not None:
            q00, q01, q11, support_r, _ = lat.brush_metric(
                float(radius) if radius is not None and float(radius) > 0 else 1.0,
                metric=metric)
        else:
            r = float(radius)
            if r <= 0.0:
                raise SeamlessInputError("radius must be positive")
            q00, q01, q11, support_r, _ = lat.brush_metric(
                r, aspect=float(aspect), angle=float(angle), rx=rx, ry=ry)
        es, et = lat.support_extents(L, q00, q01, q11)

        ids = lat.enumerate_cell_ids(
            L, x, y, metric=(q00, q01, q11),
            radius=support_r, max_cells=200000)

        premul = self._prepare_deposit(color)
        if premul is None:
            return {"cells": ids, "cell_count": len(ids), "pixels_touched": 0}

        sc, tc, i_lo, i_hi, j_lo, j_hi = lat.cell_ranges_for_stamp(
            L, x, y, es, et)

        if HAVE_NUMPY and isinstance(self._buf, _np.ndarray):
            touched = self._stamp_lattice_numpy(
                x, y, q00, q01, q11, soft, premul,
                sc, tc, i_lo, i_hi, j_lo, j_hi, policy)
        else:
            touched = self._stamp_lattice_pure(
                x, y, q00, q01, q11, soft, premul,
                sc, tc, es, et, i_lo, i_hi, j_lo, j_hi, policy)
        return {"cells": ids, "cell_count": len(ids),
                "pixels_touched": int(touched)}

    def _lattice_coverage(self, e: float, soft: float) -> float:
        if soft <= 0.0:
            return 1.0 if e <= 1.0 + 1e-9 else 0.0
        core_e = 1.0 - soft
        if e <= core_e + 1e-9:
            return 1.0
        if e >= 1.0:
            return 0.0
        return (1.0 - e) / soft

    def _stamp_lattice_pure(self, x, y, q00, q01, q11, soft, premul,
                            sc, tc, es, et, i_lo, i_hi, j_lo, j_hi, policy):
        W, H = self.width, self.height
        L = self.lattice
        ax, ay, bx, by = L.ax, L.ay, L.bx, L.by
        npx = W * H

        # Per-identity candidate pixel spans.  i shifts s -> a -> columns,
        # j shifts t -> b -> rows.
        col_spans = {}
        for i in range(i_lo, i_hi + 1):
            si = sc + i
            c0 = math.ceil(W * (si - es) - 0.5 - 1e-7)
            c1 = math.floor(W * (si + es) - 0.5 + 1e-7)
            c0 = max(0, c0)
            c1 = min(W - 1, c1)
            if c0 <= c1:
                col_spans[i] = (c0, c1, si)
        row_spans = {}
        for j in range(j_lo, j_hi + 1):
            tj = tc + j
            r0 = math.ceil(H * (tj - et) - 0.5 - 1e-7)
            r1 = math.floor(H * (tj + et) - 0.5 + 1e-7)
            r0 = max(0, r0)
            r1 = min(H - 1, r1)
            if r0 <= r1:
                row_spans[j] = (r0, r1, tj)

        def pixel_e2(col, row, si, tj):
            ds = (col + 0.5) / W - si
            dt = (row + 0.5) / H - tj
            dx = ds * ax + dt * bx
            dy = ds * ay + dt * by
            return q00 * dx * dx + 2.0 * q01 * dx * dy + q11 * dy * dy

        if policy == "lattice":
            mark = bytearray(npx)
            touched_idx = []
            best_e2 = [math.inf] * npx
            for i, (c0, c1, si) in col_spans.items():
                for j, (r0, r1, tj) in row_spans.items():
                    for row in range(r0, r1 + 1):
                        base_idx = row * W
                        for col in range(c0, c1 + 1):
                            e2 = pixel_e2(col, row, si, tj)
                            if e2 > 1.0 + 1e-7:
                                continue
                            idx = base_idx + col
                            if e2 < best_e2[idx]:
                                if not mark[idx]:
                                    mark[idx] = 1
                                    touched_idx.append(idx)
                                best_e2[idx] = e2
            for idx in touched_idx:
                cov = self._lattice_coverage(
                    math.sqrt(max(0.0, best_e2[idx])), soft)
                if cov > 0.0:
                    row, col = divmod(idx, W)
                    self._composite_pure(row, col, cov, premul)
            return len(touched_idx)

        mark = bytearray(npx)
        touched_idx = []
        union_one = [1.0] * npx
        for i, (c0, c1, si) in col_spans.items():
            for j, (r0, r1, tj) in row_spans.items():
                for row in range(r0, r1 + 1):
                    base_idx = row * W
                    for col in range(c0, c1 + 1):
                        e2 = pixel_e2(col, row, si, tj)
                        if e2 > 1.0 + 1e-7:
                            continue
                        cov = self._lattice_coverage(
                            math.sqrt(max(0.0, e2)), soft)
                        if cov <= 0.0:
                            continue
                        idx = base_idx + col
                        if not mark[idx]:
                            mark[idx] = 1
                            touched_idx.append(idx)
                            union_one[idx] = 1.0 - cov
                        else:
                            union_one[idx] *= 1.0 - cov
        for idx in touched_idx:
            cov = 1.0 - union_one[idx]
            if cov > 0.0:
                row, col = divmod(idx, W)
                self._composite_pure(row, col, cov, premul)
        return len(touched_idx)

    def _world_grid_numpy(self):
        if self._world_grid_cache is not None:
            return self._world_grid_cache
        W, H = self.width, self.height
        s = (_np.arange(W, dtype=_np.float64) + 0.5) / W
        t = (_np.arange(H, dtype=_np.float64) + 0.5) / H
        S, T = _np.meshgrid(s, t, indexing="xy")
        L = self.lattice
        wx = S * L.ax + T * L.bx
        wy = S * L.ay + T * L.by
        self._world_grid_cache = (wx, wy)
        return wx, wy

    def _stamp_lattice_numpy(self, x, y, q00, q01, q11, soft, premul,
                             sc, tc, i_lo, i_hi, j_lo, j_hi, policy):
        gx, gy = self._world_grid_numpy()
        if policy == "lattice":
            best = _np.full((self.height, self.width), _np.inf)
            for j in range(j_lo, j_hi + 1):
                cy = y + j * self.lattice.by
                cx_j = x + j * self.lattice.bx
                for i in range(i_lo, i_hi + 1):
                    cx = cx_j + i * self.lattice.ax
                    cyy = cy + i * self.lattice.ay
                    dx = gx - cx
                    dy = gy - cyy
                    e2 = q00 * dx * dx + 2.0 * q01 * dx * dy + q11 * dy * dy
                    best = _np.minimum(best, e2)
            e = _np.sqrt(_np.maximum(best, 0.0))
            if soft <= 0.0:
                cov = (e <= 1.0 + 1e-9).astype(_np.float64)
            else:
                cov = _np.clip((1.0 - e) / soft, 0.0, 1.0)
                cov = _np.where(e <= 1.0 - soft + 1e-9, 1.0, cov)
            before = int(_np.count_nonzero(cov > 0.0))
            self._composite_numpy(cov, premul)
            return before

        union_one = _np.ones((self.height, self.width), dtype=_np.float64)
        for j in range(j_lo, j_hi + 1):
            cy0 = y + j * self.lattice.by
            cx_b = x + j * self.lattice.bx
            for i in range(i_lo, i_hi + 1):
                cx = cx_b + i * self.lattice.ax
                cy = cy0 + i * self.lattice.ay
                dx = gx - cx
                dy = gy - cy
                e2 = q00 * dx * dx + 2.0 * q01 * dx * dy + q11 * dy * dy
                e = _np.sqrt(_np.maximum(e2, 0.0))
                if soft <= 0.0:
                    c = (e <= 1.0 + 1e-9).astype(_np.float64)
                else:
                    c = _np.clip((1.0 - e) / soft, 0.0, 1.0)
                    c = _np.where(e <= 1.0 - soft + 1e-9, 1.0, c)
                union_one *= 1.0 - c
        coverage = 1.0 - union_one
        touched = int(_np.count_nonzero(coverage > 0.0))
        self._composite_numpy(coverage, premul)
        return touched

    def _coverage(self, dist: float, r: float, core: float, soft_px: float) -> float:
        if dist <= core + 1e-9 or soft_px <= 0.0:
            return 1.0 if dist <= r + 1e-9 else 0.0
        if dist >= r:
            return 0.0
        return (r - dist) / soft_px

    def _stamp_numpy(self, x, y, r, core, soft_px, premul,
                     nx_lo, nx_hi, ny_lo, ny_hi, policy):
        W, H = self.width, self.height
        ys = _np.arange(H, dtype=_np.float64) + 0.5
        xs = _np.arange(W, dtype=_np.float64) + 0.5
        gy, gx = _np.meshgrid(ys, xs, indexing="ij")

        def coverage_for(cx, cy):
            dist = _np.hypot(gx - cx, gy - cy)
            if soft_px <= 0.0:
                return (dist <= r + 1e-9).astype(_np.float64)
            cov = _np.clip((r - dist) / soft_px, 0.0, 1.0)
            return _np.where(dist <= core + 1e-9, 1.0, cov)

        if policy == "lattice":
            # Lattice-basis sampling: each pixel takes the value of its NEAREST
            # periodic copy only.  The same enumerated copy set as the
            # neighbour policy is scanned, keeping the minimum distance; for a
            # hard brush this is mathematically identical to the coverage
            # union, while for soft brushes it avoids merging overlapping
            # ramps from different periods.
            nearest = _np.full((H, W), _np.inf, dtype=_np.float64)
            for ny in range(ny_lo, ny_hi + 1):
                cy = y + ny * H
                for nx in range(nx_lo, nx_hi + 1):
                    cx = x + nx * W
                    dist = _np.hypot(gx - cx, gy - cy)
                    nearest = _np.minimum(nearest, dist)
            if soft_px <= 0.0:
                cov = (nearest <= r + 1e-9).astype(_np.float64)
            else:
                cov = _np.clip((r - nearest) / soft_px, 0.0, 1.0)
                cov = _np.where(nearest <= core + 1e-9, 1.0, cov)
            self._composite_numpy(cov, premul)
            return

        # Neighbour-block policy: enumerate every intersecting period and form
        # the coverage union of all copies (essential when r >= tile size).
        # The cell range computed in stamp() already contains exactly the
        # intersecting copies, so no extra rejection predicate is used (a
        # center-outside copy can still reach into the tile and must not be
        # dropped).
        union_one = _np.ones((H, W), dtype=_np.float64)
        for ny in range(ny_lo, ny_hi + 1):
            cy = y + ny * H
            for nx in range(nx_lo, nx_hi + 1):
                cx = x + nx * W
                union_one *= 1.0 - coverage_for(cx, cy)
        self._composite_numpy(1.0 - union_one, premul)

    def _composite_numpy(self, coverage, premul):
        # One source-over deposit per pixel: periodic copies have already been
        # combined into a coverage union, so overlapping periods must not be
        # blended as if they were separate translucent layers.
        a = premul[3] * coverage
        if not _np.any(a > 0.0):
            return
        buf = self._buf
        out_a = a + buf[..., 3] * (1.0 - a)
        for k in range(3):
            buf[..., k] = premul[k] * coverage + buf[..., k] * (1.0 - a)
        buf[..., 3] = out_a
        # Premultiplied color must stay zero where coverage never reached.
        buf[..., :3] = _np.where(out_a[..., None] > 1e-12, buf[..., :3], 0.0)

    def _stamp_pure(self, x, y, r, core, soft_px, premul,
                    nx_lo, nx_hi, ny_lo, ny_hi, policy):
        W, H = self.width, self.height
        if policy == "lattice":
            # Nearest periodic copy only, over the same enumerated copy set.
            for row in range(H):
                py = row + 0.5
                for col in range(W):
                    px = col + 0.5
                    nearest = math.inf
                    for ny in range(ny_lo, ny_hi + 1):
                        cy = y + ny * H
                        for nx in range(nx_lo, nx_hi + 1):
                            cx = x + nx * W
                            d = math.hypot(px - cx, py - cy)
                            if d < nearest:
                                nearest = d
                    cov = self._coverage(nearest, r, core, soft_px)
                    if cov > 0.0:
                        self._composite_pure(row, col, cov, premul)
            return

        for row in range(H):
            py = row + 0.5
            for col in range(W):
                px = col + 0.5
                union_one = 1.0
                hit = False
                for ny in range(ny_lo, ny_hi + 1):
                    cy = y + ny * H
                    for nx in range(nx_lo, nx_hi + 1):
                        cx = x + nx * W
                        dist = math.hypot(px - cx, py - cy)
                        c = self._coverage(dist, r, core, soft_px)
                        if c > 0.0:
                            union_one *= 1.0 - c
                            hit = True
                if hit:
                    self._composite_pure(row, col, 1.0 - union_one, premul)

    def _composite_pure(self, row, col, coverage, premul):
        # Single source-over deposit using the union coverage of all periodic
        # copies intersecting this pixel (never one blend per copy).
        a = premul[3] * coverage
        if a <= 0.0:
            return
        px = self._buf[row][col]
        ia = 1.0 - a
        px[0] = premul[0] * coverage + px[0] * ia
        px[1] = premul[1] * coverage + px[1] * ia
        px[2] = premul[2] * coverage + px[2] * ia
        px[3] = a + px[3] * ia

    # ------------------------------------------------------------------ #
    # Strokes: arc-length continuous across period jumps
    # ------------------------------------------------------------------ #
    def add_stroke(
        self,
        points: Sequence[Tuple[float, float]],
        radius: float,
        color: RGBA,
        soft: float = 0.25,
        policy: str = "neighbor",
        spacing: Optional[float] = None,
        angle: float = 0.0,
        aspect: float = 1.0,
        rx: Optional[float] = None,
        ry: Optional[float] = None,
        metric: Optional[Sequence[float]] = None,
    ) -> dict:
        """Add an arc-length-resampled stroke.

        Points are world/canvas coordinates.  For the default orthogonal
        lattice the original minimum-image implementation is used; general
        lattices unwrap in cell coordinates with a true-metric shortest shift.
        """
        if len(points) < 1:
            raise SeamlessInputError("stroke needs at least one point")
        plain_circle = (
            metric is None and rx is None and ry is None
            and float(aspect) == 1.0 and float(angle) == 0.0
        )
        if self._legacy_ortho and plain_circle:
            return self._add_stroke_legacy(
                points, radius, color, soft=soft,
                policy=policy, spacing=spacing)
        return self._add_stroke_lattice(
            points, radius, color, soft=soft, policy=policy, spacing=spacing,
            angle=angle, aspect=aspect, rx=rx, ry=ry, metric=metric)

    def _add_stroke_legacy(
        self,
        points: Sequence[Tuple[float, float]],
        radius: float,
        color: RGBA,
        soft: float = 0.25,
        policy: str = "neighbor",
        spacing: Optional[float] = None,
    ) -> dict:
        if len(points) < 1:
            raise SeamlessInputError("stroke needs at least one point")
        W, H = self.width, self.height
        # Keep the raw incoming coordinates for unwrapping; only the stored
        # history record is folded into the unit cell.  Pre-wrapping here
        # would destroy the direction of the edge jump.
        raw = [(float(p[0]), float(p[1])) for p in points]
        local = [(p[0] % W, p[1] % H) for p in raw]

        # Unwrap the folded point sequence into a continuous plane using the
        # minimum-image convention.  This works for both callers that send raw
        # continuous-plane coordinates and browser points already wrapped into
        # the unit cell: modulo folding first, then each edge jump continues
        # through the short wrapped segment rather than a line across the tile.
        unwrapped: List[Tuple[float, float]] = [(local[0][0], local[0][1])]
        ox = oy = 0.0
        for i in range(1, len(local)):
            dx = local[i][0] - local[i - 1][0]
            dy = local[i][1] - local[i - 1][1]
            if dx > W / 2.0:
                dx -= W
            elif dx < -W / 2.0:
                dx += W
            if dy > H / 2.0:
                dy -= H
            elif dy < -H / 2.0:
                dy += H
            ox += dx
            oy += dy
            unwrapped.append((local[i][0] + ox, local[i][1] + oy))

        if spacing is None:
            spacing = max(0.5, min(radius * 0.25, 4.0))
        spacing = max(float(spacing), 0.05)

        # Equal-arc-length resampling along the unwrapped polyline.
        samples: List[Tuple[float, float]] = []
        samples.append(unwrapped[0])
        remainder = spacing
        for (x0, y0), (x1, y1) in zip(unwrapped[:-1], unwrapped[1:]):
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len == 0.0:
                continue
            ux, uy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
            travel = remainder
            while travel <= seg_len:
                samples.append((x0 + ux * travel, y0 + uy * travel))
                travel += spacing
            remainder = travel - seg_len
        if samples[-1] != unwrapped[-1]:
            samples.append(unwrapped[-1])

        for sx, sy in samples:
            self.stamp(sx, sy, radius, color, soft, policy)

        record = {
            # Folded points drive the UI; the raw continuous-plane points make
            # replay idempotent (re-wrapping folded points would re-route the
            # minimum-image unwrapping and change the trajectory).
            "points": [[float(a), float(b)] for a, b in local],
            "raw_points": [[float(a), float(b)] for a, b in raw],
            # Exact arc-length dab positions: basis changes transform these
            # directly instead of re-resampling changed geometry.
            "samples": [[float(a), float(b)] for a, b in samples],
            "radius": float(radius),
            "color": [float(c) for c in color],
            "soft": float(soft),
            "policy": policy,
            "spacing": float(spacing),
        }
        self.history.append(record)
        return record

    def _add_stroke_lattice(
        self,
        points: Sequence[Tuple[float, float]],
        radius: float,
        color: RGBA,
        soft: float = 0.25,
        policy: str = "neighbor",
        spacing: Optional[float] = None,
        angle: float = 0.0,
        aspect: float = 1.0,
        rx: Optional[float] = None,
        ry: Optional[float] = None,
        metric: Optional[Sequence[float]] = None,
    ) -> dict:
        if len(points) < 1:
            raise SeamlessInputError("stroke needs at least one point")
        L = self.lattice
        W, H = self.width, self.height
        raw = [(float(p[0]), float(p[1])) for p in points]

        # Fold every point into the fundamental cell first, then unwrap the
        # folded sequence with the shortest WORLD-length integer lattice
        # shift.  This keeps an edge/vertex crossing on a short wrapped
        # segment for skew bases too.
        cells = [L.world_to_cell(x, y) for x, y in raw]
        folded = [(s % 1.0, t % 1.0) for s, t in cells]

        unwrapped_cell: List[Tuple[float, float]] = [folded[0]]
        os_ = ot = 0.0
        for k in range(1, len(folded)):
            ds = folded[k][0] - folded[k - 1][0]
            dt = folded[k][1] - folded[k - 1][1]
            js, jt = lat.shortest_image_shift(L, ds, dt, search=1)
            ds += js
            dt += jt
            os_ += ds
            ot += dt
            # Accumulator already equals folded[k]-folded[0]+integer shifts,
            # so the unwrapped point is folded[0] + accumulator (adding
            # folded[k] here would double the displacement).
            unwrapped_cell.append(
                (folded[0][0] + os_, folded[0][1] + ot))

        world_pts = [L.cell_to_world(s, t) for s, t in unwrapped_cell]

        if spacing is None:
            spacing = max(0.5, min(float(radius) * 0.25, 4.0))
        spacing = max(float(spacing), 0.05)

        samples: List[Tuple[float, float]] = [world_pts[0]]
        remainder = spacing
        for (x0, y0), (x1, y1) in zip(world_pts[:-1], world_pts[1:]):
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len == 0.0:
                continue
            ux, uy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
            travel = remainder
            while travel <= seg_len:
                samples.append((x0 + ux * travel, y0 + uy * travel))
                travel += spacing
            remainder = travel - seg_len
        if samples[-1] != world_pts[-1]:
            samples.append(world_pts[-1])

        stamp_kw = dict(soft=soft, policy=policy, angle=angle, aspect=aspect,
                        rx=rx, ry=ry, metric=metric)
        last_report = None
        for sx, sy in samples:
            last_report = self.stamp(sx, sy, radius, color, **stamp_kw)

        folded_index = [[float(s * W), float(t * H)] for s, t in folded]
        record = {
            "engine": "lattice",
            "points": folded_index,
            "raw_points": [[float(a), float(b)] for a, b in raw],
            "samples": [[float(a), float(b)] for a, b in samples],
            "radius": float(radius),
            "color": [float(c) for c in color],
            "soft": float(soft),
            "policy": policy,
            "spacing": float(spacing),
            "angle": float(angle),
            "aspect": float(aspect),
            "rx": None if rx is None else float(rx),
            "ry": None if ry is None else float(ry),
            "metric": None if metric is None else [float(v) for v in metric],
        }
        if last_report is not None:
            record["last_cells"] = last_report["cells"]
        self.history.append(record)
        return record

    def _render_record(self, rec: dict) -> None:
        common = dict(
            soft=rec["soft"], policy=rec["policy"],
            angle=rec.get("angle", 0.0),
            aspect=rec.get("aspect", 1.0),
            rx=rec.get("rx"), ry=rec.get("ry"),
            metric=rec.get("metric"),
        )
        samples = rec.get("samples")
        if samples:
            # Replay exact dab positions: basis changes only alter how each
            # world dab is folded, never re-resample the trajectory.
            for x, y in samples:
                self.stamp(float(x), float(y), rec["radius"],
                           tuple(rec["color"]), **common)
            return
        # Older histories without explicit samples: rebuild the stroke.
        pts = rec.get("raw_points", rec["points"])
        self.add_stroke(
            pts, rec["radius"], tuple(rec["color"]),
            spacing=rec["spacing"], **common,
        )

    def replay(self) -> None:
        """Re-render the whole tile from the stroke history."""
        records = list(self.history)
        self.__init__(self.width, self.height, self.background,
                      lattice=self.lattice)
        self.history = records
        for rec in records:
            self._render_record(rec)

    def undo(self) -> Optional[dict]:
        if not self.history:
            return None
        rec = self.history.pop()
        records = list(self.history)
        self.__init__(self.width, self.height, self.background,
                      lattice=self.lattice)
        self.history = records
        for r in records:
            self._render_record(r)
        return rec

    def clear(self) -> None:
        self.history.clear()
        self.__init__(self.width, self.height, self.background,
                      lattice=self.lattice)

    # ------------------------------------------------------------------ #
    # Changing the repeat basis
    # ------------------------------------------------------------------ #
    def change_basis(self, new_lattice: "lat.Lattice",
                     mode: str = "rewrap") -> dict:
        """Switch repeat basis with one of two explicit semantics.

        mode="rewrap" (keep physical strokes):
            World-space trajectories and physical nib metrics are unchanged;
            history is replayed under the new lattice, so the same physical
            painting is re-wrapped by the new periodic identification.

        mode="deform" (deform existing content with the cell):
            Chart/cell coordinates of every stroke are preserved.  Raw world
            points are mapped by T = B_new B_old^{-1} and each nib metric Q is
            transformed by Q' = T^{-T} Q T^{-1}, which maps the old support
            ellipses exactly onto the deformed ones.  The current buffer is
            retained (the chart artwork is literally preserved) and history
            is rewritten so later replay/undo stays consistent.
        """
        if mode not in ("rewrap", "deform"):
            raise SeamlessInputError("basis mode must be 'rewrap' or 'deform'")
        old = self.lattice
        if not isinstance(new_lattice, lat.Lattice):
            new_lattice = lat.Lattice(*[float(v) for v in new_lattice])

        if mode == "rewrap":
            self.lattice = new_lattice
            self._world_grid_cache = None
            self.replay()
            return {"mode": mode, "lattice": new_lattice.to_json()}

        # deform: linear world map old basis -> new basis.
        det_o = old.det
        # M_old rows already available; T = B_new @ M_old
        m00, m01, m10, m11 = old.m00, old.m01, old.m10, old.m11
        n = new_lattice
        t00 = n.ax * m00 + n.bx * m10
        t01 = n.ax * m01 + n.bx * m11
        t10 = n.ay * m00 + n.by * m10
        t11 = n.ay * m01 + n.by * m11

        def map_point(x, y):
            return (t00 * x + t01 * y, t10 * x + t11 * y)

        # T^{-1} entries (2x2).
        tdet = t00 * t11 - t01 * t10
        if abs(tdet) < 1e-15:
            raise SeamlessInputError("cannot deform through a singular map")
        i00, i01 = t11 / tdet, -t01 / tdet
        i10, i11 = -t10 / tdet, t00 / tdet

        def transform_metric(q00, q01, q11):
            # Q' = T^{-T} Q T^{-1}; compute with Minv columns.
            # Minv = [[i00,i01],[i10,i11]]
            a00 = q00 * i00 + q01 * i10
            a01 = q00 * i01 + q01 * i11
            a10 = q01 * i00 + q11 * i10
            a11 = q01 * i01 + q11 * i11
            p00 = i00 * a00 + i10 * a10
            p01 = i00 * a01 + i10 * a11
            p11 = i01 * a01 + i11 * a11
            return (p00, p01, p11)

        for rec in self.history:
            rec["raw_points"] = [list(map_point(float(x), float(y)))
                                 for x, y in rec.get("raw_points",
                                                     rec["points"])]
            if rec.get("samples"):
                rec["samples"] = [list(map_point(float(x), float(y)))
                                  for x, y in rec["samples"]]
            metric = rec.get("metric")
            if metric is None:
                r = float(rec["radius"])
                ang = float(rec.get("angle", 0.0))
                asp = float(rec.get("aspect", 1.0))
                rx = rec.get("rx")
                ry = rec.get("ry")
                q00, q01, q11, _, _ = lat.brush_metric(
                    r, aspect=asp, angle=ang,
                    rx=rx, ry=ry)
            else:
                q00, q01, q11 = (float(metric[0]), float(metric[1]),
                                 float(metric[2]))
            nq = transform_metric(q00, q01, q11)
            rec["metric"] = [float(v) for v in nq]
            rec["engine"] = "lattice"
            # Folded index points keep their cell identity under deform;
            # recompute them from transformed raw points for consistency.
            folded = []
            for x, y in rec["raw_points"]:
                s, t = new_lattice.world_to_cell(x, y)
                folded.append([(s % 1.0) * self.width,
                               (t % 1.0) * self.height])
            rec["points"] = folded

        self.lattice = new_lattice
        self._legacy_ortho = False
        self._world_grid_cache = None
        # Deliberately do NOT replay: the existing buffer is the preserved
        # chart content.  Rewritten history re-renders to the same coverage.
        return {"mode": mode, "lattice": new_lattice.to_json(),
                "preserved_buffer": True}

    def preview_basis_png(self, new_lattice: "lat.Lattice", mode: str,
                          view_px: int = 512) -> bytes:
        """Non-mutating basis-change preview; returns world-view PNG bytes."""
        records = [dict(r) for r in self.history]
        bg = self.background
        W, H = self.width, self.height
        if mode == "rewrap":
            tmp = SeamlessTile(W, H, bg, lattice=new_lattice)
            for rec in records:
                tmp._render_record(rec)
        else:
            # Apply deform transforms on a throwaway tile via change_basis.
            tmp = SeamlessTile(W, H, bg, lattice=self.lattice)
            tmp.history = [dict(r) for r in self.history]
            tmp.change_basis(new_lattice, mode="deform")
        view = new_lattice.fit_world_view(512)
        img = tmp.render_world_uint8(
            view["x0"], view["y0"], view["x1"], view["y1"],
            view_px, view_px)
        from .pngio import encode_png
        return encode_png(img, view_px, view_px, 4)

    # ------------------------------------------------------------------ #
    # World-space periodic renderer (true lattice tiling)
    # ------------------------------------------------------------------ #
    def render_world_uint8(self, x0: float, y0: float,
                           x1: float, y1: float,
                           out_w: int, out_h: int):
        """Render the periodic pattern over a world rectangle.

        Every output pixel is inverse-mapped through B^{-1}, folded to the
        fundamental cell and nearest-sampled from the stored chart.  This is
        true periodic lattice geometry, not a sheared preview window.
        """
        out_w, out_h = int(out_w), int(out_h)
        unit = self.to_uint8(copies=1)
        L = self.lattice
        W, H = self.width, self.height
        if HAVE_NUMPY and isinstance(self._buf, _np.ndarray):
            cols = (_np.arange(out_w, dtype=_np.float64) + 0.5) / out_w
            rows = (_np.arange(out_h, dtype=_np.float64) + 0.5) / out_h
            wx = x0 + cols[None, :] * (x1 - x0)
            wy = y0 + rows[:, None] * (y1 - y0)
            s = L.m00 * wx + L.m01 * wy
            t = L.m10 * wx + L.m11 * wy
            sf = s - _np.floor(s)
            tf = t - _np.floor(t)
            ci = _np.clip((sf * W).astype(_np.int64), 0, W - 1)
            ri = _np.clip((tf * H).astype(_np.int64), 0, H - 1)
            return unit[ri, ci, :]

        img = [[[0, 0, 0, 0] for _ in range(out_w)] for _ in range(out_h)]
        m00, m01, m10, m11 = L.m00, L.m01, L.m10, L.m11
        for r in range(out_h):
            fy = (r + 0.5) / out_h
            wy = y0 + fy * (y1 - y0)
            row_out = img[r]
            for c in range(out_w):
                fx = (c + 0.5) / out_w
                wx = x0 + fx * (x1 - x0)
                s = m00 * wx + m01 * wy
                t = m10 * wx + m11 * wy
                sf = s - math.floor(s)
                tf = t - math.floor(t)
                ci = int(sf * W)
                ri = int(tf * H)
                if ci >= W:
                    ci = W - 1
                if ri >= H:
                    ri = H - 1
                row_out[c] = unit[ri][ci]
        return img

    def render_cell_block_uint8(self, copies: int = 3,
                                max_px: int = 640,
                                padding: float = 0.0):
        """Render an s,t in [0,copies] lattice block fitted to ``max_px``.

        Returns ``(uint8_image, width, height, world_bounds)``.  Output keeps
        a uniform world scale so nib shapes stay undistorted.
        """
        c = int(copies)
        bounds = self.lattice.cell_block_bounds(
            -padding, -padding, c + padding, c + padding)
        x0, y0, x1, y1 = bounds
        bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        scale = min(float(max_px) / bw, float(max_px) / bh)
        ow = max(1, int(math.ceil(bw * scale)))
        oh = max(1, int(math.ceil(bh * scale)))
        img = self.render_world_uint8(x0, y0, x1, y1, ow, oh)
        return img, ow, oh, bounds


    # ------------------------------------------------------------------ #
    # Export / inspection
    # ------------------------------------------------------------------ #
    def to_uint8(self, copies: int = 1) -> "object":
        """Return an un-premultiplied RGBA uint8 array (numpy or nested list).

        ``copies`` tiles the result into a (copies*H) x (copies*W) montage so
        callers can inspect periodic continuation outside the base unit.
        """
        if HAVE_NUMPY and isinstance(self._buf, _np.ndarray):
            buf = self._buf.copy()
            a = buf[..., 3:4]
            rgb = _np.where(a > 1e-9, buf[..., :3] / _np.maximum(a, 1e-9), 0.0)
            rgba = _np.concatenate([rgb, a], axis=2)
            rgba = _np.clip(rgba, 0.0, 1.0)
            tile = (rgba * 255.0 + 0.5).astype(_np.uint8)
            if copies > 1:
                tile = _np.tile(tile, (copies, copies, 1))
            return tile
        out = [[[0, 0, 0, 0] for _ in range(self.width)] for _ in range(self.height)]
        for row in range(self.height):
            for col in range(self.width):
                p = self._buf[row][col]
                if p[3] > 1e-9:
                    scale = 255.0 / p[3]
                    out[row][col] = [
                        min(255, max(0, int(p[0] * scale + 0.5))),
                        min(255, max(0, int(p[1] * scale + 0.5))),
                        min(255, max(0, int(p[2] * scale + 0.5))),
                        min(255, max(0, int(p[3] * 255.0 + 0.5))),
                    ]
        if copies > 1:
            montage = [[[0, 0, 0, 0] for _ in range(self.width * copies)]
                       for _ in range(self.height * copies)]
            for ry in range(copies):
                for rx in range(copies):
                    for row in range(self.height):
                        for col in range(self.width):
                            montage[ry * self.height + row][rx * self.width + col] = out[row][col]
            out = montage
        return out

    def snapshot_premul(self) -> "object":
        if HAVE_NUMPY and isinstance(self._buf, _np.ndarray):
            return self._buf.copy()
        return [row[:] for row in self._buf]
