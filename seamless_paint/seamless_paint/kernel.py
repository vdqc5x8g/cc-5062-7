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

try:
    import numpy as _np
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - exercised only on stripped interpreters
    _np = None
    HAVE_NUMPY = False

RGBA = Tuple[float, float, float, float]


class SeamlessInputError(ValueError):
    """Invalid painting parameters (bad radius, policy, ...).

    Raised for expected client-side input problems so the web layer can report
    them back to the painting window without logging a server stack trace.
    """


class SeamlessTile:
    """An RGBA texture tile with periodic brush stamping."""

    POLICIES = ("neighbor", "lattice")

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        background: RGBA = (1.0, 1.0, 1.0, 1.0),
    ):
        if width <= 0 or height <= 0:
            raise SeamlessInputError("tile dimensions must be positive")
        self.width = int(width)
        self.height = int(height)
        self.background = tuple(float(c) for c in background)
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
    ) -> None:
        """Deposit one circular brush dab at continuous position (x, y).

        Coordinates are NOT required to lie inside the tile: the dab is placed
        in the unwrapped plane and every periodic intersection with the tile
        is written.  ``radius`` may exceed the tile size.
        """
        if radius <= 0:
            raise SeamlessInputError("radius must be positive")
        if policy not in self.POLICIES:
            raise SeamlessInputError(f"policy must be one of {self.POLICIES}")
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
    ) -> dict:
        """Add a polyline stroke given in tile-local coordinates.

        Each point is wrapped into ``[0, W) x [0, H)``.  Consecutive points
        are unwrapped with the minimum-image convention so that a pointer
        crossing an edge continues at the opposite edge through a short
        wrapped segment instead of a line across the whole tile.  Dabs are
        placed at equal arc-length intervals, keeping the stroke density
        continuous through the jump.
        """
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
            "radius": float(radius),
            "color": [float(c) for c in color],
            "soft": float(soft),
            "policy": policy,
            "spacing": float(spacing),
        }
        self.history.append(record)
        return record

    def _render_record(self, rec: dict) -> None:
        pts = rec.get("raw_points", rec["points"])
        self.add_stroke(
            pts, rec["radius"], tuple(rec["color"]),
            soft=rec["soft"], policy=rec["policy"], spacing=rec["spacing"],
        )

    def replay(self) -> None:
        """Re-render the whole tile from the stroke history."""
        records = list(self.history)
        self.__init__(self.width, self.height, self.background)
        for rec in records:
            self._render_record(rec)

    def undo(self) -> Optional[dict]:
        if not self.history:
            return None
        rec = self.history.pop()
        records = list(self.history)
        self.__init__(self.width, self.height, self.background)
        for r in records:
            self._render_record(r)
        return rec

    def clear(self) -> None:
        self.history.clear()
        self.__init__(self.width, self.height, self.background)

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
