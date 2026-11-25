"""Persistence: unit chart, stroke history (with repeat basis) and tilings."""

from __future__ import annotations

import json
import os
from typing import Tuple

from .kernel import SeamlessTile
from .lattice import Lattice
from .pngio import write_png


def _basis_payload(tile: SeamlessTile) -> dict:
    L = tile.lattice
    legacy = getattr(tile, "_legacy_ortho", False)
    return {
        "a": [L.ax, L.ay],
        "b": [L.bx, L.by],
        "kind": "orthogonal" if legacy or L.is_orthogonal else "oblique",
        "cell_area": L.area,
        "angle_deg": L.angle_degrees,
    }


def save_project(tile: SeamlessTile, directory: str, base_name: str = "pattern",
                 write_world_tiling: bool = True) -> dict:
    """Save the unit chart, stroke history and tiled previews.

    Produces inside ``directory``:

    * ``<base_name>_unit.png`` - raw W x H chart (the deliverable);
    * ``<base_name>_history.json`` - every stroke with parameters and the
      explicit repeat basis, allowing exact re-render via
      :func:`load_project`;
    * ``<base_name>_preview3x3.png`` - 3x3 replication in chart coordinates;
    * ``<base_name>_lattice_tiled.png`` - true world-space lattice tiling
      (oblique offsets preserved), used for seam inspection.

    No edge blur or post-processing is applied: seams are correct because the
    painting itself followed the periodic coverage rules.
    """
    os.makedirs(directory, exist_ok=True)
    unit_path = os.path.join(directory, f"{base_name}_unit.png")
    history_path = os.path.join(directory, f"{base_name}_history.json")
    preview_path = os.path.join(directory, f"{base_name}_preview3x3.png")

    unit = tile.to_uint8(copies=1)
    write_png(unit_path, unit, tile.width, tile.height, 4)

    preview = tile.to_uint8(copies=3)
    write_png(preview_path, preview, tile.width * 3, tile.height * 3, 4)

    paths = {
        "unit": unit_path,
        "history": history_path,
        "preview": preview_path,
    }

    if write_world_tiling:
        tiled_path = os.path.join(directory, f"{base_name}_lattice_tiled.png")
        img, ow, oh, _ = tile.render_cell_block_uint8(3, max_px=900)
        write_png(tiled_path, img, ow, oh, 4)
        paths["lattice_tiled"] = tiled_path

    basis = _basis_payload(tile)
    payload = {
        "version": 2,
        "width": tile.width,
        "height": tile.height,
        "background": list(tile.background),
        # Canonical record of the repeat basis for importers/exporters.
        "repeat_basis": basis,
        # Same object under the kernel terminology.
        "lattice": {"a": basis["a"], "b": basis["b"]},
        "strokes": tile.history,
    }
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return paths


def _lattice_from_payload(payload: dict, width: int, height: int) -> Lattice:
    basis = payload.get("repeat_basis") or payload.get("lattice")
    if not basis:
        return Lattice.orthogonal(width, height)
    a = basis["a"]
    b = basis["b"]
    return Lattice(float(a[0]), float(a[1]), float(b[0]), float(b[1]))


def load_project(history_path: str) -> SeamlessTile:
    """Rebuild a tile exactly from a saved stroke-history JSON file."""
    with open(history_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    W = payload["width"]
    H = payload["height"]
    lattice = _lattice_from_payload(payload, W, H)
    tile = SeamlessTile(W, H, tuple(payload["background"]), lattice=lattice)
    for rec in payload["strokes"]:
        tile._render_record(rec)
    return tile


def export_montage(tile: SeamlessTile, path: str, copies: int = 3) -> Tuple[str, int, int]:
    """Export a chart-coordinate montage (used by the web UI download)."""
    img = tile.to_uint8(copies=copies)
    write_png(path, img, tile.width * copies, tile.height * copies, 4)
    return path, tile.width * copies, tile.height * copies


def export_lattice_tiling(tile: SeamlessTile, path: str, copies: int = 3,
                          max_px: int = 1024) -> Tuple[str, int, int]:
    """Export the true world-space oblique lattice tiling."""
    img, w, h, _ = tile.render_cell_block_uint8(copies, max_px=max_px)
    write_png(path, img, w, h, 4)
    return path, w, h
