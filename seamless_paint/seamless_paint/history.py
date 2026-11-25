"""Persistence: original unit tile, stroke history and tiled deliverables."""

from __future__ import annotations

import json
import os
from typing import Tuple

from .kernel import SeamlessTile
from .pngio import write_png


def save_project(tile: SeamlessTile, directory: str, base_name: str = "pattern") -> dict:
    """Save the original unit, its stroke history and a tiled preview.

    Produces three files inside ``directory``:

    * ``<base_name>_unit.png``  - the raw W x H texture unit (the deliverable);
    * ``<base_name>_history.json`` - every stroke with parameters, allowing an
      exact re-render via :func:`load_project`;
    * ``<base_name>_preview3x3.png`` - a 3x3 periodic montage for seam review.

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

    payload = {
        "version": 1,
        "width": tile.width,
        "height": tile.height,
        "background": list(tile.background),
        "strokes": tile.history,
    }
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return {
        "unit": unit_path,
        "history": history_path,
        "preview": preview_path,
    }


def load_project(history_path: str) -> SeamlessTile:
    """Rebuild a tile exactly from a saved stroke-history JSON file."""
    with open(history_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    tile = SeamlessTile(payload["width"], payload["height"], tuple(payload["background"]))
    for rec in payload["strokes"]:
        pts = rec.get("raw_points", rec["points"])
        tile.add_stroke(
            pts,
            rec["radius"],
            tuple(rec["color"]),
            soft=rec["soft"],
            policy=rec["policy"],
            spacing=rec["spacing"],
        )
    return tile


def export_montage(tile: SeamlessTile, path: str, copies: int = 3) -> Tuple[str, int, int]:
    """Export a tiled montage (used by the web UI's download buttons)."""
    img = tile.to_uint8(copies=copies)
    write_png(path, img, tile.width * copies, tile.height * copies, 4)
    return path, tile.width * copies, tile.height * copies
