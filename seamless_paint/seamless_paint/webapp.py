"""Browser painting window driven directly by the seamless kernel.

The server holds a single :class:`SeamlessTile` session.  Pointer trajectories
collected in the browser are sent as world-coordinate polylines; the kernel
performs all lattice transforms, periodic folding, arc-length resampling and
compositing.  The browser never invents pixels - every image returned is a
kernel render, and kernel errors are reported back to the on-screen status
bar instead of crashing.

Run with::

    python3 -m seamless_paint.webapp --port 8000
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .kernel import SeamlessTile, SeamlessInputError, HAVE_NUMPY
from .lattice import Lattice
from .pngio import encode_png
from .history import save_project, export_montage, export_lattice_tiling

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
VIEW_PX = 512


class PaintSession:
    """Thread-safe wrapper around the kernel plus project output paths."""

    def __init__(self, width: int = 256, height: int = 256,
                 output_dir: str = "output"):
        self.lock = threading.Lock()
        self.output_dir = output_dir
        self.tile = SeamlessTile(width, height)
        self.saved_paths: dict = {}

    def new_tile(self, width: int, height: int):
        with self.lock:
            self.tile = SeamlessTile(width, height)
            self.saved_paths = {}

    # ------------------------------------------------------------------ #
    def view_dict(self) -> dict:
        t = self.tile
        if t._legacy_ortho:
            return {
                "x0": 0.0, "y0": 0.0,
                "x1": float(t.width), "y1": float(t.height),
                "px": VIEW_PX,
                "scale_x": VIEW_PX / float(t.width),
                "scale_y": VIEW_PX / float(t.height),
                "uniform_scale": VIEW_PX / float(t.width),
            }
        v = t.lattice.fit_world_view(VIEW_PX, margin=1.04)
        half_w = (v["x1"] - v["x0"]) / 2.0
        half_h = (v["y1"] - v["y0"]) / 2.0
        return {
            "x0": v["x0"], "y0": v["y0"], "x1": v["x1"], "y1": v["y1"],
            "px": VIEW_PX,
            "scale_x": VIEW_PX / (2.0 * half_w),
            "scale_y": VIEW_PX / (2.0 * half_h),
            "uniform_scale": v["uniform_scale"],
        }

    def state_dict(self) -> dict:
        t = self.tile
        return {
            "ok": True,
            "width": t.width,
            "height": t.height,
            "strokes": len(t.history),
            "numpy": HAVE_NUMPY,
            "saved": self.saved_paths,
            "lattice": t.lattice.to_json(),
            "legacy_ortho": t._legacy_ortho,
            "view": self.view_dict(),
        }

    # ------------------------------------------------------------------ #
    def add_stroke(self, payload: dict):
        with self.lock:
            color = tuple(float(c) for c in payload["color"])
            angle = payload.get("angle")
            if angle is None and payload.get("angle_deg") is not None:
                import math
                angle = math.radians(float(payload["angle_deg"]))
            rec = self.tile.add_stroke(
                [(float(p[0]), float(p[1])) for p in payload["points"]],
                radius=float(payload["radius"]),
                color=color,
                soft=float(payload.get("soft", 0.25)),
                policy=str(payload.get("policy", "neighbor")),
                angle=float(angle or 0.0),
                aspect=float(payload.get("aspect", 1.0)),
                rx=payload.get("rx"),
                ry=payload.get("ry"),
            )
            return rec

    def set_lattice(self, payload: dict):
        with self.lock:
            L = Lattice(float(payload["ax"]), float(payload["ay"]),
                        float(payload["bx"]), float(payload["by"]))
            mode = str(payload.get("mode", "rewrap"))
            report = self.tile.change_basis(L, mode=mode)
            return report

    def basis_preview(self, payload: dict) -> dict:
        """Non-mutating compare render; returns base64 PNG data URLs."""
        with self.lock:
            L = Lattice(float(payload["ax"]), float(payload["ay"]),
                        float(payload["bx"]), float(payload["by"]))
            mode = str(payload.get("mode", "rewrap"))
            png = self.tile.preview_basis_png(L, mode=mode, view_px=384)
            b64 = base64.b64encode(png).decode("ascii")
            return {"ok": True, "mode": mode,
                    "world_png": "data:image/png;base64," + b64}

    def undo(self):
        with self.lock:
            return self.tile.undo()

    def clear(self):
        with self.lock:
            self.tile.clear()

    def render_png(self, copies: int) -> bytes:
        with self.lock:
            img = self.tile.to_uint8(copies=copies)
            w, h = self.tile.width * copies, self.tile.height * copies
        return encode_png(img, w, h, 4)

    def render_world_png(self, w: int, h: int) -> bytes:
        with self.lock:
            v = self.view_dict()
            img = self.tile.render_world_uint8(
                v["x0"], v["y0"], v["x1"], v["y1"], w, h)
        return encode_png(img, w, h, 4)

    def render_lattice_preview(self, copies: int, max_px: int):
        with self.lock:
            img, ow, oh, _ = self.tile.render_cell_block_uint8(
                copies, max_px=max_px)
        return encode_png(img, ow, oh, 4), ow, oh

    def save(self):
        with self.lock:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            paths = save_project(self.tile, self.output_dir, f"pattern_{stamp}")
            self.saved_paths = paths
            return paths

    def montage_file(self, copies: int) -> str:
        with self.lock:
            path = os.path.join(self.output_dir, f"montage_{copies}x{copies}.png")
            os.makedirs(self.output_dir, exist_ok=True)
            export_montage(self.tile, path, copies)
            return path

    def lattice_tile_file(self, copies: int) -> str:
        with self.lock:
            path = os.path.join(self.output_dir,
                                f"lattice_tiling_{copies}x{copies}.png")
            os.makedirs(self.output_dir, exist_ok=True)
            export_lattice_tiling(self.tile, path, copies)
            return path


class Handler(BaseHTTPRequestHandler):
    session: PaintSession = None  # set on server construction

    def log_message(self, fmt, *args):  # quieter console output
        pass

    # ------------------------------------------------------------------ #
    def _send_json(self, obj, status: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, exc: Exception, status: int = 400):
        self._send_json(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status
        )

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _serve_static(self, path: str):
        rel = path.lstrip("/") or "index.html"
        fs_path = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not fs_path.startswith(STATIC_DIR) or not os.path.isfile(fs_path):
            self.send_error(404)
            return
        ext = os.path.splitext(fs_path)[1]
        with open(fs_path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------ #
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        try:
            if path == "/api/state":
                with self.session.lock:
                    self._send_json(self.session.state_dict())
            elif path == "/api/tile.png":
                body = self.session.render_png(1)
                self._send_png(body)
            elif path == "/api/world.png":
                w = int(q.get("w", [VIEW_PX])[0])
                h = int(q.get("h", [VIEW_PX])[0])
                w, h = max(32, min(w, 1024)), max(32, min(h, 1024))
                self._send_png(self.session.render_world_png(w, h))
            elif path == "/api/lattice_preview.png":
                copies = max(1, min(int(q.get("copies", [3])[0]), 8))
                max_px = max(128, min(int(q.get("max", [420])[0]), 1024))
                body, _, _ = self.session.render_lattice_preview(copies, max_px)
                self._send_png(body)
            elif path == "/api/preview.png":
                copies = int(parsed.query.split("=")[1]) if "copies" in parsed.query else 3
                copies = max(1, min(copies, 8))
                self._send_png(self.session.render_png(copies))
            elif path == "/api/download/montage":
                copies = int(parsed.query.split("=")[1]) if "copies" in parsed.query else 3
                fs = self.session.montage_file(max(1, min(copies, 8)))
                self._send_file_attachment(fs, "seamless_montage.png")
            elif path == "/api/download/lattice_tile":
                copies = int(q.get("copies", [3])[0])
                fs = self.session.lattice_tile_file(max(1, min(copies, 8)))
                self._send_file_attachment(fs, "lattice_tiling.png")
            else:
                self._serve_static(path)
        except SeamlessInputError as exc:  # expected bad input: feedback, no stack
            self._send_error_json(exc, 400)
        except Exception as exc:  # unexpected kernel/render failure
            traceback.print_exc()
            self._send_error_json(exc, 500)

    def _send_file_attachment(self, fs: str, download_name: str):
        with open(fs, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Disposition",
                         f'attachment; filename="{download_name}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_png(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/stroke":
                rec = self.session.add_stroke(payload)
                self._send_json({"ok": True, "stroke": rec})
            elif path == "/api/lattice":
                report = self.session.set_lattice(payload)
                with self.session.lock:
                    state = self.session.state_dict()
                state.update({"ok": True, "change": report})
                self._send_json(state)
            elif path == "/api/basis/preview":
                self._send_json(self.session.basis_preview(payload))
            elif path == "/api/undo":
                rec = self.session.undo()
                self._send_json({"ok": True, "undone": rec is not None})
            elif path == "/api/clear":
                self.session.clear()
                self._send_json({"ok": True})
            elif path == "/api/new":
                w = int(payload.get("width", 256))
                h = int(payload.get("height", 256))
                if not (8 <= w <= 1024 and 8 <= h <= 1024):
                    raise ValueError("dimensions must be between 8 and 1024")
                self.session.new_tile(w, h)
                with self.session.lock:
                    state = self.session.state_dict()
                state["ok"] = True
                self._send_json(state)
            elif path == "/api/save":
                paths = self.session.save()
                self._send_json({"ok": True, "paths": paths})
            else:
                self.send_error(404)
        except SeamlessInputError as exc:  # expected bad input: feedback, no stack
            self._send_error_json(exc, 400)
        except Exception as exc:  # unexpected failure, still surfaced to the UI
            traceback.print_exc()
            self._send_error_json(exc, 500)


_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
}


def create_server(host: str = "127.0.0.1", port: int = 8000,
                  width: int = 256, height: int = 256,
                  output_dir: str = "output") -> ThreadingHTTPServer:
    session = PaintSession(width, height, output_dir)
    Handler.session = session
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.session = session
    return httpd


def main(argv=None):
    parser = argparse.ArgumentParser(description="Seamless pattern painting window")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output", default="output",
                        help="directory for saved textures and stroke history")
    args = parser.parse_args(argv)
    httpd = create_server(args.host, args.port, args.width, args.height, args.output)
    print(f"Seamless paint window running at http://{args.host}:{args.port}/")
    print(f"numpy accelerated kernel: {HAVE_NUMPY}")
    print(f"textures will be saved under: {os.path.abspath(args.output)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
