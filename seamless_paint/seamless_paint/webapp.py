"""Browser-based painting window driven directly by the seamless kernel.

The server holds a single :class:`SeamlessTile` session.  Pointer trajectories
collected in the browser are sent as tile-local polylines; the kernel performs
all periodic folding, arc-length resampling and compositing.  The browser
never invents pixels - every image returned is a kernel render, and kernel
errors are reported back to the on-screen status bar instead of crashing.

Run with::

    python3 -m seamless_paint.webapp --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .kernel import SeamlessTile, SeamlessInputError, HAVE_NUMPY
from .pngio import encode_png
from .history import save_project, export_montage

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
}


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

    def add_stroke(self, payload: dict):
        with self.lock:
            color = tuple(float(c) for c in payload["color"])
            rec = self.tile.add_stroke(
                [(float(p[0]), float(p[1])) for p in payload["points"]],
                radius=float(payload["radius"]),
                color=color,
                soft=float(payload.get("soft", 0.25)),
                policy=str(payload.get("policy", "neighbor")),
            )
            return rec

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
        try:
            if path == "/api/state":
                s = self.session
                with s.lock:
                    self._send_json({
                        "ok": True,
                        "width": s.tile.width,
                        "height": s.tile.height,
                        "strokes": len(s.tile.history),
                        "numpy": HAVE_NUMPY,
                        "saved": s.saved_paths,
                    })
            elif path == "/api/tile.png":
                body = self.session.render_png(1)
                self._send_png(body)
            elif path == "/api/preview.png":
                copies = int(parsed.query.split("=")[1]) if "copies" in parsed.query else 3
                copies = max(1, min(copies, 8))
                self._send_png(self.session.render_png(copies))
            elif path == "/api/download/montage":
                copies = int(parsed.query.split("=")[1]) if "copies" in parsed.query else 3
                fs = self.session.montage_file(max(1, min(copies, 8)))
                with open(fs, "rb") as fh:
                    body = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{os.path.basename(fs)}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._serve_static(path)
        except SeamlessInputError as exc:  # expected bad input: feedback, no stack
            self._send_error_json(exc, 400)
        except Exception as exc:  # unexpected kernel/render failure
            traceback.print_exc()
            self._send_error_json(exc, 500)

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
                self._send_json({"ok": True, "width": w, "height": h})
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
