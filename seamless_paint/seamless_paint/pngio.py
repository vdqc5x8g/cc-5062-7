"""Minimal PNG codec using only the Python standard library (zlib + struct).

Supports 8-bit RGB and RGBA, filter type 0 (None) per scanline, and reading
back the same kind of file.  This is deliberately small: the project must not
depend on third-party imaging libraries just to save a texture.
"""

from __future__ import annotations

import struct
import zlib
from typing import List, Sequence, Tuple

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def encode_png(pixels, width: int, height: int, channels: int = 4) -> bytes:
    """Encode an HxWxC uint8 array (numpy) or nested list to PNG bytes."""
    if channels not in (3, 4):
        raise ValueError("only RGB (3) and RGBA (4) PNGs are supported")
    color_type = 6 if channels == 4 else 2

    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter: None
        for col in range(width):
            px = pixels[row][col]
            raw.extend(int(v) & 0xFF for v in px[:channels])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    out = bytearray(PNG_SIGNATURE)
    out += _chunk(b"IHDR", ihdr)
    out += _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    out += _chunk(b"IEND", b"")
    return bytes(out)


def write_png(path: str, pixels, width: int, height: int, channels: int = 4) -> None:
    """Write an HxWxC uint8 array (numpy) or nested list to ``path``."""
    data = encode_png(pixels, width, height, channels)
    with open(path, "wb") as fh:
        fh.write(data)


def read_png(path: str) -> Tuple[List[List[List[int]]], int, int, int]:
    """Read an 8-bit RGB/RGBA PNG written by :func:`write_png`.

    Returns ``(pixels, width, height, channels)`` with pixels as nested Python
    lists (independent of numpy).  Only filter types 0..2 are needed for the
    round-trip files this project writes; filter 0 is always emitted here.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    pos = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, *_ = struct.unpack(">IIBBBBB", body)
            if bit_depth != 8 or color_type not in (2, 6):
                raise ValueError("only 8-bit RGB/RGBA PNGs supported")
        elif tag == b"IDAT":
            idat.extend(body)
        elif tag == b"IEND":
            break
    channels = 4 if color_type == 6 else 3
    stride = width * channels
    inflated = zlib.decompress(bytes(idat))
    pixels: List[List[List[int]]] = []
    prev = bytearray(stride)
    ip = 0
    for _ in range(height):
        filter_type = inflated[ip]
        ip += 1
        line = bytearray(inflated[ip:ip + stride])
        ip += stride
        if filter_type == 1:  # Sub
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + left) & 0xFF
        elif filter_type == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"unsupported PNG filter type {filter_type}")
        row = [list(line[c:c + channels]) for c in range(0, stride, channels)]
        pixels.append(row)
        prev = line
    return pixels, width, height, channels
