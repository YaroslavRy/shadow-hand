"""Shared chrome helpers for the native diagnostics UI."""

from __future__ import annotations

import cv2
import numpy as np

from .layout import DiagnosticsLayout
from .palette import DiagnosticsPalette


def build_base_canvas(
    layout: DiagnosticsLayout,
    palette: DiagnosticsPalette,
    panel_specs: list[tuple[tuple[int, int], tuple[int, int], str]],
) -> np.ndarray:
    canvas = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    _draw_background(canvas, palette)
    for origin, size, title in panel_specs:
        _draw_panel(canvas, origin, size, title, palette)
    return canvas


def put_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_background(canvas: np.ndarray, palette: DiagnosticsPalette) -> None:
    h, w = canvas.shape[:2]
    top = np.array(palette.background_top, dtype=np.float32)
    bottom = np.array(palette.background_bottom, dtype=np.float32)
    for y in range(h):
        t = y / max(1, h - 1)
        color = (1 - t) * top + t * bottom
        canvas[y, :, :] = color.astype(np.uint8)
    for x in range(0, w, 28):
        canvas[:, x:x + 1] = palette.grid_minor
    for y in range(0, h, 28):
        canvas[y:y + 1, :] = palette.grid_major


def _draw_panel(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    title: str,
    palette: DiagnosticsPalette,
) -> None:
    x, y = top_left
    w, h = size
    cv2.rectangle(canvas, (x, y), (x + w, y + h), palette.panel_border, thickness=1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), palette.panel_fill, thickness=-1)
    cv2.rectangle(canvas, (x, y), (x + w, y + 30), palette.panel_header, thickness=-1)
    cv2.line(canvas, (x + 12, y + 22), (x + 68, y + 22), palette.panel_accent, 1)
    put_text(canvas, title, (x + 78, y + 22), 0.48, palette.panel_title)
