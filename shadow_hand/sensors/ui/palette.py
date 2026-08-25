"""Visual palette for the native diagnostics UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticsPalette:
    background_top: tuple[int, int, int] = (7, 11, 18)
    background_bottom: tuple[int, int, int] = (10, 17, 27)
    grid_major: tuple[int, int, int] = (13, 20, 30)
    grid_minor: tuple[int, int, int] = (16, 25, 37)
    panel_fill: tuple[int, int, int] = (10, 16, 25)
    panel_border: tuple[int, int, int] = (32, 48, 70)
    panel_header: tuple[int, int, int] = (13, 21, 33)
    panel_accent: tuple[int, int, int] = (244, 150, 30)
    panel_title: tuple[int, int, int] = (146, 190, 240)
    text_primary: tuple[int, int, int] = (230, 236, 242)
    text_secondary: tuple[int, int, int] = (168, 196, 218)
    trace: tuple[int, int, int] = (79, 209, 197)


DEFAULT_PALETTE = DiagnosticsPalette()
