"""Geometry and sizing for the native diagnostics UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticsLayout:
    width: int = 980
    height: int = 640
    summary_origin: tuple[int, int] = (24, 24)
    summary_size: tuple[int, int] = (340, 214)
    trace_origin: tuple[int, int] = (24, 262)
    trace_size: tuple[int, int] = (430, 188)
    heatmap_origin: tuple[int, int] = (390, 24)
    heatmap_size: tuple[int, int] = (228, 384)
    actuators_origin: tuple[int, int] = (642, 24)
    actuators_size: tuple[int, int] = (314, 592)
    trace_image_size: tuple[int, int] = (388, 154)
    trace_image_origin: tuple[int, int] = (44, 286)
    heatmap_image_size: tuple[int, int] = (196, 336)
    heatmap_image_origin: tuple[int, int] = (406, 48)
    actuator_col_width: int = 146
    actuator_row_height: int = 22
    actuator_text_origin: tuple[int, int] = (662, 92)
    summary_text_origin: tuple[int, int] = (42, 70)
    summary_text_step: int = 28
    finger_row_origin: tuple[int, int] = (42, 214)
    finger_row_step: int = 20
    footer_label_origin: tuple[int, int] = (662, 52)


DEFAULT_DIAGNOSTICS_LAYOUT = DiagnosticsLayout()
