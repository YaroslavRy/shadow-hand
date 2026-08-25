"""Tiny image renderers for sensor evaluation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .dashboard import DashboardState, FINGER_ORDER, REGION_ORDER, render_finger_rows_text
from .ui import (
    DEFAULT_DIAGNOSTICS_LAYOUT,
    DEFAULT_PALETTE,
    DiagnosticsLayout,
    DiagnosticsPalette,
    build_base_canvas,
    put_text,
)


@dataclass
class SignalHistory:
    maxlen: int = 120
    total_contact: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    ema_contact: float = 0.0
    peak_hold: float = 0.0

    def __post_init__(self) -> None:
        self.total_contact = deque(self.total_contact, maxlen=self.maxlen)

    def append(self, value: float) -> None:
        value = float(value)
        self.total_contact.append(value)
        alpha = 0.88
        self.ema_contact = alpha * self.ema_contact + (1.0 - alpha) * value
        self.peak_hold = max(value, self.peak_hold * 0.965)

    def values(self) -> list[float]:
        return list(self.total_contact)

    def rolling_mean(self, window: int = 12) -> float:
        values = self.values()
        if not values:
            return 0.0
        tail = values[-max(1, window):]
        return float(sum(tail) / len(tail))

    def static_contact(self) -> float:
        return float(self.ema_contact)

    def held_peak(self) -> float:
        return float(self.peak_hold)


def render_linear_plot(history: SignalHistory, width: int = 420, height: int = 160) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (13, 22, 33)
    canvas[[0, -1], :, :] = (53, 74, 101)
    canvas[:, [0, -1], :] = (53, 74, 101)

    values = history.values()
    if len(values) < 2:
        return canvas

    max_value = max(max(values), 1e-6)
    xs = np.linspace(12, width - 12, len(values)).astype(int)
    ys = np.asarray(
        [height - 14 - int((max(0.0, value) / max_value) * (height - 30)) for value in values]
    )
    for x0, y0, x1, y1 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
        _draw_line(canvas, x0, y0, x1, y1, color=DEFAULT_PALETTE.trace)
    return canvas


def render_heatmap_image(state: DashboardState, width: int = 240, height: int = 300) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (10, 18, 28)

    def sx(value: float) -> int:
        return int(round(value * width))

    def sy(value: float) -> int:
        return int(round(value * height))

    _draw_circle(canvas, (sx(0.22), sy(0.63)), max(10, width // 12), state.normalized_regions.get("thumb_tip", 0.0))
    _draw_circle(canvas, (sx(0.30), sy(0.74)), max(12, width // 10), state.normalized_regions.get("thumb_pad", 0.0))

    x_positions = {
        "index": sx(0.42),
        "middle": sx(0.58),
        "ring": sx(0.74),
        "pinky": sx(0.90),
    }
    for finger, x in x_positions.items():
        _draw_circle(canvas, (x, sy(0.21)), max(10, width // 14), state.normalized_regions.get(f"{finger}_tip", 0.0))
        _draw_circle(canvas, (x, sy(0.43)), max(12, width // 12), state.normalized_regions.get(f"{finger}_pad", 0.0))

    _draw_circle(canvas, (sx(0.46), sy(0.70)), max(14, width // 9), state.normalized_regions.get("palm_radial", 0.0))
    _draw_circle(canvas, (sx(0.63), sy(0.76)), max(16, width // 7), state.normalized_regions.get("palm_center", 0.0))
    _draw_circle(canvas, (sx(0.81), sy(0.70)), max(14, width // 9), state.normalized_regions.get("palm_ulnar", 0.0))
    return canvas


def render_finger_table(state: DashboardState, *, max_value: float = 2.0) -> str:
    lines = ["finger bars"]
    for finger in FINGER_ORDER:
        value = float(state.finger_totals.get(finger, 0.0))
        filled = int(round(max(0.0, min(1.0, value / max_value)) * 12))
        bar = "#" * filled + "." * (12 - filled)
        lines.append(f"{finger:>6}  {bar}  {value:5.2f}")
    lines.append(f" total  {state.total_contact:5.2f}")
    return "\n".join(lines)


def render_native_diagnostics(
    state: DashboardState,
    history: SignalHistory,
    *,
    actuator_rows: list[tuple[str, float]],
    peak_sensor: tuple[str, float],
    active_sensors: int,
    width: int = DEFAULT_DIAGNOSTICS_LAYOUT.width,
    height: int = DEFAULT_DIAGNOSTICS_LAYOUT.height,
) -> np.ndarray:
    return _get_renderer(width, height).render(
        state,
        history,
        actuator_rows=actuator_rows,
        peak_sensor=peak_sensor,
        active_sensors=active_sensors,
    )


class NativeDiagnosticsRenderer:
    def __init__(
        self,
        layout: DiagnosticsLayout = DEFAULT_DIAGNOSTICS_LAYOUT,
        palette: DiagnosticsPalette = DEFAULT_PALETTE,
    ) -> None:
        self.layout = layout
        self.palette = palette
        self.base = build_base_canvas(
            layout,
            palette,
            panel_specs=[
                (layout.summary_origin, layout.summary_size, "TACTILE SUMMARY"),
                (layout.trace_origin, layout.trace_size, "CONTACT TRACE"),
                (layout.heatmap_origin, layout.heatmap_size, "TACTILE MAP"),
                (layout.actuators_origin, layout.actuators_size, "TACTILE CHANNELS"),
            ],
        )

    def render(
        self,
        state: DashboardState,
        history: SignalHistory,
        *,
        actuator_rows: list[tuple[str, float]],
        peak_sensor: tuple[str, float],
        active_sensors: int,
    ) -> np.ndarray:
        layout = self.layout
        palette = self.palette
        canvas = self.base.copy()

        info_lines = [
            f"total contact   {state.total_contact:7.3f}",
            f"static contact  {history.static_contact():7.3f}",
            f"hold peak       {history.held_peak():7.3f}",
            f"active sensors  {active_sensors:7d}",
            f"peak sensor     {peak_sensor[0]:>12s} {peak_sensor[1]:.3f}",
        ]
        for idx, line in enumerate(info_lines):
            put_text(
                canvas,
                line,
                (
                    layout.summary_text_origin[0],
                    layout.summary_text_origin[1] + idx * layout.summary_text_step,
                ),
                0.62,
                palette.text_primary,
            )

        finger_rows = render_finger_rows_text(state, max_value=0.6, width=12)
        for idx, (finger, row) in enumerate(finger_rows):
            put_text(
                canvas,
                f"{finger:>6}  {row}",
                (
                    layout.finger_row_origin[0],
                    layout.finger_row_origin[1] + idx * layout.finger_row_step,
                ),
                0.49,
                palette.text_secondary,
            )

        trace_w, trace_h = layout.trace_image_size
        trace = render_linear_plot(history, width=trace_w, height=trace_h)
        tx, ty = layout.trace_image_origin
        canvas[ty:ty + trace_h, tx:tx + trace_w] = trace

        heatmap_w, heatmap_h = layout.heatmap_image_size
        heatmap = render_heatmap_image(state, width=heatmap_w, height=heatmap_h)
        hx, hy = layout.heatmap_image_origin
        canvas[hy:hy + heatmap_h, hx:hx + heatmap_w] = heatmap

        put_text(canvas, "13 channels · raw values", layout.footer_label_origin, 0.46, palette.text_secondary)
        sensor_rows = [(name, float(state.region_values.get(name, 0.0))) for name in REGION_ORDER]
        for idx, (name, value) in enumerate(sensor_rows):
            col = idx // 7
            row = idx % 7
            x = layout.actuator_text_origin[0] + col * layout.actuator_col_width
            y = layout.actuator_text_origin[1] + row * layout.actuator_row_height
            bar_width = 8
            normalized = min(1.0, max(0.0, value / 0.35))
            filled = max(0, min(bar_width, round(normalized * bar_width)))
            bar = "=" * filled + "." * (bar_width - filled)
            short = name.replace("_", " ")
            put_text(canvas, f"{short:<12} {bar} {value: .3f}", (x, y), 0.42, palette.text_primary)

        # keep one compact actuator sanity line so we still know the control path is alive
        if actuator_rows:
            peak_act_name, peak_act_value = max(actuator_rows, key=lambda item: abs(item[1]))
            put_text(
                canvas,
                f"peak actuator {peak_act_name.replace('rh_A_', '')} {peak_act_value: .3f}",
                (42, layout.trace_origin[1] + layout.trace_size[1] + 34),
                0.42,
                palette.text_secondary,
            )

        return canvas


def _draw_line(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    color: tuple[int, int, int],
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).astype(int)
    ys = np.linspace(y0, y1, steps + 1).astype(int)
    canvas[ys.clip(0, canvas.shape[0] - 1), xs.clip(0, canvas.shape[1] - 1)] = color


def _draw_circle(
    canvas: np.ndarray,
    center: tuple[int, int],
    radius: int,
    intensity: float,
) -> None:
    value = float(max(0.0, min(1.0, intensity)))
    color = np.array(
        [25 + value * 40, 45 + value * 110, 60 + value * 195],
        dtype=np.uint8,
    )
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    cx, cy = center
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    canvas[mask] = color


@lru_cache(maxsize=4)
def _get_renderer(width: int, height: int) -> NativeDiagnosticsRenderer:
    if width == DEFAULT_DIAGNOSTICS_LAYOUT.width and height == DEFAULT_DIAGNOSTICS_LAYOUT.height:
        layout = DEFAULT_DIAGNOSTICS_LAYOUT
    else:
        layout = DiagnosticsLayout(width=width, height=height)
    return NativeDiagnosticsRenderer(layout=layout, palette=DEFAULT_PALETTE)
