"""Tiny image renderers for sensor evaluation.

Pure numpy on purpose: keeps the test loop light and avoids extra UI/runtime
dependencies when we only need quick diagnostic visuals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .dashboard import DashboardState, FINGER_ORDER, render_finger_rows_text


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
        _draw_line(canvas, x0, y0, x1, y1, color=(79, 209, 197))
    return canvas


def render_heatmap_image(state: DashboardState, width: int = 240, height: int = 300) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (10, 18, 28)
    _draw_circle(canvas, (54, 190), 16, state.normalized_regions.get("thumb_tip", 0.0))
    _draw_circle(canvas, (72, 220), 18, state.normalized_regions.get("thumb_pad", 0.0))

    x_positions = {
        "index": 82,
        "middle": 116,
        "ring": 150,
        "pinky": 184,
    }
    for finger, x in x_positions.items():
        _draw_circle(canvas, (x, 62), 14, state.normalized_regions.get(f"{finger}_tip", 0.0))
        _draw_circle(canvas, (x, 128), 16, state.normalized_regions.get(f"{finger}_pad", 0.0))

    _draw_circle(canvas, (88, 210), 22, state.normalized_regions.get("palm_radial", 0.0))
    _draw_circle(canvas, (122, 228), 28, state.normalized_regions.get("palm_center", 0.0))
    _draw_circle(canvas, (158, 210), 22, state.normalized_regions.get("palm_ulnar", 0.0))
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
    width: int = 1160,
    height: int = 760,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_background(canvas)

    _draw_panel(canvas, (24, 24), (360, 198), "TACTILE SUMMARY")
    _draw_panel(canvas, (24, 246), (506, 230), "CONTACT TRACE")
    _draw_panel(canvas, (560, 24), (264, 430), "TACTILE MAP")
    _draw_panel(canvas, (850, 24), (286, 706), "DRIVEN ACTUATORS")

    info_lines = [
        f"total contact   {state.total_contact:7.3f}",
        f"static contact  {history.static_contact():7.3f}",
        f"hold peak       {history.held_peak():7.3f}",
        f"active sensors  {active_sensors:7d}",
        f"peak sensor     {peak_sensor[0]:>12s} {peak_sensor[1]:.3f}",
    ]
    for idx, line in enumerate(info_lines):
        _put_text(canvas, line, (42, 70 + idx * 28), 0.66, (230, 236, 242))

    finger_rows = render_finger_rows_text(state, max_value=0.6, width=12)
    for idx, (finger, row) in enumerate(finger_rows):
        _put_text(
            canvas,
            f"{finger:>6}  {row}",
            (42, 212 + idx * 24),
            0.52,
            (168, 196, 218),
        )

    trace = render_linear_plot(history, width=464, height=200)
    canvas[266:466, 44:508] = trace

    heatmap = render_heatmap_image(state, width=236, height=404)
    canvas[42:446, 574:810] = heatmap

    _put_text(canvas, "coverage 20/20 driven", (868, 52), 0.5, (160, 192, 220))
    for idx, (name, value) in enumerate(actuator_rows):
        col = idx // 10
        row = idx % 10
        x = 868 + col * 132
        y = 92 + row * 28
        short = name.replace("rh_A_", "")
        _put_text(canvas, f"{short:>7} {value: .3f}", (x, y), 0.47, (220, 228, 236))

    return canvas


def _draw_background(canvas: np.ndarray) -> None:
    h, w = canvas.shape[:2]
    top = np.array([7, 11, 18], dtype=np.float32)
    bottom = np.array([11, 18, 28], dtype=np.float32)
    for y in range(h):
        t = y / max(1, h - 1)
        color = (1 - t) * top + t * bottom
        canvas[y, :, :] = color.astype(np.uint8)
    for x in range(0, w, 28):
        canvas[:, x:x + 1] = (15, 24, 36)
    for y in range(0, h, 28):
        canvas[y:y + 1, :] = (13, 20, 30)


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


def _draw_panel(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    title: str,
) -> None:
    x, y = top_left
    w, h = size
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (32, 48, 70), thickness=1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (12, 18, 28), thickness=-1)
    cv2.rectangle(canvas, (x, y), (x + w, y + 30), (14, 22, 34), thickness=-1)
    cv2.line(canvas, (x + 12, y + 22), (x + 68, y + 22), (244, 150, 30), 1)
    _put_text(canvas, title, (x + 78, y + 22), 0.48, (146, 190, 240))


def _put_text(
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
