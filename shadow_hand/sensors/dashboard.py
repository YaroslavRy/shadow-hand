"""Small HTML-first evaluation dashboard for tactile/contact experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from html import escape
from typing import Mapping

from .aggregations import aggregate_fingers, aggregate_regions, normalize_regions
from .runtime import SensorSnapshot

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky", "palm")
REGION_ORDER = (
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
    "thumb_pad",
    "index_pad",
    "middle_pad",
    "ring_pad",
    "pinky_pad",
    "palm_radial",
    "palm_center",
    "palm_ulnar",
)
HEATMAP_ROWS = (
    ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"),
    ("thumb_pad", "index_pad", "middle_pad", "ring_pad", "pinky_pad"),
    ("palm_radial", "palm_center", "palm_ulnar"),
)


@dataclass(frozen=True)
class DashboardState:
    finger_totals: dict[str, float]
    region_values: dict[str, float]
    normalized_regions: dict[str, float]
    total_contact: float
    history: tuple[float, ...]
    sensor_mode: str = "live sensordata"


@dataclass
class SensorHistory:
    maxlen: int = 60
    _values: deque[float] = field(default_factory=deque)

    def push(self, value: float) -> tuple[float, ...]:
        if self._values.maxlen != self.maxlen:
            self._values = deque(self._values, maxlen=self.maxlen)
        self._values.append(float(value))
        return tuple(self._values)


def build_dashboard_state(
    snapshot_or_values: SensorSnapshot | Mapping[str, float],
    history: tuple[float, ...] = (),
    max_value: float = 1.0,
    max_region_value: float | None = None,
    sensor_mode: str = "live sensordata",
) -> DashboardState:
    by_name = (
        snapshot_or_values.by_name
        if isinstance(snapshot_or_values, SensorSnapshot)
        else dict(snapshot_or_values)
    )
    region_scale = max_region_value if max_region_value is not None else max_value
    finger_totals = aggregate_fingers(by_name)
    region_values = aggregate_regions(by_name)
    normalized_regions = normalize_regions(region_values, max_value=region_scale)
    total_contact = sum(region_values.values())
    return DashboardState(
        finger_totals=finger_totals,
        region_values=region_values,
        normalized_regions=normalized_regions,
        total_contact=total_contact,
        history=tuple(float(v) for v in history),
        sensor_mode=sensor_mode,
    )


def render_finger_rows(finger_totals: dict[str, float], max_value: float) -> str:
    scale = max(max_value, 1e-6)
    rows = []
    for name in FINGER_ORDER:
        value = float(finger_totals.get(name, 0.0))
        width = max(4.0, min(100.0, (value / scale) * 100.0 if scale > 0 else 0.0))
        rows.append(
            "<div class='sensor-row'>"
            f"<span class='sensor-label'>{escape(name)}</span>"
            "<div class='sensor-bar'>"
            f"<div class='sensor-fill' style='width:{width:.1f}%'></div>"
            "</div>"
            f"<span class='sensor-value'>{value:.3f}</span>"
            "</div>"
        )
    return "".join(rows)


def render_finger_rows_text(
    state: DashboardState | Mapping[str, float],
    *,
    max_value: float,
    width: int = 10,
) -> list[tuple[str, str]]:
    finger_totals = state.finger_totals if isinstance(state, DashboardState) else dict(state)
    scale = max(max_value, 1e-6)
    rows = []
    for name in FINGER_ORDER:
        value = float(finger_totals.get(name, 0.0))
        filled = max(0, min(width, round((value / scale) * width)))
        bar = "=" * filled + "." * (width - filled)
        rows.append((name, f"{bar} {value:.3f}"))
    return rows


def render_heatmap_rows(region_values: dict[str, float]) -> str:
    cells = []
    for row in HEATMAP_ROWS:
        row_cells = []
        for region in row:
            intensity = max(0.0, min(1.0, float(region_values.get(region, 0.0))))
            alpha = 0.14 + 0.86 * intensity
            row_cells.append(
                "<div class='heat-cell' "
                f"title='{escape(region)}: {intensity:.3f}' "
                f"style='background:rgba(78, 245, 218, {alpha:.3f})'>"
                f"<span>{escape(region.replace('_', ' '))}</span>"
                "</div>"
            )
        cells.append(f"<div class='heat-row'>{''.join(row_cells)}</div>")
    return "".join(cells)


def render_history_svg(history: tuple[float, ...], width: int = 260, height: int = 70) -> str:
    if not history:
        return (
            f"<svg viewBox='0 0 {width} {height}' class='trace-svg'>"
            f"<rect x='0' y='0' width='{width}' height='{height}' rx='8' />"
            f"<text x='{width / 2:.0f}' y='{height / 2:.0f}' text-anchor='middle'>"
            "awaiting contact trace"
            "</text></svg>"
        )
    peak = max(max(history), 1e-6)
    if len(history) == 1:
        points = f"0,{height - ((history[0] / peak) * (height - 8) + 4):.1f}"
    else:
        step = width / (len(history) - 1)
        coords = []
        for idx, value in enumerate(history):
            x = idx * step
            y = height - ((value / peak) * (height - 8) + 4)
            coords.append(f"{x:.1f},{y:.1f}")
        points = " ".join(coords)
    return (
        f"<svg viewBox='0 0 {width} {height}' class='trace-svg'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' rx='8' />"
        f"<polyline points='{points}' /></svg>"
    )


def render_dashboard_html(state: DashboardState, max_value: float = 1.0) -> str:
    bars = render_finger_rows(state.finger_totals, max_value=max_value)
    heatmap = render_heatmap_rows(state.normalized_regions)
    trace = render_history_svg(state.history)
    return f"""
<div class="sensor-panel">
  <style>
    .sensor-panel {{
      background:#0f1722;
      border:1px solid #223247;
      border-radius:18px;
      padding:16px;
      color:#dce7f5;
      font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .sensor-grid {{
      display:grid;
      grid-template-columns:1.1fr 1fr;
      gap:14px;
    }}
    .sensor-card {{
      background:#111c2a;
      border:1px solid #1f3149;
      border-radius:14px;
      padding:12px;
    }}
    .sensor-title {{
      color:#8dc2ff;
      font-size:12px;
      letter-spacing:0.18em;
      text-transform:uppercase;
      margin:0 0 10px 0;
    }}
    .sensor-meta {{
      display:flex;
      justify-content:space-between;
      gap:12px;
      margin-bottom:12px;
      font-size:12px;
      color:#9eb3cc;
    }}
    .sensor-row {{
      display:grid;
      grid-template-columns:58px 1fr 52px;
      align-items:center;
      gap:10px;
      margin:8px 0;
    }}
    .sensor-label, .sensor-value {{
      font-size:12px;
      color:#c8d7ea;
    }}
    .sensor-bar {{
      height:10px;
      background:#0b121b;
      border:1px solid #203246;
      border-radius:999px;
      overflow:hidden;
    }}
    .sensor-fill {{
      height:100%;
      border-radius:999px;
      background:linear-gradient(90deg, #245fa5, #4ef5da);
    }}
    .heat-row {{
      display:grid;
      gap:8px;
      margin:8px 0;
    }}
    .heat-row:first-child,
    .heat-row:nth-child(2) {{
      grid-template-columns:repeat(5, minmax(0, 1fr));
    }}
    .heat-row:last-child {{
      grid-template-columns:repeat(3, minmax(0, 1fr));
    }}
    .heat-cell {{
      min-height:56px;
      border-radius:12px;
      border:1px solid rgba(142, 198, 255, 0.12);
      display:flex;
      align-items:flex-end;
      justify-content:flex-start;
      padding:8px;
      box-sizing:border-box;
    }}
    .heat-cell span {{
      font-size:10px;
      line-height:1.2;
      color:#071018;
      text-transform:uppercase;
    }}
    .trace-svg {{
      width:100%;
      height:80px;
      display:block;
    }}
    .trace-svg rect {{
      fill:#0b121b;
      stroke:#203246;
    }}
    .trace-svg polyline {{
      fill:none;
      stroke:#4ef5da;
      stroke-width:2.5;
      stroke-linecap:round;
      stroke-linejoin:round;
    }}
    .trace-svg text {{
      fill:#8097b3;
      font-size:11px;
    }}
  </style>
  <div class="sensor-meta">
    <span>sensor mode: {escape(state.sensor_mode)}</span>
    <span>total contact: {state.total_contact:.3f}</span>
  </div>
  <div class="sensor-grid">
    <div class="sensor-card">
      <div class="sensor-title">Linear Signals</div>
      {bars}
      <div class="sensor-title" style="margin-top:14px">Rolling Total</div>
      {trace}
    </div>
    <div class="sensor-card">
      <div class="sensor-title">Heatmap</div>
      {heatmap}
    </div>
  </div>
</div>
"""
