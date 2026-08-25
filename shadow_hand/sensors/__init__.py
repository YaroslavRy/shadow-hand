"""Sensor utilities for Shadow Hand tactile/contact experiments."""

from .aggregations import (
    aggregate_regions,
    aggregate_fingers,
    normalize_regions,
)
from .dashboard import (
    DashboardState,
    build_dashboard_state,
    render_finger_rows,
    render_finger_rows_text,
    render_heatmap_rows,
)
from .runtime import SensorAvailability, SensorSnapshot, build_snapshot, named_values
from .schema import SENSOR_LAYOUT, SENSOR_NAMES, SENSOR_REGIONS, sensor_names
from .ui import DEFAULT_DIAGNOSTICS_LAYOUT

try:
    from .mjcf import inspect_sensor_availability, read_named_sensordata, resolve_sensor_bindings
except ModuleNotFoundError:  # pragma: no cover - optional in pure-Python test runs
    inspect_sensor_availability = None
    read_named_sensordata = None
    resolve_sensor_bindings = None

try:
    from .plots import (
        SignalHistory,
        render_finger_table,
        render_heatmap_image,
        render_linear_plot,
        render_native_diagnostics,
    )
except ModuleNotFoundError:  # pragma: no cover - optional in pure-Python test runs
    SignalHistory = None
    render_finger_table = None
    render_heatmap_image = None
    render_linear_plot = None
    render_native_diagnostics = None

__all__ = [
    "DashboardState",
    "DEFAULT_DIAGNOSTICS_LAYOUT",
    "SENSOR_LAYOUT",
    "SENSOR_NAMES",
    "SENSOR_REGIONS",
    "SensorAvailability",
    "SensorSnapshot",
    "SignalHistory",
    "aggregate_fingers",
    "aggregate_regions",
    "build_dashboard_state",
    "build_snapshot",
    "inspect_sensor_availability",
    "named_values",
    "normalize_regions",
    "read_named_sensordata",
    "render_finger_table",
    "render_finger_rows",
    "render_finger_rows_text",
    "render_heatmap_image",
    "render_heatmap_rows",
    "render_linear_plot",
    "render_native_diagnostics",
    "resolve_sensor_bindings",
    "sensor_names",
]
