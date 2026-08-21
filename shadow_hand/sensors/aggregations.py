"""Small deterministic aggregations for plots and heatmaps."""

from __future__ import annotations

from collections import defaultdict

from .schema import SENSOR_LAYOUT


def aggregate_fingers(by_name: dict[str, float]) -> dict[str, float]:
    totals = defaultdict(float)
    for sensor in SENSOR_LAYOUT:
        totals[sensor.finger] += float(by_name.get(sensor.name, 0.0))
    return dict(totals)


def aggregate_regions(by_name: dict[str, float]) -> dict[str, float]:
    return {
        sensor.region: float(by_name.get(sensor.name, 0.0))
        for sensor in SENSOR_LAYOUT
    }


def normalize_regions(region_values: dict[str, float], max_value: float) -> dict[str, float]:
    if max_value <= 0:
        raise ValueError("max_value must be positive")
    return {
        region: max(0.0, min(1.0, float(value) / max_value))
        for region, value in region_values.items()
    }
