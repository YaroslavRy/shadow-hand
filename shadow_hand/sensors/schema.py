"""Canonical sensor layout for the first tactile/contact iteration.

This module is intentionally simple and pure-Python so we can test the sensor
pipeline before wiring it into MuJoCo XML and runtime data.sensordata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorDescriptor:
    name: str
    finger: str
    region: str


SENSOR_LAYOUT = (
    SensorDescriptor("thumb_tip", "thumb", "thumb_tip"),
    SensorDescriptor("index_tip", "index", "index_tip"),
    SensorDescriptor("middle_tip", "middle", "middle_tip"),
    SensorDescriptor("ring_tip", "ring", "ring_tip"),
    SensorDescriptor("pinky_tip", "pinky", "pinky_tip"),
    SensorDescriptor("thumb_pad", "thumb", "thumb_pad"),
    SensorDescriptor("index_pad", "index", "index_pad"),
    SensorDescriptor("middle_pad", "middle", "middle_pad"),
    SensorDescriptor("ring_pad", "ring", "ring_pad"),
    SensorDescriptor("pinky_pad", "pinky", "pinky_pad"),
    SensorDescriptor("palm_radial", "palm", "palm_radial"),
    SensorDescriptor("palm_center", "palm", "palm_center"),
    SensorDescriptor("palm_ulnar", "palm", "palm_ulnar"),
)

SENSOR_NAMES = tuple(sensor.name for sensor in SENSOR_LAYOUT)
SENSOR_REGIONS = tuple(sensor.region for sensor in SENSOR_LAYOUT)


def sensor_names() -> list[str]:
    return list(SENSOR_NAMES)
