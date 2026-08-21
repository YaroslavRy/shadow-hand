"""Runtime adapter from MuJoCo sensordata vectors to structured snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import SENSOR_LAYOUT, SENSOR_NAMES


@dataclass(frozen=True)
class SensorAvailability:
    available: bool
    expected_names: tuple[str, ...]
    resolved_names: tuple[str, ...]
    missing_names: tuple[str, ...]


@dataclass(frozen=True)
class SensorSnapshot:
    raw: tuple[float, ...]
    by_name: dict[str, float]
    availability: SensorAvailability


def _default_availability() -> SensorAvailability:
    return SensorAvailability(
        available=True,
        expected_names=tuple(SENSOR_NAMES),
        resolved_names=tuple(SENSOR_NAMES),
        missing_names=(),
    )


def named_values(values: Iterable[float], *, names: Iterable[str] | None = None) -> dict[str, float]:
    array = tuple(float(value) for value in values)
    expected_names = tuple(names or SENSOR_NAMES)
    if len(array) != len(SENSOR_LAYOUT):
        raise ValueError(
            f"expected {len(SENSOR_LAYOUT)} sensor values, got {len(array)}"
        )
    return {name: float(value) for name, value in zip(expected_names, array)}


def build_snapshot(
    values: Iterable[float],
    *,
    names: Iterable[str] | None = None,
    availability: SensorAvailability | None = None,
) -> SensorSnapshot:
    array = tuple(float(value) for value in values)
    return SensorSnapshot(
        raw=array,
        by_name=named_values(array, names=names),
        availability=availability or _default_availability(),
    )


def build_snapshot_from_data(
    data,
    *,
    names: Iterable[str] | None = None,
    availability: SensorAvailability | None = None,
) -> SensorSnapshot:
    return build_snapshot(
        data.sensordata[: len(SENSOR_LAYOUT)],
        names=names,
        availability=availability,
    )
