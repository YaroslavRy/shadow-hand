"""Helpers for validating the MuJoCo sensor wiring against the schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco

from .runtime import SensorAvailability
from .schema import SENSOR_NAMES


@dataclass(frozen=True)
class SensorValidation:
    availability: SensorAvailability
    sensor_count: int


@dataclass(frozen=True)
class SensorBindings:
    ordered_indices: tuple[int | None, ...]
    availability: SensorAvailability


def model_sensor_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = []
    for idx in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, idx)
        names.append(name or f"sensor_{idx}")
    return tuple(names)


def validate_model_sensors(model: mujoco.MjModel) -> SensorValidation:
    expected = tuple(SENSOR_NAMES)
    resolved = model_sensor_names(model)
    missing = tuple(name for name in expected if name not in resolved)
    availability = SensorAvailability(
        available=not missing,
        expected_names=expected,
        resolved_names=resolved,
        missing_names=missing,
    )
    return SensorValidation(availability=availability, sensor_count=model.nsensor)


def resolve_sensor_bindings(model: mujoco.MjModel) -> SensorBindings:
    resolved = model_sensor_names(model)
    index_by_name = {name: idx for idx, name in enumerate(resolved)}
    ordered_indices = tuple(index_by_name.get(name) for name in SENSOR_NAMES)
    missing = tuple(name for name, idx in zip(SENSOR_NAMES, ordered_indices) if idx is None)
    availability = SensorAvailability(
        available=not missing,
        expected_names=tuple(SENSOR_NAMES),
        resolved_names=resolved,
        missing_names=missing,
    )
    return SensorBindings(ordered_indices=ordered_indices, availability=availability)


def inspect_sensor_availability(model: mujoco.MjModel) -> SensorAvailability:
    return resolve_sensor_bindings(model).availability


def read_named_sensordata(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[tuple[float, ...], SensorAvailability]:
    bindings = resolve_sensor_bindings(model)
    values = []
    for idx in bindings.ordered_indices:
        values.append(0.0 if idx is None else float(data.sensordata[idx]))
    return tuple(values), bindings.availability
