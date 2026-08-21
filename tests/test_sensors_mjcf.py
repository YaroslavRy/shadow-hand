import unittest

try:
    import mujoco
except ModuleNotFoundError:  # pragma: no cover - local env may not have mujoco
    mujoco = None

if mujoco is not None:
    from shadow_hand.sensors.mjcf import inspect_sensor_availability, read_named_sensordata
else:  # pragma: no cover - pure-Python test runs
    inspect_sensor_availability = None
    read_named_sensordata = None
from shadow_hand.sensors.schema import SENSOR_NAMES
from shadow_hand.settings import SCENE_PATH


class SensorMjcfTests(unittest.TestCase):
    @unittest.skipIf(mujoco is None, "mujoco is not installed in this interpreter")
    def test_scene_exposes_expected_touch_sensors(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        availability = inspect_sensor_availability(model)

        self.assertTrue(availability.available)
        self.assertEqual(availability.resolved_names, SENSOR_NAMES)
        self.assertEqual(availability.missing_names, ())

    @unittest.skipIf(mujoco is None, "mujoco is not installed in this interpreter")
    def test_sensordata_vector_matches_schema_length(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        data = mujoco.MjData(model)
        values, availability = read_named_sensordata(model, data)

        self.assertTrue(availability.available)
        self.assertEqual(len(values), len(SENSOR_NAMES))
        self.assertEqual(list(values), [0.0] * len(SENSOR_NAMES))


if __name__ == "__main__":
    unittest.main()
