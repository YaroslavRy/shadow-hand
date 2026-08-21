import unittest

from shadow_hand.sensors.schema import SENSOR_LAYOUT, SENSOR_NAMES, sensor_names


class SensorSchemaTests(unittest.TestCase):
    def test_sensor_names_are_unique(self) -> None:
        self.assertEqual(len(SENSOR_NAMES), len(set(SENSOR_NAMES)))

    def test_first_iteration_layout_is_stable(self) -> None:
        self.assertEqual(len(SENSOR_LAYOUT), 13)
        self.assertEqual(
            SENSOR_NAMES[:5],
            (
                "thumb_tip",
                "index_tip",
                "middle_tip",
                "ring_tip",
                "pinky_tip",
            ),
        )

    def test_schema_helper_returns_list_copy(self) -> None:
        names = sensor_names()
        self.assertEqual(names, list(SENSOR_NAMES))
        self.assertIsNot(names, SENSOR_NAMES)


if __name__ == "__main__":
    unittest.main()
