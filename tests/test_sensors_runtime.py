import unittest

from shadow_hand.sensors.runtime import build_snapshot, named_values


class SensorRuntimeTests(unittest.TestCase):
    def test_named_values_maps_flat_vector_to_sensor_names(self) -> None:
        values = list(range(13))
        by_name = named_values(values)

        self.assertEqual(by_name["thumb_tip"], 0.0)
        self.assertEqual(by_name["pinky_pad"], 9.0)
        self.assertEqual(by_name["palm_ulnar"], 12.0)

    def test_named_values_rejects_wrong_vector_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 13 sensor values"):
            named_values([1.0, 2.0])

    def test_build_snapshot_keeps_raw_and_named_views(self) -> None:
        snapshot = build_snapshot([0.1] * 13)

        self.assertEqual(len(snapshot.raw), 13)
        self.assertEqual(snapshot.by_name["index_tip"], 0.1)
        self.assertTrue(snapshot.availability.available)


if __name__ == "__main__":
    unittest.main()
