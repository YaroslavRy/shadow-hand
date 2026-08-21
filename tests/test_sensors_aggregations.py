import unittest

from shadow_hand.sensors.aggregations import (
    aggregate_fingers,
    aggregate_regions,
    normalize_regions,
)


class SensorAggregationTests(unittest.TestCase):
    def test_aggregate_fingers_sums_tip_pad_and_palm_channels(self) -> None:
        by_name = {
            "thumb_tip": 0.1,
            "thumb_pad": 0.2,
            "index_tip": 1.0,
            "index_pad": 2.0,
            "middle_tip": 3.0,
            "middle_pad": 4.0,
            "ring_tip": 5.0,
            "ring_pad": 6.0,
            "pinky_tip": 7.0,
            "pinky_pad": 8.0,
            "palm_radial": 0.5,
            "palm_center": 0.75,
            "palm_ulnar": 1.25,
        }

        totals = aggregate_fingers(by_name)

        expected = {
            "thumb": 0.3,
            "index": 3.0,
            "middle": 7.0,
            "ring": 11.0,
            "pinky": 15.0,
            "palm": 2.5,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(totals[key], value)

    def test_aggregate_regions_is_heatmap_ready(self) -> None:
        by_name = {"thumb_tip": 0.5, "palm_center": 0.9}
        regions = aggregate_regions(by_name)

        self.assertEqual(regions["thumb_tip"], 0.5)
        self.assertEqual(regions["palm_center"], 0.9)
        self.assertEqual(regions["index_tip"], 0.0)

    def test_normalize_regions_clamps_to_zero_one(self) -> None:
        normalized = normalize_regions(
            {"thumb_tip": -1.0, "index_tip": 2.5, "palm_center": 10.0},
            max_value=5.0,
        )

        self.assertEqual(
            normalized,
            {
                "thumb_tip": 0.0,
                "index_tip": 0.5,
                "palm_center": 1.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
