import unittest

from shadow_hand.sensors.dashboard import build_dashboard_state, render_finger_rows_text
from shadow_hand.sensors.ui import DEFAULT_DIAGNOSTICS_LAYOUT
try:
    from shadow_hand.sensors.plots import (
        SignalHistory,
        render_finger_table,
        render_heatmap_image,
        render_linear_plot,
        render_native_diagnostics,
    )
except ModuleNotFoundError:  # pragma: no cover - numpy may be absent in base interpreter
    SignalHistory = None
    render_finger_table = None
    render_heatmap_image = None
    render_linear_plot = None
    render_native_diagnostics = None


class SensorDashboardTests(unittest.TestCase):
    @unittest.skipIf(SignalHistory is None, "numpy-backed plot helpers unavailable")
    def test_signal_history_keeps_fixed_window(self) -> None:
        history = SignalHistory(maxlen=3)
        for value in [0.1, 0.2, 0.3, 0.4]:
            history.append(value)
        self.assertEqual(history.values(), [0.2, 0.3, 0.4])
        self.assertGreater(history.static_contact(), 0.0)
        self.assertGreater(history.held_peak(), 0.0)

    @unittest.skipIf(render_finger_table is None, "numpy-backed plot helpers unavailable")
    def test_render_finger_table_contains_named_rows(self) -> None:
        state = build_dashboard_state(
            {
                "thumb_tip": 0.1,
                "index_tip": 0.2,
                "middle_tip": 0.3,
                "ring_tip": 0.4,
                "pinky_tip": 0.5,
                "palm_center": 0.6,
            },
            max_region_value=1.0,
        )
        table = render_finger_table(state, max_value=1.0)
        self.assertIn("thumb", table)
        self.assertIn("palm", table)
        self.assertIn("total", table)

    @unittest.skipIf(render_linear_plot is None, "numpy-backed plot helpers unavailable")
    def test_plot_and_heatmap_render_numpy_images(self) -> None:
        history = SignalHistory(maxlen=4)
        for value in [0.0, 0.1, 0.4, 0.2]:
            history.append(value)
        state = build_dashboard_state(
            {"thumb_tip": 0.2, "index_tip": 0.4, "palm_center": 0.6},
            max_region_value=1.0,
        )
        plot = render_linear_plot(history)
        heatmap = render_heatmap_image(state)
        self.assertEqual(plot.shape, (160, 420, 3))
        self.assertEqual(heatmap.shape, (300, 240, 3))

    @unittest.skipIf(render_native_diagnostics is None, "numpy-backed plot helpers unavailable")
    def test_native_diagnostics_renders_panel_image(self) -> None:
        history = SignalHistory(maxlen=6)
        for value in [0.0, 0.2, 0.4, 0.3]:
            history.append(value)
        state = build_dashboard_state(
            {"thumb_tip": 0.2, "index_tip": 0.4, "palm_center": 0.6},
            max_region_value=1.0,
        )
        image = render_native_diagnostics(
            state,
            history,
            actuator_rows=[("rh_A_WRJ1", 0.1), ("rh_A_FFJ0", 0.4)],
            peak_sensor=("palm_center", 0.6),
            active_sensors=3,
        )
        self.assertEqual(
            image.shape,
            (
                DEFAULT_DIAGNOSTICS_LAYOUT.height,
                DEFAULT_DIAGNOSTICS_LAYOUT.width,
                3,
            ),
        )

    def test_native_diagnostics_layout_is_module_backed(self) -> None:
        self.assertEqual(DEFAULT_DIAGNOSTICS_LAYOUT.width, 980)
        self.assertEqual(DEFAULT_DIAGNOSTICS_LAYOUT.height, 640)

    def test_dashboard_state_keeps_contact_summary_without_numpy(self) -> None:
        state = build_dashboard_state(
            {"thumb_tip": 0.2, "thumb_pad": 0.3, "index_tip": 1.0, "palm_center": 0.8},
            max_region_value=1.0,
        )
        self.assertAlmostEqual(state.finger_totals["thumb"], 0.5)
        self.assertAlmostEqual(state.finger_totals["index"], 1.0)
        self.assertAlmostEqual(state.finger_totals["palm"], 0.8)
        self.assertAlmostEqual(state.total_contact, 2.3)

    def test_native_text_rows_support_fixed_width_bars(self) -> None:
        state = build_dashboard_state(
            {"thumb_tip": 0.2, "thumb_pad": 0.3, "index_tip": 1.0, "palm_center": 0.8},
            max_region_value=1.0,
        )
        rows = render_finger_rows_text(state, max_value=1.0, width=5)
        self.assertEqual(rows[0][0], "thumb")
        self.assertIn("==", rows[0][1])
        self.assertIn("0.500", rows[0][1])


if __name__ == "__main__":
    unittest.main()
