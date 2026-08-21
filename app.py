"""Browser demo for Shadow Hand retargeting.

Designed for Hugging Face Spaces, serving both execution paths:
  /server   webcam frames are retargeted and rendered server-side (this module)
  /browser  the static MuJoCo-WASM build, which runs entirely in the client
"""

import logging
import threading
import time
import json

import cv2
import gradio as gr
import mediapipe as mp
import mujoco
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from shadow_hand.model import extract_shadow_hand_targets
from shadow_hand.sensors import (
    SignalHistory,
    build_dashboard_state,
    build_snapshot,
    read_named_sensordata,
    render_finger_table,
    render_heatmap_image,
    render_linear_plot,
)
from shadow_hand.settings import ACTUATOR_MAP, PROJECT_ROOT, SCENE_PATH

SCENE_PARAMS = json.loads((PROJECT_ROOT / "scene_params.json").read_text())
SERVER_SCENE_PARAMS = SCENE_PARAMS["server"]

# Keep the full Menagerie checkout out of the deployment. Locally we retain
# SCENE_PATH; the Space instead carries this small, self-contained asset copy.
SPACE_SCENE_PATH = PROJECT_ROOT / "space_assets" / "shadow_hand" / "my_scene.xml"
log = logging.getLogger(__name__)


class ShadowHandRenderer:
    """A serialized, headless MuJoCo + MediaPipe inference pipeline."""

    def __init__(self, width: int = 400, height: int = 300) -> None:
        scene_path = SPACE_SCENE_PATH if SPACE_SCENE_PATH.exists() else SCENE_PATH
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
        )
        self.actuator_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in ACTUATOR_MAP
        }
        self.lock = threading.Lock()
        self.frames_processed = 0
        self.sensor_history = SignalHistory(maxlen=120)

    def _set_camera(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        target_x: float,
        target_y: float,
        target_z: float,
    ) -> None:
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self.camera.distance = distance
        self.camera.lookat[:] = (target_x, target_y, target_z)

    def _render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def process(
        self,
        frame: np.ndarray,
        azimuth: float = SERVER_SCENE_PARAMS["azimuth"],
        elevation: float = SERVER_SCENE_PARAMS["elevation"],
        distance: float = SERVER_SCENE_PARAMS["distance"],
        target_x: float = SERVER_SCENE_PARAMS["target"][0],
        target_y: float = SERVER_SCENE_PARAMS["target"][1],
        target_z: float = SERVER_SCENE_PARAMS["target"][2],
    ):
        """Retarget the latest camera frame and return preview + render."""
        if frame is None:
            sensor_plot, sensor_heatmap, sensor_table, sensor_status = self._sensor_outputs()
            return (
                None,
                None,
                "Waiting for a webcam frame.",
                sensor_status,
                sensor_table,
                sensor_plot,
                sensor_heatmap,
            )

        with self.lock:
            started = time.perf_counter()
            self._set_camera(
                azimuth, elevation, distance, target_x, target_y, target_z
            )
            # Keep CPU inference and browser round trips light on CPU Basic.
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            # Mirror to match the familiar selfie-view coordinate convention.
            image = cv2.flip(frame, 1)
            results = self.hands.process(image)
            annotated = image.copy()
            keypoints = None

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                keypoints = np.asarray([[p.x, p.y, p.z] for p in hand.landmark])
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated, hand, mp.solutions.hands.HAND_CONNECTIONS
                )
            inference_ms = (time.perf_counter() - started) * 1_000

            if keypoints is None:
                status = "No hand detected — keep one hand visible to the camera."
            else:
                targets = extract_shadow_hand_targets(keypoints)
                for name, value in targets.items():
                    self.data.ctrl[self.actuator_ids[name]] = value
                # A short settle is enough for a responsive visual preview.
                simulation_started = time.perf_counter()
                for _ in range(4):
                    mujoco.mj_step(self.model, self.data)
                simulation_ms = (time.perf_counter() - simulation_started) * 1_000
                status = "Hand detected · 20 Shadow Hand actuators retargeted."
            if keypoints is None:
                simulation_ms = 0.0

            sensor_plot, sensor_heatmap, sensor_table, sensor_status = self._sensor_outputs()
            render_started = time.perf_counter()
            rendered = self._render()
            render_ms = (time.perf_counter() - render_started) * 1_000
            status += f"  |  infer {inference_ms:.0f} ms · sim {simulation_ms:.0f} ms · render {render_ms:.0f} ms"
            self.frames_processed += 1
            if self.frames_processed % 5 == 0:
                total_ms = (time.perf_counter() - started) * 1_000
                log.info(
                    "frame=%d inference=%.0fms render=%.0fms total=%.0fms",
                    self.frames_processed, inference_ms, render_ms, total_ms,
                )
            preview = cv2.resize(annotated, (160, 120), interpolation=cv2.INTER_AREA)
            return preview, rendered, status, sensor_status, sensor_table, sensor_plot, sensor_heatmap

    def update_camera(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        target_x: float,
        target_y: float,
        target_z: float,
    ):
        """Reframe the already-simulated pose without rerunning MediaPipe."""
        with self.lock:
            self._set_camera(
                azimuth, elevation, distance, target_x, target_y, target_z
            )
            _, _, sensor_table, sensor_status = self._sensor_outputs()
            return self._render(), "Scene camera updated.", sensor_status, sensor_table

    def _sensor_outputs(self):
        sensor_values, availability = read_named_sensordata(self.model, self.data)
        snapshot = build_snapshot(sensor_values, availability=availability)
        state = build_dashboard_state(snapshot.by_name, max_region_value=0.35)
        self.sensor_history.append(state.total_contact)
        sensor_plot = render_linear_plot(self.sensor_history)
        sensor_heatmap = render_heatmap_image(state)
        sensor_table = render_finger_table(state, max_value=0.6)
        if availability.available:
            sensor_status = (
                f"{len(availability.resolved_names)} tactile sensors live · "
                f"total contact {state.total_contact:.3f}"
            )
        else:
            missing = ", ".join(availability.missing_names[:4])
            suffix = "..." if len(availability.missing_names) > 4 else ""
            sensor_status = (
                f"tactile sensors incomplete · missing {len(availability.missing_names)} "
                f"({missing}{suffix})"
            )
        return sensor_plot, sensor_heatmap, sensor_table, sensor_status


PIPELINE = ShadowHandRenderer()


with gr.Blocks(title="Shadow Hand Teleoperation") as demo:
    gr.Markdown(
        """# 🦾 Shadow Hand Teleoperation

Move one hand in front of your camera. MediaPipe extracts 21 landmarks, which
are geometrically retargeted to a 20-actuator MuJoCo Shadow Hand.

Choose a path:
- `/browser`: fastest demo, MuJoCo-WASM + webcam stay in your browser
- `/server`: fallback demo, webcam frames are uploaded and rendered on this Space

The native local app remains the source of truth for future tactile/contact
sensor experiments.

Roadmap:
- tactile/contact sensors
- optimization loop: user pose -> robot pose -> robot hand pose estimate -> minimize pose error
"""
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            camera = gr.Image(
                label="Live webcam",
                sources=["webcam"],
                streaming=True,
                type="numpy",
                image_mode="RGB",
                height=180,
            )
            landmarks = gr.Image(
                label="Detected keypoints",
                type="numpy",
                height=140,
            )
        with gr.Column(scale=3, min_width=640):
            rendered = gr.Image(label="Shadow Hand simulation", type="numpy", height=560)
            with gr.Accordion("Scene controls", open=True):
                with gr.Row():
                    azimuth = gr.Slider(-180, 180, value=SERVER_SCENE_PARAMS["azimuth"], step=1, label="Rotate left / right")
                    elevation = gr.Slider(-85, 85, value=SERVER_SCENE_PARAMS["elevation"], step=1, label="Rotate up / down")
                    distance = gr.Slider(0.25, 1.4, value=SERVER_SCENE_PARAMS["distance"], step=0.01, label="Zoom")
                with gr.Row():
                    target_x = gr.Slider(-0.25, 0.25, value=SERVER_SCENE_PARAMS["target"][0], step=0.01, label="Move target X")
                    target_y = gr.Slider(-0.25, 0.25, value=SERVER_SCENE_PARAMS["target"][1], step=0.01, label="Move target Y")
                    target_z = gr.Slider(-0.05, 0.45, value=SERVER_SCENE_PARAMS["target"][2], step=0.01, label="Move target Z")
    status = gr.Textbox(label="Status", interactive=False)
    sensor_status = gr.Textbox(label="Tactile sensors", interactive=False)
    with gr.Row():
        sensor_table = gr.Textbox(label="Finger signals", interactive=False, lines=8, max_lines=8, scale=1)
        sensor_plot = gr.Image(label="Linear signal plot", type="numpy", height=160, scale=2)
        sensor_heatmap = gr.Image(label="Heatmap", type="numpy", height=160, scale=1)

    render_inputs = [camera, azimuth, elevation, distance, target_x, target_y, target_z]
    render_outputs = [landmarks, rendered, status, sensor_status, sensor_table, sensor_plot, sensor_heatmap]
    camera.change(
        PIPELINE.process,
        inputs=render_inputs,
        outputs=render_outputs,
        trigger_mode="always_last",
        concurrency_id="hand-pipeline",
    )
    camera.stream(
        PIPELINE.process,
        inputs=render_inputs,
        outputs=render_outputs,
        stream_every=0.1,
        trigger_mode="always_last",
        concurrency_id="hand-pipeline",
    )
    controls = render_inputs[1:]
    for control in controls:
        control.change(
            PIPELINE.update_camera,
            inputs=controls,
            outputs=[rendered, status, sensor_status, sensor_table],
            trigger_mode="always_last",
            concurrency_id="hand-pipeline",
        )

    gr.Markdown(
        """### Notes

This is a visual, server-rendered demo—not a low-latency robot controller: your
camera frames are uploaded and the simulation is rendered here, one visitor at a
time. **[Try the browser version →](/browser)** — it runs MuJoCo-WASM entirely on
your machine at a higher rate, and no video leaves your browser.

For native webcam teleoperation, run `uv run mjpython -m shadow_hand.main`
locally. No camera frames are recorded by this Space.
"""
    )


# Both execution paths are served from this one Space:
#   /browser  -> static MuJoCo-WASM build (dist/), physics in the client
#   /server   -> this Gradio app, physics rendered here and streamed back
# The WASM build is only present when dist/ has been built (the Dockerfile does
# this in a node stage); locally the app still runs, just without /browser.
def build_app():
    fastapi_app = FastAPI()
    dist = PROJECT_ROOT / "dist"

    if (dist / "index.html").exists():
        fastapi_app.mount("/browser", StaticFiles(directory=dist, html=True), name="browser")
        log.info("browser WASM build mounted at /browser")
    else:
        log.warning("dist/ not built; /browser is unavailable (run `npm run build`)")

    @fastapi_app.get("/")
    def _root():
        # Prefer the client-side path when it is available: it is faster and
        # does not consume Space CPU per visitor.
        return RedirectResponse("/browser/" if (dist / "index.html").exists() else "/server")

    # Keep only one pending event: interactive video must prefer freshness.
    demo.queue(default_concurrency_limit=1, max_size=1)
    return gr.mount_gradio_app(fastapi_app, demo, path="/server")


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=7860)
