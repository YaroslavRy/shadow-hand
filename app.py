"""Browser demo for Shadow Hand retargeting.

Designed for Hugging Face Spaces: a webcam/uploaded RGB frame is retargeted to
the Shadow Hand and rendered server-side without a native MuJoCo viewer.
"""

import threading

import cv2
import gradio as gr
import mediapipe as mp
import mujoco
import numpy as np

from shadow_hand.model import extract_shadow_hand_targets
from shadow_hand.settings import ACTUATOR_MAP, PROJECT_ROOT, SCENE_PATH


# Keep the full Menagerie checkout out of the deployment. Locally we retain
# SCENE_PATH; the Space instead carries this small, self-contained asset copy.
SPACE_SCENE_PATH = PROJECT_ROOT / "space_assets" / "shadow_hand" / "my_scene.xml"


class ShadowHandRenderer:
    """A serialized, headless MuJoCo + MediaPipe inference pipeline."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
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

    def process(
        self,
        frame: np.ndarray,
        azimuth: float = 140,
        elevation: float = -15,
        distance: float = 0.52,
        target_x: float = 0.0,
        target_y: float = 0.0,
        target_z: float = 0.20,
    ):
        """Return annotated camera frame, rendered hand, and a compact status."""
        if frame is None:
            return None, None, "Waiting for a webcam frame or image upload."

        with self.lock:
            self.camera.azimuth = azimuth
            self.camera.elevation = elevation
            self.camera.distance = distance
            self.camera.lookat[:] = (target_x, target_y, target_z)
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

            if keypoints is None:
                status = "No hand detected — keep one hand visible to the camera."
            else:
                targets = extract_shadow_hand_targets(keypoints)
                for name, value in targets.items():
                    self.data.ctrl[self.actuator_ids[name]] = value
                # A short settle is enough for a responsive visual preview.
                for _ in range(10):
                    mujoco.mj_step(self.model, self.data)
                status = "Hand detected · 20 Shadow Hand actuators retargeted."

            self.renderer.update_scene(self.data, camera=self.camera)
            rendered = self.renderer.render()
            return annotated, rendered, status


PIPELINE = ShadowHandRenderer()


with gr.Blocks(title="Shadow Hand Teleoperation") as demo:
    gr.Markdown(
        """# 🦾 Shadow Hand Teleoperation

Move one hand in front of your camera. MediaPipe extracts 21 landmarks, which
are geometrically retargeted to a 20-actuator MuJoCo Shadow Hand.
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
            landmarks = gr.Image(label="Detected keypoints", type="numpy", height=180)
        with gr.Column(scale=3, min_width=640):
            rendered = gr.Image(label="Shadow Hand simulation", type="numpy", height=560)
            with gr.Accordion("Scene controls", open=True):
                with gr.Row():
                    azimuth = gr.Slider(-180, 180, value=140, step=1, label="Rotate left / right")
                    elevation = gr.Slider(-85, 85, value=-15, step=1, label="Rotate up / down")
                    distance = gr.Slider(0.25, 1.2, value=0.52, step=0.01, label="Zoom")
                with gr.Row():
                    target_x = gr.Slider(-0.25, 0.25, value=0.0, step=0.01, label="Move target X")
                    target_y = gr.Slider(-0.25, 0.25, value=0.0, step=0.01, label="Move target Y")
                    target_z = gr.Slider(-0.05, 0.45, value=0.20, step=0.01, label="Move target Z")
    status = gr.Textbox(label="Status", interactive=False)

    render_inputs = [camera, azimuth, elevation, distance, target_x, target_y, target_z]
    render_outputs = [landmarks, rendered, status]
    camera.change(PIPELINE.process, inputs=render_inputs, outputs=render_outputs)
    camera.stream(PIPELINE.process, inputs=render_inputs, outputs=render_outputs)
    for control in render_inputs[1:]:
        control.change(PIPELINE.process, inputs=render_inputs, outputs=render_outputs)

    gr.Markdown(
        """### Notes

This is a visual, server-rendered demo—not a low-latency robot controller.
For native webcam teleoperation, run `uv run mjpython -m shadow_hand.main`
locally. No camera frames are recorded by this Space.
"""
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
