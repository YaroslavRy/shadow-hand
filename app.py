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

    def process(self, frame: np.ndarray):
        """Return annotated camera frame, rendered hand, and a compact status."""
        if frame is None:
            return None, None, "Waiting for a webcam frame or image upload."

        with self.lock:
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
                # Let position actuators settle before rendering this pose.
                for _ in range(20):
                    mujoco.mj_step(self.model, self.data)
                status = "Hand detected · 20 Shadow Hand actuators retargeted."

            self.renderer.update_scene(self.data)
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
        source = gr.Image(
            label="Camera / image input",
            sources=["webcam", "upload"],
            streaming=True,
            type="numpy",
            image_mode="RGB",
        )
        rendered = gr.Image(label="Headless MuJoCo render", type="numpy")
    landmarks = gr.Image(label="MediaPipe landmarks", type="numpy")
    status = gr.Textbox(label="Status", interactive=False)

    source.change(PIPELINE.process, inputs=source, outputs=[landmarks, rendered, status])
    source.stream(PIPELINE.process, inputs=source, outputs=[landmarks, rendered, status])

    gr.Markdown(
        """### Notes

This is a visual, server-rendered demo—not a low-latency robot controller.
For native webcam teleoperation, run `uv run mjpython -m shadow_hand.main`
locally. No camera frames are recorded by this Space.
"""
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
