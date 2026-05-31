"""MediaPipe hand-tracking wrapper used by the Shadow Hand control loop.

Owns the camera and MediaPipe `Hands` instance as module-level singletons.
Provides a `FrameTracker` background thread that runs camera + MediaPipe in
parallel with the main loop and publishes the latest result via a single-slot
queue. The main loop reads non-blocking and stays at its target rate even when
MediaPipe inference takes 30-50 ms per frame.
"""
import logging
import queue
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

log = logging.getLogger(__name__)

mp_hands = mp.solutions.hands
hands = None
cap = None


def init_camera(camera_index: int = 0, width: int = 640, height: int = 480) -> None:
    """Open the webcam and create a MediaPipe Hands detector. Idempotent."""
    global hands, cap
    if cap is not None:
        return
    log.info("opening camera index=%d (%dx%d)", camera_index, width, height)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera index {camera_index}")


def release() -> None:
    global cap
    if cap is not None:
        cap.release()
        cap = None
    cv2.destroyAllWindows()


def safe_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return vec * 0.0
    return vec / norm


class FrameTracker:
    """Background thread: camera + MediaPipe -> single-slot queue of latest result.

    Decouples perception from the control loop. The control loop calls
    `latest()` non-blocking; the tracker thread runs as fast as the camera
    and MediaPipe allow, dropping stale frames so the consumer only ever
    sees the freshest result.

    Each result is `(frame_with_skeleton, keypoints, handedness)`:
      - frame:      BGR numpy array, already mirrored, with skeleton drawn
                    when a hand was detected. Send straight to a preview window.
      - keypoints:  (21, 3) numpy array of normalized xyz landmarks, or None.
      - handedness: "Left" / "Right" string, or None.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_processed = 0
        self.frames_dropped = 0

    def start(self) -> None:
        if cap is None or hands is None:
            raise RuntimeError("call init_camera() before FrameTracker.start()")
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="frame-tracker", daemon=True
        )
        self._thread.start()
        log.info("frame tracker thread started")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning("frame tracker did not stop within %.1fs", timeout)
            self._thread = None

    def latest(self):
        """Return the freshest (frame, keypoints, handedness) or None if no
        new result is available since the last call."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            keypoints = None
            handedness = None
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                keypoints = np.array([[p.x, p.y, p.z] for p in lm.landmark])
                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # Single-slot queue: drop the previous frame if the consumer
            # hasn't picked it up yet.
            try:
                self._queue.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((frame, keypoints, handedness))
            except queue.Full:
                # Race: consumer put one back? Fine, just drop ours.
                pass
            self.frames_processed += 1
        log.info("frame tracker stopped (processed=%d, dropped=%d)",
                 self.frames_processed, self.frames_dropped)
