"""Shadow Hand teleoperation entry point.

Run with:
    mjpython -m shadow_hand.main

(`mjpython` is required on macOS for MuJoCo's interactive viewer.)
"""

import argparse
import collections
import csv
import logging
import multiprocessing as mproc
import queue as python_queue
import signal
import threading
import time

import cv2
import mujoco
import numpy as np
from mujoco import viewer

from . import tracking
from .logging_setup import configure_logging
from .mano_pipeline import SynergyProjector
from .model import compute_signals, extract_shadow_hand_targets
from .sensors import (
    DEFAULT_DIAGNOSTICS_LAYOUT,
    SignalHistory,
    build_dashboard_state,
    build_snapshot,
    read_named_sensordata,
    render_native_diagnostics,
)
from .settings import (
    ACTUATOR_MAP,
    DATA_DIR,
    FPS,
    HAND_LOG_CSV,
    SCENE_PATH,
    SMOOTHING_FACTOR,
)

log = logging.getLogger("shadow_hand.main")


def smooth_targets(new_targets, prev_targets, factor):
    smoothed = {}
    for k, v in new_targets.items():
        prev_value = prev_targets.get(k, 0.0)
        smoothed[k] = factor * prev_value + (1 - factor) * v
    return smoothed


def cv2_viewer_process(q):
    """Subprocess that owns the OpenCV preview window. macOS-safe."""
    # The subprocess is a fresh Python process - configure its own logging.
    sub_log = logging.getLogger("shadow_hand.cv2")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    window = "MediaPipe Neural Input"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    # Show a placeholder so the window is visible immediately on macOS,
    # even before the first real frame arrives.
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        "Waiting for frames...",
        (140, 240),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (70, 255, 0),
        2,
    )
    cv2.imshow(window, placeholder)
    cv2.waitKey(1)
    sub_log.info("cv2 preview window opened")

    try:
        while True:
            try:
                frame = q.get(timeout=0.05)
                if frame is None:
                    sub_log.info("shutdown sentinel received")
                    break
                cv2.imshow(window, frame)
            except python_queue.Empty:
                cv2.waitKey(1)
            except Exception:
                sub_log.exception("error in cv2 viewer loop")
            cv2.waitKey(1)
    except KeyboardInterrupt:
        sub_log.info("KeyboardInterrupt -> exiting cv2 subprocess")
    finally:
        cv2.destroyAllWindows()


def sensor_panel_process(q):
    """Subprocess that owns the native diagnostics window."""
    sub_log = logging.getLogger("shadow_hand.sensor_panel")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    window = "Shadow Hand Diagnostics"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    placeholder = np.zeros(
        (DEFAULT_DIAGNOSTICS_LAYOUT.height, DEFAULT_DIAGNOSTICS_LAYOUT.width, 3),
        dtype=np.uint8,
    )
    cv2.putText(
        placeholder,
        "Waiting for sensor/joint diagnostics...",
        (250, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (120, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(window, placeholder)
    cv2.waitKey(1)
    sub_log.info("sensor diagnostics window opened")

    try:
        while True:
            try:
                frame = q.get(timeout=0.05)
                if frame is None:
                    sub_log.info("shutdown sentinel received")
                    break
                cv2.imshow(window, frame)
            except python_queue.Empty:
                cv2.waitKey(1)
            except Exception:
                sub_log.exception("error in sensor diagnostics loop")
            cv2.waitKey(1)
    except KeyboardInterrupt:
        sub_log.info("KeyboardInterrupt -> exiting sensor diagnostics subprocess")
    finally:
        cv2.destroyAllWindows()


def _build_slider_specs(model, data, smoothing_default):
    """Wire calibration slider actuators -> live multiplicative scales.

    Reads `data.ctrl[i]` on the position actuators defined in my_scene.xml
    (`ctl_curl_scale`, `ctl_spread_scale`, etc.) and remaps each to a
    parameter range. Drag them in the viewer's right-side Control panel.
    Returns a `read()` closure that produces a dict of current values.
    """
    SLIDER_CTRL_RANGE = 0.20  # ctrlrange of the actuators in my_scene.xml
    specs = {
        # key: (actuator_name, max_param, default_param)
        "curl": ("ctl_curl_scale", 2.0, 1.0),
        "spread": ("ctl_spread_scale", 2.0, 1.0),
        "thumb": ("ctl_thumb_scale", 2.0, 1.0),
        "smoothing": ("ctl_smoothing", 0.95, smoothing_default),
        "synergy_blend": ("ctl_synergy_blend", 1.0, 0.0),
    }
    bindings = {}
    for key, (act_name, max_p, default_p) in specs.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if aid == -1:
            log.warning("slider actuator %r not found in model; skipping", act_name)
            continue
        bindings[key] = (aid, max_p)
        data.ctrl[aid] = (default_p / max_p) * SLIDER_CTRL_RANGE
    mujoco.mj_forward(model, data)
    log.info("calibration sliders wired (Control panel): %s", sorted(bindings.keys()))

    def read():
        out = {}
        for key, (aid, max_p) in bindings.items():
            ctrl = float(data.ctrl[aid])
            out[key] = max(0.0, min(max_p, ctrl / SLIDER_CTRL_RANGE * max_p))
        return out

    return read


def parse_args():
    p = argparse.ArgumentParser(description="Shadow Hand teleop via MediaPipe.")
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    p.add_argument(
        "--no-record", action="store_true", help="Do not write data/hand_log.csv"
    )
    p.add_argument(
        "--no-cv2", action="store_true", help="Do not open the OpenCV preview window"
    )
    p.add_argument(
        "--no-sensor-panel",
        action="store_true",
        help="Do not open the separate native sensor/joint diagnostics window",
    )
    return p.parse_args()


def main():
    args = parse_args()
    configure_logging(level=args.log_level)
    log.info("starting Shadow Hand teleop")
    log.info("scene path: %s", SCENE_PATH)

    shutdown = threading.Event()

    def _request_shutdown(signum, _frame):
        log.info("received signal %d; requesting shutdown", signum)
        shutdown.set()

    # Signal handlers are best-effort here: under `mjpython` the GUI runs on
    # the OS main thread and Python signals don't reach the worker thread
    # cleanly. Real shutdown paths under mjpython are:
    #   - press ESC or Q in the MuJoCo window (key_callback below)
    #   - close the MuJoCo window
    #   - Ctrl+C in terminal kills the cv2 subprocess; we detect that and
    #     treat it as a shutdown signal too.
    signal.signal(signal.SIGTERM, _request_shutdown)
    try:
        signal.signal(signal.SIGINT, _request_shutdown)
    except ValueError:
        pass  # not in main thread under some mjpython builds

    # CV2 preview subprocess (optional)
    cv2_proc = None
    cv2_queue = None
    if not args.no_cv2:
        mproc.set_start_method("spawn", force=True)
        cv2_queue = mproc.Queue(maxsize=3)
        cv2_proc = mproc.Process(target=cv2_viewer_process, args=(cv2_queue,))
        cv2_proc.start()
        log.info("cv2 preview subprocess started (pid=%d)", cv2_proc.pid)

    sensor_proc = None
    sensor_queue = None
    if not args.no_sensor_panel:
        mproc.set_start_method("spawn", force=True)
        sensor_queue = mproc.Queue(maxsize=1)
        sensor_proc = mproc.Process(target=sensor_panel_process, args=(sensor_queue,))
        sensor_proc.start()
        log.info("sensor diagnostics subprocess started (pid=%d)", sensor_proc.pid)

    tracking.init_camera()
    tracker = tracking.FrameTracker()
    tracker.start()

    if not SCENE_PATH.exists():
        log.error("scene file not found: %s", SCENE_PATH)
        raise SystemExit(1)

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    model.opt.gravity[2] = -9.81  # Needed so the threaded target can hang and swing.

    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    log.info("loaded model: %d actuators", model.nu)
    log.debug("actuators: %s", actuator_names)

    # Pre-resolve actuator ids once instead of mj_name2id every frame.
    act_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ACTUATOR_MAP
    }
    missing = [n for n, i in act_ids.items() if i == -1]
    if missing:
        log.warning("actuators not present in model: %s", missing)

    read_sliders = _build_slider_specs(model, data, SMOOTHING_FACTOR)

    # Synergy projector (Santello 5-DOF prior). Off by default; the
    # `ctl_synergy_blend` slider mixes raw mapping (0) with projected (1).
    synergy = SynergyProjector(k=5, blend=0.0)

    # CSV recording
    csv_file = None
    csv_writer = None
    if not args.no_record:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        csv_file = open(HAND_LOG_CSV, "w", newline="")
        csv_writer = csv.writer(csv_file)
        headers = ["timestamp_ms"]
        for i in range(21):
            headers.extend([f"mp_{i}_x", f"mp_{i}_y", f"mp_{i}_z"])
        headers.extend(actuator_names)
        csv_writer.writerow(headers)
        log.info("recording -> %s", HAND_LOG_CSV)

    previous_targets = {}
    last_csv_flush = time.time()
    last_perf_report = time.time()
    frames_since_report = 0
    loop_ms_accum = 0.0

    # Per-stage timing accumulators - cleared every perf report.
    stage_ns = collections.defaultdict(float)
    fresh_frames_since_report = 0
    DIAGNOSTICS_EVERY_N = 6  # ~5 Hz at 30 FPS; enough for tactile readout, lighter on laptops.
    frame_idx = 0

    # Cached latest tracker output. The control loop runs faster than
    # MediaPipe, so most iterations reuse the previous result.
    last_frame = None
    last_keypoints = None
    last_handedness = None
    last_tracking_status = "tracker starting"
    last_retarget_status = "waiting for hand"
    last_target_peak = 0.0
    last_target_mean = 0.0
    sensor_history = SignalHistory(maxlen=120)
    last_smoothed_targets = {name: 0.0 for name in ACTUATOR_MAP}

    # ESC or Q in the MuJoCo viewer is the most reliable shutdown path
    # under mjpython, since this callback fires on the GUI thread.
    def viewer_key_callback(keycode):
        # 256 = GLFW ESCAPE, 81 = 'Q'
        if keycode in (256, 81):
            log.info("viewer key %d pressed; requesting shutdown", keycode)
            shutdown.set()

    with viewer.launch_passive(
        model, data, key_callback=viewer_key_callback
    ) as viewer_obj:
        viewer_obj.cam.azimuth = 140
        viewer_obj.cam.elevation = -10
        viewer_obj.cam.distance = 1
        viewer_obj.cam.lookat[:] = [0, 0, 0.2]

        perf = time.perf_counter

        try:
            while viewer_obj.is_running() and not shutdown.is_set():
                loop_start = time.time()

                # Health-check the cv2 subprocess. If it died (e.g. user
                # Ctrl+C in terminal killed it), treat that as shutdown.
                if cv2_proc is not None and not cv2_proc.is_alive():
                    log.warning(
                        "cv2 subprocess died (exit=%s); shutting down",
                        cv2_proc.exitcode,
                    )
                    cv2_proc = None
                    shutdown.set()
                if sensor_proc is not None and not sensor_proc.is_alive():
                    log.warning(
                        "sensor diagnostics subprocess died (exit=%s); shutting down",
                        sensor_proc.exitcode,
                    )
                    sensor_proc = None
                    shutdown.set()

                # ---- Sliders ----
                t0 = perf()
                slider_vals = read_sliders()
                scales = {
                    "curl": slider_vals.get("curl", 1.0),
                    "spread": slider_vals.get("spread", 1.0),
                    "thumb": slider_vals.get("thumb", 1.0),
                }
                live_smoothing = slider_vals.get("smoothing", SMOOTHING_FACTOR)
                live_synergy_blend = slider_vals.get("synergy_blend", 0.0)
                stage_ns["sliders"] += perf() - t0

                # ---- Native diagnostics window (throttled) ----
                if sensor_queue is not None and frame_idx % DIAGNOSTICS_EVERY_N == 0:
                    t0 = perf()
                    sensor_values, availability = read_named_sensordata(model, data)
                    sensor_snapshot = build_snapshot(
                        sensor_values,
                        availability=availability,
                    )
                    sensor_state = build_dashboard_state(
                        sensor_snapshot.by_name,
                        max_region_value=0.35,
                    )
                    active_threshold = 1e-4
                    active = [
                        (name, value)
                        for name, value in sensor_snapshot.by_name.items()
                        if value > active_threshold
                    ]
                    if active:
                        peak_name, peak_value = max(active, key=lambda item: item[1])
                    else:
                        peak_name, peak_value = "none", 0.0
                    sensor_history.append(sensor_state.total_contact)
                    if sensor_queue is not None:
                        actuator_rows = sorted(last_smoothed_targets.items())
                        panel = render_native_diagnostics(
                            sensor_state,
                            sensor_history,
                            actuator_rows=actuator_rows,
                            peak_sensor=(peak_name, peak_value),
                            active_sensors=len(active),
                        )
                        try:
                            sensor_queue.put_nowait(panel)
                        except python_queue.Full:
                            pass
                    stage_ns["hud"] += perf() - t0

                # ---- Get latest tracker result (non-blocking) ----
                t0 = perf()
                fresh = tracker.latest()
                stage_ns["tracker_get"] += perf() - t0
                if fresh is not None:
                    last_frame, last_keypoints, last_handedness = fresh
                    fresh_frames_since_report += 1
                    if last_keypoints is not None:
                        hand_label = last_handedness or "unknown"
                        last_tracking_status = f"hand ready ({hand_label})"
                    else:
                        last_tracking_status = "camera live · no hand detected"
                elif last_frame is None:
                    last_tracking_status = "waiting for first tracker frame"

                frame = last_frame
                keypoints = last_keypoints

                if keypoints is not None:
                    # ---- Mapping ----
                    t0 = perf()
                    raw_targets = extract_shadow_hand_targets(keypoints, scales=scales)
                    # Optional Santello-style synergy regularization.
                    # blend=0 -> bypass (raw mapping), blend=1 -> full project.
                    if live_synergy_blend > 1e-3:
                        projected = synergy.project(
                            raw_targets, blend=live_synergy_blend
                        )
                    else:
                        projected = raw_targets
                    smoothed_targets = smooth_targets(
                        projected, previous_targets, live_smoothing
                    )
                    previous_targets = smoothed_targets.copy()
                    last_smoothed_targets = smoothed_targets.copy()

                    for act_name, val in smoothed_targets.items():
                        aid = act_ids.get(act_name, -1)
                        if aid != -1:
                            data.ctrl[aid] = val
                    target_values = list(smoothed_targets.values())
                    if target_values:
                        last_target_peak = max(abs(v) for v in target_values)
                        last_target_mean = sum(abs(v) for v in target_values) / len(target_values)
                    else:
                        last_target_peak = 0.0
                        last_target_mean = 0.0
                    last_retarget_status = f"targets updated · {len(smoothed_targets)} actuators"
                    stage_ns["mapping"] += perf() - t0

                    # ---- CSV ----
                    if csv_writer is not None:
                        t0 = perf()
                        now_ms = int(time.time() * 1000)
                        row = [now_ms]
                        for kp in keypoints:
                            row.extend([kp[0], kp[1], kp[2]])
                        for name in actuator_names:
                            row.append(f"{smoothed_targets.get(name, 0.0):.4f}")
                        csv_writer.writerow(row)
                        stage_ns["csv"] += perf() - t0
                else:
                    if frame is None:
                        last_tracking_status = "no camera frame yet"
                        last_retarget_status = "no frame -> no retarget"
                    else:
                        last_tracking_status = "camera live · no hand detected"
                        last_retarget_status = "holding last pose"
                    last_target_peak = 0.0
                    last_target_mean = 0.0
                    last_smoothed_targets = {name: 0.0 for name in ACTUATOR_MAP}

                # ---- CV2 frame send (forward whatever frame we have) ----
                if cv2_queue is not None and frame is not None:
                    try:
                        cv2_queue.put_nowait(frame)
                    except python_queue.Full:
                        pass  # Drop frame; preview is best-effort.

                # ---- Physics ----
                t0 = perf()
                steps_per_frame = 16
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                stage_ns["mj_step"] += perf() - t0

                # ---- Viewer sync (render) ----
                t0 = perf()
                viewer_obj.sync()
                stage_ns["sync"] += perf() - t0

                # Periodic CSV flush so a kill doesn't lose the last seconds.
                if csv_file is not None and (time.time() - last_csv_flush) > 1.0:
                    csv_file.flush()
                    last_csv_flush = time.time()

                # Per-loop timing accounting
                loop_time = time.time() - loop_start
                loop_ms_accum += loop_time * 1000.0
                frames_since_report += 1
                frame_idx += 1

                if (time.time() - last_perf_report) > 5.0:
                    n = max(1, frames_since_report)
                    elapsed = time.time() - last_perf_report
                    avg_ms = loop_ms_accum / n
                    breakdown = " ".join(
                        f"{k}={v*1000.0/n:.2f}" for k, v in sorted(stage_ns.items())
                    )
                    log.info(
                        "perf: %.1f ms/loop, %d loops, %d fresh tracker frames over %.1fs | %s (ms/loop)",
                        avg_ms,
                        frames_since_report,
                        fresh_frames_since_report,
                        elapsed,
                        breakdown,
                    )
                    last_perf_report = time.time()
                    frames_since_report = 0
                    fresh_frames_since_report = 0
                    loop_ms_accum = 0.0
                    stage_ns.clear()

                # Throttle to FPS cap
                if loop_time < (1.0 / FPS):
                    time.sleep((1.0 / FPS) - loop_time)
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt -> shutting down")
        except Exception:
            log.exception("fatal error in main loop")
            raise
        finally:
            log.info("cleaning up")
            if csv_file is not None:
                try:
                    csv_file.flush()
                    csv_file.close()
                except Exception:
                    log.exception("error closing csv file")

            if cv2_proc is not None and cv2_proc.is_alive():
                try:
                    cv2_queue.put_nowait(None)  # graceful sentinel
                except Exception:
                    pass
                cv2_proc.join(timeout=1.0)
                if cv2_proc.is_alive():
                    log.warning("cv2 subprocess did not exit; terminating")
                    cv2_proc.terminate()
                    cv2_proc.join(timeout=1.0)
                if cv2_proc.is_alive():
                    log.warning("cv2 subprocess still alive; sending SIGKILL")
                    cv2_proc.kill()
                    cv2_proc.join(timeout=1.0)
            if sensor_proc is not None and sensor_proc.is_alive():
                try:
                    sensor_queue.put_nowait(None)
                except Exception:
                    pass
                sensor_proc.join(timeout=1.0)
                if sensor_proc.is_alive():
                    log.warning("sensor diagnostics subprocess did not exit; terminating")
                    sensor_proc.terminate()
                    sensor_proc.join(timeout=1.0)
                if sensor_proc.is_alive():
                    log.warning("sensor diagnostics subprocess still alive; sending SIGKILL")
                    sensor_proc.kill()
                    sensor_proc.join(timeout=1.0)

            try:
                tracker.stop()
            except Exception:
                log.exception("error stopping frame tracker")

            try:
                tracking.release()
            except Exception:
                log.exception("error releasing camera")

            log.info("shutdown complete")


if __name__ == "__main__":
    main()
