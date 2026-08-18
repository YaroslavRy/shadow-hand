# Shadow Hand teleop — architecture

## Concurrency schematic

```
┌────────────────────────────── main process (mjpython) ───────────────────────────────┐
│                                                                                       │
│   ┌──── thread: frame-tracker ────┐         ┌──── thread: control loop ────┐          │
│   │                               │         │                              │          │
│   │  cap.read()       ~33 ms      │         │  read_sliders()              │          │
│   │  cv2.flip                     │  Queue  │  HUD set_texts (throttled)   │          │
│   │  cv2.cvtColor                 │ maxsize │  tracker.latest()            │          │
│   │  hands.process()  ~30-50 ms   │   = 1   │  extract_targets()           │          │
│   │  draw skeleton                │ ──────► │  synergy.project() (opt)     │          │
│   │  put_latest(frame, kp, hand)  │         │  smooth_targets()            │          │
│   │                               │         │  data.ctrl[i] = val          │          │
│   │  runs as fast as MediaPipe    │         │  csv_writer.writerow         │          │
│   │  allows (~15-20 fps)          │         │  cv2_queue.put_nowait(frame)─┼──┐       │
│   │                               │         │  mj_step × 16                │  │       │
│   │                               │         │  viewer.sync()               │  │       │
│   │                               │         │  loop targets 30 fps         │  │       │
│   └───────────────────────────────┘         └──────────────────────────────┘  │       │
│                                                                               │       │
└───────────────────────────────────────────────────────────────────────────────┼───────┘
                                                                                │
                                                                                │ mp.Queue
                                                                                │ maxsize=3
                                                                                ▼
                                  ┌──────── subprocess: cv2 preview ────────┐
                                  │  q.get(timeout=0.05)                    │
                                  │  cv2.imshow("MediaPipe Neural Input")   │
                                  │  cv2.waitKey(1) ── pumps macOS UI       │
                                  └─────────────────────────────────────────┘
```

## Per-frame data pipeline

```
   ┌─ webcam frame ─┐
   │                ▼
   │     ┌── tracking.py ──┐
   │     │  cv2.flip       │
   │     │  MediaPipe      │
   │     │  draw skeleton  │   ← runs in frame-tracker thread
   │     └────────┬────────┘
   │              │ (frame, 21x3 keypoints, handedness)  via Queue(1)
   │              ▼
   │     ┌── model.py ─────────────────────────┐
   │     │  compute_signals(keypoints)          │   raw scalars:
   │     │    spread_index / ring / pinky       │     palm_x/z, curl_*,
   │     │     ← SIGNED 2D angle (image plane)  │     spread_*, thumb_*
   │     │    curl_*, thumb_* (3D OK here)      │
   │     └──────────────────┬──────────────────┘
   │                        │
   │                        ▼
   │     ┌── settings.py: ACTUATOR_MAP ────────┐   per actuator:
   │     │  target = clip(signal*gain*scale     │      signal name,
   │     │                + offset, joint_lim)  │      gain, offset, clip
   │     └──────────────────┬──────────────────┘
   │                        │ targets: dict[actuator → ctrl]
   │                        ▼
   │     ┌── mano_pipeline.py (optional) ──────┐
   │     │  SynergyProjector.project(targets,   │   project onto 5-DOF
   │     │     blend = ctl_synergy_blend)       │   Santello synergy basis
   │     └──────────────────┬──────────────────┘
   │                        │
   │                        ▼
   │     ┌── main.py: smooth_targets() ────────┐   EMA blend with
   │     │  alpha = ctl_smoothing               │   previous frame
   │     └──────────────────┬──────────────────┘
   │                        │
   │                        ▼
   │            data.ctrl[i] = val           ─────────►  mj_step × 16  →  viewer.sync()
   │                        │                                  │
   │                        ▼                                  │
   │       cv2_queue.put_nowait(frame)  →  preview subprocess  │
   │                                                           ▼
   │                                              [ MuJoCo viewer window ]
   ▼
   slider feedback (ctl_curl_scale, ctl_spread_scale, ctl_thumb_scale,
   ctl_smoothing, ctl_synergy_blend) flows back from Control panel via
   data.ctrl[i] of the slider actuators → read_sliders() at top of loop.
```

## What the system does

Tracks a hand in front of the webcam, converts the 21 MediaPipe landmarks
into Shadow Hand actuator targets, and drives a MuJoCo simulation in real
time. Five live sliders in the MuJoCo viewer let you tune the mapping
gains, the EMA smoothing factor, and an optional Santello-style synergy
prior while watching the result. CSV logging captures keypoints and
actuator targets for offline analysis.

## High-level overview

Three workers, two queues, one shared `mujoco.MjData`:

- **Frame-tracker thread** — owns the camera and MediaPipe. Runs as fast as
  inference allows and publishes the freshest result (frame, keypoints,
  handedness) via a single-slot `Queue`. Stale frames are dropped at the
  producer.
- **Control loop** (main thread under `mjpython`) — reads the freshest
  perception result non-blocking, runs the mapping pipeline, applies
  actuator targets, advances physics 16 sub-steps, syncs the viewer, and
  ships the annotated frame to the preview subprocess. Targets 30 fps.
- **cv2 preview subprocess** — owns the OpenCV preview window. macOS
  requires GUI windows on a dedicated process; the control loop just
  forwards frames via `multiprocessing.Queue(maxsize=3)`.

The mapping itself is two clean layers:

- **Signal layer** ([`shadow_hand/model.py:compute_signals`](../shadow_hand/model.py))
  — turns keypoints into a dict of named scalars (palm direction
  components, curl per finger, signed spread per finger, thumb
  opposition/abduction/bend).
- **Mapping layer** ([`shadow_hand/settings.py:ACTUATOR_MAP`](../shadow_hand/settings.py))
  — one table row per actuator: `{signal, gain, offset, clip}`. Formula:
  `target = clip(signal * gain * scale + offset, *joint_limits)`. Group
  scales (`curl`, `spread`, `thumb`) come from live sliders.

Two optional layers can plug in *between* the mapping and `data.ctrl`:

- **Synergy projection** ([`shadow_hand/mano_pipeline.py`](../shadow_hand/mano_pipeline.py))
  — projects the 20-DOF target onto a 5-DOF subspace derived from
  Santello (1998) hand-posture PCA, then reconstructs. Blended live by
  the `ctl_synergy_blend` slider.
- **MANO front-end** (scaffolded, not wired) — fit MANO to MediaPipe
  for biomechanically clean angles. Requires `manotorch` and the MPI
  model file. See [`docs/RETARGETING.md`](RETARGETING.md).

## Mid-detailed overview

### Concurrency model

A single-threaded loop would run at `min(camera_fps, mediapipe_fps,
mj_step_fps, render_fps)`. With MediaPipe at 30-50 ms on CPU, that caps
the simulation at ~15 fps. Splitting perception into its own thread lets
the control loop run at 30 fps regardless of inference latency. Both
OpenCV and MediaPipe release the GIL during C++ work, so a Python thread
overlaps cleanly with `mj_step` — no need for the cost of a subprocess.

The two queues:

| Queue | Type | Maxsize | Semantics |
|---|---|---|---|
| `FrameTracker._queue` | `queue.Queue` (in-process) | 1 | **Latest-value.** Producer drops the previous slot before publishing. Consumer reads non-blocking; reuses previous result if empty. |
| `cv2_queue` | `multiprocessing.Queue` (cross-process) | 3 | **Bounded buffer.** Producer `put_nowait`; drops frame on `Full`. Preview is best-effort. |

### Signal layer (compute_signals)

Converts 21 MediaPipe landmarks into named scalars. Two important
geometry choices:

- **Spread signals use a SIGNED 2D angle in the image plane.** The 3D
  angle was inverting under hand depth jitter (z noise from MediaPipe).
  The signed form lets the robot also spread *inward* (crossed
  fingers, horns sign).
- **Curl signals use the unsigned 3D distance ratio** (`1 - L_straight /
  L_bent`). Z noise doesn't matter much here because the ratio is
  scale-invariant.
- **Thumb signals use 3D angles** between consecutive segments — z is
  load-bearing for opposition/abduction.

### Mapping layer (ACTUATOR_MAP)

20 rows, one per Shadow Hand actuator. Each row says: which named
signal feeds this actuator, what gain/offset to apply, and what joint
limits to clip to. The clip ranges match the actuator `ctrlrange` from
`right_hand.xml` exactly — overshoot would have the controller fight
itself.

Group scales `curl`, `spread`, `thumb` multiply the gain at runtime so
the live sliders can re-tune sensitivity without restarting. The
`smoothing` slider drives the EMA factor.

Sign conventions worth knowing:
- `FFJ4`/`RFJ4`/`LFJ4` (spread joints) use **negative ctrl = outward**
  on this Shadow Hand. The XML axes on the four knuckle joints are
  mirrored (`0 -1 0` for index/middle, `0 1 0` for ring/pinky) so the
  same numerical sign gives symmetric physical motion across fingers.

### Control panel sliders

Five `<position>` actuators in `my_scene.xml` drive slide-joint widgets
in the scene. They appear in the MuJoCo viewer's right-side Control
panel as native draggable sliders:

| Actuator | Parameter range | Default | What it scales |
|---|---|---|---|
| `ctl_curl_scale` | 0.0 — 2.0 | 1.0 | All `curl_*` gains |
| `ctl_spread_scale` | 0.0 — 2.0 | 1.0 | All `spread_*` gains |
| `ctl_thumb_scale` | 0.0 — 2.0 | 1.0 | All `thumb_*` gains |
| `ctl_smoothing` | 0.0 — 0.95 | 0.15 | EMA smoothing factor |
| `ctl_synergy_blend` | 0.0 — 1.0 | 0.0 | Synergy projection mix |

The slide-joint widgets themselves are visual indicators of the current
value; the actual UI is the right-side panel.

### Synergy projection (opt-in)

`SynergyProjector` in `mano_pipeline.py` implements a Santello-style
regularizing prior:

```
target_20d        = extract_shadow_hand_targets(...)
weights_5d        = U  @ target_20d
target_constrained = U.T @ weights_5d
out = (1 - blend) * target_20d + blend * target_constrained
```

where `U` is a 5×20 basis whose rows approximate the canonical
post-Santello synergies (overall aperture, thumb opposition, precision
pinch, radial-ulnar gradient, distal-vs-proximal flexion). Blend=0
bypasses entirely; blend=1 is full projection. The basis is
hand-crafted; a real pipeline would PCA it from motion-capture data.

### Lifecycle

```
init_camera()          ── cv2.VideoCapture + MediaPipe Hands
FrameTracker.start()   ── launch perception thread
launch_passive(...)    ── MuJoCo viewer up; key_callback wired for ESC/Q
control loop runs      ── until ESC/Q, viewer closed, cv2 subprocess
                          dies, or shutdown event set
finally:
  csv_file.flush() + close()
  cv2_queue.put_nowait(None)              ── graceful sentinel
  cv2_proc.join → terminate → kill        ── escalating timeouts
  tracker.stop()                          ── join perception thread
  tracking.release()                      ── close camera
```

The `finally` order matters: stop the tracker thread *before* releasing
the camera, otherwise it could call `cap.read()` on a closed handle.

### Shutdown signals under mjpython

`mjpython` runs Python on a worker thread while the Cocoa GUI holds the
OS main thread. Unix signals (SIGINT from Ctrl+C) don't reliably deliver
to Python's signal handler in this layout. The reliable shutdown paths,
in order of preference:

1. **ESC or Q** in the MuJoCo viewer (`key_callback` runs on the GUI thread).
2. **Close the MuJoCo window** (`viewer.is_running()` flips to False).
3. **Ctrl+C in terminal** — kills the cv2 subprocess; the control loop
   notices `cv2_proc.is_alive() == False` and triggers shutdown.
4. **`pkill -9 -f shadow_hand.main`** — last resort.
