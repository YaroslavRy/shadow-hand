# Recent changes

A summary of what changed in the Shadow Hand teleop pipeline. Use as a
starting point for README updates / release notes.

## Project structure

- Split monolithic root layout into three packages:
  - **`shadow_hand/`** — active Shadow Hand teleop (MuJoCo)
  - **`pybullet_hand/`** — legacy PyBullet teleop, archived
  - **`experiments/`** — unrelated research scripts and notebooks
  - **`tests/`** — `test_*.py` files
- Removed: root-level `to_del.py`.
- Updated `pybullet_hand/main.py` to use script-relative URDF paths.

## Configuration / data discipline

- Magic numbers extracted from `shadow_hand_model.py` into a single
  `ACTUATOR_MAP` table in `shadow_hand/settings.py`. Each row:
  `{signal, gain, offset, clip}`.
- `compute_signals()` produces named scalar signals from MediaPipe
  keypoints; `extract_shadow_hand_targets()` is now a small loop over the
  table (no more per-actuator hand-tuned expressions).
- Signals grouped (`curl`, `spread`, `thumb`) — each group is scaled by a
  live calibration multiplier.
- All file paths centralized in `settings.py` (`PROJECT_ROOT`, `ASSETS_DIR`,
  `DATA_DIR`, `LOGS_DIR`, `SCENE_PATH`, `HAND_LOG_CSV`, `RUN_LOG`).

## Concurrency architecture

- **`FrameTracker`** background thread in `shadow_hand/tracking.py`:
  camera read + MediaPipe inference + skeleton draw, publishing the
  latest result via `Queue(maxsize=1)`.
- Control loop reads via `tracker.latest()` non-blocking. Old frames are
  dropped at the producer, so the consumer always sees the freshest
  result. MediaPipe latency no longer caps the simulation rate.
- Result: loop time dropped from ~96 ms/loop (10 fps) to target 30 fps.

## Live calibration UI

- Added four sliders driving multiplicative scales:
  `curl`, `spread`, `thumb`, `smoothing`.
- Implemented as MuJoCo `<position>` actuators on slide-joint widgets in
  `assets/mujoco_menagerie/shadow_hand/my_scene.xml`. Sliders appear in
  the viewer's right-side **Control panel** as native draggable controls.
- HUD overlay (top-left) shows current calibration values. Live signal
  HUD (bottom-left) shows raw spread/curl/thumb values for debugging.
- HUD updates throttled to ~6 Hz so they don't compete with the render lock.

## Logging

- New `shadow_hand/logging_setup.py`:
  - Console handler at `--log-level` (default INFO)
  - Rotating file handler at DEBUG (`logs/run.log`, 10 MB × 5 backups)
  - Per-module loggers (`shadow_hand.main`, `shadow_hand.tracking`,
    `shadow_hand.cv2`)
- Replaced `print()` and silent `except: pass` blocks.
- Periodic performance reports every 5 s with per-stage timing breakdown.
- CSV file flushed every 1 s to avoid losing recent rows on a hard kill.

## Shutdown and error handling

- Multiple shutdown paths added (`mjpython` is fussy about signals on
  macOS, so we needed several):
  1. **ESC** or **Q** in the MuJoCo viewer (`key_callback`)
  2. Closing the MuJoCo window
  3. Ctrl+C in terminal — kills cv2 subprocess; control loop detects this
     and shuts down
  4. SIGINT/SIGTERM handlers (best-effort under mjpython)
- cv2 subprocess teardown is now escalating: sentinel → `terminate()` →
  `kill()`.
- Frame tracker thread joined *before* camera release to avoid reading
  a closed `VideoCapture`.

## Entry point

- Run with `mjpython -m shadow_hand.main` (was `python main_mujoco.py`).
- CLI flags: `--log-level`, `--no-record`, `--no-cv2`.

## Misc

- Removed legacy import of `pybullet` from the MediaPipe utilities; the
  Shadow Hand path no longer requires PyBullet installed.
- Pre-resolved MuJoCo actuator IDs once at startup; the per-frame
  `mj_name2id` calls are gone.

## Browser WASM path — render, tracking and retargeting parity

Detail and rationale in [`WASM.md`](WASM.md).

### Rendering (`web/app.js`)

- Geom world transforms were built with the translation in the bottom row of
  a row-major `Matrix4.set()`. It belongs in the fourth column; the old form
  dropped the translation and left a projective matrix, collapsing every geom
  toward the origin.
- Placement used `convert · M · convert⁻¹`. Mesh vertices stay in MuJoCo's
  Z-up local frame, so only `convert · M` is correct — the trailing inverse
  rotated each part about its own origin and rendered the hand exploded.
- The render loop only started from the "Enable camera" handler, so the WebGL
  view stayed blank until the camera was granted. It now starts at load.
- `forearm_*` meshes were skipped, removing the mount the hand sits on; the
  Three.js floor was at `y=0` while `my_scene.xml` puts the ground plane at
  `z=-0.2`. Both corrected (22 → 24 visual geoms).
- Camera framing derived its span from palm + fingertips and cropped the base;
  it now frames the model bounding box.

### Tracking (`web/app.js`)

- Inference could run before `getUserMedia()` resolved, handing MediaPipe a
  0×0 frame. That fails its `ImageToTensor` ROI check and permanently poisons
  the graph, so tracking never recovered. Inference is now gated on
  `readyState >= 2` and non-zero video dimensions.
- "No hand detected" and "tracker threw" were indistinguishable; they are now
  separate status messages, and stale overlay keypoints get cleared.
- Detection/tracking confidences aligned with `tracking.py` (0.7 / 0.5, was
  0.2), and inference raised to 30 Hz to match the native control loop.

### Retargeting (`web/retarget.js`, new)

- The browser previously used an ad-hoc mapping: 13 actuators, invented gains,
  no wrist, no spread, no `LFJ5`, thumb from raw pixel deltas, no smoothing.
- Replaced with a direct port of `settings.py` `ACTUATOR_MAP` +
  `model.compute_signals()` + `main.smooth_targets()`. All 20 actuators.
- `tracking.py` mirrors the frame before inference, so `mirrorLandmarks()`
  applies `x -> 1 - x` to match the sign conventions the map assumes.
- Physics now steps off the wall clock at the model's 0.002 s timestep
  (previously 4 fixed steps per rendered frame) and steps every frame whether
  or not a hand was seen, matching the native loop.
- Verified numerically against the Python mapping: 300 samples × 20 actuators,
  max absolute delta `3.9e-15`.

### Deployment — both paths in one Space

- The Space previously served only the Gradio server-rendered path. It now
  serves both, with a link to switch:
  - `/browser` — static MuJoCo-WASM build, physics in the client
  - `/server` — the existing Gradio app, unchanged in behaviour
  - `/` — redirects to `/browser`
- `Dockerfile` is now two-stage: `node:20-slim` runs `npm ci && npm run build`,
  and the Python stage copies `dist/` in.
- `app.py` wraps the Gradio `Blocks` in FastAPI (`gr.mount_gradio_app`) and
  mounts `dist/` with `StaticFiles`. If `dist/` is missing, `/browser` is not
  mounted and `/` falls through to `/server`, so the server path never depends
  on the browser build.
- `.dockerignore` (an allow-list) additionally permits `package.json`,
  `package-lock.json`, `index.html`, `web/` and `public/`; `node_modules/` and
  `dist/` stay excluded and are produced inside the image.
