# ShadowHand — context handoff

## What this is
Educational project: real-time teleoperation of the DeepMind Menagerie
Shadow Hand in MuJoCo, driven today by webcam→MediaPipe, designed to
accept EMG/EEG/video signals tomorrow. The architecture funnels every
input through one function (`compute_signals`) so the input side is a
single swap.

## Repo
- Root: `/Users/corvie/Documents/Projects/ShadowHand`
- Git: `main` branch, clean working tree against last commit `98e1a24`
- Python: 3.10 (pinned via `.python-version`), uv-managed `.venv`
- Platform: macOS x86_64 (darwin 22.6.0); `mjpython` required for MuJoCo viewer

## Active code (path: `shadow_hand/`)
- `main.py`        control loop @30 Hz, MuJoCo viewer, CSV logging, perf telemetry,
                   cv2 preview subprocess. Still wires up live-calibration sliders
                   (`_build_slider_specs`, lines 95–123, 210, 280–291) — user said
                   "sliders are removed" but the code is still there.
- `tracking.py`    camera + MediaPipe in a background thread; single-slot Queue
                   publishes (frame, 21×3 keypoints, handedness) to the control loop.
- `model.py`       `compute_signals(keypoints)` → dict of named scalars
                   (`palm_x/z`, `curl_<finger>`, `spread_<finger>` signed-2D,
                   `thumb_oppose`, `thumb_abduct`, `thumb_mid_bend`, `thumb_dist_bend`).
                   `extract_shadow_hand_targets(keypoints, scales)` applies ACTUATOR_MAP.
- `settings.py`    Single source of truth: ACTUATOR_MAP — per-actuator
                   `(signal, gain, offset, clip)`. Paths, FPS=30, SMOOTHING_FACTOR=0.15.
- `mano_pipeline.py` SynergyProjector (Santello 5-DOF prior, opt-in via blend);
                   MANOFitter scaffold (not used yet).
- `logging_setup.py` rotating file logging at `logs/run.log`.

## Archived / aux
- `pybullet_hand/`   legacy PyBullet+URDF path (kept for reference).
- `experiments/`     notebooks, scratch (rnn, diffusion, drawing).
- `assets/mujoco_menagerie/shadow_hand/` Menagerie XML model + `my_scene.xml`
                     (still has `ctl_curl_scale`, `ctl_spread_scale`,
                     `ctl_thumb_scale`, `ctl_smoothing`, `ctl_synergy_blend`
                     position actuators at lines 77–81).
- `data/hand_log.csv` per-frame log: timestamp_ms + 21×(x,y,z) + 20 actuator targets.

## Per-frame pipeline
landmarks → compute_signals → ACTUATOR_MAP (clip(signal·gain+offset)) →
SynergyProjector (optional, blend) → smooth_targets (EMA α=SMOOTHING_FACTOR) →
data.ctrl[i] → mj_step×16 → viewer.sync().

## Concurrency
- Thread A: tracking.FrameTracker (camera + MediaPipe, ~15–20 fps).
- Thread B: control loop in main.py (~30 fps), reads `tracker.latest()`
  non-blocking, runs MuJoCo, writes CSV.
- Subprocess: cv2 preview window (mp.Queue maxsize=3, macOS-safe under mjpython).

## Run
```bash
uv venv --python 3.10                          # if .venv missing or wrong version
uv pip install mediapipe mujoco opencv-python numpy
uv run mjpython -m shadow_hand.main            # NOT `python main.py` — root main.py is a stub
Flags: --no-record, --no-cv2, --log-level DEBUG.
Exit: ESC or Q in the MuJoCo window.
```

Key signals (geometry, all from MediaPipe 21 landmarks)
curl(finger) = 1 − ‖tip−mcp‖ / (‖pip−mcp‖+‖dip−pip‖+‖tip−dip‖); ×2 clipped to [0,1].
spread(finger) = signed atan2 between (middle_pip−middle_mcp) and (finger_pip−finger_mcp) in XY plane. Sign distinguishes fan vs cross. spread_index negated to normalize sign across left/right of middle.
thumb_oppose = 1 − ‖thumb_tip − pinky_mcp‖ / (1.5·palm_width).
thumb_abduct/mid_bend/dist_bend = unsigned 3D angles between consecutive thumb bones.
palm_x, palm_z = wrist→middle_MCP unit vector components → wrist actuators.
Active task state
README.md was just fully rewritten (uv install, Mermaid diagrams, theory cheatsheet, EMG/EEG roadmap). No slider section in the new README.
Code still references sliders; cleanup pending.
requirements.txt is outdated (pinned mediapipe 0.10.5, missing mujoco). Project should migrate to pyproject.toml deps or unpinned requirements.
Open items / next steps
Strip slider code: _build_slider_specs, read_sliders calls, slider HUD text block in main.py:298–312, and the 5 <position> actuators in assets/mujoco_menagerie/shadow_hand/my_scene.xml:77–81. Replace slider-derived scales/live_smoothing/live_synergy_blend with static constants pulled from settings.py.
Update requirements.txt or move deps into pyproject.toml (uv add). Add mujoco. Drop the strict mediapipe==0.10.5 pin.
Optional: replace root main.py (stub) with a forwarder to shadow_hand.main so python main.py works.
Future direction (user-stated)
Replace MediaPipe input with EMG envelopes / EEG features / learned video→pose. Swap point: compute_signals(features) in shadow_hand/model.py.
The hand_log.csv recorder is the training scaffold (time-aligned landmarks + actuator targets) for supervising a future biosignal decoder.
Docs already in repo
docs/ARCHITECTURE.md — concurrency + pipeline diagrams (canonical).
docs/RETARGETING.md — MANO / dex-retargeting upgrade paths.
docs/CHANGES.md — changelog.
User preferences
Prefers terse, structured responses with brief examples + brief theory + diagrams.
Uses uv exclusively for Python env mgmt; let uv resolve versions.
macOS x86_64 → some opencv/mediapipe builds need version pinning when no wheel exists.