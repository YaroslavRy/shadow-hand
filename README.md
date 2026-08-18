---
title: Shadow Hand Teleoperation
emoji: 🦾
colorFrom: blue
colorTo: indigo
sdk: docker
suggested_hardware: cpu-basic
---

# Shadow Hand Teleoperation

**Webcam hand pose → MediaPipe landmarks → retargeting → real MuJoCo Shadow Hand.**

![Shadow Hand teleoperation workbench](assets/screen3.png)

This is a research PoC for Shadow Hand teleoperation with the real [DeepMind MuJoCo Menagerie Shadow Hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand). The core loop is webcam hand pose -> MediaPipe landmarks -> retargeting -> MuJoCo actuators -> Shadow Hand motion, using the real Menagerie MJCF and `mj_step` physics.

## Space paths

```mermaid
flowchart LR
  C[Webcam] --> MP[MediaPipe: 21 landmarks]
  MP --> R[Geometric retargeting]
  R --> A[20 Shadow Hand actuator targets]
  A --> M[MuJoCo Menagerie model]
  M --> V[Rendered Shadow Hand]
```

| Path | What it does |
|---|---|
| **`/browser`** | Fastest demo. MediaPipe + MuJoCo-WASM run in your browser; webcam frames stay local. |
| **`/server`** | Fallback demo. Frames are uploaded to this Space and rendered server-side. |
| **Native local** | Source-of-truth research path for sensors, tuning, and future experiments. |

`/` opens the browser path by default.

Near-term roadmap:
- tactile/contact sensors in the native local MuJoCo path
- an optimization layer: user pose -> robot pose -> robot hand pose estimate -> minimize user/robot pose error

The WASM path now reproduces the native model's visual and retargeting behaviour locally: it compiles the same MJCF (matching `nq 29` / `nu 25`), steps the same `mj_step` physics, and its retargeting matches the Python mapping to floating-point rounding. See [`docs/WASM.md`](docs/WASM.md).

## Current experiment status

- [x] Menagerie Shadow Hand MJCF and mesh assets
- [x] 20 actuator targets from MediaPipe geometry
- [x] MuJoCo stepping and native local viewer
- [x] Per-frame landmark / actuator CSV logging
- [x] Browser-side MediaPipe prototype
- [x] Browser-side MuJoCo-WASM renderer and retargeting parity (verified: max actuator delta 3.9e-15 over 300 samples)
- [ ] Tactile/contact sensors and contact-readout experiments

## Run the reference system locally

The local application is the source of truth for physics and future sensor work. It requires Python 3.10, MuJoCo, and a webcam.

```bash
uv venv --python 3.10
uv pip install mediapipe mujoco opencv-python numpy
uv run mjpython -m shadow_hand.main
```

Press `ESC` or `Q` in the MuJoCo window to quit. Recorded samples are written to `data/hand_log.csv` unless `--no-record` is passed.

## Retargeting experiment

Each video frame produces 21 MediaPipe points. `compute_signals()` extracts finger curl, signed spread, and thumb-opposition signals. The mapping table in [`shadow_hand/settings.py`](shadow_hand/settings.py) converts those signals to the 20 position-actuator targets, then an EMA smooths the targets before the simulation advances.

```text
21 landmarks → curl / spread / thumb-opposition → ACTUATOR_MAP + optional synergy prior → data.ctrl[20] → mj_step → Shadow Hand state
```

The CSV log keeps the time-aligned landmarks and actuator targets. It is the dataset scaffold for future EMG/EEG decoders and learned retargeting.

## Browser WASM workbench

```bash
npm install
npm run dev -- --port 5173
```

Open `http://127.0.0.1:5173`. It loads the same Menagerie XML and mesh assets into official MuJoCo WASM and compiles the MJCF in-browser; webcam frames never leave the machine.

The retargeting in [`web/retarget.js`](web/retarget.js) is a direct port of `settings.py` / `model.py` and is checked numerically against the Python implementation — 300 random landmark sets, 20 actuators each, max delta `3.9e-15`. Gains live in `shadow_hand/settings.py`; change them there and port, rather than tuning the two sides independently.

`npm run build` produces a fully static `dist/` (~17 MB, mostly the MuJoCo WASM binary). Architecture, parity method, coordinate-frame pitfalls and known differences are documented in [`docs/WASM.md`](docs/WASM.md).

## Project map

```text
shadow_hand/       native MuJoCo reference application
space_assets/      Menagerie Shadow Hand MJCF and visual assets
web/               browser MuJoCo-WASM path (engine, retargeting, render)
public/mujoco/     scene assets served to the browser build
data/              recorded landmark / actuator samples
docs/              experiment notes and design rationale
```

## References

- DeepMind, [MuJoCo Menagerie: Shadow Hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand)
- Santello, Flanders & Soechting (1998), *Postural Hand Synergies for Tool Use*
- Google, [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
