---
title: Shadow Hand Teleoperation
emoji: 🦾
colorFrom: blue
colorTo: indigo
sdk: docker
suggested_hardware: cpu-basic
---

# Shadow Hand Teleoperation

## Abstract

This project is a research PoC for vision-driven teleoperation of the MuJoCo Menagerie Shadow Hand, with the longer-term goal of building a richer bio-inspired robotics stack around it.

The current system already closes a minimal perception-to-action loop:

```text
human hand -> MediaPipe hand pose estimation -> retargeting -> Shadow Hand actuator commands -> MuJoCo simulation
```

The system uses the MuJoCo Menagerie Shadow Hand model and runs it in MuJoCo simulation.

The broader purpose is to turn this into a foundation for future work on:

- tactile sensing
- self-calibration
- signal fusion and perception
- action models
- reinforcement learning
- bio-inspired robot manipulation

In other words, this repository is not only a teleoperation demo. It is the beginning of a modular experimental platform for studying how perception, control, touch, and learned policies can be combined in one reproducible MuJoCo system.

## Research purpose

We are building toward a layered robotics stack:

```text
vision
  + tactile sensing
  + calibration / adaptation
  + action modeling
  + policy learning
  = embodied manipulation research platform
```

Near-term scientific goals:

1. Establish a stable teleoperation baseline.
2. Add tactile/contact sensing and interpretable diagnostics.
3. Introduce self-calibration and pose-alignment optimization.
4. Build signal fusion modules across vision, touch, and future modalities.
5. Use the resulting state/action interface for action models and RL.

Longer-term direction:

```text
multimodal sensing -> fused state estimate -> calibrated hand model
-> action prior / action model -> closed-loop control -> manipulation skill learning
```

## System overview

Current data path:

```text
camera frame
  -> MediaPipe landmarks (Hand pose estimation)
  -> geometric retargeting
  -> 20 actuator targets
  -> MuJoCo Shadow Hand dynamics
  -> rendered robot state
```

Planned extended data path:

```text
vision + touch + calibration signals
  -> fusion / perception layer
  -> robot state estimate
  -> optimization / self-calibration
  -> action model / policy
  -> robot control
```

Planned closed-loop path:

```text
camera + touch
  -> fusion / perception
  -> user-hand estimate
  -> retargeting
  -> self-calibration module
  -> robot command
  -> MuJoCo robot state
  -> contact + pose feedback
  -> updated calibration / corrected control
```

## Execution paths

| Path | Role |
|---|---|
| `/browser` | Fast demo path. MediaPipe + MuJoCo-WASM run in the browser; webcam frames stay local. |
| `/server` | Fallback visualization path. Frames are processed and rendered on the Space. |
| Native local app | Source-of-truth research path for sensors, debugging, tuning, and future experiments. |

Default behavior:

```text
/  -> /browser
/server -> Gradio server-rendered app
/browser -> static MuJoCo-WASM app
```

If the browser page is opened from the Hugging Face static domain, its server link should redirect to the main Space host.

## Current capabilities

- real MuJoCo Menagerie Shadow Hand assets
- MediaPipe-to-actuator retargeting
- native MuJoCo teleoperation app
- browser MuJoCo-WASM teleoperation path
- CSV logging of landmarks and actuator signals
- tactile/contact sensing in the native path
- native diagnostics for:
  - per-sensor activity
  - rolling linear traces
  - heatmap-style contact summaries
  - contact totals and channel values

## Tactile checkpoint

As of August 24, 2026, the project includes a first tactile sensing iteration in the native MuJoCo app:

- touch sensors attached to fingertip, finger-pad, and palm regions
- a suspended dynamic object for contact and grasp experiments
- a separate diagnostics window for quick tactile inspection
- modular UI files for palette, layout, and rendering behavior

This is still an experimental first step. The tactile system is intended as a base for later:

- fused perception
- grasp-state estimation
- contact-aware control
- action-model supervision
- RL reward shaping

## Current experiment status

- [x] Menagerie Shadow Hand MJCF and mesh assets
- [x] 20 actuator targets from MediaPipe geometry
- [x] MuJoCo stepping and native local viewer
- [x] Per-frame landmark / actuator CSV logging
- [x] Browser-side MuJoCo-WASM renderer
- [x] Tactile/contact sensors in the native path
- [ ] Self-calibration layer
- [ ] Sensor fusion layer
- [ ] Action-model interface
- [ ] RL training loop

## Local usage

### Native reference app

```bash
uv sync
uv run mjpython -m shadow_hand.main
```

Useful variants:

```bash
uv run mjpython -m shadow_hand.main --no-cv2
uv run mjpython -m shadow_hand.main --no-sensor-panel
```

The native app is the main research path.

### Browser MuJoCo-WASM app

```bash
npm install
npm run dev -- --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

This path runs MediaPipe and MuJoCo-WASM locally in the browser.

## Project map

```text
shadow_hand/       native MuJoCo app, tracking, retargeting, sensors
web/               browser MuJoCo-WASM path
public/mujoco/     browser-served MuJoCo scene assets
space_assets/      Space-specific MuJoCo assets
assets/            screenshots and bundled Menagerie assets
docs/              experiment notes, architecture, WASM notes, plans
tests/             behavior checks for sensors, runtime, and UI slices
```

## Why this matters

A useful robot-hand research platform should support more than just kinematic imitation. It should let us study how multiple signals can be combined into a richer control system.

This repo is meant to become a small but serious foundation for that direction:

```text
teleoperation baseline
  -> tactile grounding
  -> multimodal fusion
  -> self-calibration
  -> action abstraction
  -> policy learning
```

That makes it relevant not only as a demo, but as infrastructure for future work in bio-inspired robotics, sensor fusion, contact-rich manipulation, and learning-based control.

## References

- DeepMind, [MuJoCo Menagerie: Shadow Hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand)
- Google, [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- Santello, Flanders & Soechting (1998), *Postural Hand Synergies for Tool Use*
