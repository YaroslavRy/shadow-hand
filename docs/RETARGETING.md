# Hand retargeting — approaches, theory, references

Background reading for upgrading the MediaPipe → Shadow Hand pipeline
beyond the current direct-joint mapping.

---

## 1. What we have today: direct joint mapping

Per-finger geometric features (curl, spread, opposition) → robot
actuator targets via affine functions, defined in
`shadow_hand/settings.py:ACTUATOR_MAP`:

```
target = clip(signal * gain * scale + offset, joint_limits)
```

**Pros:** simple, fast, predictable, easy to tune.
**Cons:** ignores the robot's actual kinematics. Doesn't generalize to
fine manipulation. Per-joint tuning is brittle to camera position and
hand size.

---

## 2. MANO — parametric human-hand model

**M**odel of **A**rticulated and **N**on-rigid def**O**rmations of the hand,
from MPI Tübingen (Romero, Tzionas, Black 2017).

MANO is a parametric model:
```
M(θ, β) -> 778 vertices, 16 joints
```
where:
- **`θ`** (theta) — 45-dim pose vector (15 joints × 3 axis-angle params)
- **`β`** (beta) — 10-dim shape vector (PCA of human hand shapes)

### Why MANO is better than raw MediaPipe for control

| Property | MediaPipe 21-keypoint | MANO 16-joint |
|---|---|---|
| Output | 3D landmark positions | Joint angles in axis-angle |
| Anatomy-aware | No | Yes (real human kinematics) |
| Smoothness over time | Jittery in z-depth | Smooth (joints can't pop) |
| Shape-invariant | No (depends on hand size) | Yes (β decouples shape) |
| Suitable for control | Needs heuristics | Direct joint mapping |

### How to fit MANO to MediaPipe

Standard pipeline:
1. Get 21 MediaPipe keypoints (xyz).
2. Solve `min_θ,β ‖ J_MANO(θ, β) - keypoints_human ‖²`
   with regularizers on θ and β.
3. Output: pose vector θ which has biomechanically valid joint angles.

Libraries:
- **`manotorch`** — PyTorch implementation. https://github.com/lixiny/manotorch
- **`smplx`** — official MPI release, includes MANO. https://github.com/vchoutas/smplx
- **`MANO_PYTORCH`** — lighter weight reference impl.

You need to **download the MANO model file** (`MANO_RIGHT.pkl`) from
https://mano.is.tue.mpg.de/ (free, license-gated, requires registration).

---

## 3. Hand synergies (Santello 1998)

> Santello, Flanders & Soechting (1998): *"Postural hand synergies for tool use."*
> Journal of Neuroscience, 18(23):10105–10115.

The classic paper. Asked 5 subjects to mime grasping ~57 common objects;
recorded 15-DOF hand postures. PCA on the dataset showed:

- **PC1** accounts for **~50%** of variance.
- **First 2 PCs** account for **~80%**.
- **First 5 PCs** account for **~98%**.

In other words: humans use about **5 effective degrees of freedom** out
of the ~20 available in the hand. The remaining DOFs are highly
correlated — for daily tasks they're never independent.

### Why this matters for robot control

Instead of mapping to 20 independent Shadow Hand actuators, project the
target vector onto a 5-dim "synergy space" first, then reconstruct.
Benefits:

- **Smoother motion** — high-frequency jitter in any single DOF gets
  averaged out across correlated DOFs.
- **More natural poses** — the projected output stays close to
  realistic hand configurations, even when MediaPipe is noisy.
- **Reduced calibration burden** — instead of tuning 20 per-actuator
  gains, you tune the 5 synergy weights.
- **Failure-tolerant** — if one finger's landmarks are occluded, the
  synergy basis interpolates from the others.

### What a synergy basis looks like

Five 20-dim unit vectors `U_1 ... U_5`, each capturing a coordinated
hand motion:

| Synergy | Description |
|---|---|
| U_1 | Overall grasp aperture (close all fingers together) |
| U_2 | Thumb opposition (thumb vs. fingers) |
| U_3 | Precision pinch (index vs. middle/ring/pinky) |
| U_4 | Radial-ulnar gradient (index/middle vs. ring/pinky) |
| U_5 | Distal-vs-proximal flexion bias |

In practice, the basis comes from PCA on motion-capture data. A
hand-crafted approximation works for prototypes (see
`shadow_hand/mano_pipeline.py:DEFAULT_SYNERGY_BASIS`).

### Application as a regularizing prior

```
target_20d        = extract_shadow_hand_targets(...)
weights_5d        = U @ target_20d      # project
target_regularized = U.T @ weights_5d   # reconstruct
```

The reconstruction is constrained to lie in a 5-dim subspace —
"forgetting" the noise components but keeping the dominant motion.

---

## 4. Optimization-based retargeting

The production standard, used in AnyTeleop, DexPilot, and most
academic dexterous-manipulation work.

### The optimization problem

```
q* = argmin_q  ‖ FK_robot(q) − KP_human ‖² + λ_smooth ‖ q − q_prev ‖² + λ_limits L(q)
```

- `q` — robot joint angle vector (20-dim for Shadow Hand)
- `FK_robot` — forward kinematics; computes fingertip / fingertip-vector
  positions from joint angles
- `KP_human` — target keypoints from MediaPipe / MANO
- Solved with Levenberg-Marquardt or sequential quadratic programming
  at every frame (~1-5 ms with a warm start)

Two retargeting modes typically supported:
- **Position retargeting** — match absolute 3D fingertip positions.
  Best for grasping.
- **Vector retargeting** — match relative vectors between fingertips
  (thumb-to-index, etc.). Best for gestures and pinching.

### Library: `dex-retargeting`

https://github.com/dexsuite/dex-retargeting

Out-of-the-box configs for Shadow Hand, Allegro, Inspire, SCHUNK,
LEAP Hand, and others. Used by the AnyTeleop project (Berkeley 2023).

```python
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.constants import RobotName, RetargetingType

config = RetargetingConfig(
    type=RetargetingType.position,
    robot_name=RobotName.shadow,
    urdf_path="assets/.../shadow_hand.urdf",
    target_link_names=["rh_fftip", "rh_mftip", "rh_rftip",
                       "rh_lftip", "rh_thtip"],
    target_origin_link_name="rh_palm",
    target_link_human_indices=[[8, 12, 16, 20, 4]],  # MediaPipe tips
)
retargeter = config.build()
q = retargeter.retarget(mediapipe_xyz)
```

---

## 5. Learning-based retargeting

Neural network maps human keypoints → robot joint angles, supervised
on motion-capture or self-supervised on simulation.

**Pros:** handles complex correlations naturally, can be smooth.
**Cons:** opaque, needs data, hard to debug failure modes.

Recent work: diffusion-policy-based teleoperation, generative
retargeting models. Out of scope for a personal project.

---

## 6. References

### Primary papers

- **Santello, Flanders & Soechting (1998)** — *Postural hand synergies for tool use.*
  *J. Neuroscience*, 18(23):10105–10115.
  Classic synergy paper. Establishes ~5-DOF empirical dimensionality.

- **Romero, Tzionas & Black (2017)** — *Embodied Hands: Modeling and Capturing Hands and Bodies Together.*
  *SIGGRAPH Asia.* The MANO paper.
  https://mano.is.tue.mpg.de/

- **Handa et al. (2019)** — *DexPilot: Vision-Based Teleoperation of Dexterous Robotic Hand-Arm System.*
  NVIDIA. Earlier position-based retargeting.
  https://arxiv.org/abs/1910.03135

- **Sivakumar, Shaw & Pathak (2022)** — *Robotic Telekinesis: Learning a Robotic Hand Imitator by Watching Humans on YouTube.*
  CMU/Meta. Vision-only retargeting via optimization.
  https://robotic-telekinesis.github.io/

- **Qin et al. (2023)** — *AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System.*
  Berkeley. The current SOTA open-source pipeline.
  https://yzqin.github.io/anyteleop

### Related: hand synergy follow-ups

- **Mason, Gomez & Ebner (2001)** — confirmed Santello's findings with
  different task sets.
- **Catalano et al. (2014)** — *Adaptive synergies for the design and control of the Pisa/IIT SoftHand.*
  Engineered a soft robotic hand with hardware-level synergies.
- **Ficuciello et al. (2011)** — applied synergies to the
  DLR-Hand II for grasp planning.

### Software / repos

- `dex-retargeting` — https://github.com/dexsuite/dex-retargeting
- `manotorch` — https://github.com/lixiny/manotorch
- `smplx` — https://github.com/vchoutas/smplx
- `pinocchio` — https://github.com/stack-of-tasks/pinocchio
- `pytorch_kinematics` — https://github.com/UM-ARM-Lab/pytorch_kinematics

---

## 7. Upgrade paths from current code

In rough order of effort:

1. **Synergy prior (1-2 hr)** — implemented in `shadow_hand/mano_pipeline.py`.
   No new dependencies. Wraps the existing `extract_shadow_hand_targets`
   output and projects it onto a 5-dim subspace. Can be toggled at runtime.

2. **`dex-retargeting` swap (few hours)** — add `dex-retargeting` to
   `requirements.txt`, replace `extract_shadow_hand_targets` with a call
   into the library. Keep the queue / viewer / slider infrastructure.

3. **MANO front-end (1 day)** — fit MANO to MediaPipe per frame, use
   MANO joint positions / angles as input to either direct mapping or
   `dex-retargeting`. Requires downloading MANO_RIGHT.pkl.

4. **Learning-based** — collect teleop data, train a policy. Multi-week effort.
