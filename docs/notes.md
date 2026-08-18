# What the magic numbers actually mean

Reference for the values in `ACTUATOR_MAP` (settings.py).

## Anatomy of one row

For the original line:

```python
targets["rh_A_WRJ2"] = float(np.clip(-palm_v[0] * 1.5, -0.52, 0.17))
```

| Piece | Meaning |
|---|---|
| `palm_v[0]` | x-component of the unit vector wrist → middle-MCP (sideways palm tilt, range `[-1, 1]`) |
| `-` | sign flip for handedness / coordinate convention |
| `* 1.5` | **gain** — sensitivity. `>1` amplifies, `<1` dampens |
| `-0.52, 0.17` | **clip** — NOT arbitrary; these are the joint limits copied from [right_hand.xml:15](assets/mujoco_menagerie/shadow_hand/right_hand.xml#L15) (`range="-0.523599 0.174533"` ≈ -30° to +10°). WRJ2 physically can't go past these. |

Every actuator follows the pattern:

```
output = clip(signal * gain + offset, joint_min, joint_max)
```

## Other recurring constants

| Value | Meaning |
|---|---|
| `1.57` | ≈ π/2 — max bend for a single 90° knuckle (FFJ3, MFJ3, RFJ3, LFJ3) |
| `3.14` | ≈ π — max bend for `*J0` joints (coupled MCP+DIP, ctrlrange `[0, π]`, see [right_hand.xml:61](assets/mujoco_menagerie/shadow_hand/right_hand.xml#L61)) |
| `0.34` | ≈ knuckle spread limit (±0.349 rad in XML) |
| `0.10` | `SPREAD_BASE` — natural resting fan of human fingers (~6°), subtracted so a relaxed hand maps to "fingers parallel" on the robot |
| `1.5`  | spread sensitivity gain (`k_spread`) |
| Thumb (`* 2.0`, `- 0.2`, `* 4.0 - 0.5`) | empirical scale + offset; no deeper geometric meaning, just hand-tuned to look right |


## Notes and additional information

https://github.com/dexsuite/dex-retargeting
