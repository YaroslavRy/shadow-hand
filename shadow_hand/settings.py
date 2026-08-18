from pathlib import Path


# ----- Paths (resolved at import time, relative to project root) -----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
SCENE_PATH = ASSETS_DIR / "mujoco_menagerie" / "shadow_hand" / "my_scene.xml"
HAND_LOG_CSV = DATA_DIR / "hand_log.csv"
RUN_LOG = LOGS_DIR / "run.log"


# MediaPipe finger landmarks: [MCP, PIP, DIP, TIP] indices into the 21-point hand model.
FINGER_LANDMARKS = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20],
}


# Visual loop FPS cap (main.py).
FPS = 30.0

# Frame-to-frame smoothing for actuator targets. 0 = no smoothing, 1 = frozen.
SMOOTHING_FACTOR = 0.1


# ------------------------------------------------------------------
# ACTUATOR_MAP
# ------------------------------------------------------------------
# Each entry maps a Shadow Hand actuator to a single named input signal:
#
#       target = clip(signal * gain + offset, *clip)
#
# - `signal` keys are produced by compute_signals() in shadow_hand/model.py.
# - `clip` ranges come from joint limits in
#       assets/mujoco_menagerie/shadow_hand/right_hand.xml
#   Do NOT widen them past the XML limits or the controller fights itself.
# - For spread joints, the offset bakes in a "natural resting fan" subtraction
#   (~6 deg) so a relaxed hand maps to fingers-parallel on the robot.
# ------------------------------------------------------------------
ACTUATOR_MAP = {
    # Wrist (palm direction components in MediaPipe frame).
    # Gains kept conservative because MediaPipe's z-coordinate is noisy
    # and the MCP joints sit slightly forward of the wrist anatomically,
    # so palm_z is non-zero even at perceived neutral. High pitch gain
    # turns that baseline + noise into visible drift.
    "rh_A_WRJ2": {
        "signal": "palm_x",
        "gain": -1.0,
        "offset": 0.00,
        "clip": (-0.524, 0.175),
    },
    "rh_A_WRJ1": {
        "signal": "palm_z",
        "gain": -0.5,
        "offset": 0.00,
        "clip": (-0.700, 0.490),
    },
    # Index finger (FF)
    # Spread mapping uses a SIGNED 2D angle so the robot can spread both
    # ways. Convention:
    #   +signal (outward fan)   -> -ctrl (outward spread on robot)
    #   -signal (crossed inward) -> +ctrl (inward crossing on robot)
    # The negative gain inverts because the XML axis on FFJ4 (`0 -1 0`)
    # flips the intuitive sign relative to ctrl.
    "rh_A_FFJ4": {
        "signal": "spread_index",
        "gain": -0.9,
        "offset": 0.00,
        "clip": (-0.340, 0.340),
    },
    "rh_A_FFJ3": {
        "signal": "curl_index",
        "gain": 1.57,
        "offset": 0.00,
        "clip": (0.000, 1.570),
    },
    "rh_A_FFJ0": {
        "signal": "curl_index",
        "gain": 3.14,
        "offset": 0.00,
        "clip": (0.000, 3.140),
    },
    # Middle finger (MF) - palm-aligned, no driven spread
    "rh_A_MFJ4": {
        "signal": "zero",
        "gain": 0.0,
        "offset": 0.00,
        "clip": (-0.349, 0.349),
    },
    "rh_A_MFJ3": {
        "signal": "curl_middle",
        "gain": 1.57,
        "offset": 0.00,
        "clip": (0.000, 1.570),
    },
    "rh_A_MFJ0": {
        "signal": "curl_middle",
        "gain": 3.14,
        "offset": 0.00,
        "clip": (0.000, 3.140),
    },
    # Ring finger (RF) - same convention as FFJ4. spread_ring is sign-flipped
    # in compute_signals() so + still means outward, matching index.
    "rh_A_RFJ4": {
        "signal": "spread_ring",
        "gain": -0.9,
        "offset": 0.00,
        "clip": (-0.340, 0.340),
    },
    "rh_A_RFJ3": {
        "signal": "curl_ring",
        "gain": 1.57,
        "offset": 0.00,
        "clip": (0.000, 1.570),
    },
    "rh_A_RFJ0": {
        "signal": "curl_ring",
        "gain": 3.14,
        "offset": 0.00,
        "clip": (0.000, 3.140),
    },
    # Pinky (LF) - has an extra metacarpal LFJ5
    "rh_A_LFJ5": {
        "signal": "curl_pinky",
        "gain": 0.4,
        "offset": 0.00,
        "clip": (0.000, 0.785),
    },
    # Pinky fans out wider than the others in the image plane (anatomy),
    # so its gain magnitude is smaller to keep its dynamic range balanced.
    "rh_A_LFJ4": {
        "signal": "spread_pinky",
        "gain": -0.45,
        "offset": 0.00,
        "clip": (-0.340, 0.340),
    },
    "rh_A_LFJ3": {
        "signal": "curl_pinky",
        "gain": 1.57,
        "offset": 0.00,
        "clip": (0.000, 1.570),
    },
    "rh_A_LFJ0": {
        "signal": "curl_pinky",
        "gain": 3.14,
        "offset": 0.00,
        "clip": (0.000, 3.140),
    },
    # Thumb - each joint has its own dedicated geometric signal
    "rh_A_THJ5": {
        "signal": "thumb_oppose",
        "gain": 4.0,
        "offset": -0.50,
        "clip": (-1.050, 1.050),
    },
    "rh_A_THJ4": {
        "signal": "thumb_abduct",
        "gain": 1.5,
        "offset": -0.20,
        "clip": (0.000, 1.220),
    },
    "rh_A_THJ3": {
        "signal": "zero",
        "gain": 0.0,
        "offset": 0.00,
        "clip": (-0.209, 0.209),
    },
    "rh_A_THJ2": {
        "signal": "thumb_mid_bend",
        "gain": 2.0,
        "offset": 0.00,
        "clip": (-0.700, 0.700),
    },
    "rh_A_THJ1": {
        "signal": "thumb_dist_bend",
        "gain": 2.0,
        "offset": 0.00,
        "clip": (-0.262, 1.570),
    },
}
