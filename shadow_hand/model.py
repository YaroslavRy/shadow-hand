"""Map MediaPipe hand keypoints -> Shadow Hand actuator targets via ACTUATOR_MAP."""
import numpy as np

from .settings import ACTUATOR_MAP, FINGER_LANDMARKS
from .tracking import safe_normalize


# Group each signal into a calibration bucket. The slider widgets in
# my_scene.xml multiply the gains in each bucket at runtime.
SIGNAL_GROUPS = {
    "curl_index": "curl",
    "curl_middle": "curl",
    "curl_ring": "curl",
    "curl_pinky": "curl",
    "spread_index": "spread",
    "spread_ring": "spread",
    "spread_pinky": "spread",
    "thumb_oppose": "thumb",
    "thumb_abduct": "thumb",
    "thumb_mid_bend": "thumb",
    "thumb_dist_bend": "thumb",
}


def calculate_finger_curl(keypoints, finger_points):
    mcp, pip, dip, tip = [keypoints[i] for i in finger_points]
    straight = np.linalg.norm(tip - mcp)
    bent = (
        np.linalg.norm(pip - mcp)
        + np.linalg.norm(dip - pip)
        + np.linalg.norm(tip - dip)
    )
    curl = 1.0 - (straight / (bent + 1e-6))
    return max(0.0, min(1.0, curl * 2.0))


def _angle(v1, v2):
    return float(
        np.arccos(
            np.clip(np.dot(safe_normalize(v1), safe_normalize(v2)), -1.0, 1.0)
        )
    )


def _angle_2d(v1, v2):
    """Unsigned angle in the XY (image) plane. Z is discarded."""
    a = np.array([v1[0], v1[1], 0.0])
    b = np.array([v2[0], v2[1], 0.0])
    return float(
        np.arccos(
            np.clip(np.dot(safe_normalize(a), safe_normalize(b)), -1.0, 1.0)
        )
    )


def _signed_angle_2d(v1, v2):
    """Signed angle from v1 to v2 in the XY (image) plane.

    Returns positive when v2 is rotated clockwise from v1 (in MediaPipe
    image coordinates, where y points down). The sign lets the spread
    mapping distinguish outward spread from inward crossing - unsigned
    angles can't tell those apart, so the robot can only spread one way.
    """
    a = safe_normalize(np.array([v1[0], v1[1], 0.0]))
    b = safe_normalize(np.array([v2[0], v2[1], 0.0]))
    cross_z = a[0] * b[1] - a[1] * b[0]
    dot = a[0] * b[0] + a[1] * b[1]
    return float(np.arctan2(cross_z, dot))


def compute_signals(keypoints):
    """Pre-compute every named signal that ACTUATOR_MAP feeds from."""
    wrist = keypoints[0]
    palm_v = safe_normalize(keypoints[9] - wrist)

    curls = {
        name.lower(): calculate_finger_curl(keypoints, idx)
        for name, idx in FINGER_LANDMARKS.items()
    }

    v_mid = safe_normalize(keypoints[10] - keypoints[9])

    v_t_prox = keypoints[2] - keypoints[1]
    v_t_mid = keypoints[3] - keypoints[2]
    v_t_dist = keypoints[4] - keypoints[3]
    v_idx_root = keypoints[5] - wrist

    thumb_to_pinky = np.linalg.norm(keypoints[17] - keypoints[4])
    palm_width = np.linalg.norm(keypoints[17] - keypoints[5])
    opp_ratio = 1.0 - (thumb_to_pinky / (palm_width * 1.5 + 1e-6))

    return {
        "zero": 0.0,
        "palm_x": float(palm_v[0]),
        "palm_z": float(palm_v[2]),
        "curl_index": curls["index"],
        "curl_middle": curls["middle"],
        "curl_ring": curls["ring"],
        "curl_pinky": curls["pinky"],
        # 2D signed image-plane angles. Convention: + = finger displaced
        # OUTWARD from middle, - = crossed INWARD over middle.
        #
        # For a right hand in the cv2-flipped frame, the layout left->right
        # is: thumb, index, middle, ring, pinky. Index sits on the LEFT
        # side of middle - its raw signed angle is naturally NEGATIVE when
        # the finger spreads outward, so we negate it. Ring and pinky sit
        # on the RIGHT side of middle - their raw signed angles are
        # already POSITIVE when spreading outward, no negation needed.
        "spread_index": -_signed_angle_2d(v_mid, keypoints[6] - keypoints[5]),
        "spread_ring":   _signed_angle_2d(v_mid, keypoints[14] - keypoints[13]),
        "spread_pinky":  _signed_angle_2d(v_mid, keypoints[18] - keypoints[17]),
        "thumb_dist_bend": _angle(v_t_mid, v_t_dist),
        "thumb_mid_bend": _angle(v_t_prox, v_t_mid),
        "thumb_abduct": _angle(v_idx_root, v_t_prox),
        "thumb_oppose": float(opp_ratio),
    }


def extract_shadow_hand_targets(keypoints, scales=None):
    if keypoints is None:
        return {}
    scales = scales or {}
    signals = compute_signals(keypoints)
    targets = {}
    for actuator, cfg in ACTUATOR_MAP.items():
        group = SIGNAL_GROUPS.get(cfg["signal"])
        scale = scales.get(group, 1.0)
        v = signals[cfg["signal"]] * cfg["gain"] * scale + cfg["offset"]
        targets[actuator] = float(np.clip(v, *cfg["clip"]))
    return targets
