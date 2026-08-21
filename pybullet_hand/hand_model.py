import numpy as np
import cv2
import mediapipe as mp
from .settings import FINGER_LANDMARKS

FINGER_NAMES = list(FINGER_LANDMARKS.keys())

mp_hands = mp.solutions.hands
hands = None
cap = None

def init_camera():
    global hands, cap
    if cap is None:
        hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
        )
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def get_hand_keypoints():
    ret, frame = cap.read()
    if not ret:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        return keypoints
    else:
        return None


def calculate_finger_curl(keypoints, finger_points):
    """
    Calculate finger curl using distance-based method
    More reliable than angle-based for MediaPipe landmarks
    """
    # Get the four points for this finger
    mcp, pip, dip, tip = [keypoints[i] for i in finger_points]

    # Calculate the "straight" distance (MCP to tip)
    straight_distance = np.linalg.norm(tip - mcp)

    # Calculate the "bent" distance (sum of segments)
    bent_distance = (
        np.linalg.norm(pip - mcp)
        + np.linalg.norm(dip - pip)
        + np.linalg.norm(tip - dip)
    )

    # Curl ratio: 0 = straight, 1 = fully curled
    # When straight: bent_distance ≈ straight_distance, ratio ≈ 0
    # When curled: bent_distance > straight_distance, ratio approaches 1
    curl_ratio = 1 - (straight_distance / bent_distance) if bent_distance > 0 else 0
    curl_ratio = np.clip(curl_ratio, 0, 1)

    return curl_ratio


def extract_finger_angles(keypoints):
    angles = []
    for finger_idx, finger in enumerate(FINGER_NAMES):
        finger_points = FINGER_LANDMARKS[finger]
        curl_ratio = calculate_finger_curl(keypoints, finger_points)
        if finger_idx == 0:  # Thumb
            joint1_angle = np.clip(curl_ratio * 1.8 - 0.2, -0.3, 1.5)
            joint2_angle = np.clip(curl_ratio * 1.6 - 0.1, -0.3, 1.3)
        else:
            joint1_angle = np.clip(curl_ratio * 1.6 - 0.1, -0.2, 1.5)
            joint2_angle = np.clip(curl_ratio * 1.4 - 0.05, -0.2, 1.3)
        angles.extend([joint1_angle, joint2_angle])
    return angles


def angle_between_points(a, b, c):
    """Calculate angle at point b between points a and c"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.arccos(cosine_angle)


def extract_finger_angles_alternative(keypoints):
    angles = []
    for finger_idx, finger in enumerate(FINGER_NAMES):
        finger_points = FINGER_LANDMARKS[finger]
        angle1 = angle_between_points(
            keypoints[finger_points[0]],
            keypoints[finger_points[1]],
            keypoints[finger_points[2]],
        )
        angle2 = angle_between_points(
            keypoints[finger_points[1]],
            keypoints[finger_points[2]],
            keypoints[finger_points[3]],
        )
        bend1 = np.pi - angle1
        bend2 = np.pi - angle2
        if finger_idx == 0:  # Thumb
            scaled1 = np.clip((bend1 * 1.8) - 0.3, -0.4, 1.6)
            scaled2 = np.clip((bend2 * 1.6) - 0.2, -0.3, 1.4)
        else:
            scaled1 = np.clip((bend1 * 2.2) - 0.2, -0.3, 1.7)
            scaled2 = np.clip((bend2 * 2.0) - 0.1, -0.2, 1.5)

        angles.extend([scaled1, scaled2])

    return angles


def get_palm_orientation(keypoints):
    """
    Calculate palm orientation - simplified approach for better stability
    """
    wrist = keypoints[0]
    middle_mcp = keypoints[9]
    index_mcp = keypoints[5]
    pinky_mcp = keypoints[17]

    # Simple approach: just use wrist to middle finger for main direction
    forward = middle_mcp - wrist
    forward = forward / np.linalg.norm(forward)

    # Side direction: wrist to index-pinky midpoint
    side = ((index_mcp + pinky_mcp) / 2) - wrist
    side = side / np.linalg.norm(side)

    # Up direction: cross product
    up = np.cross(forward, side)
    up = up / np.linalg.norm(up)

    return forward, side, up


def get_simple_hand_rotation(keypoints):
    """
    Fixed hand rotation - direct mapping with proper coordinate system
    """
    wrist = keypoints[0]
    index_mcp = keypoints[5]
    pinky_mcp = keypoints[17]
    middle_mcp = keypoints[9]

    # Method 1: Use knuckle line (more stable)
    knuckle_vector = index_mcp - pinky_mcp
    knuckle_angle = np.arctan2(knuckle_vector[1], knuckle_vector[0])

    # Method 2: Use palm direction (alternative)
    palm_vector = middle_mcp - wrist
    palm_angle = np.arctan2(palm_vector[1], palm_vector[0])

    # Use knuckle line for rotation (more reliable)
    angle = knuckle_angle

    # CRITICAL FIX: Coordinate system correction
    # MediaPipe: Y increases downward, X increases right
    # PyBullet: Standard XYZ where Z is up
    # We need to adjust for this coordinate difference

    # Flip Y component to match standard coordinates
    angle = -angle

    # Add 90 degree offset so that horizontal knuckles = 0 rotation
    angle = angle + np.pi / 2

    # Normalize angle to [-π, π]
    angle = np.arctan2(np.sin(angle), np.cos(angle))

    import pybullet as p
    euler = [0, 0, angle]
    return p.getQuaternionFromEuler(euler)


def debug_finger_curl(keypoints, finger_idx, finger_points):
    """
    Debug function to print curl values for a specific finger
    """
    curl = calculate_finger_curl(keypoints, finger_points)
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    print(f"{finger_names[finger_idx]}: curl = {curl:.3f}")
    return curl


def release():
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def _safe_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return vec * 0.0
    return vec / norm


def _signed_angle_2d(a, b):
    """Signed angle from vector a to b in XY plane."""
    a2 = _safe_normalize(np.array([a[0], a[1], 0.0]))
    b2 = _safe_normalize(np.array([b[0], b[1], 0.0]))
    dot = np.clip(np.dot(a2, b2), -1.0, 1.0)
    cross_z = np.cross(a2, b2)[2]
    return np.arctan2(cross_z, dot)


def extract_hand_joint_targets(keypoints, use_angle_method=False):
    """
    Build named joint targets for the extended hand URDF.

    Returns:
        dict[str, float]: Mapping from URDF joint name to target angle (radians).
    """
    wrist = keypoints[0]
    middle_mcp = keypoints[9]

    targets = {}

    for finger_idx, finger in enumerate(FINGER_NAMES):
        mcp_i, pip_i, dip_i, tip_i = FINGER_LANDMARKS[finger]
        mcp = keypoints[mcp_i]
        pip = keypoints[pip_i]
        dip = keypoints[dip_i]
        tip = keypoints[tip_i]

        # Finger spread (side-to-side at MCP) relative to middle finger direction.
        # Kept conservative so it does not destabilize when landmarks jitter.
        finger_vec = mcp - wrist
        middle_vec = middle_mcp - wrist
        spread = np.clip(_signed_angle_2d(middle_vec, finger_vec) * 0.35, -0.35, 0.35)

        mcp_bend = np.pi - angle_between_points(wrist, mcp, pip)
        pip_bend = np.pi - angle_between_points(mcp, pip, dip)
        dip_bend = np.pi - angle_between_points(pip, dip, tip)

        if not use_angle_method:
            # Curl ratio is robust for webcam landmarks; convert into staged phalange bends.
            curl_ratio = calculate_finger_curl(keypoints, FINGER_LANDMARKS[finger])
            mcp_flex = np.clip(curl_ratio * 1.3 - 0.08, -0.20, 1.25)
            pip_flex = np.clip(curl_ratio * 1.45 - 0.05, -0.15, 1.45)
            dip_flex = np.clip(curl_ratio * 1.20 - 0.03, -0.10, 1.10)
        else:
            mcp_flex = np.clip(mcp_bend * 1.2 - 0.08, -0.20, 1.25)
            pip_flex = np.clip(pip_bend * 1.1 - 0.06, -0.15, 1.45)
            # Couple DIP slightly to PIP for realism; pure DIP estimate is often noisy.
            dip_flex = np.clip((0.45 * pip_bend + 0.55 * dip_bend) * 1.0 - 0.04, -0.10, 1.10)

        prefix = finger.lower()
        if finger == "Thumb":
            targets["thumb_spread_joint"] = np.clip(spread * -1.2, -0.55, 0.55)
            targets["thumb_joint1"] = np.clip(mcp_flex * 1.10, -0.30, 1.30)
            targets["thumb_joint2"] = np.clip(pip_flex * 1.00, -0.20, 1.20)
            targets["thumb_joint3"] = np.clip(dip_flex * 0.95, -0.20, 1.00)
        else:
            targets[f"{prefix}_spread_joint"] = spread
            targets[f"{prefix}_joint1"] = mcp_flex
            targets[f"{prefix}_joint2"] = pip_flex
            targets[f"{prefix}_joint3"] = dip_flex

    return targets
