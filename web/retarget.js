// Faithful port of the desktop retargeting pipeline so the browser build tracks
// the same poses as the Python implementation:
//   shadow_hand/settings.py  -> ACTUATOR_MAP, FINGER_LANDMARKS, SMOOTHING_FACTOR
//   shadow_hand/model.py     -> compute_signals / extract_shadow_hand_targets
//   shadow_hand/main.py      -> smooth_targets, steps_per_frame
// Gains, offsets and clip ranges are tied to the joint limits in right_hand.xml;
// they must stay in sync with settings.py rather than be retuned here.

export const SMOOTHING_FACTOR = 0.1;
// Python steps 16 x 0.002s per 30 FPS frame, i.e. real time.
export const PHYSICS_TIMESTEP = 0.002;

const FINGER_LANDMARKS = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

// Which calibration bucket scales each signal (sliders default to 1.0).
const SIGNAL_GROUPS = {
  curl_index: "curl", curl_middle: "curl", curl_ring: "curl", curl_pinky: "curl",
  spread_index: "spread", spread_ring: "spread", spread_pinky: "spread",
  thumb_oppose: "thumb", thumb_abduct: "thumb",
  thumb_mid_bend: "thumb", thumb_dist_bend: "thumb",
};

export const ACTUATOR_MAP = {
  rh_A_WRJ2: { signal: "palm_x", gain: -1.0, offset: 0, clip: [-0.524, 0.175] },
  rh_A_WRJ1: { signal: "palm_z", gain: -0.5, offset: 0, clip: [-0.700, 0.490] },

  rh_A_FFJ4: { signal: "spread_index", gain: -0.9, offset: 0, clip: [-0.340, 0.340] },
  rh_A_FFJ3: { signal: "curl_index", gain: 1.57, offset: 0, clip: [0, 1.570] },
  rh_A_FFJ0: { signal: "curl_index", gain: 3.14, offset: 0, clip: [0, 3.140] },

  rh_A_MFJ4: { signal: "zero", gain: 0, offset: 0, clip: [-0.349, 0.349] },
  rh_A_MFJ3: { signal: "curl_middle", gain: 1.57, offset: 0, clip: [0, 1.570] },
  rh_A_MFJ0: { signal: "curl_middle", gain: 3.14, offset: 0, clip: [0, 3.140] },

  rh_A_RFJ4: { signal: "spread_ring", gain: -0.9, offset: 0, clip: [-0.340, 0.340] },
  rh_A_RFJ3: { signal: "curl_ring", gain: 1.57, offset: 0, clip: [0, 1.570] },
  rh_A_RFJ0: { signal: "curl_ring", gain: 3.14, offset: 0, clip: [0, 3.140] },

  rh_A_LFJ5: { signal: "curl_pinky", gain: 0.4, offset: 0, clip: [0, 0.785] },
  rh_A_LFJ4: { signal: "spread_pinky", gain: -0.45, offset: 0, clip: [-0.340, 0.340] },
  rh_A_LFJ3: { signal: "curl_pinky", gain: 1.57, offset: 0, clip: [0, 1.570] },
  rh_A_LFJ0: { signal: "curl_pinky", gain: 3.14, offset: 0, clip: [0, 3.140] },

  rh_A_THJ5: { signal: "thumb_oppose", gain: 4.0, offset: -0.50, clip: [-1.050, 1.050] },
  rh_A_THJ4: { signal: "thumb_abduct", gain: 1.5, offset: -0.20, clip: [0, 1.220] },
  rh_A_THJ3: { signal: "zero", gain: 0, offset: 0, clip: [-0.209, 0.209] },
  rh_A_THJ2: { signal: "thumb_mid_bend", gain: 2.0, offset: 0, clip: [-0.700, 0.700] },
  rh_A_THJ1: { signal: "thumb_dist_bend", gain: 2.0, offset: 0, clip: [-0.262, 1.570] },
};

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = v => Math.hypot(v[0], v[1], v[2]);
const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

function safeNormalize(v) {
  const n = norm(v);
  return n < 1e-6 ? [0, 0, 0] : [v[0] / n, v[1] / n, v[2] / n];
}

function angle(v1, v2) {
  const a = safeNormalize(v1), b = safeNormalize(v2);
  return Math.acos(clamp(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1, 1));
}

// Signed angle from v1 to v2 in the image plane; the sign is what lets a spread
// mapping tell outward fanning apart from crossing inward over the middle.
function signedAngle2d(v1, v2) {
  const a = safeNormalize([v1[0], v1[1], 0]), b = safeNormalize([v2[0], v2[1], 0]);
  return Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
}

function fingerCurl(kp, points) {
  const [mcp, pip, dip, tip] = points.map(i => kp[i]);
  const straight = norm(sub(tip, mcp));
  const bent = norm(sub(pip, mcp)) + norm(sub(dip, pip)) + norm(sub(tip, dip));
  return clamp((1 - straight / (bent + 1e-6)) * 2, 0, 1);
}

export function computeSignals(kp) {
  const wrist = kp[0];
  const palm = safeNormalize(sub(kp[9], wrist));
  const curls = {};
  for (const [name, idx] of Object.entries(FINGER_LANDMARKS)) curls[name] = fingerCurl(kp, idx);

  const vMid = safeNormalize(sub(kp[10], kp[9]));
  const vThumbProx = sub(kp[2], kp[1]);
  const vThumbMid = sub(kp[3], kp[2]);
  const vThumbDist = sub(kp[4], kp[3]);
  const vIndexRoot = sub(kp[5], wrist);

  const thumbToPinky = norm(sub(kp[17], kp[4]));
  const palmWidth = norm(sub(kp[17], kp[5]));

  return {
    zero: 0,
    palm_x: palm[0],
    palm_z: palm[2],
    curl_index: curls.index,
    curl_middle: curls.middle,
    curl_ring: curls.ring,
    curl_pinky: curls.pinky,
    // + = displaced outward from middle, - = crossed inward. Index sits on the
    // opposite side of middle from ring/pinky, so its raw sign is negated.
    spread_index: -signedAngle2d(vMid, sub(kp[6], kp[5])),
    spread_ring: signedAngle2d(vMid, sub(kp[14], kp[13])),
    spread_pinky: signedAngle2d(vMid, sub(kp[18], kp[17])),
    thumb_dist_bend: angle(vThumbMid, vThumbDist),
    thumb_mid_bend: angle(vThumbProx, vThumbMid),
    thumb_abduct: angle(vIndexRoot, vThumbProx),
    thumb_oppose: 1 - thumbToPinky / (palmWidth * 1.5 + 1e-6),
  };
}

export function extractTargets(kp, scales = {}) {
  const signals = computeSignals(kp);
  const targets = {};
  for (const [actuator, cfg] of Object.entries(ACTUATOR_MAP)) {
    const scale = scales[SIGNAL_GROUPS[cfg.signal]] ?? 1;
    targets[actuator] = clamp(signals[cfg.signal] * cfg.gain * scale + cfg.offset, cfg.clip[0], cfg.clip[1]);
  }
  return targets;
}

export function smoothTargets(next, previous, factor) {
  const out = {};
  for (const [name, value] of Object.entries(next)) out[name] = factor * (previous[name] ?? 0) + (1 - factor) * value;
  return out;
}

// tracking.py mirrors the frame (cv2.flip) before MediaPipe runs, so every sign
// convention above assumes mirrored landmarks. The browser feeds the raw camera
// frame, so mirror x here to land in the same coordinate frame.
export function mirrorLandmarks(landmarks) {
  return landmarks.map(p => [1 - p.x, p.y, p.z]);
}
