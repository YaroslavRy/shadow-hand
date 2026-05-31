"""MANO-inspired pipeline: synergy prior + optional MANO front-end.

Two independent pieces, both opt-in. The default `shadow_hand.main` pipeline
is unchanged; this module is imported only if you want to layer additional
biomechanical structure on top of the existing direct mapping.

------------------------------------------------------------------
1) SynergyProjector
------------------------------------------------------------------
Projects a 20-dim Shadow Hand target vector onto a 5-dim "synergy"
subspace then reconstructs. Based on Santello, Flanders & Soechting
(1998): humans use ~5 effective DOFs out of the hand's 20+.

This is a regularizing prior. It smooths noise, suppresses unrealistic
poses, and lets you control the hand with fewer tunable knobs. No new
dependencies — just numpy.

Wire it into main.py like:

    from .mano_pipeline import SynergyProjector
    proj = SynergyProjector(k=5)
    ...
    raw_targets   = extract_shadow_hand_targets(keypoints, scales=scales)
    proj_targets  = proj.project(raw_targets)  # opt-in regularization
    smoothed      = smooth_targets(proj_targets, previous_targets, ...)

------------------------------------------------------------------
2) MANOFitter
------------------------------------------------------------------
Scaffold for fitting the MANO parametric hand model to MediaPipe
keypoints. Replaces the geometric `compute_signals()` pipeline with a
proper kinematic fit. Outputs joint angles in axis-angle form, which
can then be retargeted to the Shadow Hand via dex-retargeting (or a
custom MANO->Shadow joint map).

Requires `manotorch` and the MANO model file. See docs/RETARGETING.md.

The class raises a clear ImportError / FileNotFoundError if those are
missing, so the rest of the codebase doesn't need to care.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .settings import ACTUATOR_MAP

log = logging.getLogger(__name__)


# ======================================================================
# Shadow Hand actuator order. Used to lay out target vectors consistently.
# ======================================================================

ACTUATOR_ORDER = list(ACTUATOR_MAP.keys())
N_ACTUATORS = len(ACTUATOR_ORDER)  # 20

_ACT_INDEX = {name: i for i, name in enumerate(ACTUATOR_ORDER)}


def targets_dict_to_vector(targets: Dict[str, float]) -> np.ndarray:
    v = np.zeros(N_ACTUATORS, dtype=np.float64)
    for name, val in targets.items():
        idx = _ACT_INDEX.get(name)
        if idx is not None:
            v[idx] = val
    return v


def vector_to_targets_dict(v: np.ndarray) -> Dict[str, float]:
    return {ACTUATOR_ORDER[i]: float(v[i]) for i in range(N_ACTUATORS)}


# ======================================================================
# 1) Synergy projector
# ======================================================================

# Hand-crafted approximation of the first five Santello-style synergies,
# adapted for the Shadow Hand's 20-actuator order. Each row is a unit
# direction in actuator space.
#
# This is NOT trained from motion-capture data; it's a reasonable
# initialization derived from the canonical grasp postures. Replace with
# a basis learned via PCA on real telemetry once you have it.
#
# Columns follow ACTUATOR_ORDER. For reference, the order is:
#   [WRJ2, WRJ1, THJ5..THJ1, FFJ4, FFJ3, FFJ0, MFJ4, MFJ3, MFJ0,
#    RFJ4, RFJ3, RFJ0, LFJ5, LFJ4, LFJ3, LFJ0]
#
# (Verify with: print(shadow_hand.mano_pipeline.ACTUATOR_ORDER))

def _build_default_synergy_basis() -> np.ndarray:
    """Construct a (5, 20) basis. Each row is normalized to unit norm."""
    # Initialize with zeros; we'll fill in the meaningful entries by
    # name to keep this readable.
    rows = []

    def vec(spec: Dict[str, float]) -> np.ndarray:
        v = np.zeros(N_ACTUATORS)
        for name, val in spec.items():
            if name in _ACT_INDEX:
                v[_ACT_INDEX[name]] = val
        return v

    # PC1: Overall grasp aperture - close all four fingers + thumb in.
    rows.append(vec({
        "rh_A_FFJ3": 1.0, "rh_A_FFJ0": 1.0,
        "rh_A_MFJ3": 1.0, "rh_A_MFJ0": 1.0,
        "rh_A_RFJ3": 1.0, "rh_A_RFJ0": 1.0,
        "rh_A_LFJ3": 1.0, "rh_A_LFJ0": 1.0,
        "rh_A_THJ2": 0.5, "rh_A_THJ1": 0.5,
    }))

    # PC2: Thumb opposition (thumb folds across, fingers stay).
    rows.append(vec({
        "rh_A_THJ5": 1.0, "rh_A_THJ4": 0.6,
        "rh_A_THJ2": 0.3, "rh_A_THJ1": 0.3,
        # slight finger curl as counter-pressure
        "rh_A_FFJ3": 0.2, "rh_A_MFJ3": 0.2,
    }))

    # PC3: Precision pinch (index vs middle/ring/pinky).
    rows.append(vec({
        "rh_A_FFJ3": 1.0, "rh_A_FFJ0": 0.8,
        "rh_A_MFJ3": -0.6, "rh_A_MFJ0": -0.6,
        "rh_A_RFJ3": -0.6, "rh_A_RFJ0": -0.6,
        "rh_A_LFJ3": -0.6, "rh_A_LFJ0": -0.6,
    }))

    # PC4: Radial-ulnar gradient (index/middle vs ring/pinky).
    rows.append(vec({
        "rh_A_FFJ3": 0.7, "rh_A_FFJ0": 0.7,
        "rh_A_MFJ3": 0.4, "rh_A_MFJ0": 0.4,
        "rh_A_RFJ3": -0.4, "rh_A_RFJ0": -0.4,
        "rh_A_LFJ3": -0.7, "rh_A_LFJ0": -0.7,
        "rh_A_LFJ5": -0.4,
    }))

    # PC5: Distal-vs-proximal flexion bias (curl from tips vs from knuckles).
    rows.append(vec({
        # J0 = coupled MCP+DIP (distal), J3 = MCP (proximal). Push them apart.
        "rh_A_FFJ0": 1.0, "rh_A_FFJ3": -0.5,
        "rh_A_MFJ0": 1.0, "rh_A_MFJ3": -0.5,
        "rh_A_RFJ0": 1.0, "rh_A_RFJ3": -0.5,
        "rh_A_LFJ0": 1.0, "rh_A_LFJ3": -0.5,
    }))

    basis = np.vstack(rows)
    # Row-wise unit normalization.
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return basis / norms


DEFAULT_SYNERGY_BASIS = _build_default_synergy_basis()


class SynergyProjector:
    """Projects 20-DOF targets onto a k-dim synergy subspace and back.

    The projected output stays close to "natural" hand configurations,
    even when input has noisy or inconsistent finger angles. Trades
    expressive precision for smoothness/realism.

    Parameters
    ----------
    k : int
        Number of synergies to keep (1 .. 5). Lower = more aggressive
        regularization. 5 reproduces ~98% of natural hand poses
        according to Santello 1998.
    basis : (k_max, 20) array, optional
        Custom basis. Each row should be a unit vector in actuator
        space. Defaults to `DEFAULT_SYNERGY_BASIS`.
    blend : float in [0, 1]
        How much of the projected output to mix with the raw input.
        0.0 = pure raw (no projection), 1.0 = pure projected.
        Useful for smoothly transitioning between paths during testing.
    """

    def __init__(
        self,
        k: int = 5,
        basis: Optional[np.ndarray] = None,
        blend: float = 1.0,
    ) -> None:
        if basis is None:
            basis = DEFAULT_SYNERGY_BASIS
        if basis.shape[1] != N_ACTUATORS:
            raise ValueError(
                f"basis must have {N_ACTUATORS} columns, got {basis.shape[1]}"
            )
        k = max(1, min(k, basis.shape[0]))
        self.U = basis[:k]  # (k, 20)
        self.k = k
        self.blend = float(np.clip(blend, 0.0, 1.0))
        log.info("SynergyProjector ready (k=%d, blend=%.2f)", k, blend)

    def project(
        self,
        targets: Dict[str, float],
        blend: Optional[float] = None,
    ) -> Dict[str, float]:
        """Take a {actuator: value} dict, return a regularized one.

        Args
        ----
        targets : dict
            Per-actuator target values from the upstream mapper.
        blend : float, optional
            Override `self.blend` for this call. Useful when the blend
            is driven by a live UI slider.
        """
        b = self.blend if blend is None else float(np.clip(blend, 0.0, 1.0))
        v = targets_dict_to_vector(targets)
        weights = self.U @ v
        reconstructed = self.U.T @ weights
        out = (1.0 - b) * v + b * reconstructed
        return vector_to_targets_dict(out)


# ======================================================================
# 2) MANO front-end (scaffold)
# ======================================================================
#
# This is a thin wrapper around manotorch. It is opt-in and not imported
# by the rest of the codebase. To use it:
#
#   pip install manotorch torch
#
# Download MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/ (requires
# free registration) and pass its path to MANOFitter(mano_path=...).
#
# Then in main.py:
#
#   fitter = MANOFitter(mano_path="path/to/MANO_RIGHT.pkl")
#   ...
#   joint_pos = fitter.fit(keypoints)   # (16, 3) MANO joints in xyz
#   # then either:
#   #   - feed joint_pos into dex_retargeting, or
#   #   - map joint angles directly to Shadow Hand actuators
# ======================================================================


class MANOFitter:
    """Fit MANO to MediaPipe keypoints. Requires manotorch.

    Args
    ----
    mano_path : str | Path
        Path to MANO_RIGHT.pkl (download from mano.is.tue.mpg.de).
    n_iter : int
        Optimization iterations per frame. 30-50 is reasonable.
    """

    def __init__(self, mano_path: str | Path, n_iter: int = 30) -> None:
        try:
            import torch  # noqa: F401
            from manotorch.manolayer import ManoLayer
        except ImportError as e:
            raise ImportError(
                "MANOFitter requires `pip install manotorch torch`. "
                "See docs/RETARGETING.md."
            ) from e

        mano_path = Path(mano_path)
        if not mano_path.exists():
            raise FileNotFoundError(
                f"MANO model file not found at {mano_path}. "
                "Download MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/ "
                "(free, license-gated)."
            )

        import torch as _torch
        self._torch = _torch
        self._mano = ManoLayer(
            mano_assets_root=str(mano_path.parent),
            use_pca=False,        # use full 45-dim pose (more flexible)
            flat_hand_mean=False,
            side="right",
        )
        # Trainable params: shape (1,10), pose (1,48), global (1,3), trans (1,3)
        self.theta = _torch.zeros(1, 48, requires_grad=True)
        self.beta = _torch.zeros(1, 10, requires_grad=True)
        self.trans = _torch.zeros(1, 3, requires_grad=True)
        self.n_iter = n_iter

        # Correspondence: MediaPipe has 21 landmarks, MANO has 16 joints.
        # The MediaPipe order is [wrist, thumb_cmc..thumb_tip,
        #   index_mcp..index_tip, middle_mcp..middle_tip,
        #   ring_mcp..ring_tip, pinky_mcp..pinky_tip].
        # MANO joint order is [wrist, index_mcp..index_tip,
        #   middle, ring, pinky, thumb] (approx; check manotorch docs).
        # The exact mapping varies by MANO version - check before using.
        self._mp_to_mano = None  # TODO: fill in 16-index mapping

    def fit(self, mediapipe_keypoints: np.ndarray) -> np.ndarray:
        """Fit MANO to one frame of MediaPipe keypoints.

        Args
        ----
        mediapipe_keypoints : (21, 3) array of xyz in MediaPipe units.

        Returns
        -------
        (16, 3) array of fitted MANO joint positions.
        """
        # This is the place to implement the per-frame Adam / LBFGS
        # optimization loop. Pseudocode:
        #
        #   target = self._torch.tensor(mediapipe_keypoints[mapping])
        #   optimizer = torch.optim.Adam([self.theta, self.trans], lr=1e-2)
        #   for _ in range(self.n_iter):
        #       output = self._mano(self.theta, self.beta)
        #       pred = output.joints[0] + self.trans
        #       loss = ((pred - target) ** 2).sum() \
        #              + 1e-3 * (self.theta ** 2).sum()
        #       optimizer.zero_grad(); loss.backward(); optimizer.step()
        #   return pred.detach().cpu().numpy()
        raise NotImplementedError(
            "MANOFitter.fit() is a scaffold. Implement the optimization "
            "loop and the MediaPipe<->MANO joint correspondence. See "
            "docs/RETARGETING.md or manotorch examples."
        )
