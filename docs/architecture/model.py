"""Hand-curated architecture tree for ShadowHand.

A single tree: every node has a `parent` (None = top level). The viewer
drills from top-level nodes down through children until it reaches a
module, then into that module's code symbols (added automatically by the
AST scan in build.py — you don't list functions/classes here).

This project exposes two top-level subtrees:

  shadow_hand/   the real file structure (packages → modules)
  Concurrency    a logical view of the runtime: which threads/processes
                 exist and how data flows between them. This grouping
                 can't be inferred from imports alone, so it's curated.

Edges are import/data-flow relationships. The viewer aggregates them to
whatever level you're currently looking at.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_NAME = "ShadowHand"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    parent: Optional[str] = None
    kind: str = "module"
    file: Optional[str] = None
    symbol: Optional[str] = None
    line: Optional[int] = None


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str = ""


NODES: list[Node] = [
    # ======================================================================
    # Subtree 1 — the source package as it sits on disk.
    # ======================================================================
    Node("shadow_hand", "shadow_hand/", parent=None, kind="package",
         file="shadow_hand"),

    Node("sh/main.py", "main.py", parent="shadow_hand", kind="process",
         file="shadow_hand/main.py", symbol="main"),
    Node("sh/tracking.py", "tracking.py", parent="shadow_hand", kind="thread",
         file="shadow_hand/tracking.py", symbol="FrameTracker"),
    Node("sh/model.py", "model.py", parent="shadow_hand", kind="module",
         file="shadow_hand/model.py", symbol="compute_signals"),
    Node("sh/settings.py", "settings.py", parent="shadow_hand", kind="module",
         file="shadow_hand/settings.py", symbol="ACTUATOR_MAP"),
    Node("sh/mano_pipeline.py", "mano_pipeline.py", parent="shadow_hand", kind="module",
         file="shadow_hand/mano_pipeline.py", symbol="SynergyProjector"),
    Node("sh/logging_setup.py", "logging_setup.py", parent="shadow_hand", kind="module",
         file="shadow_hand/logging_setup.py", symbol="configure_logging"),

    # ======================================================================
    # Subtree 2 — logical runtime / concurrency view.
    # ======================================================================
    Node("concurrency", "Concurrency", parent=None, kind="package"),

    Node("c_webcam", "Webcam", parent="concurrency", kind="external"),
    Node("c_tracker", "FrameTracker thread", parent="concurrency", kind="thread",
         file="shadow_hand/tracking.py", symbol="FrameTracker"),
    Node("c_loop", "Control loop", parent="concurrency", kind="thread",
         file="shadow_hand/main.py", symbol="main"),
    Node("c_cv2", "cv2 preview subprocess", parent="concurrency", kind="process",
         file="shadow_hand/main.py", symbol="cv2_viewer_process"),
    Node("c_mujoco", "MuJoCo viewer", parent="concurrency", kind="external"),
    Node("c_csv", "hand_log.csv", parent="concurrency", kind="asset"),
]


EDGES: list[Edge] = [
    # ---- module imports (subtree 1) --------------------------------------
    Edge("sh/main.py", "sh/tracking.py", "imports"),
    Edge("sh/main.py", "sh/model.py", "imports"),
    Edge("sh/main.py", "sh/settings.py", "imports"),
    Edge("sh/main.py", "sh/mano_pipeline.py", "imports"),
    Edge("sh/main.py", "sh/logging_setup.py", "imports"),
    Edge("sh/model.py", "sh/settings.py", "imports"),
    Edge("sh/model.py", "sh/tracking.py", "safe_normalize"),
    Edge("sh/mano_pipeline.py", "sh/settings.py", "imports"),

    # ---- runtime data flow (subtree 2) -----------------------------------
    Edge("c_webcam", "c_tracker", "BGR frames"),
    Edge("c_tracker", "c_loop", "Queue(1): frame, kp"),
    Edge("c_loop", "c_cv2", "mp.Queue(3): frame"),
    Edge("c_loop", "c_mujoco", "data.ctrl + sync"),
    Edge("c_loop", "c_csv", "row / frame"),
]
