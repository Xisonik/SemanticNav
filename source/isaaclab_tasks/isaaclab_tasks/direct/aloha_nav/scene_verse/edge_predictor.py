"""
SceneVerse-based heuristic edge predictor service.

Wraps the SceneVerse SSG relationship heuristics (support, proximity,
hanging/above-below) and exposes a simple API that takes object
centres + extents and returns integer relation IDs per ordered pair.

Frame convention  (matches IsaacSim / aloha_nav):
    X – right,  Y – forward,  Z – up  (gravity along –Z)

The SceneVerse heuristics use the same convention internally:
    • z_min / z_max  → vertical extent
    • XY-plane        → floor / footprint
    • camera_angle    → rotates XY for direction labels (we use 0 =
      identity, so directional labels are in world frame)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# --- import SceneVerse relationship modules ----------------------------------
from .ssg_data.script.ObjNode import ObjNode
from .relationships.support import cal_support_relations
from .relationships.proximity import cal_proximity_relationships
from .relationships.hanging import cal_hanging_relationships

# =============================================================================
# Relation vocabulary
# =============================================================================

SV_RELATION_LABELS: List[str] = [
    "none",            # 0  – no detected relation
    "supported_by",    # 1  – obj rests on anchor (anchor supports obj)
    "supports",        # 2  – obj supports anchor (obj is underneath)
    "embedded",        # 3  – obj is embedded in anchor
    "inside",          # 4  – obj is inside anchor
    "above",           # 5  – obj is above anchor (non-contact vertical)
    "below",           # 6  – obj is below anchor (non-contact vertical)
    "beside",          # 7  – obj close with overlapping footprint
    "near",            # 8  – obj close, non-overlapping
    "far",             # 9  – obj far away
]
NUM_SV_RELATIONS: int = len(SV_RELATION_LABELS)
MAX_SV_REL_ID: float = float(NUM_SV_RELATIONS - 1)  # 9

_LABEL_TO_ID: Dict[str, int] = {l: i for i, l in enumerate(SV_RELATION_LABELS)}


# =============================================================================
# Helper: classify a raw SceneVerse relation string → integer ID
# =============================================================================

def _classify_relation_string(rel_str: str) -> int:
    """Map a raw SceneVerse relation string to the canonical integer ID.

    The SceneVerse heuristics produce varied strings like
    ``"support"``, ``"on"``, ``"embd into"``, ``"inside"``,
    ``"above"``, ``"below"``, ``"beside"``, ``"hung on"``,
    ``"3 o'clock direction near"``, ``"to the left of"``, etc.

    We bucket them into the 10-class vocabulary above.
    """
    s = rel_str.strip().lower()

    # --- support / on ---
    if s == "support":
        return _LABEL_TO_ID["supports"]      # src supports tgt
    if s in ("on", "resting on", "placed on", "supported by", "on the top of"):
        return _LABEL_TO_ID["supported_by"]  # src is on tgt

    # --- embed / inside ---
    if "embd" in s or "embed" in s:
        return _LABEL_TO_ID["embedded"]
    if "inside" in s:
        return _LABEL_TO_ID["inside"]

    # --- vertical ---
    if s == "above" or "higher" in s:
        return _LABEL_TO_ID["above"]
    if s == "below" or "lower" in s:
        return _LABEL_TO_ID["below"]

    # --- hanging (treat as supported_by: obj hangs from anchor) ---
    if "hung" in s or "hang" in s:
        return _LABEL_TO_ID["supported_by"]

    # --- proximity: close / beside / under(overlap) ---
    if s in ("beside", "close to", "adjacent to", "next to"):
        return _LABEL_TO_ID["beside"]
    if s == "under":
        # "under" in SceneVerse proximity = significant footprint overlap
        return _LABEL_TO_ID["beside"]

    # --- proximity: direction + far / near ---
    if "far" in s:
        return _LABEL_TO_ID["far"]
    if "near" in s:
        return _LABEL_TO_ID["near"]

    # --- oppo-support ("on" reversed) ---
    if "oppo" in s:
        return _LABEL_TO_ID["supported_by"]

    # --- directional (left / right / front / behind) → map to near ---
    if any(d in s for d in ("left", "right", "front", "behind", "o'clock")):
        return _LABEL_TO_ID["near"]

    # Fallback
    return _LABEL_TO_ID["none"]


# =============================================================================
# Service class
# =============================================================================

class SceneVerseEdgePredictorService:
    """Heuristic edge predictor using SceneVerse SSG relationship rules.

    Unlike VL-SAT (neural), this is purely geometry-based: AABBs are
    compared via support-overlap, footprint-containment, proximity, and
    above/below checks.  No trained weights are involved.

    Usage::

        svc = SceneVerseEdgePredictorService()
        rel_ids = svc.predict_pairwise_relation_ids(centers, extents)
        # rel_ids: dict[(i,j)] → int  in  [0, 9]
    """

    # camera_angle = 0 → identity rotation in the XY plane.
    # Directional proximity labels will be relative to world +X axis.
    CAMERA_ANGLE: float = 0.0

    # ------------------------------------------------------------------ #
    #  public API
    # ------------------------------------------------------------------ #

    def predict_pairwise_relation_ids(
        self,
        centers: np.ndarray,
        extents: np.ndarray,
        names: Optional[List[str]] = None,
    ) -> Dict[Tuple[int, int], int]:
        """Predict relations for *all* ordered pairs.

        Args:
            centers:  (N, 3) object centres in world frame.
            extents:  (N, 3) full AABB extents (width, depth, height).
            names:    optional list[N] of object names (for label-aware
                      filtering inside SceneVerse heuristics).

        Returns:
            Dictionary ``{(src_local_idx, tgt_local_idx): rel_id}``
            where ``rel_id`` is in ``[0, NUM_SV_RELATIONS-1]``.
        """
        N = len(centers)
        if N < 2:
            return {}

        # ---- build ObjNode dict keyed by local index ----
        obj_dict: Dict[int, ObjNode] = {}
        for i in range(N):
            c = centers[i]
            e = extents[i]
            label = names[i] if names else f"obj_{i}"
            half = e / 2.0
            obj_dict[i] = ObjNode(
                id=i,
                label=label,
                position=c,
                x_min=c[0] - half[0],
                x_max=c[0] + half[0],
                y_min=c[1] - half[1],
                y_max=c[1] + half[1],
                z_min=c[2] - half[2],
                z_max=c[2] + half[2],
            )

        # ---- estimate scene_high (used by hanging / proximity) ----
        all_z = np.array([obj_dict[i].z_max for i in obj_dict])
        all_z_min = np.array([obj_dict[i].z_min for i in obj_dict])
        scene_high = max(float(all_z.max() - all_z_min.min()), 0.5)

        # ---- run SceneVerse heuristics ----
        # 1. support / embed / inside
        support_rels, embedded_rels, hanging_objs = cal_support_relations(
            obj_dict, self.CAMERA_ANGLE
        )

        # 2. hanging / above / below  (for objects not in hanging_objs)
        hanging_rels = cal_hanging_relationships(
            obj_dict, hanging_objs, self.CAMERA_ANGLE, scene_high
        )

        # 3. proximity (operates on all object ids)
        all_ids = list(obj_dict.keys())
        proximity_rels = cal_proximity_relationships(
            all_ids, self.CAMERA_ANGLE, obj_dict, scene_high
        )

        # ---- collect per-pair: pick highest-priority relation ----
        # Process in descending priority order; first registration wins.
        # Priority:  support/embed/inside  >  hanging/above/below  >  proximity
        pair_rel: Dict[Tuple[int, int], int] = {}

        def _register(src: int, tgt: int, rel_str: str) -> None:
            rid = _classify_relation_string(rel_str)
            if rid == 0:  # "none" — don't overwrite anything
                return
            key = (src, tgt)
            if key not in pair_rel:
                pair_rel[key] = rid

        # 1) support & embed (highest priority)
        for src, tgt, rel_str in support_rels:
            _register(src, tgt, rel_str)
        for src, tgt, rel_str in embedded_rels:
            _register(src, tgt, rel_str)

        # 2) hanging / above / below
        for rel in hanging_rels:
            if isinstance(rel[0], list):
                for sub in rel:
                    _register(sub[0], sub[1], sub[2])
            else:
                _register(rel[0], rel[1], rel[2])

        # 3) proximity (lowest priority — only fills gaps)
        for src, tgt, rel_str in proximity_rels:
            _register(src, tgt, rel_str)

        return pair_rel

    def predict_pair_relation_ids_for_anchor(
        self,
        centers: np.ndarray,
        extents: np.ndarray,
        anchor_local_idx: int,
        names: Optional[List[str]] = None,
    ) -> Dict[int, int]:
        """Predict relations between every object and a single anchor.

        Returns:
            ``{local_obj_idx: rel_id}`` for each object ≠ anchor.
        """
        all_pairs = self.predict_pairwise_relation_ids(centers, extents, names)
        result: Dict[int, int] = {}
        N = len(centers)
        for i in range(N):
            if i == anchor_local_idx:
                continue
            # direction: (object, anchor)
            result[i] = all_pairs.get((i, anchor_local_idx), 0)
        return result
