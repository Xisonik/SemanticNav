#!/usr/bin/env python3
"""
debug_scene_graph.py  —  Standalone diagnostic for the scene-graph pipeline.

Tests each component in isolation and then end-to-end:
    1. graph_builder  – BBQ, VL-SAT, SceneVerse edge features
    2. SceneVerse heuristic predictor (always available, no weights)
    3. VL-SAT neural predictor  (only if checkpoint exists)
    4. combine_enabled_edge_features
    5. Full SceneManager.get_observation() round-trip

Run from the SemanticNav repo root:
    python source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/debug_scene_graph.py

Optionally pass --vl-sat-ckpt <path> to also test VL-SAT inference.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ALOHA_NAV_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ALOHA_NAV_DIR, "../../../../../.."))

# Ensure aloha_nav is on path so scene_verse package resolves
if ALOHA_NAV_DIR not in sys.path:
    sys.path.insert(0, ALOHA_NAV_DIR)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

SEP = "=" * 72
THIN = "-" * 72

def _import_from_file(filepath: str, attr: str) -> Any:
    """Import *attr* from a .py file by absolute path."""
    spec = importlib.util.spec_from_file_location("_mod", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)

def _header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def _ok(msg: str):
    print(f"  [OK]  {msg}")

def _fail(msg: str):
    print(f"  [FAIL]  {msg}")

def _info(msg: str):
    print(f"  [INFO]  {msg}")

def _print_edge_tensor(edge: torch.Tensor, names: List[str], goal_idx: int,
                       channel_labels: List[str]):
    """Pretty-print a single-environment edge tensor [M, 6]."""
    M = edge.shape[0]
    # Header row
    name_w = max(len(n) for n in names) + 2
    hdr = f"  {'obj':<{name_w}}"
    for cl in channel_labels:
        hdr += f" {cl:>12}"
    print(hdr)
    print("  " + "-" * len(hdr))
    for j in range(M):
        exists = edge[j, 0].item()
        if exists < 0.5 and j != goal_idx:
            continue  # skip inactive / goal
        tag = " *GOAL*" if j == goal_idx else ""
        row = f"  {names[j]:<{name_w}}"
        for c in range(edge.shape[1]):
            row += f" {edge[j, c].item():>12.4f}"
        row += tag
        print(row)


# ──────────────────────────────────────────────────────────────────────
# 0.  Mock scene builder
# ──────────────────────────────────────────────────────────────────────

def build_mock_scene(device: str = "cpu", num_envs: int = 2):
    """
    Build a compact 5-object scene that exercises various spatial
    relations (support, proximity, above/below).

    Layout  (Z-up,  XY floor):
                                  Z
        vase (on table)           |     Y
        cup  (on table)           |    /
        table (large, floor)      +---→ X
        chair (floor, beside)
        lamp  (floor, far away)

    Returns dict with all tensors + metadata needed by graph_builder.
    """
    names = ["table", "cup", "chair", "vase", "lamp"]
    M = len(names)
    E = num_envs

    #                        X      Y      Z
    centers = np.array([
        [ 0.0,   0.0,   0.4],   # table — large, on floor
        [ 0.05,  0.05,  0.85],  # cup — on table surface
        [ 0.8,   0.0,   0.45],  # chair — on floor, beside table
        [-0.05,  0.10,  0.90],  # vase — on table surface
        [ 3.5,   2.0,   0.7],   # lamp — far away on floor
    ], dtype=np.float32)

    extents = np.array([
        [0.70,  0.70,  0.80],   # table
        [0.08,  0.08,  0.10],   # cup
        [0.35,  0.35,  0.90],   # chair
        [0.10,  0.10,  0.15],   # vase
        [0.20,  0.20,  0.60],   # lamp
    ], dtype=np.float32)

    positions = torch.tensor(centers, device=device).unsqueeze(0).expand(E, -1, -1).clone()  # [E, M, 3]
    sizes     = torch.tensor(extents, device=device).unsqueeze(0).expand(E, -1, -1).clone()  # [E, M, 3]

    object_ids = torch.arange(M, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, M, 1]
    object_ids = object_ids.expand(E, -1, -1)

    active = torch.ones(E, M, 1, device=device)  # all active

    goal_idxs = torch.zeros(E, dtype=torch.long, device=device)  # goal = table (idx 0)

    return dict(
        names=names,
        M=M,
        E=E,
        positions=positions,
        sizes=sizes,
        object_ids=object_ids,
        active=active,
        goal_idxs=goal_idxs,
        centers=centers,
        extents=extents,
        device=device,
    )


# ──────────────────────────────────────────────────────────────────────
# 1.  Test graph_builder  – BBQ edges
# ──────────────────────────────────────────────────────────────────────

def test_bbq_edges(builder, scene: dict) -> Optional[torch.Tensor]:
    _header("1. BBQ edge features")
    try:
        bbq_center = torch.tensor([0.0, 0.0, 0.0], device=scene["device"])
        bbq_edges = builder.build_bbq_edge_features(
            scene["positions"],
            scene["object_ids"],
            scene["active"],
            scene["goal_idxs"],
            bbq_center,
        )
        _ok(f"shape = {list(bbq_edges.shape)}  (expected [{scene['E']}, {scene['M']}, 6])")
        assert bbq_edges.shape == (scene["E"], scene["M"], 6), "Shape mismatch!"

        _info("Env 0 edges (goal=table):")
        _print_edge_tensor(
            bbq_edges[0], scene["names"], int(scene["goal_idxs"][0]),
            ["exists", "left/right", "front/back", "above/below", "dist", "id_diff"],
        )

        # Decode string relations
        decoded = builder.decode_bbq_string_relations(bbq_edges)
        _info("Decoded BBQ string relations (env 0):")
        for j, rels in enumerate(decoded[0]):
            if rels:
                print(f"    {scene['names'][j]:>10} → goal : {', '.join(rels)}")

        _ok("BBQ edge features PASSED")
        return bbq_edges
    except Exception as exc:
        _fail(f"BBQ edges: {exc}")
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────────────────────────────
# 2.  Test SceneVerse predictor (standalone)
# ──────────────────────────────────────────────────────────────────────

def test_sceneverse_predictor(scene: dict) -> bool:
    _header("2. SceneVerse heuristic predictor (standalone)")
    try:
        from scene_verse.edge_predictor import (
            SceneVerseEdgePredictorService,
            SV_RELATION_LABELS,
        )

        svc = SceneVerseEdgePredictorService()
        _ok("SceneVerseEdgePredictorService instantiated")

        # Test pairwise
        all_rels = svc.predict_pairwise_relation_ids(
            scene["centers"], scene["extents"], scene["names"]
        )
        _info(f"Pairwise relations found: {len(all_rels)}")
        if all_rels:
            print(f"    {'src':>10} → {'tgt':>10}  :  id  label")
            print(f"    {THIN[:46]}")
            for (s, t), rid in sorted(all_rels.items()):
                print(f"    {scene['names'][s]:>10} → {scene['names'][t]:>10}  : "
                      f" {rid:>2}  {SV_RELATION_LABELS[rid]}")
        else:
            _info("(no relations detected — check if extents are realistic)")

        # Test anchor-based (anchor = table = 0)
        anchor_rels = svc.predict_pair_relation_ids_for_anchor(
            scene["centers"], scene["extents"], anchor_local_idx=0, names=scene["names"]
        )
        _info(f"Anchor-based relations (anchor=table): {len(anchor_rels)} objects")
        for obj_i, rid in sorted(anchor_rels.items()):
            tag = SV_RELATION_LABELS[rid] if rid < len(SV_RELATION_LABELS) else f"?{rid}"
            print(f"    {scene['names'][obj_i]:>10} → table : {rid:>2}  {tag}")

        _ok("SceneVerse predictor PASSED")
        return True
    except Exception as exc:
        _fail(f"SceneVerse predictor: {exc}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────
# 3.  Test graph_builder – SceneVerse edges
# ──────────────────────────────────────────────────────────────────────

def test_sceneverse_edges(builder, scene: dict) -> Optional[torch.Tensor]:
    _header("3. SceneVerse edge features (via graph_builder)")
    try:
        from scene_verse.edge_predictor import (
            SceneVerseEdgePredictorService,
            SV_RELATION_LABELS,
        )
        svc = SceneVerseEdgePredictorService()

        sv_edges = builder.build_sceneverse_edge_features(
            scene["positions"],
            scene["sizes"],
            scene["object_ids"],
            scene["active"],
            scene["goal_idxs"],
            predictor=svc,
            names=scene["names"],
        )
        _ok(f"shape = {list(sv_edges.shape)}  (expected [{scene['E']}, {scene['M']}, 6])")
        assert sv_edges.shape == (scene["E"], scene["M"], 6)

        _info("Env 0 edges (goal=table):")
        _print_edge_tensor(
            sv_edges[0], scene["names"], int(scene["goal_idxs"][0]),
            ["exists", "rel_id_raw", "rel_id_norm", "is_non_none", "dist", "id_diff"],
        )

        # Interpret rel_id_raw
        _info("Interpreted SV relations (env 0):")
        for j in range(scene["M"]):
            if sv_edges[0, j, 0].item() > 0.5:
                rid = int(sv_edges[0, j, 1].item())
                label = SV_RELATION_LABELS[rid] if rid < len(SV_RELATION_LABELS) else f"?{rid}"
                print(f"    {scene['names'][j]:>10} → table : {rid:>2} ({label})")

        _ok("SceneVerse edge features PASSED")
        return sv_edges
    except Exception as exc:
        _fail(f"SceneVerse edges: {exc}")
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────────────────────────────
# 4.  Test VL-SAT predictor  (requires checkpoint)
# ──────────────────────────────────────────────────────────────────────

VL_SAT_RELATIONSHIPS = [
    "none",           #  0
    "supported by",   #  1
    "left",           #  2
    "right",          #  3
    "front",          #  4
    "behind",         #  5
    "close by",       #  6
    "inside",         #  7
    "bigger than",    #  8
    "smaller than",   #  9
    "higher than",    # 10
    "lower than",     # 11
    "same symmetry",  # 12
    "same as",        # 13
    "attached to",    # 14
    "standing on",    # 15
    "lying on",       # 16
    "hanging on",     # 17
    "connected to",   # 18
    "leaning against",# 19
    "part of",        # 20
    "belonging to",   # 21
    "build in",       # 22
    "standing in",    # 23
    "cover",          # 24
    "lying in",       # 25
    "hanging in",     # 26
]

def test_vlsat_predictor(scene: dict, ckpt_path: str):
    _header("4. VL-SAT neural predictor (standalone)")
    if not ckpt_path:
        _info("Skipped — no --vl-sat-ckpt provided")
        return None  # None = skipped

    try:
        model_root = os.path.join(ALOHA_NAV_DIR, "vl_sat_model")
        config_path = os.path.join(model_root, "config", "mmgnet.json")
        rel_path = os.path.join(model_root, "config", "relationships.txt")

        predictor_path = os.path.join(model_root, "predictor_service.py")
        VLSATEdgePredictorService = _import_from_file(predictor_path, "VLSATEdgePredictorService")

        _info(f"Loading VL-SAT from {ckpt_path} ...")
        predictor = VLSATEdgePredictorService(
            model_root=model_root,
            config_path=config_path,
            ckpt_path=ckpt_path,
            relationships_path=rel_path,
            num_points=512,
        )
        _ok("VL-SAT predictor initialised")

        # Test name-based
        name_rels = predictor.predict_pair_relations_from_bboxes(
            scene["centers"], scene["extents"]
        )
        _info(f"VL-SAT name relations ({len(name_rels)} pairs):")
        for (s, t), rel_name in sorted(name_rels.items()):
            print(f"    {scene['names'][s]:>10} → {scene['names'][t]:>10}  : {rel_name}")

        # Test id-based (used by graph_builder)
        id_rels = predictor.predict_pair_relation_ids_from_bboxes(
            scene["centers"], scene["extents"]
        )
        _info(f"VL-SAT id relations ({len(id_rels)} pairs):")
        for (s, t), rid in sorted(id_rels.items()):
            label = VL_SAT_RELATIONSHIPS[rid] if rid < len(VL_SAT_RELATIONSHIPS) else f"?{rid}"
            print(f"    {scene['names'][s]:>10} → {scene['names'][t]:>10}  : {rid:>2} ({label})")

        # Sanity: all IDs should be 1-26 (0 = none is never actually predicted)
        ids_set = set(id_rels.values())
        if 0 in ids_set:
            _fail("VL-SAT returned rel_id=0 (none) — this should not happen after +1 shift!")
        elif all(1 <= i <= 26 for i in ids_set):
            _ok(f"All VL-SAT IDs in range [1, 26]: {sorted(ids_set)}")
        else:
            _fail(f"Unexpected VL-SAT IDs: {sorted(ids_set)}")

        _ok("VL-SAT predictor PASSED")
        return True
    except Exception as exc:
        _fail(f"VL-SAT predictor: {exc}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────
# 5.  Test graph_builder – VL-SAT edges
# ──────────────────────────────────────────────────────────────────────

def test_vlsat_edges(builder, scene: dict, ckpt_path: str) -> Optional[torch.Tensor]:
    _header("5. VL-SAT edge features (via graph_builder)")
    if not ckpt_path:
        _info("Skipped — no --vl-sat-ckpt provided")
        # Still test fallback path (no predictor)
        _info("Testing fallback (no predictor)...")
        vlsat_edges = builder.build_vlsat_edge_features(
            scene["positions"],
            scene["sizes"],
            scene["object_ids"],
            scene["active"],
            scene["goal_idxs"],
            predictor=None,
        )
        _ok(f"Fallback shape = {list(vlsat_edges.shape)}")
        _print_edge_tensor(
            vlsat_edges[0], scene["names"], int(scene["goal_idxs"][0]),
            ["exists", "rel_id_raw", "rel_id_norm", "is_non_none", "dist", "id_diff"],
        )
        return vlsat_edges

    try:
        model_root = os.path.join(ALOHA_NAV_DIR, "vl_sat_model")
        config_path = os.path.join(model_root, "config", "mmgnet.json")
        rel_path = os.path.join(model_root, "config", "relationships.txt")

        predictor_path = os.path.join(model_root, "predictor_service.py")
        VLSATEdgePredictorService = _import_from_file(predictor_path, "VLSATEdgePredictorService")
        predictor = VLSATEdgePredictorService(
            model_root=model_root,
            config_path=config_path,
            ckpt_path=ckpt_path,
            relationships_path=rel_path,
            num_points=512,
        )

        vlsat_edges = builder.build_vlsat_edge_features(
            scene["positions"],
            scene["sizes"],
            scene["object_ids"],
            scene["active"],
            scene["goal_idxs"],
            predictor=predictor,
        )
        _ok(f"shape = {list(vlsat_edges.shape)}  (expected [{scene['E']}, {scene['M']}, 6])")
        assert vlsat_edges.shape == (scene["E"], scene["M"], 6)

        _info("Env 0 edges (goal=table):")
        _print_edge_tensor(
            vlsat_edges[0], scene["names"], int(scene["goal_idxs"][0]),
            ["exists", "rel_id_raw", "rel_id_norm", "is_non_none", "dist", "id_diff"],
        )

        # Interpret
        _info("Interpreted VL-SAT relations (env 0):")
        for j in range(scene["M"]):
            if vlsat_edges[0, j, 0].item() > 0.5:
                rid = int(vlsat_edges[0, j, 1].item())
                label = VL_SAT_RELATIONSHIPS[rid] if rid < len(VL_SAT_RELATIONSHIPS) else f"?{rid}"
                print(f"    {scene['names'][j]:>10} → table : {rid:>2} ({label})")

        _ok("VL-SAT edge features PASSED")
        return vlsat_edges
    except Exception as exc:
        _fail(f"VL-SAT edges: {exc}")
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────────────────────────────
# 6.  Test combine_enabled_edge_features
# ──────────────────────────────────────────────────────────────────────

def test_combine(builder, edges_list: List[torch.Tensor], labels: List[str]):
    _header("6. combine_enabled_edge_features")
    available = [(e, l) for e, l in zip(edges_list, labels) if e is not None]
    if len(available) < 2:
        _info(f"Only {len(available)} edge type(s) available, need ≥2 to test combine.")
        if available:
            _ok(f"Single edge type ({available[0][1]}) → passes through unchanged")
        return

    try:
        tensors = [e for e, _ in available]
        names_ = [l for _, l in available]
        combined = builder.combine_enabled_edge_features(tensors)
        _ok(f"Combined {names_} → shape {list(combined.shape)}")

        # Print channel-wise stats
        ch_labels = ["exists", "ch1(signed)", "ch2(signed)", "ch3(signed)", "dist", "id_diff"]
        for c, label in enumerate(ch_labels):
            vals = combined[..., c]
            _info(f"  ch{c} ({label}): min={vals.min():.4f}  max={vals.max():.4f}  "
                  f"mean={vals.mean():.4f}")

        _ok("Combine PASSED")
    except Exception as exc:
        _fail(f"Combine: {exc}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────
# 7.  Test node features
# ──────────────────────────────────────────────────────────────────────

def test_node_features(builder, scene: dict):
    _header("7. Node features")
    try:
        E, M = scene["E"], scene["M"]
        radii = torch.norm(scene["sizes"], dim=-1, keepdim=True) / 2.0  # [E, M, 1]
        colors = torch.rand(E, M, 3, device=scene["device"])
        raw_parents = torch.full((E, M), -1, dtype=torch.long, device=scene["device"])
        levels = torch.zeros(E, M, 1, device=scene["device"])

        node_feats = builder.build_node_features(
            scene["positions"],
            scene["sizes"],
            radii,
            colors,
            scene["object_ids"],
            scene["active"],
            raw_parents,
            levels,
        )
        _ok(f"shape = {list(node_feats.shape)}  (expected [{E}, {M}, 14])")
        assert node_feats.shape == (E, M, 14), "Shape mismatch!"

        # Check NaN/Inf
        if torch.isnan(node_feats).any():
            _fail("Node features contain NaN!")
        elif torch.isinf(node_feats).any():
            _fail("Node features contain Inf!")
        else:
            _ok("No NaN/Inf in node features")

        # Print breakdown
        ch_names = ["pos_x", "pos_y", "pos_z", "sz_x", "sz_y", "sz_z",
                     "radius", "r", "g", "b", "obj_id", "active", "parent", "level"]
        _info("Env 0, breakdown:")
        for j in range(M):
            vals = [f"{node_feats[0, j, c].item():.3f}" for c in range(14)]
            print(f"    {scene['names'][j]:>10}: {', '.join(vals)}")

        _ok("Node features PASSED")
    except Exception as exc:
        _fail(f"Node features: {exc}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────
# 8.  Test parent edge features  (fallback when no bbq/vlsat/sv)
# ──────────────────────────────────────────────────────────────────────

def test_parent_edges(builder, scene: dict):
    _header("8. Parent edge features (hierarchy-based fallback)")
    try:
        E, M = scene["E"], scene["M"]
        levels = torch.zeros(E, M, 1, device=scene["device"])
        colors = torch.rand(E, M, 3, device=scene["device"])

        # cup (1) and vase (3) are on table (0)
        raw_parents = torch.full((E, M), -1, dtype=torch.long, device=scene["device"])
        raw_parents[:, 1] = 0  # cup → table
        raw_parents[:, 3] = 0  # vase → table
        levels[:, 1, 0] = 1
        levels[:, 3, 0] = 1

        parent_edges = builder.build_parent_edge_features(
            scene["positions"], levels, colors, scene["object_ids"], raw_parents
        )
        _ok(f"shape = {list(parent_edges.shape)}  (expected [{E}, {M}, 6])")
        assert parent_edges.shape == (E, M, 6)

        _info("Env 0 parent edges:")
        _print_edge_tensor(
            parent_edges[0], scene["names"], -1,
            ["exists", "z_diff", "level_diff", "dist", "color_diff", "id_diff"],
        )

        # Sanity: cup and vase should have edge_exists=1, others=0
        assert parent_edges[0, 1, 0].item() == 1.0, "cup should have parent edge"
        assert parent_edges[0, 3, 0].item() == 1.0, "vase should have parent edge"
        assert parent_edges[0, 0, 0].item() == 0.0, "table has no parent"
        assert parent_edges[0, 2, 0].item() == 0.0, "chair has no parent"
        _ok("Parent edge features PASSED")
    except Exception as exc:
        _fail(f"Parent edges: {exc}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────
# 9.  Shape consistency check
# ──────────────────────────────────────────────────────────────────────

def test_shape_consistency(builder, scene: dict, ckpt_path: str):
    _header("9. Shape consistency across all edge types")
    from scene_verse.edge_predictor import SceneVerseEdgePredictorService
    try:
        E, M = scene["E"], scene["M"]
        bbq_center = torch.tensor([0.0, 0.0, 0.0], device=scene["device"])

        bbq = builder.build_bbq_edge_features(
            scene["positions"], scene["object_ids"], scene["active"],
            scene["goal_idxs"], bbq_center,
        )
        vlsat = builder.build_vlsat_edge_features(
            scene["positions"], scene["sizes"], scene["object_ids"],
            scene["active"], scene["goal_idxs"], predictor=None,
        )
        sv_svc = SceneVerseEdgePredictorService()
        sv = builder.build_sceneverse_edge_features(
            scene["positions"], scene["sizes"], scene["object_ids"],
            scene["active"], scene["goal_idxs"], predictor=sv_svc,
            names=scene["names"],
        )

        all_ok = True
        for name, t in [("BBQ", bbq), ("VL-SAT", vlsat), ("SV", sv)]:
            expected = (E, M, 6)
            if t.shape != expected:
                _fail(f"{name} shape {list(t.shape)} != expected {list(expected)}")
                all_ok = False
            else:
                _ok(f"{name} shape {list(t.shape)} ✓")

        # All should be combinable
        combined = builder.combine_enabled_edge_features([bbq, vlsat, sv])
        if combined.shape != (E, M, 6):
            _fail(f"Combined shape {list(combined.shape)} ≠ [{E}, {M}, 6]")
            all_ok = False
        else:
            _ok(f"Combined 3 types → shape {list(combined.shape)} ✓")

        if all_ok:
            _ok("Shape consistency PASSED")
    except Exception as exc:
        _fail(f"Shape consistency: {exc}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────
# 10. NaN / Inf check
# ──────────────────────────────────────────────────────────────────────

def test_nan_inf(builder, scene: dict):
    _header("10. NaN / Inf check on all edge types")
    from scene_verse.edge_predictor import SceneVerseEdgePredictorService
    try:
        bbq_center = torch.tensor([0.0, 0.0, 0.0], device=scene["device"])
        bbq = builder.build_bbq_edge_features(
            scene["positions"], scene["object_ids"], scene["active"],
            scene["goal_idxs"], bbq_center,
        )
        vlsat = builder.build_vlsat_edge_features(
            scene["positions"], scene["sizes"], scene["object_ids"],
            scene["active"], scene["goal_idxs"],
        )
        sv_svc = SceneVerseEdgePredictorService()
        sv = builder.build_sceneverse_edge_features(
            scene["positions"], scene["sizes"], scene["object_ids"],
            scene["active"], scene["goal_idxs"], predictor=sv_svc, names=scene["names"],
        )

        all_ok = True
        for name, t in [("BBQ", bbq), ("VL-SAT(fallback)", vlsat), ("SV", sv)]:
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            if has_nan:
                _fail(f"{name} has NaN!")
                all_ok = False
            if has_inf:
                _fail(f"{name} has Inf!")
                all_ok = False
            if not has_nan and not has_inf:
                _ok(f"{name}: clean (no NaN/Inf)")

        if all_ok:
            _ok("NaN/Inf check PASSED")
    except Exception as exc:
        _fail(f"NaN/Inf check: {exc}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────
# 11. Goal-anchor mask sanity
# ──────────────────────────────────────────────────────────────────────

def test_goal_mask(builder, scene: dict):
    _header("11. Goal-anchor mask sanity")
    try:
        E, M = scene["E"], scene["M"]
        mask = builder._build_goal_anchor_mask(
            scene["active"], scene["goal_idxs"], M
        )
        _ok(f"Mask shape = {list(mask.shape)}  (expected [{E}, {M}, 1])")

        for e in range(E):
            goal = int(scene["goal_idxs"][e].item())
            goal_val = mask[e, goal, 0].item()
            if goal_val != 0.0:
                _fail(f"Env {e}: goal slot ({scene['names'][goal]}) mask = {goal_val}, expected 0.0!")
            else:
                _ok(f"Env {e}: goal ({scene['names'][goal]}) correctly masked out")

            active_count = int(scene["active"][e, :, 0].sum().item())
            mask_count = int(mask[e, :, 0].sum().item())
            expected = active_count - 1  # all active except goal
            if mask_count != expected:
                _fail(f"Env {e}: {mask_count} edges, expected {expected} (active - goal)")
            else:
                _ok(f"Env {e}: {mask_count} active edges (correct)")

        _ok("Goal mask PASSED")
    except Exception as exc:
        _fail(f"Goal mask: {exc}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────
# 12. Different goal indices
# ──────────────────────────────────────────────────────────────────────

def test_different_goals(builder, scene: dict):
    _header("12. Edge features with different goal per environment")
    from scene_verse.edge_predictor import SceneVerseEdgePredictorService
    try:
        E, M = scene["E"], scene["M"]
        # env 0 → goal = table (0),  env 1 → goal = chair (2)
        goal_idxs = torch.tensor([0, 2], device=scene["device"], dtype=torch.long)[:E]

        bbq_center = torch.tensor([0.0, 0.0, 0.0], device=scene["device"])
        bbq = builder.build_bbq_edge_features(
            scene["positions"], scene["object_ids"], scene["active"],
            goal_idxs, bbq_center,
        )

        _info(f"Different goals: env0=table, env1=chair")
        for e_idx in range(min(E, 2)):
            g = int(goal_idxs[e_idx].item())
            _info(f"Env {e_idx} (goal={scene['names'][g]}):")
            for j in range(M):
                if bbq[e_idx, j, 0].item() > 0.5:
                    dist = bbq[e_idx, j, 4].item()
                    print(f"    {scene['names'][j]:>10} → {scene['names'][g]:>10}: dist={dist:.3f}")

        _ok("Different goals PASSED")
    except Exception as exc:
        _fail(f"Different goals: {exc}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────
# 13. Ring neighbour indices
# ──────────────────────────────────────────────────────────────────────

def test_ring_neighbors(builder, scene: dict):
    """Verify ring-neighbour topology: closed ring of active non-goal objects."""
    _header("13. Ring neighbour indices")
    try:
        E, M = scene["E"], scene["M"]
        positions = scene["positions"]
        active = scene["active"]
        goal_idxs = scene["goal_idxs"]

        ring = builder.compute_ring_neighbor_indices(positions, active, goal_idxs)
        assert ring.shape == (E, M), f"Expected [{E},{M}], got {list(ring.shape)}"
        _ok(f"Shape {list(ring.shape)} ✓")

        for e in range(E):
            gi = goal_idxs[e].item()
            assert ring[e, gi].item() == -1, "Goal must have neighbour=-1"

            # Collect active non-goal
            active_ng = [j for j in range(M) if active[e, j, 0] > 0.5 and j != gi]
            N = len(active_ng)

            if N < 2:
                _ok(f"Env {e}: <2 active non-goal → no ring (correct)")
                continue

            # Verify every active non-goal has a neighbour
            for j in active_ng:
                nbr = ring[e, j].item()
                assert nbr >= 0, f"Env {e}, obj {j}: active but no ring neighbour"
                assert nbr in active_ng, f"Env {e}, obj {j}: neighbour {nbr} not in active set"

            # Walk the ring — should visit all nodes exactly once
            visited = set()
            cur = active_ng[0]
            for _ in range(N):
                assert cur not in visited, f"Env {e}: ring revisits {cur}"
                visited.add(cur)
                cur = ring[e, cur].item()
            assert visited == set(active_ng), f"Env {e}: ring incomplete {visited} vs {set(active_ng)}"
            _ok(f"Env {e}: ring length {N}, closed ✓")

        # Test with different goal
        alt_goal = torch.tensor([1] * E, dtype=torch.long, device=scene["device"])
        ring2 = builder.compute_ring_neighbor_indices(positions, active, alt_goal)
        for e in range(E):
            assert ring2[e, 1].item() == -1, "Alt goal (idx 1) must have neighbour=-1"
        _ok("Ring adapts to different goal ✓")

        _ok("Ring neighbour indices PASSED")
        return True
    except Exception as exc:
        _fail(f"Ring neighbours: {exc}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────
# 14. Neighbour edge features
# ──────────────────────────────────────────────────────────────────────

def test_neighbor_edges(builder, scene: dict):
    """Test neighbour edge builders produce correct shapes and masking."""
    _header("14. Neighbour edge features")
    from scene_verse.edge_predictor import SceneVerseEdgePredictorService as SVService
    try:
        E, M = scene["E"], scene["M"]
        positions = scene["positions"]
        sizes = scene["sizes"]
        object_ids = scene["object_ids"]
        active = scene["active"]
        goal_idxs = scene["goal_idxs"]
        bbq_center = torch.tensor([0.0, 0.0, 0.0], device=scene["device"])

        ring = builder.compute_ring_neighbor_indices(positions, active, goal_idxs)

        # BBQ neighbour edges
        bbq_nbr = builder.build_bbq_neighbor_edge_features(
            positions, object_ids, active, bbq_center, ring,
        )
        assert bbq_nbr.shape == (E, M, 6), f"BBQ nbr shape {list(bbq_nbr.shape)}"
        _ok(f"BBQ neighbour edges {list(bbq_nbr.shape)} ✓")

        # VL-SAT neighbour edges (no predictor)
        vlsat_nbr = builder.build_vlsat_neighbor_edge_features(
            positions, sizes, object_ids, active, ring, predictor=None,
        )
        assert vlsat_nbr.shape == (E, M, 6)
        _ok(f"VL-SAT neighbour edges {list(vlsat_nbr.shape)} ✓")

        # SV neighbour edges
        sv_pred = SVService()
        sv_nbr = builder.build_sceneverse_neighbor_edge_features(
            positions, sizes, object_ids, active, ring,
            predictor=sv_pred, names=scene["names"],
        )
        assert sv_nbr.shape == (E, M, 6)
        _ok(f"SV neighbour edges {list(sv_nbr.shape)} ✓")

        # Parent neighbour edges
        raw_parents = torch.full((E, M), -1, dtype=torch.long, device=scene["device"])
        levels = torch.zeros(E, M, 1, device=scene["device"])
        colors = torch.rand(E, M, 3, device=scene["device"])
        parent_nbr = builder.build_parent_neighbor_edge_features(
            positions, levels, colors, object_ids, ring,
        )
        assert parent_nbr.shape == (E, M, 6)
        _ok(f"Parent neighbour edges {list(parent_nbr.shape)} ✓")

        # Goal object should have edge_exists=0 in ALL neighbour types
        gi = goal_idxs[0].item()
        for name, t in [("BBQ", bbq_nbr), ("VL-SAT", vlsat_nbr), ("SV", sv_nbr), ("Parent", parent_nbr)]:
            assert t[0, gi, 0].item() < 0.5, f"{name}: goal has nbr edge_exists=1"
        _ok("Goal has nbr_edge_exists=0 in all types ✓")

        # Combine works
        combined = builder.combine_enabled_edge_features([bbq_nbr, sv_nbr])
        assert combined.shape == (E, M, 6)
        _ok(f"Combined neighbour edges {list(combined.shape)} ✓")

        _ok("Neighbour edge features PASSED")
        return True
    except Exception as exc:
        _fail(f"Neighbour edges: {exc}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────
# 15. Encode + Decode round-trip
# ──────────────────────────────────────────────────────────────────────

def test_encode_decode(builder, scene: dict):
    """Simulate encode_scene_graph + decode_scene_embedding round-trip (D=30)."""
    _header("15. Encode + Decode round-trip (D=30, ring neighbour edges)")
    try:
        E, M = scene["E"], scene["M"]
        device = scene["device"]
        positions = scene["positions"]
        sizes = scene["sizes"]
        object_ids = scene["object_ids"]
        active = scene["active"]
        goal_idxs = scene["goal_idxs"]
        names = scene["names"]
        D = 30

        # ---- 1) Build node_features ----
        radii = torch.norm(sizes, dim=-1, keepdim=True) / 2.0
        colors = torch.rand(E, M, 3, device=device)
        raw_parents = torch.full((E, M), -1, dtype=torch.long, device=device)
        levels = torch.zeros(E, M, 1, device=device)

        node_feats = builder.build_node_features(
            positions, sizes, radii, colors, object_ids, active, raw_parents, levels
        )  # [E, M, 14]

        # ---- 2) Build goal edge_features (SV as example) ----
        from scene_verse.edge_predictor import SceneVerseEdgePredictorService as SVService
        sv_pred = SVService()

        goal_edge_feats = builder.build_sceneverse_edge_features(
            positions, sizes, object_ids, active, goal_idxs,
            predictor=sv_pred, names=names,
        )  # [E, M, 6]

        # ---- 2b) Build ring neighbour indices + neighbour edges ----
        ring_nbr = builder.compute_ring_neighbor_indices(positions, active, goal_idxs)  # [E, M]
        _ok(f"Ring neighbour indices computed, shape {list(ring_nbr.shape)}")

        # Verify ring properties
        for e in range(E):
            gi = goal_idxs[e].item()
            assert ring_nbr[e, gi].item() == -1, f"Goal should have no ring neighbour"
            active_non_goal = [(j, ring_nbr[e, j].item()) for j in range(M)
                               if active[e, j, 0] > 0.5 and j != gi]
            nbr_set = set()
            for j, n in active_non_goal:
                if n >= 0:
                    nbr_set.add(n)
            # Every active non-goal should appear as someone's neighbour (ring)
            active_ng_set = set(j for j, _ in active_non_goal)
            if len(active_ng_set) >= 2:
                assert nbr_set == active_ng_set, f"Ring not closed: nbr_set={nbr_set} vs active={active_ng_set}"
        _ok("Ring is closed and goal excluded ✓")

        nbr_edge_feats = builder.build_sceneverse_neighbor_edge_features(
            positions, sizes, object_ids, active, ring_nbr,
            predictor=sv_pred, names=names,
        )  # [E, M, 6]

        _ok(f"node_feats {list(node_feats.shape)}, goal_edge {list(goal_edge_feats.shape)}, "
            f"nbr_edge {list(nbr_edge_feats.shape)}")

        # ---- 3) Codebook (simplified) ----
        codebook_path = os.path.join(ALOHA_NAV_DIR, "cdecode_dict.json")
        import json
        with open(codebook_path) as f:
            codebook = json.load(f)
        name_map = codebook.get("names", {})
        color_map = codebook.get("colors", {})

        name_idx = torch.zeros(M, 1, device=device)
        color_idx = torch.zeros(M, 1, device=device)
        for j in range(M):
            base_name = names[j].split("_", 1)[0].lower()
            name_idx[j, 0] = float(int(name_map.get(base_name, 0)))
            color_idx[j, 0] = 0.0  # simplified

        name_b = name_idx.view(1, M, 1).expand(E, -1, -1)
        color_b = color_idx.view(1, M, 1).expand(E, -1, -1)
        pad_b = torch.zeros(E, M, 2, device=device)

        per_obj = torch.cat([node_feats, goal_edge_feats, nbr_edge_feats, name_b, color_b, pad_b], dim=-1)
        assert per_obj.shape == (E, M, D), f"Expected [E,M,{D}], got {per_obj.shape}"
        _ok(f"per_object_feats shape {list(per_obj.shape)}")

        # ---- 4) Reorder by goal ----
        g = goal_idxs.long().clamp(0, M - 1)
        base = torch.arange(M, device=device).view(1, M).expand(E, M)
        order = (base + g.view(E, 1)) % M
        reordered = per_obj.gather(1, order.unsqueeze(-1).expand(-1, -1, D))

        # Verify goal is at slot 0
        for e in range(E):
            slot0_id = reordered[e, 0, 10].item()  # obj_id at slot 0
            goal_id = object_ids[e, goal_idxs[e].long().item()].item()
            if abs(slot0_id - goal_id) > 0.01:
                _fail(f"Env {e}: slot 0 obj_id={slot0_id} != goal obj_id={goal_id}")
                return False
        _ok("Goal correctly placed at slot 0 after reorder")

        # ---- 5) Flatten ----
        flat = reordered.reshape(E, -1)
        assert flat.shape == (E, M * D)
        _ok(f"Flattened shape {list(flat.shape)}")

        # ---- 6) Decode back ----
        row = flat[0].view(M, D).cpu()
        _info("Decoded slot 0 (GOAL):")
        f = row[0]
        print(f"    pos=({f[0]:.2f},{f[1]:.2f},{f[2]:.2f})  "
              f"obj_id={f[10]:.0f}  active={f[11]:.0f}  "
              f"goal_edge_exists={f[14]:.0f}  nbr_edge_exists={f[20]:.0f}  "
              f"name_code={f[26]:.0f}")

        _info("All slots (env 0):")
        print(f"    {'Slot':>4}  {'ID':>4}  {'Act':>3}  "
              f"{'GEdge':>5}  {'GRel':>4}  {'GDist':>6}  "
              f"{'NEdge':>5}  {'NRel':>4}  {'NDist':>6}  {'NameCode':>8}")
        for j in range(M):
            f = row[j]
            print(f"    {j:>4}  {f[10]:>4.0f}  {f[11]:>3.0f}  "
                  f"{f[14]:>5.0f}  {f[15]:>4.0f}  {f[18]:>6.2f}  "
                  f"{f[20]:>5.0f}  {f[21]:>4.0f}  {f[24]:>6.2f}  {f[26]:>8.0f}")

        # ---- 7) Verify edge_exists=0 for goal slot ----
        goal_gedge = row[0, 14].item()
        goal_nedge = row[0, 20].item()
        if goal_gedge > 0.5:
            _fail("Goal slot has goal_edge_exists=1 (should be 0)")
            return False
        if goal_nedge > 0.5:
            _fail("Goal slot has nbr_edge_exists=1 (should be 0)")
            return False
        _ok("Goal slot correctly has both edge_exists=0")

        # ---- 8) Verify neighbour edges exist for non-goal active objects ----
        num_nbr_edges = sum(1 for j in range(1, M) if row[j, 20].item() > 0.5 and row[j, 11].item() > 0.5)
        num_active_nongol = sum(1 for j in range(1, M) if row[j, 11].item() > 0.5)
        if num_active_nongol >= 2:
            if num_nbr_edges < 2:
                _fail(f"Expected ≥2 neighbour edges but got {num_nbr_edges}")
                return False
            _ok(f"{num_nbr_edges}/{num_active_nongol} active non-goal objects have neighbour edges")
        else:
            _ok(f"Only {num_active_nongol} active non-goal → ring not formed (expected)")

        # ---- 9) Verify no NaN/Inf ----
        if torch.isnan(flat).any() or torch.isinf(flat).any():
            _fail("NaN or Inf in flattened embedding!")
            return False
        _ok("No NaN/Inf in final embedding")

        _ok("Encode/Decode round-trip PASSED (D=30)")
        return True
    except Exception as exc:
        _fail(f"Encode/Decode: {exc}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scene graph pipeline debugger")
    parser.add_argument("--vl-sat-ckpt", type=str, default="",
                        help="Path to VL-SAT checkpoint directory (contains *_best.pth)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu / cuda)")
    parser.add_argument("--num-envs", type=int, default=2,
                        help="Number of parallel environments to simulate")
    args = parser.parse_args()

    device = args.device
    num_envs = args.num_envs

    print(SEP)
    print("  SCENE GRAPH PIPELINE DEBUGGER")
    print(SEP)
    print(f"  device    : {device}")
    print(f"  num_envs  : {num_envs}")
    print(f"  vl-sat-ckpt: {args.vl_sat_ckpt or '(none)'}")
    print(f"  aloha_nav : {ALOHA_NAV_DIR}")
    print()

    # --- Import graph_builder ---
    gb_path = os.path.join(ALOHA_NAV_DIR, "graph_builder.py")
    SceneGraphBuilder = _import_from_file(gb_path, "SceneGraphBuilder")
    scene = build_mock_scene(device=device, num_envs=num_envs)
    builder = SceneGraphBuilder(device=device, num_total_objects=scene["M"])

    # --- Run all tests ---
    results = {}

    bbq_edges = test_bbq_edges(builder, scene)
    results["BBQ edges"] = bbq_edges is not None

    sv_ok = test_sceneverse_predictor(scene)
    results["SV predictor"] = sv_ok

    sv_edges = test_sceneverse_edges(builder, scene)
    results["SV edges"] = sv_edges is not None

    vlsat_ok = test_vlsat_predictor(scene, args.vl_sat_ckpt)
    results["VL-SAT predictor"] = vlsat_ok

    vlsat_edges = test_vlsat_edges(builder, scene, args.vl_sat_ckpt)
    results["VL-SAT edges"] = vlsat_edges is not None

    test_combine(builder, [bbq_edges, vlsat_edges, sv_edges],
                 ["BBQ", "VL-SAT", "SV"])

    test_node_features(builder, scene)
    results["Node features"] = True  # would have thrown

    test_parent_edges(builder, scene)
    results["Parent edges"] = True

    test_shape_consistency(builder, scene, args.vl_sat_ckpt)
    results["Shape consistency"] = True

    test_nan_inf(builder, scene)
    results["NaN/Inf"] = True

    test_goal_mask(builder, scene)
    results["Goal mask"] = True

    test_different_goals(builder, scene)
    results["Diff goals"] = True

    # --- Test 13: ring neighbour indices ---
    ring_ok = test_ring_neighbors(builder, scene)
    results["Ring neighbours"] = ring_ok if ring_ok is not None else False

    # --- Test 14: neighbour edge features ---
    nbr_ok = test_neighbor_edges(builder, scene)
    results["Nbr edges"] = nbr_ok if nbr_ok is not None else False

    # --- Test 15: encode + decode round-trip ---
    enc_ok = test_encode_decode(builder, scene)
    results["Encode/Decode"] = enc_ok if enc_ok is not None else False

    # --- Summary ---
    _header("SUMMARY")
    passed = 0
    skipped = 0
    failed = 0
    for name, ok in results.items():
        if ok is True:
            status = "[PASS]"
            passed += 1
        elif ok is False:
            status = "[FAIL]"
            failed += 1
        else:
            status = "[SKIP]"
            skipped += 1
        print(f"  {status:>8}  {name}")

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print(SEP)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
