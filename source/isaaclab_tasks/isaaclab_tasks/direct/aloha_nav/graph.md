# Scene Graph Embedding — Technical Reference

## Overview

The scene graph encodes every object in the simulation into a fixed-size flat tensor that is fed to the RL policy as `obs["graph"]`.  The graph has a **star + ring** topology:

```
        ┌───── goal ─────┐
        │   │   │   │   │
        a   b   c   d   e      ← star (goal edges)
        │               │
        └─b─c─d─e───────┘      ← ring (neighbour edges, angular order)
```

- **Star edges** — every active non-goal object has a directed relation edge **to the goal**.
- **Ring edges** — active non-goal objects are sorted by angle (`atan2`) around the goal and linked in a cycle: `a→b→c→d→e→a`.

The goal object itself has **no edges** (both `edge_exists` channels are 0).

---

## Tensor Layout

Each object is encoded into **30 features**.  With `M = 21` total object slots the observation tensor is:

```
obs["graph"]  →  flat tensor of shape [M × 30] = [630]
```

### Per-object feature breakdown (30 channels)

| Channel | Field | Description |
|---------|-------|-------------|
| 0–2 | `pos_x, pos_y, pos_z` | Local position (env-relative) |
| 3–5 | `size_x, size_y, size_z` | Bounding box extents |
| 6 | `radius` | Half-diagonal of bbox |
| 7–9 | `R, G, B` | Object colour (float 0–1) |
| 10 | `obj_type_id` | Integer type ID (see codebook) |
| 11 | `active` | 1.0 if object is active, 0.0 if deactivated |
| 12 | `parent_idx` | Index of supporting surface (-1 if none) |
| 13 | `surface_level` | Hierarchy depth (0 = floor, 1 = on surface, …) |
| **14** | **`goal_edge_exists`** | 1.0 if this object has a valid goal edge |
| 15–17 | `goal_ch1–ch3` | Goal edge relation channels (meaning depends on mode) |
| 18 | `goal_dist` | Euclidean distance to goal |
| 19 | `goal_id_diff` | `obj_type_id − goal_type_id` |
| **20** | **`nbr_edge_exists`** | 1.0 if this object has a ring neighbour edge |
| 21–23 | `nbr_ch1–ch3` | Neighbour edge relation channels (same encoding as goal edge) |
| 24 | `nbr_dist` | Euclidean distance to ring neighbour |
| 25 | `nbr_id_diff` | `obj_type_id − neighbour_type_id` |
| 26 | `name_code` | Codebook integer for the object name |
| 27 | `color_code` | Codebook integer for the quantised colour |
| 28–29 | `pad` | Zero padding (reserved) |

---

## Edge Modes

Exactly **one** edge mode is active at a time, configured in `scene_items.json`.  The mode determines how channels 15–17 (goal edge) and 21–23 (neighbour edge) are filled.

### BBQ edges (`"bbq_edge": true`)

Directional relations computed in a local coordinate frame anchored at the goal / neighbour.

| Channel | Meaning | Values |
|---------|---------|--------|
| ch1 | left / right | −1 = left, +1 = right, 0 = aligned |
| ch2 | front / back | −1 = front, +1 = back |
| ch3 | above / below | −1 = above, +1 = below |

### VL-SAT edges (`"vl_sat_edge": true`)

Semantic relations predicted by the VL-SAT neural model (requires a checkpoint).

| Channel | Meaning | Values |
|---------|---------|--------|
| ch1 | `rel_id_raw` | Integer relation ID 0–26 (float) |
| ch2 | `rel_id_norm` | `rel_id_raw / 26.0` |
| ch3 | `rel_is_non_none` | 1.0 if relation ≠ "none" |

**VL-SAT relation labels** (index → label):

| ID | Label | ID | Label | ID | Label |
|----|-------|----|-------|----|-------|
| 0 | none | 9 | smaller than | 18 | connected to |
| 1 | supported by | 10 | higher than | 19 | leaning against |
| 2 | left | 11 | lower than | 20 | part of |
| 3 | right | 12 | same symmetry as | 21 | belonging to |
| 4 | front | 13 | same as | 22 | build in |
| 5 | behind | 14 | attached to | 23 | standing in |
| 6 | close by | 15 | standing on | 24 | cover |
| 7 | inside | 16 | lying on | 25 | lying in |
| 8 | bigger than | 17 | hanging on | 26 | hanging in |

### SceneVerse edges (`"sv_edge": true`)

Heuristic rule-based relations (no checkpoint needed).

| Channel | Meaning | Values |
|---------|---------|--------|
| ch1 | `rel_id_raw` | Integer relation ID 0–9 (float) |
| ch2 | `rel_id_norm` | `rel_id_raw / 9.0` |
| ch3 | `rel_is_non_none` | 1.0 if relation ≠ "none" |

**SceneVerse relation labels**:

| ID | Label |
|----|-------|
| 0 | none |
| 1 | supported_by |
| 2 | supports |
| 3 | embedded |
| 4 | inside |
| 5 | above |
| 6 | below |
| 7 | beside |
| 8 | near |
| 9 | far |

### Parent fallback (all edge flags `false`)

Geometric relation between child and its supporting surface.

| Channel | Meaning |
|---------|---------|
| ch1 | z-axis difference |
| ch2 | surface level difference |
| ch3 | planar distance |

---

## Ring Neighbour Ordering

Non-goal active objects are sorted by **angular position** around the goal:

```python
delta = obj_pos[:2] - goal_pos[:2]
angle = atan2(delta.y, delta.x)
# sorted ascending → ring: obj[0]→obj[1]→…→obj[N-1]→obj[0]
```

This produces a spatially coherent ring where adjacent objects in the ring are also angularly adjacent around the goal.

---

## Goal Reordering

Before flattening, all object slots are **rotated** so that the goal is always at **slot 0**:

```python
order[j] = (j + goal_index) % M
```

This means the policy always sees the goal at a fixed location in the tensor, regardless of which object is the goal.

---

## Codebook (`cdecode_dict.json`)

### Object name codes (`name_code`, channel 26)

| Code | Name | Code | Name |
|------|------|------|------|
| 0 | air | 8 | lamp |
| 1 | box | 9 | standard |
| 2 | cabinet | 10 | table |
| 3 | chair | 11 | teddy |
| 4 | clock | 12 | trashcan |
| 5 | crestwood | 13 | vase |
| 6 | desk | 14 | yucca |
| 7 | ladder | 15 | bowl |

### Colour codes (`color_code`, channel 27)

| Code | Colour | Code | Colour |
|------|--------|------|--------|
| 0 | black | 4 | orange |
| 1 | brown | 5 | red |
| 2 | gray | 6 | yellow |
| 3 | green | 7 | white |

---

## Configuration (`scene_items.json`)

Top-level keys that control the graph:

```jsonc
{
  "bbq_edge": false,          // enable BBQ directional edges
  "vl_sat_edge": true,        // enable VL-SAT neural edges
  "sv_edge": false,            // enable SceneVerse heuristic edges
  "bbq_center_point": [0,0,0], // reference origin for BBQ frame
  "vl_sat_ckpt_path": "",      // path to VL-SAT checkpoint dir
  // ... object definitions ...
}
```

> **Only one** of `bbq_edge`, `vl_sat_edge`, `sv_edge` should be `true`.  
> If all are `false`, the parent-hierarchy fallback is used.

---

## Key API Calls

### Build and get the graph observation

```python
# In aloha_env._reset_idx():
self.scene_embeddings[env_ids] = self.scene_manager.encode_scene_graph(env_ids)
# Returns: [len(env_ids), 630]  (30 × 21 objects)
```

### Access in the policy

```python
obs["graph"]  # shape: [num_envs, 630]
```

### Decode for debugging

```python
# From within aloha_env (e.g. in a breakpoint or callback):
self.debug_scene_graph_embedding(env_id=0)

# Or directly:
self.scene_manager.decode_scene_embedding(
    self.scene_embeddings, env_idx=0, verbose=True
)
```

This prints a human-readable table showing each slot's name, goal edge relation, neighbour edge relation, distances, and codebook codes.

### Get raw graph components

```python
obs_dict = self.scene_manager.get_graph_obs(env_ids)
# Returns dict with:
#   "node_features"           → [E, M, 14]
#   "edge_features"           → [E, M, 6]   (goal edges)
#   "neighbor_edge_features"  → [E, M, 6]   (ring neighbour edges)
```

---

## Lifecycle

1. **Scene reset** (`_reset_idx`) — objects are randomised, then `encode_scene_graph()` is called **once**.
2. **During episode** — `scene_embeddings` is **static** (the scene doesn't change mid-episode).
3. **Policy reads** `obs["graph"]` every step — always the same tensor until next reset.

---

## File Map

| File | Responsibility |
|------|---------------|
| `graph_builder.py` | All tensor-level edge/node builders, ring computation |
| `scene_manager.py` | `SceneManager` orchestrates graph build, encode, decode |
| `aloha_env.py` | Stores `scene_embeddings`, exposes `obs["graph"]` |
| `scene_items.json` | Scene & edge configuration |
| `cdecode_dict.json` | Name/colour codebook |
| `vl_sat_model/` | VL-SAT neural predictor (config, weights, service) |
| `scene_verse/` | SceneVerse heuristic predictor |
| `debug_scene_graph.py` | Standalone test suite (15 tests) |

---

## Running the Test Suite

```bash
cd SemanticNav
python source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/debug_scene_graph.py

# With VL-SAT checkpoint:
python source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/debug_scene_graph.py \
    --vl-sat-ckpt path/to/checkpoint_dir
```

Expected output: **13 passed, 0 failed, 1 skipped** (VL-SAT tests skip without checkpoint).
