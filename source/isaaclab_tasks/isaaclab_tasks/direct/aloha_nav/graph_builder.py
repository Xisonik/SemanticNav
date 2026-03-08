from __future__ import annotations

from typing import List, Optional

import torch


class SceneGraphBuilder:
    def __init__(self, device: str, num_total_objects: int):
        self.device = device
        self.num_total_objects = num_total_objects

    @staticmethod
    @torch.no_grad()
    def safe_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)

    @torch.no_grad()
    def build_node_features(
        self,
        positions: torch.Tensor,
        sizes: torch.Tensor,
        radii: torch.Tensor,
        colors: torch.Tensor,
        object_ids: torch.Tensor,
        active: torch.Tensor,
        raw_parents: torch.Tensor,
        levels: torch.Tensor,
    ) -> torch.Tensor:
        parents_feat = raw_parents.unsqueeze(-1).float()
        return torch.cat(
            [positions, sizes, radii, colors, object_ids, active, parents_feat, levels],
            dim=-1,
        )  # (E, M, 14)

    @torch.no_grad()
    def build_parent_edge_features(
        self,
        positions: torch.Tensor,
        levels: torch.Tensor,
        colors: torch.Tensor,
        object_ids: torch.Tensor,
        raw_parents: torch.Tensor,
    ) -> torch.Tensor:
        device = positions.device
        E, M, _ = positions.shape

        edge_exists = (raw_parents >= 0).float().unsqueeze(-1)
        valid_mask = (raw_parents >= 0)

        z_diff = torch.zeros(E, M, 1, device=device)
        level_diff = torch.zeros_like(z_diff)
        dist = torch.zeros_like(z_diff)
        color_diff_norm = torch.zeros_like(z_diff)
        id_diff = torch.zeros_like(z_diff)

        if valid_mask.any():
            batch_idx = torch.arange(E, device=device)[:, None].expand(-1, M)[valid_mask]
            obj_idx = torch.arange(M, device=device)[None, :].expand(E, -1)[valid_mask]
            parent_idx = raw_parents[valid_mask].long()

            z_diff[valid_mask] = positions[batch_idx, obj_idx, 2:3] - positions[batch_idx, parent_idx, 2:3]
            level_diff[valid_mask] = levels[batch_idx, obj_idx] - levels[batch_idx, parent_idx]

            child_xy = positions[batch_idx, obj_idx, :2]
            parent_xy = positions[batch_idx, parent_idx, :2]
            dist[valid_mask] = torch.norm(child_xy - parent_xy, dim=-1, keepdim=True)

            child_color = colors[batch_idx, obj_idx]
            parent_color = colors[batch_idx, parent_idx]
            color_diff_norm[valid_mask] = torch.norm(child_color - parent_color, dim=-1, keepdim=True)

            child_id = object_ids[batch_idx, obj_idx]
            parent_id = object_ids[batch_idx, parent_idx]
            id_diff[valid_mask] = child_id - parent_id

        return torch.cat([edge_exists, z_diff, level_diff, dist, color_diff_norm, id_diff], dim=-1)

    @torch.no_grad()
    def bbq_relative_components(
        self,
        target_pos: torch.Tensor,
        anchor_pos: torch.Tensor,
        center_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        world_up = torch.zeros_like(anchor_pos)
        world_up[..., 2] = 1.0

        forward = self.safe_normalize(anchor_pos - center_point)
        right = torch.linalg.cross(forward, world_up, dim=-1)

        right_norm = torch.linalg.norm(right, dim=-1, keepdim=True)
        fallback_up = torch.zeros_like(anchor_pos)
        fallback_up[..., 1] = 1.0
        right_fallback = torch.linalg.cross(forward, fallback_up, dim=-1)
        right = torch.where(right_norm > 1e-6, right, right_fallback)
        right = self.safe_normalize(right)

        up = self.safe_normalize(torch.linalg.cross(right, forward, dim=-1))
        rel = target_pos - anchor_pos

        rel_x = torch.sum(rel * right, dim=-1, keepdim=True)
        rel_y = torch.sum(rel * up, dim=-1, keepdim=True)
        rel_z = torch.sum(rel * forward, dim=-1, keepdim=True)
        return rel_x, rel_y, rel_z

    @torch.no_grad()
    def _build_goal_anchor_mask(
        self,
        active: torch.Tensor,
        goal_idxs: torch.Tensor,
        M: int,
    ) -> torch.Tensor:
        node_indices = torch.arange(M, device=active.device).view(1, M).expand(active.shape[0], M)
        valid_mask = active.squeeze(-1).bool() & (node_indices != goal_idxs.view(active.shape[0], 1))
        return valid_mask.unsqueeze(-1).float()

    @torch.no_grad()
    def build_bbq_edge_features(
        self,
        positions: torch.Tensor,
        object_ids: torch.Tensor,
        active: torch.Tensor,
        goal_idxs: torch.Tensor,
        center_point: torch.Tensor,
    ) -> torch.Tensor:
        E, M, _ = positions.shape
        batch = torch.arange(E, device=positions.device)

        goal_idxs = goal_idxs.long().clamp(0, M - 1)
        goal_pos = positions[batch, goal_idxs]
        goal_id = object_ids[batch, goal_idxs]

        target_pos = positions
        anchor_pos = goal_pos.unsqueeze(1).expand(-1, M, -1)
        center = center_point.view(1, 1, 3).expand(E, M, 3)

        rel_x, rel_y, rel_z = self.bbq_relative_components(target_pos, anchor_pos, center)

        left_right = torch.sign(rel_x)
        front_back = torch.sign(rel_z)
        above_below = torch.sign(rel_y)

        valid_mask_f = self._build_goal_anchor_mask(active, goal_idxs, M)

        edge_exists = valid_mask_f
        left_right = left_right * valid_mask_f
        front_back = front_back * valid_mask_f
        above_below = above_below * valid_mask_f
        dist = torch.linalg.norm(target_pos - anchor_pos, dim=-1, keepdim=True) * valid_mask_f
        id_diff = (object_ids - goal_id.unsqueeze(1)) * valid_mask_f

        return torch.cat([edge_exists, left_right, front_back, above_below, dist, id_diff], dim=-1)

    @torch.no_grad()
    def build_vlsat_edge_features(
        self,
        positions: torch.Tensor,
        sizes: torch.Tensor,
        object_ids: torch.Tensor,
        active: torch.Tensor,
        goal_idxs: torch.Tensor,
        predictor: Optional[object] = None,
    ) -> torch.Tensor:
        E, M, _ = positions.shape
        batch = torch.arange(E, device=positions.device)

        goal_idxs = goal_idxs.long().clamp(0, M - 1)
        goal_pos = positions[batch, goal_idxs]
        goal_id = object_ids[batch, goal_idxs]

        rel_id_raw = torch.zeros(E, M, 1, device=positions.device)
        rel_id_norm = torch.zeros(E, M, 1, device=positions.device)
        rel_is_non_none = torch.zeros(E, M, 1, device=positions.device)

        # Model outputs 26 classes (0-25, no "none").  predictor_service
        # shifts them +1 so that 0 = "none / not-computed", 1-26 = real.
        max_rel_id = 26.0

        if predictor is not None:
            for e in range(E):
                active_mask = active[e, :, 0] > 0.5
                if active_mask.sum().item() < 2:
                    continue

                idx_active = torch.nonzero(active_mask, as_tuple=False).view(-1)
                centers = positions[e, idx_active].detach().cpu().numpy()
                extents = sizes[e, idx_active].detach().cpu().numpy()

                try:
                    pair_rel_ids = predictor.predict_pair_relation_ids_from_bboxes(centers, extents)
                except Exception:
                    pair_rel_ids = {}

                local_goal = torch.nonzero(idx_active == goal_idxs[e], as_tuple=False)
                if local_goal.numel() == 0:
                    continue
                g_local = int(local_goal[0, 0].item())

                for local_i, global_i in enumerate(idx_active.tolist()):
                    if global_i == int(goal_idxs[e].item()):
                        continue

                    rel_id = float(pair_rel_ids.get((local_i, g_local), 0))
                    rel_id_raw[e, global_i, 0] = rel_id
                    rel_id_norm[e, global_i, 0] = rel_id / max_rel_id
                    rel_is_non_none[e, global_i, 0] = 1.0 if rel_id > 0 else 0.0
        else:
            rel = positions - goal_pos.unsqueeze(1)
            rel_id_raw = torch.zeros_like(rel[..., 0:1])
            rel_id_norm = torch.zeros_like(rel[..., 0:1])
            rel_is_non_none = (torch.linalg.norm(rel, dim=-1, keepdim=True) > 0).float()

        valid_mask_f = self._build_goal_anchor_mask(active, goal_idxs, M)

        edge_exists = valid_mask_f
        rel_id_raw = rel_id_raw * valid_mask_f
        rel_id_norm = rel_id_norm * valid_mask_f
        rel_is_non_none = rel_is_non_none * valid_mask_f
        rel = positions - goal_pos.unsqueeze(1)
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True) * valid_mask_f
        id_diff = (object_ids - goal_id.unsqueeze(1)) * valid_mask_f

        return torch.cat([edge_exists, rel_id_raw, rel_id_norm, rel_is_non_none, dist, id_diff], dim=-1)

    @torch.no_grad()
    def build_sceneverse_edge_features(
        self,
        positions: torch.Tensor,
        sizes: torch.Tensor,
        object_ids: torch.Tensor,
        active: torch.Tensor,
        goal_idxs: torch.Tensor,
        predictor: Optional[object] = None,
        names: Optional[list] = None,
    ) -> torch.Tensor:
        """Build SV edge features using SceneVerse heuristic predictor.

        Encoding (same 6-channel layout as VL-SAT edges):
            [edge_exists, rel_id_raw, rel_id_norm, rel_is_non_none, dist, id_diff]

        ``rel_id_raw``  – integer class id in [0, 9] as float.
        ``rel_id_norm`` – class id normalised to [0, 1].
        ``rel_is_non_none`` – 1.0 if relation is not "none", else 0.0.
        """
        E, M, _ = positions.shape
        batch = torch.arange(E, device=positions.device)

        goal_idxs = goal_idxs.long().clamp(0, M - 1)
        goal_pos = positions[batch, goal_idxs]
        goal_id = object_ids[batch, goal_idxs]

        rel_id_raw = torch.zeros(E, M, 1, device=positions.device)
        rel_id_norm = torch.zeros(E, M, 1, device=positions.device)
        rel_is_non_none = torch.zeros(E, M, 1, device=positions.device)

        max_rel_id = 9.0  # NUM_SV_RELATIONS - 1

        if predictor is not None:
            for e in range(E):
                active_mask = active[e, :, 0] > 0.5
                if active_mask.sum().item() < 2:
                    continue

                idx_active = torch.nonzero(active_mask, as_tuple=False).view(-1)
                centers = positions[e, idx_active].detach().cpu().numpy()
                extents = sizes[e, idx_active].detach().cpu().numpy()

                obj_names = None
                if names is not None:
                    obj_names = [names[g] for g in idx_active.tolist()]

                # Find local index of goal within active objects
                local_goal = torch.nonzero(idx_active == goal_idxs[e], as_tuple=False)
                if local_goal.numel() == 0:
                    continue
                g_local = int(local_goal[0, 0].item())

                try:
                    obj_rels = predictor.predict_pair_relation_ids_for_anchor(
                        centers, extents, g_local, obj_names
                    )
                except Exception:
                    obj_rels = {}

                for local_i, global_i in enumerate(idx_active.tolist()):
                    if global_i == int(goal_idxs[e].item()):
                        continue

                    rel_id = float(obj_rels.get(local_i, 0))
                    rel_id_raw[e, global_i, 0] = rel_id
                    rel_id_norm[e, global_i, 0] = rel_id / max_rel_id
                    rel_is_non_none[e, global_i, 0] = 1.0 if rel_id > 0 else 0.0
        else:
            # Fallback: use simple geometric heuristic (no predictor)
            rel = positions - goal_pos.unsqueeze(1)
            rel_id_raw = torch.zeros_like(rel[..., 0:1])
            rel_id_norm = torch.zeros_like(rel[..., 0:1])
            rel_is_non_none = (torch.linalg.norm(rel, dim=-1, keepdim=True) > 0).float()

        valid_mask_f = self._build_goal_anchor_mask(active, goal_idxs, M)

        edge_exists = valid_mask_f
        rel_id_raw = rel_id_raw * valid_mask_f
        rel_id_norm = rel_id_norm * valid_mask_f
        rel_is_non_none = rel_is_non_none * valid_mask_f
        rel = positions - goal_pos.unsqueeze(1)
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True) * valid_mask_f
        id_diff = (object_ids - goal_id.unsqueeze(1)) * valid_mask_f

        return torch.cat([edge_exists, rel_id_raw, rel_id_norm, rel_is_non_none, dist, id_diff], dim=-1)

    @torch.no_grad()
    def combine_enabled_edge_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        if len(features) == 1:
            return features[0]

        stacked = torch.stack(features, dim=0)  # [K,E,M,6]
        exists = torch.max(stacked[..., 0:1], dim=0).values

        signed = torch.sign(torch.sum(stacked[..., 1:4], dim=0))

        dist_num = torch.sum(stacked[..., 4:5], dim=0)
        id_num = torch.sum(stacked[..., 5:6], dim=0)
        count = torch.clamp(torch.sum(stacked[..., 0:1], dim=0), min=1.0)
        dist = dist_num / count
        id_diff = id_num / count

        return torch.cat([exists, signed, dist, id_diff], dim=-1)

    # ------------------------------------------------------------------
    # Ring-neighbour topology
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_ring_neighbor_indices(
        self,
        positions: torch.Tensor,   # [E, M, 3]
        active: torch.Tensor,      # [E, M, 1]
        goal_idxs: torch.Tensor,   # [E]
    ) -> torch.Tensor:
        """Compute a ring ordering of active non-goal objects.

        Active non-goal objects are sorted by ``atan2(dy, dx)`` relative to
        the goal position.  Object *i* in the sorted ring connects to object
        ``(i+1) % N``.  Goal and inactive objects receive ``-1`` (no
        neighbour).

        Returns:
            ``[E, M]`` long tensor – ring neighbour global index (or -1).
        """
        E, M, _ = positions.shape
        device = positions.device
        batch = torch.arange(E, device=device)

        goal_idxs = goal_idxs.long().clamp(0, M - 1)
        goal_pos = positions[batch, goal_idxs]  # [E, 3]

        neighbor_indices = torch.full((E, M), -1, dtype=torch.long, device=device)

        for e in range(E):
            active_mask = active[e, :, 0] > 0.5
            gi = goal_idxs[e].item()

            non_goal_mask = active_mask.clone()
            non_goal_mask[gi] = False
            idx_active = torch.nonzero(non_goal_mask, as_tuple=False).view(-1)

            if len(idx_active) < 2:
                continue

            # angular sort around goal
            delta = positions[e, idx_active, :2] - goal_pos[e, :2].unsqueeze(0)
            angles = torch.atan2(delta[:, 1], delta[:, 0])
            sorted_order = torch.argsort(angles)
            sorted_indices = idx_active[sorted_order]

            N = len(sorted_indices)
            for i in range(N):
                neighbor_indices[e, sorted_indices[i].item()] = sorted_indices[(i + 1) % N].item()

        return neighbor_indices

    # ------------------------------------------------------------------
    # Neighbour-edge builders  (same 6-ch layout as goal-edge builders)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_bbq_neighbor_edge_features(
        self,
        positions: torch.Tensor,
        object_ids: torch.Tensor,
        active: torch.Tensor,
        center_point: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """BBQ directional edge from each object to its ring neighbour."""
        E, M, _ = positions.shape
        device = positions.device

        has_nbr = (neighbor_indices >= 0)  # [E, M]
        safe_nbr = neighbor_indices.clamp(min=0)
        batch_idx = torch.arange(E, device=device).unsqueeze(1).expand(E, M)

        nbr_pos = positions[batch_idx, safe_nbr]   # [E, M, 3]
        nbr_id = object_ids[batch_idx, safe_nbr]   # [E, M, 1]

        center = center_point.view(1, 1, 3).expand(E, M, 3)
        rel_x, rel_y, rel_z = self.bbq_relative_components(positions, nbr_pos, center)

        mask = has_nbr.unsqueeze(-1).float()
        edge_exists = mask
        left_right = torch.sign(rel_x) * mask
        front_back = torch.sign(rel_z) * mask
        above_below = torch.sign(rel_y) * mask
        dist = torch.linalg.norm(positions - nbr_pos, dim=-1, keepdim=True) * mask
        id_diff = (object_ids - nbr_id) * mask

        return torch.cat([edge_exists, left_right, front_back, above_below, dist, id_diff], dim=-1)

    @torch.no_grad()
    def build_vlsat_neighbor_edge_features(
        self,
        positions: torch.Tensor,
        sizes: torch.Tensor,
        object_ids: torch.Tensor,
        active: torch.Tensor,
        neighbor_indices: torch.Tensor,
        predictor: Optional[object] = None,
    ) -> torch.Tensor:
        """VL-SAT relation edge from each object to its ring neighbour."""
        E, M, _ = positions.shape
        device = positions.device

        has_nbr = (neighbor_indices >= 0)
        safe_nbr = neighbor_indices.clamp(min=0)
        batch_idx = torch.arange(E, device=device).unsqueeze(1).expand(E, M)
        nbr_pos = positions[batch_idx, safe_nbr]
        nbr_id = object_ids[batch_idx, safe_nbr]

        rel_id_raw = torch.zeros(E, M, 1, device=device)
        rel_id_norm = torch.zeros(E, M, 1, device=device)
        rel_is_non_none = torch.zeros(E, M, 1, device=device)
        max_rel_id = 26.0

        if predictor is not None:
            for e in range(E):
                active_mask = active[e, :, 0] > 0.5
                if active_mask.sum().item() < 2:
                    continue

                idx_active = torch.nonzero(active_mask, as_tuple=False).view(-1)
                centers = positions[e, idx_active].detach().cpu().numpy()
                extents = sizes[e, idx_active].detach().cpu().numpy()

                try:
                    pair_rel_ids = predictor.predict_pair_relation_ids_from_bboxes(centers, extents)
                except Exception:
                    pair_rel_ids = {}

                global_to_local = {int(g): l for l, g in enumerate(idx_active.tolist())}

                for local_i, global_i in enumerate(idx_active.tolist()):
                    nbr_global = neighbor_indices[e, global_i].item()
                    if nbr_global < 0:
                        continue
                    nbr_local = global_to_local.get(nbr_global, -1)
                    if nbr_local < 0:
                        continue

                    rel_id = float(pair_rel_ids.get((local_i, nbr_local), 0))
                    rel_id_raw[e, global_i, 0] = rel_id
                    rel_id_norm[e, global_i, 0] = rel_id / max_rel_id
                    rel_is_non_none[e, global_i, 0] = 1.0 if rel_id > 0 else 0.0

        mask = has_nbr.unsqueeze(-1).float()
        edge_exists = mask
        rel_id_raw = rel_id_raw * mask
        rel_id_norm = rel_id_norm * mask
        rel_is_non_none = rel_is_non_none * mask
        dist = torch.linalg.norm(positions - nbr_pos, dim=-1, keepdim=True) * mask
        id_diff = (object_ids - nbr_id) * mask

        return torch.cat([edge_exists, rel_id_raw, rel_id_norm, rel_is_non_none, dist, id_diff], dim=-1)

    @torch.no_grad()
    def build_sceneverse_neighbor_edge_features(
        self,
        positions: torch.Tensor,
        sizes: torch.Tensor,
        object_ids: torch.Tensor,
        active: torch.Tensor,
        neighbor_indices: torch.Tensor,
        predictor: Optional[object] = None,
        names: Optional[list] = None,
    ) -> torch.Tensor:
        """SceneVerse relation edge from each object to its ring neighbour."""
        E, M, _ = positions.shape
        device = positions.device

        has_nbr = (neighbor_indices >= 0)
        safe_nbr = neighbor_indices.clamp(min=0)
        batch_idx = torch.arange(E, device=device).unsqueeze(1).expand(E, M)
        nbr_pos = positions[batch_idx, safe_nbr]
        nbr_id = object_ids[batch_idx, safe_nbr]

        rel_id_raw = torch.zeros(E, M, 1, device=device)
        rel_id_norm = torch.zeros(E, M, 1, device=device)
        rel_is_non_none = torch.zeros(E, M, 1, device=device)
        max_rel_id = 9.0

        if predictor is not None:
            for e in range(E):
                active_mask = active[e, :, 0] > 0.5
                if active_mask.sum().item() < 2:
                    continue

                idx_active = torch.nonzero(active_mask, as_tuple=False).view(-1)
                centers = positions[e, idx_active].detach().cpu().numpy()
                extents = sizes[e, idx_active].detach().cpu().numpy()

                obj_names = None
                if names is not None:
                    obj_names = [names[g] for g in idx_active.tolist()]

                global_to_local = {int(g): l for l, g in enumerate(idx_active.tolist())}

                for local_i, global_i in enumerate(idx_active.tolist()):
                    nbr_global = neighbor_indices[e, global_i].item()
                    if nbr_global < 0:
                        continue
                    nbr_local = global_to_local.get(nbr_global, -1)
                    if nbr_local < 0:
                        continue

                    try:
                        obj_rels = predictor.predict_pair_relation_ids_for_anchor(
                            centers, extents, nbr_local, obj_names
                        )
                    except Exception:
                        obj_rels = {}

                    rel_id = float(obj_rels.get(local_i, 0))
                    rel_id_raw[e, global_i, 0] = rel_id
                    rel_id_norm[e, global_i, 0] = rel_id / max_rel_id
                    rel_is_non_none[e, global_i, 0] = 1.0 if rel_id > 0 else 0.0

        mask = has_nbr.unsqueeze(-1).float()
        edge_exists = mask
        rel_id_raw = rel_id_raw * mask
        rel_id_norm = rel_id_norm * mask
        rel_is_non_none = rel_is_non_none * mask
        dist = torch.linalg.norm(positions - nbr_pos, dim=-1, keepdim=True) * mask
        id_diff = (object_ids - nbr_id) * mask

        return torch.cat([edge_exists, rel_id_raw, rel_id_norm, rel_is_non_none, dist, id_diff], dim=-1)

    @torch.no_grad()
    def build_parent_neighbor_edge_features(
        self,
        positions: torch.Tensor,
        levels: torch.Tensor,
        colors: torch.Tensor,
        object_ids: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Geometric edge (z_diff, level_diff, dist, color_diff) to ring neighbour."""
        E, M, _ = positions.shape
        device = positions.device

        has_nbr = (neighbor_indices >= 0)
        safe_nbr = neighbor_indices.clamp(min=0)
        batch_idx = torch.arange(E, device=device).unsqueeze(1).expand(E, M)

        z_diff = torch.zeros(E, M, 1, device=device)
        level_diff = torch.zeros_like(z_diff)
        dist = torch.zeros_like(z_diff)
        color_diff_norm = torch.zeros_like(z_diff)
        id_diff = torch.zeros_like(z_diff)

        if has_nbr.any():
            obj_idx = torch.arange(M, device=device).unsqueeze(0).expand(E, M)
            z_diff[has_nbr] = (positions[batch_idx, obj_idx, 2:3] - positions[batch_idx, safe_nbr, 2:3])[has_nbr]
            level_diff[has_nbr] = (levels[batch_idx, obj_idx] - levels[batch_idx, safe_nbr])[has_nbr]

            child_xy = positions[batch_idx, obj_idx, :2]
            nbr_xy = positions[batch_idx, safe_nbr, :2]
            dist[has_nbr] = torch.norm(child_xy - nbr_xy, dim=-1, keepdim=True)[has_nbr]

            child_color = colors[batch_idx, obj_idx]
            nbr_color = colors[batch_idx, safe_nbr]
            color_diff_norm[has_nbr] = torch.norm(child_color - nbr_color, dim=-1, keepdim=True)[has_nbr]

            child_id = object_ids[batch_idx, obj_idx]
            nbr_id_t = object_ids[batch_idx, safe_nbr]
            id_diff[has_nbr] = (child_id - nbr_id_t)[has_nbr]

        edge_exists = has_nbr.unsqueeze(-1).float()
        return torch.cat([edge_exists, z_diff, level_diff, dist, color_diff_norm, id_diff], dim=-1)

    @torch.no_grad()
    def decode_bbq_string_relations(self, edge_features: torch.Tensor) -> List[List[List[str]]]:
        E, M, _ = edge_features.shape
        relations: List[List[List[str]]] = []

        for e in range(E):
            env_rels: List[List[str]] = []
            for j in range(M):
                rels: List[str] = []
                exists = float(edge_features[e, j, 0].item()) > 0.5
                if exists:
                    lr = float(edge_features[e, j, 1].item())
                    fb = float(edge_features[e, j, 2].item())
                    ud = float(edge_features[e, j, 3].item())

                    if lr < 0:
                        rels.append("left")
                    elif lr > 0:
                        rels.append("right")

                    if fb < 0:
                        rels.append("front")
                    elif fb > 0:
                        rels.append("back")

                    if ud < 0:
                        rels.append("above")
                    elif ud > 0:
                        rels.append("below")

                env_rels.append(rels)
            relations.append(env_rels)

        return relations
