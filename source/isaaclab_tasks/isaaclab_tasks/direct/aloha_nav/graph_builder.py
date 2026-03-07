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

        left_right = torch.zeros(E, M, 1, device=positions.device)
        front_back = torch.zeros(E, M, 1, device=positions.device)
        above_below = torch.zeros(E, M, 1, device=positions.device)

        if predictor is not None:
            for e in range(E):
                active_mask = active[e, :, 0] > 0.5
                if active_mask.sum().item() < 2:
                    continue

                idx_active = torch.nonzero(active_mask, as_tuple=False).view(-1)
                centers = positions[e, idx_active].detach().cpu().numpy()
                extents = sizes[e, idx_active].detach().cpu().numpy()

                try:
                    pair_rel = predictor.predict_pair_relations_from_bboxes(centers, extents)
                except Exception:
                    pair_rel = {}

                local_goal = torch.nonzero(idx_active == goal_idxs[e], as_tuple=False)
                if local_goal.numel() == 0:
                    continue
                g_local = int(local_goal[0, 0].item())

                for local_i, global_i in enumerate(idx_active.tolist()):
                    if global_i == int(goal_idxs[e].item()):
                        continue

                    rel_name = pair_rel.get((local_i, g_local), "none").strip().lower()

                    if rel_name == "left":
                        left_right[e, global_i, 0] = -1.0
                    elif rel_name == "right":
                        left_right[e, global_i, 0] = 1.0

                    if rel_name == "front":
                        front_back[e, global_i, 0] = -1.0
                    elif rel_name in {"behind", "back"}:
                        front_back[e, global_i, 0] = 1.0

                    if rel_name in {"higher than", "above"}:
                        above_below[e, global_i, 0] = -1.0
                    elif rel_name in {"lower than", "below"}:
                        above_below[e, global_i, 0] = 1.0
        else:
            rel = positions - goal_pos.unsqueeze(1)
            left_right = torch.sign(rel[..., 0:1])
            front_back = torch.sign(rel[..., 1:2])
            above_below = torch.sign(rel[..., 2:3])

        valid_mask_f = self._build_goal_anchor_mask(active, goal_idxs, M)

        edge_exists = valid_mask_f
        left_right = left_right * valid_mask_f
        front_back = front_back * valid_mask_f
        above_below = above_below * valid_mask_f
        rel = positions - goal_pos.unsqueeze(1)
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True) * valid_mask_f
        id_diff = (object_ids - goal_id.unsqueeze(1)) * valid_mask_f

        return torch.cat([edge_exists, left_right, front_back, above_below, dist, id_diff], dim=-1)

    @torch.no_grad()
    def build_sceneverse_edge_features(
        self,
        positions: torch.Tensor,
        object_ids: torch.Tensor,
        active: torch.Tensor,
        goal_idxs: torch.Tensor,
    ) -> torch.Tensor:
        E, M, _ = positions.shape
        batch = torch.arange(E, device=positions.device)

        goal_idxs = goal_idxs.long().clamp(0, M - 1)
        goal_pos = positions[batch, goal_idxs]
        goal_id = object_ids[batch, goal_idxs]

        rel = positions - goal_pos.unsqueeze(1)
        abs_rel = rel.abs()
        dom = torch.argmax(abs_rel, dim=-1)

        left_right = torch.zeros(E, M, 1, device=positions.device)
        front_back = torch.zeros(E, M, 1, device=positions.device)
        above_below = torch.zeros(E, M, 1, device=positions.device)

        left_right[dom == 0] = torch.sign(rel[..., 0:1][dom == 0])
        front_back[dom == 1] = torch.sign(rel[..., 1:2][dom == 1])
        above_below[dom == 2] = torch.sign(rel[..., 2:3][dom == 2])

        valid_mask_f = self._build_goal_anchor_mask(active, goal_idxs, M)

        edge_exists = valid_mask_f
        left_right = left_right * valid_mask_f
        front_back = front_back * valid_mask_f
        above_below = above_below * valid_mask_f
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True) * valid_mask_f
        id_diff = (object_ids - goal_id.unsqueeze(1)) * valid_mask_f

        return torch.cat([edge_exists, left_right, front_back, above_below, dist, id_diff], dim=-1)

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
