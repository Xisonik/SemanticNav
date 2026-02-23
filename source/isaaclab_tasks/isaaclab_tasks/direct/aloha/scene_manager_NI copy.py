# -*- coding: utf-8 -*-
"""
Scene Manager V3
----------------
Полностью переразложенный модуль с чётким разделением ответственности:

- SceneManager: отвечает за векторизованное состояние сцены, раскладки, выбор цели,
  размещение робота и предоставление данных.
- SceneGraph: инкапсулирует *всю* графовую логику: построение граф-наблюдения,
  расчёт пространственных отношений и сборку текстовых промптов для навигации.

⚙️ Совместимость сохранена:
- Формы тензоров и имена публичных методов не менялись, но теперь `get_graph_obs` делегирует в SceneGraph.
- Добавлены новые методы: `compute_relations` и `get_navigation_prompts` (делегируют в SceneGraph).

📌 Обновление графа:
- Граф не копирует тензоры, а работает с менеджером по ссылке.
- После любых изменений сцены менеджер вызывает `self.graph.refresh()`.

Как использовать в aloha_env.py:
  from scene_manager_v3 import SceneManager
  sm = SceneManager(num_envs, config_path, device)
  ...
  prompts = sm.get_navigation_prompts(env_ids, radius=5.0)
  text_embeds = clip.encode_text(prompts)

"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch
import math
import random
import json
from collections import defaultdict
from tabulate import tabulate
import importlib.util

# =====================
# Placement strategies
# =====================

def import_class_from_path(module_path, class_name):
    spec = importlib.util.spec_from_file_location("custom_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_obj = getattr(module, class_name)
    return class_obj

module_path = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/placement_strategies.py"
PlacementStrategy = import_class_from_path(module_path, "PlacementStrategy")
GridPlacement = import_class_from_path(module_path, "GridPlacement")
OnSurfacePlacement = import_class_from_path(module_path, "OnSurfacePlacement")


# =====================
# Vocab & color helpers
# =====================
class RelationVocab:
    LABELS = [
        'in_front_of',  # +x
        'behind',       # -x
        'left_of',      # +y
        'right_of',     # -y
        'above',        # +z
        'below',        # -z
        'inside',       # AABB containment
        'overlapping',  # AABB intersects but not containing
    ]
    TO_ID = {k: i for i, k in enumerate(LABELS)}


class ColorQuantizer:
    """Квантует RGB в 7 базовых цветов (L2 в RGB)."""
    """Квантует RGB в 10 базовых цветов (L2 в RGB)."""
    BASE = torch.tensor([
        [1.0, 0.0, 0.0],   # red
        [0.0, 1.0, 0.0],   # green
        [0.0, 0.0, 1.0],   # blue
        [1.0, 1.0, 0.0],   # yellow
        [1.0, 0.65, 0.0],  # orange
        [0.5, 0.0, 0.5],   # purple
        [1.0, 1.0, 1.0],   # white
        [0.5, 0.5, 0.5],   # gray
        [0.0, 0.0, 0.0],   # black
        [0.6, 0.3, 0.0],   # brown
    ], dtype=torch.float32)

    NAMES = [
        'red', 'green', 'blue',
        'yellow', 'orange', 'purple',
        'white', 'gray', 'black', 'brown'
    ]

    @classmethod
    def rgb_to_name(cls, rgb: torch.Tensor) -> str:
        rgb = rgb.to(dtype=torch.float32).view(1, 3)
        base = cls.BASE.to(device=rgb.device, dtype=rgb.dtype)
        d = torch.cdist(rgb, base)  # [1,10]  ✅
        idx = int(torch.argmin(d, dim=1).item())
        return cls.NAMES[idx]

# ==============
# SceneGraph
# ==============
class SceneGraph:
    """Вся графовая логика: наблюдения, отношения, промпты."""
    def __init__(self, manager: 'SceneManager'):
        self.m = manager
        self._dirty = True  # если нужна инвалидация кэшей в будущем

    def refresh(self):
        """Вызывается менеджером после изменений сцены. Сейчас ничего не кэшируем, но оставляем хук."""
        self._dirty = True

    # ---------- Graph observation ----------
    @torch.no_grad()
    def get_observation(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        m = self.m
        device = m.device
        if env_ids is None:
            env_ids = torch.arange(m.num_envs, device=device)
        E = len(env_ids)

        # --- node features ---
        positions   = m.positions[env_ids]                            # (E, M, 3)
        sizes       = m.sizes.expand(E, -1, -1)                             # (E, M, 3)
        radii       = m.radii.expand(E, -1).unsqueeze(-1)                   # (E, M, 1)
        colors      = m.colors.expand(E, -1, -1)                           # (E, M, 3)
        object_ids  = m.object_ids.expand(E, -1).unsqueeze(-1).float()    # (E, M, 1)
        active      = m.active[env_ids].unsqueeze(-1).float()             # (E, M, 1)

        raw_parents = m.on_surface_idx[env_ids]                                    # (E, M)  int
        parents_feat= raw_parents.unsqueeze(-1).float()                   # (E, M, 1) — ТОЛЬКО для node_features

        levels      = m.surface_level[env_ids].unsqueeze(-1).float()        # (E, M, 1)

        node_features = torch.cat(
            [positions, sizes, radii, colors, object_ids, active, parents_feat, levels],
            dim=-1
        )  # (E, M, 14)

        # --- edge features (используем СЫРЫЕ индексы, без деления) ---
        edge_exists = (raw_parents >= 0).float().unsqueeze(-1)                     # (E, M, 1)
        valid_mask  = (raw_parents >= 0)                                           # (E, M)

        z_diff = torch.zeros(E, m.num_total_objects, 1, device=device)
        level_diff = torch.zeros_like(z_diff)
        dist = torch.zeros_like(z_diff)
        color_diff_norm = torch.zeros_like(z_diff)
        id_diff = torch.zeros_like(z_diff)

        if valid_mask.any():
            batch_idx = torch.arange(E, device=device)[:, None].expand(-1, m.num_total_objects)[valid_mask]
            obj_idx   = torch.arange(m.num_total_objects, device=device)[None, :].expand(E, -1)[valid_mask]
            parent_idx= raw_parents[valid_mask].long()

            # z diff
            z_diff[valid_mask] = positions[batch_idx, obj_idx, 2:3] - positions[batch_idx, parent_idx, 2:3]
            # level diff
            level_diff[valid_mask] = levels[batch_idx, obj_idx] - levels[batch_idx, parent_idx]
            # xy distance
            child_xy  = positions[batch_idx, obj_idx, :2]
            parent_xy = positions[batch_idx, parent_idx, :2]
            dist[valid_mask] = torch.norm(child_xy - parent_xy, dim=-1, keepdim=True)
            # color diff
            child_color  = colors[batch_idx, obj_idx]
            parent_color = colors[batch_idx, parent_idx]
            color_diff_norm[valid_mask] = torch.norm(child_color - parent_color, dim=-1, keepdim=True)
            # id diff
            child_id  = object_ids[batch_idx, obj_idx]
            parent_id = object_ids[batch_idx, parent_idx]
            id_diff[valid_mask] = child_id - parent_id

        edge_features = torch.cat([edge_exists, z_diff, level_diff, dist, color_diff_norm, id_diff], dim=-1)  # (E, M, 6)
        return {"node_features": node_features, "edge_features": edge_features}


    # ---------- Spatial relations ----------
    @torch.no_grad()
    def compute_relations(
        self,
        env_ids: torch.Tensor,
        reference: str | int = 'goal',
        *,
        use_local_frame: bool = True,
        reference_yaws: Optional[torch.Tensor] = None,
        radius: Optional[float] = 5.0,
        include_inactive: bool = False,
    ) -> List[Dict[str, int]]:
        m = self.m
        pos = m.positions[env_ids]                 # [E, M, 3]
        sizes = m.sizes[0]                         # [M, 3]
        active = m.active[env_ids].bool()          # [E, M]
        names = m.names

        E, M = pos.shape[:2]

        # Resolve reference index per env
        if isinstance(reference, int):
            ref_idx = torch.full((E,), int(reference), device=pos.device, dtype=torch.long)
        else:
            if reference in ('goal', 'robot'):
                if reference == 'goal':
                    ref_idx = m.active_goal_indices[env_ids]
                else:
                    ref_idx = m.robot_global_index_tensor.expand(E)
            else:
                if reference not in m.object_map:
                    raise KeyError(f"Unknown reference name: {reference}")
                ref_idx = m.object_map[reference]['indices'][0].expand(E)

        batch_idx = torch.arange(E, device=pos.device).view(E, 1).expand(E, M)
        ref_pos = pos[batch_idx, ref_idx.view(E, 1).expand(E, M)]
        deltas = ref_pos - pos   # [E, M, 3]

        # Rotate to local frame if needed
        if use_local_frame and reference_yaws is not None:
            cy = torch.cos(-reference_yaws).view(E, 1)
            sy = torch.sin(-reference_yaws).view(E, 1)
            x = deltas[..., 0]
            y = deltas[..., 1]
            x_r = x * cy + y * -sy
            y_r = x * sy + y *  cy
            deltas = torch.stack([x_r, y_r, deltas[..., 2]], dim=-1)

        dist = torch.linalg.norm(deltas, dim=-1)  # [E, M]
        mask = torch.ones_like(dist, dtype=torch.bool)
        if radius is not None:
            mask &= (dist <= radius)
        if not include_inactive:
            mask &= active

        # Exclude the reference itself
        ref_mask = torch.zeros_like(mask)
        ref_mask[torch.arange(E, device=pos.device), ref_idx] = True
        mask &= ~ref_mask

        abs_d = deltas.abs()
        dom_axis = torch.argmax(abs_d, dim=-1)  # 0/1/2
        signs = torch.sign(deltas).to(torch.int8)

        rel_id = torch.full((E, M), -1, device=pos.device, dtype=torch.long)
        # X axis
        xpos = (dom_axis == 0) & (signs[..., 0] >= 0)
        xneg = (dom_axis == 0) & (signs[..., 0] <  0)
        rel_id[xpos] = RelationVocab.TO_ID['in_front_of']
        rel_id[xneg] = RelationVocab.TO_ID['behind']
        # Y axis
        ypos = (dom_axis == 1) & (signs[..., 1] >= 0)
        yneg = (dom_axis == 1) & (signs[..., 1] <  0)
        rel_id[ypos] = RelationVocab.TO_ID['left_of']
        rel_id[yneg] = RelationVocab.TO_ID['right_of']
        # Z axis
        zpos = (dom_axis == 2) & (signs[..., 2] >= 0)
        zneg = (dom_axis == 2) & (signs[..., 2] <  0)
        rel_id[zpos] = RelationVocab.TO_ID['above']
        rel_id[zneg] = RelationVocab.TO_ID['below']

        # Inside / overlap
        half = sizes / 2.0
        half_b = half.unsqueeze(0).expand(E, M, 3)
        other_to_ref = -deltas
        inside = (other_to_ref.abs() <= half_b + 1e-6).all(dim=-1)
        rel_id[inside] = RelationVocab.TO_ID['inside']

        ref_sizes = sizes[ref_idx]              # [E,3]
        ref_half_b = (ref_sizes / 2.0).view(E, 1, 3).expand(E, M, 3)
        overlap = (deltas.abs() <= (ref_half_b + half_b) + 1e-6).all(dim=-1) & ~inside
        rel_id[overlap] = RelationVocab.TO_ID['overlapping']

        rel_id = torch.where(mask, rel_id, torch.full_like(rel_id, -1))

        out: List[Dict[str, int]] = []
        for e in range(E):
            dct: Dict[str, int] = {}
            valid = rel_id[e] >= 0
            idxs = torch.nonzero(valid, as_tuple=False).view(-1)
            for j in idxs.tolist():
                dct[names[j]] = int(rel_id[e, j].item())
            out.append(dct)
        return out

    # ---------- Prompt builder ----------
    @torch.no_grad()
    def build_navigation_prompt(
        self,
        env_ids: torch.Tensor,
        goal_name: Optional[str] = None,
        radius: float = 5.0,
        use_local_frame: bool = True,
        reference_yaws: Optional[torch.Tensor] = None,
    ) -> List[str]:
        m = self.m
        E = len(env_ids)
        goal_idxs = m.active_goal_indices[env_ids]   # [E]
        names = m.names
        colors = m.colors[0]                         # [M,3] on (cpu/cuda)

        # словарь для быстрого поиска индекса по имени (O(1) вместо names.index(...))
        name_to_idx = {n: i for i, n in enumerate(names)}

        rel_dicts = self.compute_relations(
            env_ids=env_ids,
            reference='goal',
            use_local_frame=use_local_frame,
            reference_yaws=reference_yaws,
            radius=radius,
            include_inactive=False,
        )

        prompts: List[str] = []
        for k in range(E):
            g_idx = int(goal_idxs[k].item())
            # имя цели без суффикса "_i"
            g_name_raw = goal_name if goal_name is not None else names[g_idx]
            g_name = g_name_raw.split('_', 1)[0]
            # цвет цели -> базовое название
            g_color = ColorQuantizer.rgb_to_name(colors[g_idx])

            # Собираем фразы отношений с ЦВЕТОМ каждого объекта
            rels = rel_dicts[k]              # {obj_name: relation_id}
            phrases: List[str] = []
            for obj_name, rid in rels.items():
                obj_idx = name_to_idx.get(obj_name, None)
                if obj_idx is None:
                    continue
                obj_simple = obj_name.split('_', 1)[0]
                obj_color = ColorQuantizer.rgb_to_name(colors[obj_idx])

                label = RelationVocab.LABELS[rid]
                if label == 'in_front_of':
                    phrases.append(f"in front of {obj_color} {obj_simple}")
                elif label == 'behind':
                    phrases.append(f"behind {obj_color} {obj_simple}")
                elif label == 'left_of':
                    phrases.append(f"left from {obj_color} {obj_simple}")
                elif label == 'right_of':
                    phrases.append(f"right from {obj_color} {obj_simple}")
                elif label == 'above':
                    phrases.append(f"above {obj_color} {obj_simple}")
                elif label == 'below':
                    phrases.append(f"below {obj_color} {obj_simple}")
                elif label == 'inside':
                    phrases.append(f"inside {obj_color} {obj_simple}")
                elif label == 'overlapping':
                    phrases.append(f"overlapping {obj_color} {obj_simple}")

            rel_part = ''
            if phrases:
                # перечисление через запятую: "... that is behind yellow table, left from green vase"
                rel_part = ' that is ' + ', '.join(phrases)

            # финальный промпт: "Move to yellow bowl that is behind green table, left from blue vase"
            prompt = f"Move to {g_color} {g_name}{rel_part}"
            prompts.append(prompt)

        return prompts



# ==============
# SceneManager
# ==============
class SceneManager:
    def __init__(self, num_envs: int, config_path: str, device: str):
        self.num_envs = num_envs
        self.device = device
        with open(config_path, 'r') as f:
            raw = json.load(f)
        self.raw_config = raw
        self.config = raw['objects']
        self.type_placements_cfg = raw.get('type_placements', {})

    # в __init__ SceneManager:
        base = {
            'red':[1,0,0],'green':[0,1,0],'blue':[0,0,1],
            'yellow':[1,1,0],'orange':[1,0.65,0],'purple':[0.5,0,0.5],
            'white':[1,1,1],'gray':[0.5,0.5,0.5],'black':[0,0,0],'brown':[0.6,0.3,0],
        }
        self.colors_dict = {k: base[k] for k in ['red','green','blue','yellow','gray','black','brown']}  # любое подмножество

        # --- Векторизованная структура данных ---
        self.num_total_objects = sum(obj['count'] for obj in self.config)
        self.object_ids = torch.zeros(1, self.num_total_objects, device=self.device)
        self.object_map: Dict[str, Dict] = {}
        self.type_map = defaultdict(list)

        self.positions = torch.zeros(self.num_envs, self.num_total_objects, 3, device=self.device)
        self.sizes = torch.zeros(1, self.num_total_objects, 3, device=self.device)
        self.radii = torch.zeros(1, self.num_total_objects, device=self.device)
        self.colors = torch.ones(1, self.num_total_objects, 3, device=self.device)
        self.names: List[str] = []
        self.active = torch.zeros(self.num_envs, self.num_total_objects, dtype=torch.bool, device=self.device)
        self.on_surface_idx = torch.full((self.num_envs, self.num_total_objects), -1, dtype=torch.long, device=self.device)
        self.surface_level = torch.zeros(self.num_envs, self.num_total_objects, dtype=torch.long, device=self.device)

        self._initialize_object_data()  # создаёт default_positions и positions
        self.default_positions = self.positions.clone()

        self.placement_strategies = self._initialize_strategies()

        self.robot_radius = 0.5
        self.room_bounds = {'x_min': -4, 'x_max': 4, 'y_min': -4, 'y_max': 4}
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)

        # Индекс активной цели на каждую сцену (E,), плюс индекс робота (если есть)
        self.active_goal_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if 'robot' in self.object_map:
            self.robot_global_index = int(self.object_map['robot']['indices'][0].item())
        else:
            self.robot_global_index = 0
        # тензор для удобства бродкаста
        self.robot_global_index_tensor = torch.tensor(self.robot_global_index, device=self.device, dtype=torch.long)

        # Дискретные углы для размещения робота
        n_angles = 36
        angle_step = 2 * math.pi / n_angles
        self.discrete_angles = torch.arange(0, 2 * math.pi, angle_step, device=self.device)
        self.candidate_vectors = torch.stack([torch.cos(self.discrete_angles), torch.sin(self.discrete_angles)], dim=1)

        # Инициализируем графовый помощник
        self.graph = SceneGraph(self)

    # ----------- Работа с примами (заглушка) -----------
    def update_prims(self):
        pass

    # ----------- Векторные данные сцены -----------
    def get_scene_data_dict(self):
        return {
            "positions": self.positions,
            "sizes": self.sizes.expand(self.num_envs, -1, -1),
            "radii": self.radii.expand(self.num_envs, -1),
            "active": self.active,
            "on_surface_idx": self.on_surface_idx,
            "surface_level": self.surface_level,
            "colors": self.colors.expand(self.num_envs, -1, -1),   # 👈 добавить
            "object_ids": self.object_ids.expand(self.num_envs, -1),  # (по желанию) 👈
            "names": self.names,
        }


    # ----------- Фиксированная раскладка -----------
    def apply_fixed_positions(self, env_ids: torch.Tensor, positions_config: List[dict]):
        self.active[env_ids] = False
        self.positions[env_ids] = self.default_positions[env_ids]
        self.on_surface_idx[env_ids] = -1
        self.surface_level[env_ids] = 0
        scene_data = self.get_scene_data_dict()
        for env_id in env_ids:
            env_dict = positions_config[env_id.item()]
            for obj_name, pos_list in env_dict.items():
                if obj_name not in self.object_map:
                    continue
                indices = self.object_map[obj_name]["indices"]
                for i, pos in enumerate(pos_list):
                    if i >= len(indices):
                        print("[WARN] Too many instances for", obj_name)
                        break
                    scene_data["positions"][env_id.item(), indices[i]] = torch.tensor(pos, device=self.device)
                    scene_data["active"][env_id.item(), indices[i]] = True
                    scene_data["on_surface_idx"][env_id.item(), indices[i]] = -1
                    scene_data["surface_level"][env_id.item(), indices[i]] = 0
        self.chose_active_goal_state(env_ids)
        self.graph.refresh()

    # ----------- Инициализация объектов -----------
    def _initialize_object_data(self):
        start_idx = 0
        default_pos_tensor = torch.zeros(1, self.num_total_objects, 3, device=self.device)

        # "Кладбище"
        graveyard_start_x = -8.0
        graveyard_start_y = 6.0
        spacing = 1.1
        max_per_row = 14
        for i in range(self.num_total_objects):
            row = i // max_per_row
            col = i % max_per_row
            default_pos_tensor[0, i, 0] = graveyard_start_x + col * spacing
            default_pos_tensor[0, i, 1] = graveyard_start_y + row * spacing
            default_pos_tensor[0, i, 2] = 0.0

        for obj_cfg in self.config:
            name = obj_cfg['name']
            count = obj_cfg['count']
            indices = torch.arange(start_idx, start_idx + count, device=self.device, dtype=torch.long)
            types = set(obj_cfg['type'])

            info = obj_cfg.get("info", {}) or {}
            info_color = info.get("color", None)

            if isinstance(info_color, str):
                color_name = info_color.strip().lower()
                if color_name in self.colors_dict:
                    color_rgb = torch.tensor(self.colors_dict[color_name], device=self.device, dtype=torch.float32)
                    self.colors[0, indices] = color_rgb  # один и тот же цвет на все инстансы этого типа
                else:
                    print(f"[WARN] Unknown color '{info_color}' for '{obj_cfg['name']}', fallback to gray")
                    self.colors[0, indices] = torch.tensor(self.colors_dict["gray"], device=self.device)

            self.object_map[name] = {'indices': indices, 'types': types, 'count': count}
            for type_str in types:
                self.type_map[type_str].extend(indices.tolist())

            self.names.extend([f"{name}_{i}" for i in range(count)])

            size_tensor = torch.tensor(obj_cfg['size'], device=self.device)
            self.sizes[0, indices] = size_tensor
            self.radii[0, indices] = torch.norm(size_tensor[:2] / 2)
            start_idx += count

        for type_str, indices in self.type_map.items():
            self.type_map[type_str] = torch.tensor(sorted(indices), device=self.device, dtype=torch.long)
        
        self.type_vocab = sorted(self.type_map.keys())      # стабильный порядок типов
        self.num_types = len(self.type_vocab)

        self.default_positions = default_pos_tensor.expand(self.num_envs, -1, -1)
        id_map = {"table": 1, "bowl": 2, "chair": 3, "cabinet": 4}
        for name, data in self.object_map.items():
            obj_id = id_map.get(name, 0)
            self.object_ids[0, data['indices']] = obj_id
        self.positions = self.default_positions.clone()

    # ----------- Стратегии раскладки -----------
    def _initialize_strategies(self):
        strategies_by_type = {}

        def _indices_for_types(type_names):
            if isinstance(type_names, str):
                type_names = [type_names]
            acc = []
            for t in type_names:
                inds = self.type_map.get(t, torch.tensor([], dtype=torch.long, device=self.device))
                if len(inds):
                    acc.extend(inds.tolist())
            return sorted(set(acc))

        if self.type_placements_cfg:
            for t, t_cfg in self.type_placements_cfg.items():
                stype = t_cfg["strategy"]
                if stype == "grid":
                    strategies_by_type[t] = GridPlacement(self.device, t_cfg["grid_coordinates"])
                elif stype == "on_surface":
                    surf_types = t_cfg.get("surface_types", ["surface_provider"])
                    surf_inds = _indices_for_types(surf_types)
                    strategies_by_type[t] = OnSurfacePlacement(self.device, surf_inds, t_cfg["margin"])
        if not strategies_by_type:
            print("[ ERR ] WE HAVE AN ERROR IN _initialize_strategies")
        return strategies_by_type

    # ----------- Рандомизация сцены -----------
    def randomize_scene(self, env_ids: torch.Tensor, mess: bool = False, use_obstacles: bool = False, use_staff: bool = False, all_defoult: bool = True):
        device = self.device
        self.active[env_ids] = False
        self.positions[env_ids] = self.default_positions[env_ids]
        self.on_surface_idx[env_ids] = -1
        self.surface_level[env_ids] = 0
        if all_defoult:
            self.graph.refresh()
            return

        scene_data  = self.get_scene_data_dict()
        type_strats = self.placement_strategies
        placement_order = [t for t in ["surface_provider", "surface_only", "movable_obstacle", "staff_obstacle"] if t in type_strats]
        idx_by_type = {
            "surface_provider": self.type_map.get("surface_provider", torch.tensor([], dtype=torch.long, device=device)),
            "surface_only":     self.type_map.get("surface_only",     torch.tensor([], dtype=torch.long, device=device)),
            "movable_obstacle": self.type_map.get("movable_obstacle", torch.tensor([], dtype=torch.long, device=device)),
            "staff_obstacle":   self.type_map.get("staff_obstacle",   torch.tensor([], dtype=torch.long, device=device)),
        }

        def _sample_available_for_env(env_id: int, eligible: torch.Tensor, k: int) -> torch.Tensor:
            if eligible.numel() == 0 or k <= 0:
                return torch.empty((1, 0), dtype=torch.long, device=device)
            avail_mask = ~self.active[env_id, eligible]
            if avail_mask.sum().item() == 0:
                return torch.empty((1, 0), dtype=torch.long, device=device)
            elig_avail = eligible[avail_mask]
            Ma = elig_avail.numel()
            kk = min(k, Ma)
            perm = torch.randperm(Ma, device=device)[:kk]
            picked = elig_avail[perm]
            return picked.view(1, -1)

        def _apply_strategy_one_env(p_type: str, env_id: int, obj_idx_row: torch.Tensor):
            if obj_idx_row.numel() == 0:
                return
            env_row = torch.tensor([env_id], dtype=torch.long, device=device)
            type_strats[p_type].apply(env_row, obj_idx_row.to(torch.long), scene_data, mess)

        for env_id in env_ids.tolist():
            max_prov = idx_by_type["surface_provider"].numel()
            max_surf = idx_by_type["surface_only"].numel()
            max_mov  = idx_by_type["movable_obstacle"].numel()
            max_staff= idx_by_type["staff_obstacle"].numel()

            if max_surf > 0:
                k_surface_only = int(torch.randint(1, max_surf + 1, (1,), device=device).item())
            else:
                k_surface_only = 0
                print("[ ERR ] max_surf == 0")

            if max_prov > 0:
                low = max(1, k_surface_only)
                low = min(low, max_prov)
                k_providers = int(torch.randint(low, max_prov + 1, (1,), device=device).item())
            else:
                print("[ ERR ] max_prov == 0")
                k_providers = 0
                k_surface_only = 0

            k_movable = int(torch.randint(9, 10, (1,), device=device).item()) if (use_obstacles and max_mov > 0) else 0
            k_staff   = 7 #int(torch.randint(int(max_mov/2) if max_mov>0 else 0, max_staff + 1, (1,), device=device).item()) if (use_staff and max_staff > 0) else 0

            for p_type in placement_order:
                if p_type == "surface_provider":
                    elig = idx_by_type["surface_provider"]
                    picked = _sample_available_for_env(env_id, elig, k_providers)
                    _apply_strategy_one_env("surface_provider", env_id, picked)
                elif p_type == "movable_obstacle":
                    elig = idx_by_type["movable_obstacle"]
                    so = set(idx_by_type["surface_only"].tolist()) if idx_by_type["surface_only"].numel() > 0 else set()
                    if so and elig.numel() > 0:
                        elig = torch.tensor([i for i in elig.tolist() if i not in so], dtype=torch.long, device=device)
                    picked = _sample_available_for_env(env_id, elig, k_movable)
                    _apply_strategy_one_env("movable_obstacle", env_id, picked)
                elif p_type == "surface_only":
                    elig = idx_by_type["surface_only"]
                    picked = _sample_available_for_env(env_id, elig, k_surface_only)
                    _apply_strategy_one_env("surface_only", env_id, picked)
                elif p_type == "staff_obstacle":
                    elig = idx_by_type["staff_obstacle"]
                    picked = _sample_available_for_env(env_id, elig, k_staff)
                    _apply_strategy_one_env("staff_obstacle", env_id, picked)

        self.chose_active_goal_state(env_ids)
        self.graph.refresh()

    # ----------- Помощники для плана пути -----------
    def get_active_obstacle_positions_for_path_planning(self, env_ids: torch.Tensor) -> list:
        obs_indices = self.type_map.get("movable_obstacle", torch.tensor([], dtype=torch.long))
        if len(obs_indices) == 0:
            return [[] for _ in env_ids]
        active_mask = self.active[env_ids][:, obs_indices]
        positions = self.positions[env_ids][:, obs_indices].cpu().numpy()
        output_list = []
        for i in range(len(env_ids)):
            active_positions = positions[i, active_mask[i].cpu().numpy()]
            rounded_pos = [(round(p[0], 1), round(p[1], 1), round(p[2], 1)) for p in active_positions]
            output_list.append(sorted(rounded_pos))
        return output_list

    # ----------- Графовые эмбеддинги (плоские) -----------

    # ----------- Графовые эмбеддинги (плоские) -----------
    # def get_graph_embedding(self, robot_position, env_ids: torch.Tensor) -> torch.Tensor:
    #     """
    #     Эмбеддинг сцены для env_ids.

    #     На объект:
    #     active(1) + pos(3) + is_goal(1) + type_id(1)  =>  6 признаков.
    #     type_id:
    #         0 - прочее
    #         1 - movable_obstacle
    #         2 - possible_goal
    #         3 - surface_provider

    #     Возвращает [E, M*6].
    #     """
    #     device = self.device
    #     env_ids = env_ids.to(device)
    #     E = len(env_ids)
    #     M = self.num_total_objects

    #     # --- базовые признаки ---
    #     env_active  = self.active[env_ids].float().unsqueeze(-1)      # [E, M, 1]
    #     env_pos     = self.positions[env_ids][:, :, :2]  #- robot_position[env_ids, None, :]                        # [E, M, 3]

    #     # --- бинарный флаг цели ---
    #     goal_idx = self.active_goal_indices[env_ids]                  # [E]
    #     all_idx = torch.arange(M, device=device).view(1, M).expand(E, -1)  # [E, M]
    #     is_goal = (all_idx == goal_idx.view(E, 1)).float().unsqueeze(-1)   # [E, M, 1]

    #     # --- type_id на объект (0..3) ---
    #     # 0 - прочее
    #     # 1 - movable_obstacle
    #     # 2 - possible_goal
    #     # 3 - surface_provider
    #     type_id = torch.zeros(M, device=device)                       # [M]

    #     if 'movable_obstacle' in self.type_map:
    #         idxs = self.type_map['movable_obstacle']                  # [K]
    #         if idxs.numel() > 0:
    #             type_id[idxs.long()] = 1.0

    #     if 'possible_goal' in self.type_map:
    #         idxs = self.type_map['possible_goal']
    #         if idxs.numel() > 0:
    #             type_id[idxs.long()] = 2.0

    #     # расширяем до батча: [E, M, 1]
    #     type_id_feat = type_id.view(1, M, 1).expand(E, -1, -1)

    #     # --- маскирование неактивных: ставим -13 ---
    #     inactive_mask = (~self.active[env_ids]).unsqueeze(-1)  # [E, M, 1]
    #     fill = -13.0

    #     env_pos      = torch.where(inactive_mask, torch.full_like(env_pos, fill), env_pos)
    #     is_goal      = torch.where(inactive_mask, torch.full_like(is_goal, fill), is_goal)
    #     type_id_feat = torch.where(inactive_mask, torch.full_like(type_id_feat, fill), type_id_feat)

    #     # --- сборка ---
        
    #     embed = torch.cat(
    #         [env_active, env_pos, is_goal, type_id_feat],
    #         dim=-1
    #     )  # [E, M, 6]
    #     return embed.reshape(E, -1)

    def get_graph_embedding(self, robot_position, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Компактный граф-эмбеддинг сцены для выбранных env_ids.

        Слоты (всего 7):
            0: цель
            1: стол (surface_provider)
            2–4: 3 ближайших препятствия (movable_obstacle)
            5–6: 2 ближайших объекта интереса (possible_goal, можно заменить на нужный тип)

        Каждый слот = (dx, dy), где
            dx, dy = object_xy - robot_xy   (координаты в системе робота)

        Если подходящих объектов нет (например, мало препятствий в радиусе 4 м),
        соответствующие слоты остаются (0, 0).

        Возвращает тензор формы [E, 7 * 2].
        """
        device = self.device
        env_ids = env_ids.to(device)
        E = len(env_ids)
        M = self.num_total_objects

        # позиции всех объектов во всех средах: [E, M, 2] в мировой системе
        env_pos = self.positions[env_ids][:, :, :2]  # [E, M, 2]

        # позиции робота: [E, 2]
        # предполагаем, что robot_position имеет форму [num_envs, 2] в тех же координатах
        robot_xy = robot_position[env_ids]  # [E, 2]

        # индекс цели в self.positions для каждого env: [E]
        goal_idx = self.active_goal_indices[env_ids]  # [E]

        # типы объектов по глобальным индексам: [M]
        type_id_global = torch.zeros(M, device=device)

        if "movable_obstacle" in self.type_map:
            mo_idxs = self.type_map["movable_obstacle"]
            if mo_idxs.numel() > 0:
                type_id_global[mo_idxs.long()] = 1.0

        if "possible_goal" in self.type_map:
            pg_idxs = self.type_map["possible_goal"]
            if pg_idxs.numel() > 0:
                type_id_global[pg_idxs.long()] = 2.0

        if "surface_provider" in self.type_map:
            sp_idxs = self.type_map["surface_provider"]
            if sp_idxs.numel() > 0:
                type_id_global[sp_idxs.long()] = 3.0  # например, стол

        # индексы по типам (глобальные, одинаковые для всех env)
        movable_idxs = self.type_map.get(
            "movable_obstacle",
            torch.empty(0, device=device, dtype=torch.long)
        )
        poi_idxs = self.type_map.get(
            "possible_goal",
            torch.empty(0, device=device, dtype=torch.long)
        )
        surface_idxs = self.type_map.get(
            "surface_provider",
            torch.empty(0, device=device, dtype=torch.long)
        )

        # выход: [E, 7 слотов, 2 фичи]
        num_slots = 7
        embed = torch.zeros(E, num_slots, 2, device=device)

        max_dist = 4.0  # радиус вокруг цели, в котором ищем препятствия и POI

        for e in range(E):
            # ---------- общие штуки для этого env ----------
            # позиция цели в мировых коорд.
            g_idx = int(goal_idx[e].item())
            goal_xy = env_pos[e, g_idx, :]  # [2]

            # позиция робота
            r_xy = robot_xy[e]  # [2]

            # ---------- слот 0: цель ----------
            # dx, dy = goal_xy - robot_xy
            embed[e, 0, :] = goal_xy - r_xy

            # ---------- слот 1: стол (surface_provider) ----------
            if surface_idxs.numel() > 0:
                sp_pos = env_pos[e, surface_idxs, :]       # [Ns, 2]
                sp_dists = torch.norm(sp_pos - goal_xy.unsqueeze(0), dim=-1)  # [Ns]
                sp_min_idx = torch.argmin(sp_dists)
                sp_global_idx = int(surface_idxs[sp_min_idx].item())
                obj_xy = env_pos[e, sp_global_idx, :]      # [2]
                embed[e, 1, :] = obj_xy - r_xy

            # ---------- слоты 2–4: 3 ближайших препятствия ----------
            if movable_idxs.numel() > 0:
                mo_pos = env_pos[e, movable_idxs, :]               # [Ko, 2]
                mo_dists_goal = torch.norm(mo_pos - goal_xy.unsqueeze(0), dim=-1)  # [Ko]

                # фильтр по радиусу от цели
                within_mask = mo_dists_goal <= max_dist
                if within_mask.any():
                    mo_pos_near = mo_pos[within_mask]              # [K', 2]
                    mo_idxs_near = movable_idxs[within_mask]       # [K']

                    # сортировка по (x, y)
                    sort_keys = mo_pos_near[:, 0] * 10.0 + mo_pos_near[:, 1]
                    _, sorted_idx = torch.sort(sort_keys, dim=0)
                    mo_idxs_sorted = mo_idxs_near[sorted_idx]      # [K']

                    k = min(3, mo_idxs_sorted.shape[0])
                    for i in range(k):
                        gidx = int(mo_idxs_sorted[i].item())
                        obj_xy = env_pos[e, gidx, :]               # [2]
                        slot = 2 + i
                        embed[e, slot, :] = obj_xy - r_xy

            # ---------- слоты 5–6: 2 ближайших объекта интереса (POI) ----------
            # здесь я использую type "possible_goal" как "объекты интереса";
            # при желании можно заменить на другой список индексов
            if poi_idxs.numel() > 0:
                poi_pos = env_pos[e, poi_idxs, :]                  # [Kp, 2]
                poi_dists_goal = torch.norm(poi_pos - goal_xy.unsqueeze(0), dim=-1)  # [Kp]

                within_mask = poi_dists_goal <= max_dist
                if within_mask.any():
                    poi_pos_near = poi_pos[within_mask]
                    poi_idxs_near = poi_idxs[within_mask]

                    sort_keys = poi_pos_near[:, 0] * 10.0 + poi_pos_near[:, 1]
                    _, sorted_idx = torch.sort(sort_keys, dim=0)
                    poi_idxs_sorted = poi_idxs_near[sorted_idx]

                    k = min(2, poi_idxs_sorted.shape[0])
                    for i in range(k):
                        gidx = int(poi_idxs_sorted[i].item())
                        obj_xy = env_pos[e, gidx, :]               # [2]
                        slot = 5 + i
                        embed[e, slot, :] = obj_xy - r_xy

        # финальный эмбеддинг: [E, 14]
        # print(embed[0], len(embed), len(embed[0]), len(embed[0][0]))
        return embed.reshape(E, -1)


    def get_blocking_tangent(
        self,
        env_ids: torch.Tensor,
        points_xy: torch.Tensor,
        motion_dirs: torch.Tensor,
        obstacle_types: Optional[List[str]] = ["movable_obstacle"],
        safety_margin: float = 0.2,
        no_hit_dist: float = 1e6,
    ):
        """
        Для каждой среды находит препятствие, которое первым пересечёт луч из точки
        в направлении motion_dirs, и возвращает касательное направление объезда.

        Аргументы:
        - env_ids: [E]
        - points_xy: [E, 2] — позиции робота (XY) в глобальных координатах
        - motion_dirs: [E, 2] — желаемое направление движения (например, текущая линейная скорость)
        - obstacle_types: какие типы считать препятствиями
        - safety_margin: добавка к радиусу препятствия (учёт радиуса робота, запас)
        - no_hit_dist: "расстояние", если пересечений нет

        Возвращает:
        - tangent_dirs: [E, 2] — касательный юнит-вектор (0,0 если нет пересечения)
        - hit_dists: [E] — расстояние до точки пересечения вдоль луча
        - has_hit: [E] bool — есть ли блокирующее препятствие
        """
        device = self.device
        env_ids = env_ids.to(device)
        points_xy = points_xy.to(device)
        motion_dirs = motion_dirs.to(device)

        E = len(env_ids)
        if points_xy.shape != (E, 2):
            raise ValueError(f"points_xy must be [E,2], got {points_xy.shape}")
        if motion_dirs.shape != (E, 2):
            raise ValueError(f"motion_dirs must be [E,2], got {motion_dirs.shape}")

        # --- индексы препятствий ---
        if obstacle_types is None:
            obstacle_types = ["movable_obstacle"]

        obs_idx_list = []
        for t in obstacle_types:
            idxs = self.type_map.get(t, None)
            if idxs is not None and idxs.numel() > 0:
                obs_idx_list.extend(idxs.tolist())

        if len(obs_idx_list) == 0:
            tangent_dirs = torch.zeros(E, 2, device=device)
            hit_dists = torch.full((E,), no_hit_dist, device=device)
            has_hit = torch.zeros(E, dtype=torch.bool, device=device)
            return tangent_dirs, hit_dists, has_hit

        obs_indices = torch.tensor(sorted(set(obs_idx_list)), device=device, dtype=torch.long)  # [K]
        K = obs_indices.numel()

        # центры и радиусы препятствий
        obs_pos_xy = self.positions[env_ids][:, obs_indices, :2]  # [E, K, 2]
        obs_r = self.radii.expand(self.num_envs, -1)[env_ids][:, obs_indices]  # [E, K]
        # эффективный радиус = препятствие + радиус робота + margin
        eff_r = obs_r + self.robot_radius + safety_margin         # [E, K]

        active_mask = self.active[env_ids][:, obs_indices]        # [E, K]
        if not active_mask.any():
            tangent_dirs = torch.zeros(E, 2, device=device)
            hit_dists = torch.full((E,), no_hit_dist, device=device)
            has_hit = torch.zeros(E, dtype=torch.bool, device=device)
            return tangent_dirs, hit_dists, has_hit

        # маскируем неактивные очень далеко
        big = no_hit_dist
        obs_pos_xy = torch.where(
            active_mask.unsqueeze(-1),
            obs_pos_xy,
            torch.full_like(obs_pos_xy, big),
        )
        eff_r = torch.where(
            active_mask,
            eff_r,
            torch.full_like(eff_r, -big),
        )

        # --- подготавливаем желаемое направление движения ---
        # если скорость почти нулевая — берём направление на цель
        goal_xy = self.goal_positions[env_ids][:, :2]                  # [E, 2]
        goal_vec = goal_xy - points_xy                                 # [E, 2]
        goal_dir = goal_vec / (torch.norm(goal_vec, dim=-1, keepdim=True) + 1e-6)

        speed = torch.norm(motion_dirs, dim=-1, keepdim=True)          # [E,1]
        small_speed = speed < 1e-3
        dir_unit = motion_dirs / (speed + 1e-6)
        dir_unit = torch.where(small_speed, goal_dir, dir_unit)        # [E,2]

        # --- луч p(t) = p0 + t * d, t >= 0 ---
        # для каждого препятствия решаем пересечение луча с окружностью
        # || p0 + t d - c ||^2 = R^2  → t^2 + 2*(d·(p0-c)) t + (||p0-c||^2 - R^2) = 0

        # f = p0 - c
        f = points_xy.unsqueeze(1) - obs_pos_xy                        # [E, K, 2]
        b = 2.0 * torch.sum(dir_unit.unsqueeze(1) * f, dim=-1)         # [E, K]
        c_val = torch.sum(f * f, dim=-1) - eff_r * eff_r               # [E, K]
        disc = b * b - 4.0 * c_val                                     # [E, K]
        has_real = disc >= 0.0

        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))

        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        big_tensor = torch.full_like(t1, big)
        t1_pos = torch.where(has_real & (t1 > 0.0), t1, big_tensor)
        t2_pos = torch.where(has_real & (t2 > 0.0), t2, big_tensor)
        t = torch.minimum(t1_pos, t2_pos)                              # [E, K]

        # минимальный t по препятствиям
        t_min, idx_min = torch.min(t, dim=1)                           # [E], [E]
        has_hit = t_min < big * 0.5

        # точка пересечения
        p_hit = points_xy + dir_unit * t_min.unsqueeze(-1)             # [E, 2]
        batch_idx = torch.arange(E, device=device)
        c_min = obs_pos_xy[batch_idx, idx_min]                         # [E, 2]
        r_min = eff_r[batch_idx, idx_min]                              # [E]

        # радиальный вектор и нормаль
        n_vec = (p_hit - c_min) / (r_min.unsqueeze(-1) + 1e-6)         # [E,2]

        # две касательные: влево и вправо
        t_left = torch.stack([-n_vec[:, 1], n_vec[:, 0]], dim=-1)      # [E,2]
        t_right = -t_left                                              # [E,2]

        # выбираем ту, которая лучше согласована с направлением на цель
        dot_left = torch.sum(t_left * goal_dir, dim=-1)                # [E]
        dot_right = torch.sum(t_right * goal_dir, dim=-1)              # [E]
        choose_left = dot_left >= dot_right

        tangent = torch.where(choose_left.unsqueeze(-1), t_left, t_right)  # [E,2]

        # где пересечений нет — зануляем
        tangent = torch.where(has_hit.unsqueeze(-1), tangent, torch.zeros_like(tangent))
        hit_dists = torch.where(has_hit, t_min, torch.full_like(t_min, no_hit_dist))

        return tangent, hit_dists, has_hit
    
    def get_clearance_radius(
        self,
        env_ids: torch.Tensor,
        points: torch.Tensor,
        obstacle_types: Optional[List[str]] = ["movable_obstacle"],
        default_clearance: float = 999.0,
    ) -> torch.Tensor:
        """
        Возвращает для каждой сцены радиус свободного пространства от заданной точки
        до ближайшего препятствия (по поверхности), в плоскости XY.

        Аргументы:
        - env_ids: [E] тензор индексов сред.
        - points:  [E, 3] или [E, 2] точки в глобальной системе координат.
                   points[i] соответствует env_ids[i].
        - obstacle_types: список типов объектов, которые считаются препятствиями.
                          Если None, по умолчанию берём несколько "разумных" типов:
                          movable_obstacle, staff_obstacle, surface_provider, surface_only.
        - default_clearance: значение, которое возвращаем, если в сцене нет активных препятствий.

        Возвращает:
        - clearance: [E] — минимальное расстояние от точки до поверхности ближайшего препятствия.
                     Если препятствий нет, будет default_clearance.
        """
        device = self.device
        env_ids = env_ids.to(device)
        points = points.to(device)

        E = len(env_ids)
        M = self.num_total_objects

        # --- приведём points к [E, 2] ---
        if points.dim() == 1:
            # один point, одна env
            points = points.view(1, -1)
        if points.size(-1) == 3:
            point_xy = points[:, :2]  # [E, 2]
        elif points.size(-1) == 2:
            point_xy = points
        else:
            raise ValueError(f"points must have shape [E,2] or [E,3], got {points.shape}")

        if point_xy.shape[0] != E:
            raise ValueError(
                f"points batch dim ({point_xy.shape[0]}) must match env_ids len ({E})"
            )

        # --- выбираем индексы препятствий ---
        if obstacle_types is None:
            # разумный дефолт под твой конфиг
            obstacle_types = ["movable_obstacle"] #, "staff_obstacle", "surface_provider", "surface_only"]

        obs_idx_list = []
        for t in obstacle_types:
            idxs = self.type_map.get(t, None)
            if idxs is not None and idxs.numel() > 0:
                obs_idx_list.extend(idxs.tolist())

        if len(obs_idx_list) == 0:
            # Никаких объектов таких типов нет — возвращаем просто default_clearance
            return torch.full((E,), default_clearance, device=device)

        obs_indices = torch.tensor(sorted(set(obs_idx_list)), device=device, dtype=torch.long)  # [K]
        K = obs_indices.numel()

        # --- вытаскиваем позиции и радиусы препятствий ---
        # pos: [E, K, 2], r: [E, K]
        obs_pos_xy = self.positions[env_ids][:, obs_indices, :2]  # [E, K, 2]
        obs_r = self.radii.expand(self.num_envs, -1)[env_ids][:, obs_indices]  # [E, K]

        # учитываем только активные препятствия
        active_mask = self.active[env_ids][:, obs_indices]  # [E, K]
        if not active_mask.any():
            return torch.full((E,), default_clearance, device=device)

        # маскируем неактивные "фейковыми" большими расстояниями
        # (чтобы они не влияли на минимум)
        big = default_clearance + 1000.0
        obs_pos_xy = torch.where(
            active_mask.unsqueeze(-1),
            obs_pos_xy,
            torch.full_like(obs_pos_xy, big),
        )
        obs_r = torch.where(
            active_mask,
            obs_r,
            torch.full_like(obs_r, -big),  # чтоб d - r было большим
        )

        # --- считаем расстояния от точки до центров и вычитаем радиус ---
        # point_xy: [E, 2] → [E, 1, 2] для broadcast
        p_xy = point_xy.unsqueeze(1)  # [E, 1, 2]
        dists = torch.norm(obs_pos_xy - p_xy, dim=-1)  # [E, K]
        clearance_all = dists - obs_r                  # [E, K]

        # если точка "внутри" препятствия, clearance может быть отрицательным
        # часто полезно clamp'нуть снизу нулём, но оставляю на твой вкус:
        # clearance_all = torch.clamp_min(clearance_all, 0.0)

        # --- минимум по препятствиям ---
        clearance, _ = torch.min(clearance_all, dim=1)  # [E]

        # Если все препятствия были "задвинули" в big, то min ~= big → можем заменить на default_clearance
        clearance = torch.where(
            torch.isfinite(clearance),
            clearance,
            torch.full_like(clearance, default_clearance),
        )

        return clearance

    # def get_graph_embedding(self, env_ids: torch.Tensor) -> torch.Tensor:
    #     """
    #     Эмбеддинг сцены для env_ids.
    #     На объект:
    #     active(1) + pos(3) + size(3) + radius(1) + object_id(1) + is_goal(1) +
    #     color_one_hot(10) + type_multi_hot(T)  =>  20 + T признаков.
    #     Возвращает [E, M*(20+T)].
    #     """
    #     device = self.device
    #     env_ids = env_ids.to(device)
    #     E = len(env_ids)
    #     M = self.num_total_objects
    #     T = getattr(self, "num_types", len(getattr(self, "type_map", {})))

    #     # --- базовые признаки ---
    #     env_active  = self.active[env_ids].float().unsqueeze(-1)                 # [E, M, 1]
    #     env_pos     = self.positions[env_ids]                                    # [E, M, 3]
    #     env_sizes   = self.sizes.expand(E, -1, -1)                               # [E, M, 3]
    #     env_radii   = self.radii.expand(E, -1).unsqueeze(-1)                     # [E, M, 1]
    #     env_obj_ids = self.object_ids.expand(E, -1).unsqueeze(-1)                # [E, M, 1]

    #     # --- one-hot цвета (10) ---
    #     base_colors = ColorQuantizer.BASE.to(device=device, dtype=torch.float32) # [10,3]
    #     obj_colors  = self.colors[0].to(device=device, dtype=torch.float32)      # [M,3]
    #     dists = torch.cdist(obj_colors.unsqueeze(0), base_colors.unsqueeze(0)).squeeze(0)  # [M,10]
    #     color_idx = torch.argmin(dists, dim=1)                                   # [M]
    #     color_one_hot = torch.zeros(M, base_colors.shape[0], device=device)      # [M,10]
    #     color_one_hot.scatter_(1, color_idx.view(-1, 1), 1.0)
    #     color_one_hot = color_one_hot.unsqueeze(0).expand(E, -1, -1)             # [E, M, 10]

    #     # --- бинарный флаг цели ---
    #     goal_idx = self.active_goal_indices[env_ids]                              # [E]
    #     all_idx = torch.arange(M, device=device).view(1, M).expand(E, -1)        # [E, M]
    #     is_goal = (all_idx == goal_idx.view(E, 1)).float().unsqueeze(-1)         # [E, M, 1]

    #     # --- multi-hot типов (T) ---
    #     # фиксированный порядок self.type_vocab, соберём [M, T]
    #     if not hasattr(self, "type_vocab"):
    #         self.type_vocab = sorted(self.type_map.keys())
    #         self.num_types = len(self.type_vocab)
    #         T = self.num_types

    #     type_multi_hot = torch.zeros(M, T, device=device)                         # [M, T]
    #     for t_idx, t_name in enumerate(self.type_vocab):
    #         idxs = self.type_map.get(t_name, None)
    #         if idxs is None or idxs.numel() == 0:
    #             continue
    #         type_multi_hot[idxs.long(), t_idx] = 1.0
    #     type_multi_hot = type_multi_hot.unsqueeze(0).expand(E, -1, -1)           # [E, M, T]

    #     # --- маскирование неактивных (по желанию, чтобы “мёртвые” = нули) ---
    #     inactive_mask = (~self.active[env_ids]).unsqueeze(-1)  # [E, M, 1]
    #     fill = -13.0

    #     env_pos       = torch.where(inactive_mask, torch.full_like(env_pos, fill), env_pos)
    #     env_sizes     = torch.where(inactive_mask, torch.full_like(env_sizes, fill), env_sizes)
    #     env_radii     = torch.where(inactive_mask, torch.full_like(env_radii, fill), env_radii)
    #     env_obj_ids   = torch.where(inactive_mask, torch.full_like(env_obj_ids, fill), env_obj_ids)
    #     color_one_hot = torch.where(inactive_mask, torch.full_like(color_one_hot, fill), color_one_hot)
    #     is_goal       = torch.where(inactive_mask, torch.full_like(is_goal, fill), is_goal)
    #     type_multi_hot= torch.where(inactive_mask, torch.full_like(type_multi_hot, fill), type_multi_hot)

    #     # --- сборка ---
    #     embed = torch.cat(
    #         [env_active, env_pos, is_goal, type_multi_hot],
    #         dim=-1
    #     )  # [E, M, 20+T]
    #     # print(embed, len(embed[0]))
    #     return embed.reshape(E, -1)

    # ----------- Отладочный принт -----------
    def print_graph_info(self, env_id: int):
        print(f"\n=== Scene Information (Env ID: {env_id}) ===")
        positions = self.positions[env_id]
        active_states = self.active[env_id]
        surface_indices = self.on_surface_idx[env_id]
        surface_levels = self.surface_level[env_id]

        # возьмём RGB с устройства менеджера; ColorQuantizer сам приведёт device/dtype
        colors = self.colors[0]   # [M,3]

        table_data = []
        for i in range(self.num_total_objects):
            name = self.names[i]
            pos = positions[i]
            types = ", ".join([t for t, inds in self.type_map.items() if i in inds])

            rgb = colors[i]
            color_name = ColorQuantizer.rgb_to_name(rgb)
            rgb_str = f"({float(rgb[0]):.2f}, {float(rgb[1]):.2f}, {float(rgb[2]):.2f})"

            row = [
                i, name, types,
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                f"{self.radii[0, i]:.2f}",
                str(active_states[i].item()),
                surface_indices[i].item(),
                surface_levels[i].item(),
                color_name,
                rgb_str,
            ]
            table_data.append(row)

        headers = [
            "ID", "Name", "Types", "Position", "Radius",
            "Active", "On Surface", "Surface Level",
            "Color", "RGB"  # 👈 новые колонки
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


    # ----------- Выбор активной цели -----------
    def chose_active_goal_state(self, env_ids: torch.Tensor):
        goal_indices = self.type_map.get("possible_goal", torch.tensor([], dtype=torch.long, device=self.device))
        if len(goal_indices) == 0:
            print(f"[WARNING] No objects of type 'possible_goal' found in config.")
            self.goal_positions[env_ids] = torch.tensor([-3.75, 0.0, 0.1], device=self.device)
            return
        active_goal_mask = self.active[env_ids][:, goal_indices].float()
        any_active = active_goal_mask.sum(dim=1) > 0
        if not all(any_active):
            print("NO GOAL", any_active)
        chosen_goal_rel_idx = torch.multinomial(active_goal_mask + 1e-9, 1).squeeze(-1)
        chosen_goal_idx = goal_indices[chosen_goal_rel_idx]
        self.goal_positions[env_ids] = self.positions[env_ids, chosen_goal_idx]
        # ВАЖНО: сохраняем индексы цели
        self.active_goal_indices[env_ids] = chosen_goal_idx

    def get_active_goal_state(self, env_ids: torch.Tensor):
        return self.goal_positions[env_ids]

    # ----------- Размещение робота -----------
    def place_robot_for_goal(self, env_ids: torch.Tensor, mean_dist: float, min_dist: float, max_dist: float, angle_error: float):
        num_envs = len(env_ids)
        goal_pos = self.goal_positions[env_ids]
        # print("goal: ", goal_pos)
        is_floor_obstacle = (self.active[env_ids] == True) & (self.on_surface_idx[env_ids] == -1)
        obstacle_pos_all = self.positions[env_ids, :, :2].clone()
        _ = self.radii.expand(self.num_envs, -1)[env_ids]  # kept for possible future use
        inf_pos = torch.full_like(obstacle_pos_all, 999.0)
        obstacle_pos = torch.where(is_floor_obstacle.unsqueeze(-1), obstacle_pos_all, inf_pos)

        mean_dist_with_shift = mean_dist + 1.31
        # print("mean_dist_with_shift", mean_dist_with_shift)
        radii = torch.normal(mean=mean_dist_with_shift, std=0.1, size=(num_envs, 1), device=self.device).clamp_(min_dist, max_dist)
        # print("radii: ", radii)
        candidates = goal_pos[:, None, :2] + radii.unsqueeze(1) * self.candidate_vectors
        # print("candidates: ", candidates)
        bounds = self.room_bounds
        in_bounds_mask = (
            (candidates[..., 0] >= bounds['x_min'] + self.robot_radius) &
            (candidates[..., 0] <= bounds['x_max'] - self.robot_radius) &
            (candidates[..., 1] >= bounds['y_min'] + self.robot_radius) &
            (candidates[..., 1] <= bounds['y_max'] - self.robot_radius)
        )
        # print("in_bounds_mask: ", in_bounds_mask)
        in_bounds_mask_float = in_bounds_mask.float() + 1e-9
        chosen_angle_idx = torch.multinomial(in_bounds_mask_float, 1).squeeze(-1)
        batch_indices = torch.arange(num_envs, device=self.device)
        final_robot_positions = candidates[batch_indices, chosen_angle_idx]

        no_valid_pos_mask = ~in_bounds_mask.any(dim=1)
        if torch.any(no_valid_pos_mask):
            fallback_pos = goal_pos[:, :2] + torch.tensor([max_dist, 0.0], device=self.device)
            final_robot_positions[no_valid_pos_mask] = fallback_pos[no_valid_pos_mask]

        # final_robot_positions = torch.zeros_like(final_robot_positions, device=self.device)

        direction_to_goal = goal_pos[:, :2] - final_robot_positions
        base_yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
        error = (torch.rand(num_envs, device=self.device) - 0.5) * 2 * angle_error
        # final_yaw = base_yaw + error
        final_yaw = torch.zeros_like(base_yaw + error, device=self.device)
        final_yaw = torch.full_like(base_yaw + error, math.pi / 2, device=self.device)
        # final_robot_positions = torch.zeros_like(final_robot_positions, device=self.device)
        robot_quats = torch.zeros(num_envs, 4, device=self.device)
        robot_quats[:, 0] = torch.cos(final_yaw / 2.0)
        robot_quats[:, 3] = torch.sin(final_yaw / 2.0)

        self.remove_colliding_obstacles(env_ids, final_robot_positions)
        return final_robot_positions, robot_quats

    def remove_colliding_obstacles(self, env_ids: torch.Tensor, robot_positions: torch.Tensor):
        """
        Удаляет colliding active объекты (все типы) с роботом.
        - Проверяет dist < (robot_radius + obj_r + 0.2) для всех active obj.
        - Деактивирует и перемещает в default_pos.
        """
        obs_indices = torch.arange(self.num_total_objects, device=self.device)  # ВСЕ индексы (не только movable)
        if len(obs_indices) == 0:
            return
        E = len(env_ids)
        obs_pos = self.positions[env_ids][:, obs_indices, :2]  # [E, M, 2]
        obs_r = self.radii.expand(E, -1)[:, obs_indices]       # [E, M]
        active_obs_mask = self.active[env_ids][:, obs_indices] # [E, M] — только active

        # Маскируем неактивные: их pos/r игнорируем (inf dist)
        obs_pos = torch.where(active_obs_mask.unsqueeze(-1), obs_pos, 
                            torch.full_like(obs_pos, 999.0))
        obs_r = torch.where(active_obs_mask, obs_r, 
                            torch.full_like(obs_r, 999.0))

        dists = torch.norm(obs_pos - robot_positions[:, None, :2], dim=2)  # [E, M]
        coll_mask = dists < (self.robot_radius + 0.4 + 0.2)  # [E, M]
            
        if coll_mask.any():
            # print("robot pos: ", robot_positions[:, None, :2])
            # print("collisiona mask: ", coll_mask)
            # Деактивируем colliding
            batch_idx, obs_idx = torch.where(coll_mask)
            # print("obs_idx :", obs_idx)
            env_batch_idx = env_ids[batch_idx]
            obs_indices_sel = obs_indices[obs_idx]
            
            self.active[env_batch_idx, obs_indices_sel] = False
            # Перемещаем в default
            default_pos = self.default_positions[env_batch_idx, obs_indices_sel]
            self.positions[env_batch_idx, obs_indices_sel] = default_pos
            
            # Лог (опционально, для дебага)
            # print(f"[COLL DEBUG] Deactivated {coll_mask.sum().item()} obstacles in envs {env_ids.tolist()}")

    # ----------- Делегаты в SceneGraph -----------
    def get_graph_obs(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return self.graph.get_observation(env_ids)

    @torch.no_grad()
    def compute_relations(self, env_ids: torch.Tensor, reference: str | int = 'goal', *,
                          use_local_frame: bool = True, reference_yaws: Optional[torch.Tensor] = None,
                          radius: Optional[float] = 5.0, include_inactive: bool = False) -> List[Dict[str, int]]:
        return self.graph.compute_relations(env_ids, reference, use_local_frame=use_local_frame,
                                            reference_yaws=reference_yaws, radius=radius,
                                            include_inactive=include_inactive)

    @torch.no_grad()
    def get_navigation_prompts(self, env_ids: torch.Tensor, goal_name: Optional[str] = None, radius: float = 5.0,
                               use_local_frame: bool = True, reference_yaws: Optional[torch.Tensor] = None) -> List[str]:
        return self.graph.build_navigation_prompt(env_ids, goal_name=goal_name, radius=radius,
                                                  use_local_frame=use_local_frame, reference_yaws=reference_yaws)
