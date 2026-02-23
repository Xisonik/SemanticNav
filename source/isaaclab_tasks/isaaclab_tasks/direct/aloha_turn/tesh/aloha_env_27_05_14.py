# env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
 
import gymnasium as gym
import torch
import math
import numpy as np
import os
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from .control_manager import ControlManager
import omni.kit.commands
import omni.usd

##
# Pre-defined configs
##
from isaaclab_assets.robots.aloha import ALOHA_CFG
from isaaclab.markers import CUBOID_MARKER_CFG

class WheeledRobotEnvWindow(BaseEnvWindow):
    def __init__(self, env: 'WheeledRobotEnv', window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class WheeledRobotEnvCfg(DirectRLEnvCfg):
    episode_length_s = 60.0
    decimation = 4
    action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        shape=(2,)
    )
    # Observation space is now the ResNet18 embedding size (512)
    observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(514,), dtype="float32")
    state_space = 0
    debug_vis = True
    use_controller = False

    ui_window_class_type = WheeledRobotEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.02,
            dynamic_friction=0.02,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.02,
            dynamic_friction=0.02,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=20, replicate_physics=True)
    robot: ArticulationCfg = ALOHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheel_radius = 0.068
    wheel_distance = 0.34
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/fl_link6/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.01, 0.0, 0.05), rot=(1.0, 0.0, 0.0, -0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=224,  # ResNet18 expects 224x224 images
        height=224,
    )
    kitchen = sim_utils.UsdFileCfg(
        usd_path="/home/mipt/Downloads/assets/assets/scenes/scenes_sber_kitchen_for_BBQ/kitchen_new_simple.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=False,
            rigid_body_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        filter_prim_paths_expr=["/World/envs/env_.*"],
    )
    chairs = [
        sim_utils.UsdFileCfg(
            usd_path="/home/mipt/Downloads/assets/scenes/obstacles/chair1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # Стол неподвижен
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        sim_utils.UsdFileCfg(
            usd_path="/home/mipt/Downloads/assets/scenes/obstacles/chair2.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # Стол неподвижен
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
    ]

    table = sim_utils.UsdFileCfg(
        usd_path="/home/mipt/Downloads/assets/assets/scenes/scenes_sber_kitchen_for_BBQ/table/table.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,  # Стол неподвижен
            rigid_body_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,  # Отключаем коллизии для безопасности
        ),
    )

    # Конфигурация миски (цели)
    bowl = sim_utils.UsdFileCfg(
        usd_path="/home/mipt/Downloads/assets/assets/objects/bowl.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,  # Миска неподвижна
            rigid_body_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,  # Отключаем коллизии
        ),
    )

    # Позиция цели в локальных координатах
    chair_positions = [(1.0, 2.0, 0.0), (2.0, 1.0, 0.0)]
    chair_update_period = 100

class WheeledRobotEnv(DirectRLEnv):
    cfg: WheeledRobotEnvCfg

    def __init__(self, cfg: WheeledRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actions = torch.ones((self.num_envs, 2), device=self.device)
        self._actions[:, 1] = 0.0
        self._left_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._right_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["base_penalty", "moves", "goal_reached_reward", "out_of_bounds_penalty", "contact_penalty"]
        }
        self._left_wheel_id = self._robot.find_joints("left_wheel")[0]
        self._right_wheel_id = self._robot.find_joints("right_wheel")[0]

        self.set_debug_vis(self.cfg.debug_vis)
        self.Debug = True
        self.event_history = torch.zeros((self.num_envs, 50), dtype=torch.float, device=self.device)
        self.event_history_index = 0
        self.event_history_filled = False
        self.event_update_counter = 0
        self.episode_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.control_manager = ControlManager(self.num_envs, self.device)
        self.count = 0
        self._debug_log_enabled = True
        self._debug_envs_to_log = list(range(min(5, self.num_envs)))
        self._inconsistencies = []
        self._debug_step_counter = 0
        self.step_counter = 0
        self._debug_log_frequency = 10
        self.use_controller = self.cfg.use_controller
        self._screenshot_dir = "/home/mipt/Downloads/IsaacLab-main/logs/camera_images/screenshots"
        os.makedirs(self._screenshot_dir, exist_ok=True)

        self.previous_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Initialize ResNet18 for image embeddings
        self.resnet18 = models.resnet18(pretrained=True).to(self.device)
        self.resnet18.eval()  # Set to evaluation mode
        # Remove the final fully connected layer to get embeddings
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        # Image preprocessing for ResNet18
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.success_rate = 0
        self._step_update_counter = 0
        self.mean_radius = 4
        self._obstacle_update_counter = 0

        self.level = 1
        self.sim = SimulationContext.instance()
        self.obstacle_positions = None

    def _setup_scene(self):
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        from isaaclab.sensors import ContactSensor
        import time
        from pxr import Usd
        self.obstacle_positions = None
        self.chair_prims = [[] for _ in range(self.cfg.scene.num_envs)]
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=True)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        spawn_from_usd(
            prim_path="/World/envs/env_.*/Kitchen",
            cfg=self.cfg.kitchen,
            translation=(5.0, 4.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        # Спавн стола
        spawn_from_usd(
            prim_path="/World/envs/env_.*/Table",
            cfg=self.cfg.table,
            translation=(-4.5, 0.0, 0.0),  # Стол в центре локальной системы
            orientation=(0.7071, 0.0, 0.0, 0.7071),
        )
        goal_pos = (-4.5, 0, 0.6)  # z=0.8 для поверхности стола
        spawn_from_usd(
            prim_path="/World/envs/env_.*/Bowl",
            cfg=self.cfg.bowl,
            translation=goal_pos,
            orientation=(0.0, 0.0, 0.7071, 0.7071),
        )
        # spawn_from_usd(
        #     prim_path="/World/envs/env_1/Chair",
        #     cfg=self.cfg.chairs[0],
        #     translation=(0, 0, 0.6),
        #     orientation=(0.0, 0.0, 0.0, 1.0),
        # )
        # Проверка созданных примитивов
        stage = omni.usd.get_context().get_stage()
        for env_id in range(self.cfg.scene.num_envs):
            for prim_path in [
                f"/World/envs/env_{env_id}/Kitchen",
                f"/World/envs/env_{env_id}/Table",
                f"/World/envs/env_{env_id}/Bowl",
            ]:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    raise RuntimeError(f"Failed to create prim at {prim_path}")
                # print(f"Created prim {prim_path}, Type: {prim.GetTypeName()}")
        import random

        # self._update_chairs()
        # Список для хранения prim-путей стульев для последующего управления
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        # self.step_counter += 1
        # if self.step_counter > 100 and False:
        #     self.control_manager.change_radius_values()
        #     self.step_counter = 0
        root_lin_vel_w = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1).unsqueeze(-1)
        root_ang_vel_w = self._robot.data.root_ang_vel_w[:, 2].unsqueeze(-1)
        # Get RGB images from the tiled camera
        camera_data = self._tiled_camera.data.output["rgb"].clone() / 255.0  # Shape: (num_envs, 224, 224, 3)
        # Reshape and transpose to (num_envs, 3, 224, 224) for ResNet18
        camera_data = camera_data.permute(0, 3, 1, 2).to(self.device)
        # Apply ImageNet normalization
        camera_data = self.transform(camera_data)

        
        # Pass through ResNet18 to get embeddings
        with torch.no_grad():
            embeddings = self.resnet18(camera_data).squeeze(-1).squeeze(-1)  # Shape: (num_envs, 512)
        obs = torch.cat([embeddings, root_lin_vel_w, root_ang_vel_w], dim=-1)
        observations = {"policy": obs}
        
        return observations

    # The rest of the methods (_pre_physics_step, _apply_action, _get_rewards, etc.) remain unchanged
    # as they are not affected by the observation space change.

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        r = self.cfg.wheel_radius
        L = self.cfg.wheel_distance
        
        if self.use_controller:
            control_actions = self.control_manager.compute_control(
                self._robot.data.root_pos_w,
                self._robot.data.root_state_w[:, 3:7]
            )
            linear_speed = control_actions[:, 0]
            angular_speed = control_actions[:, 1]
            self._actions[:, 0] = (linear_speed * 2.0) - 1
            self._actions[:, 1] = angular_speed
        else:
            linear_speed = 0.6*(self._actions[:, 0] + 1.0) # [num_envs], всегда > 0
            angular_speed = 0.5*self._actions[:, 1]  # [num_envs], оставляем как есть от RL
        
        self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
        self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r
        # self._left_wheel_vel = torch.zeros(1, device=self.device)
        # self._right_wheel_vel = torch.zeros(1, device=self.device)

        return self._actions

    def _apply_action(self):
        wheel_velocities = torch.stack([self._left_wheel_vel, self._right_wheel_vel], dim=1).unsqueeze(-1)
        self._robot.set_joint_velocity_target(wheel_velocities, joint_ids=[self._left_wheel_id, self._right_wheel_id])

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        lin_vel_reward = lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
        ang_vel = torch.abs(self._robot.data.root_ang_vel_w[:, 2])
        ang_vel_reward = ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        out_of_bounds_penalty_scale = -10.0
        out_of_bounds = distance_to_goal > 8.0
        out_of_bounds_penalty = out_of_bounds.float() * out_of_bounds_penalty_scale
        goal_reached = self.goal_reached(distance_to_goal)
        goal_reached_reward_scale = 15.0
        goal_reached_reward = goal_reached.float() * goal_reached_reward_scale
        moves = 100*(self.previous_distance_to_goal-distance_to_goal)
        self.previous_distance_to_goal = distance_to_goal
        rewards = {
            "goal_reached_reward": goal_reached_reward,
            "out_of_bounds_penalty": out_of_bounds_penalty,
            "moves":moves,
        }
        #reward = (-1 + goal_reached_reward + lin_vel_reward + ang_vel_reward + out_of_bounds_penalty + moves)
        reward = (-0.7 + goal_reached_reward + lin_vel_reward + ang_vel_reward + out_of_bounds_penalty)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def goal_reached(self, distance_to_goal: torch.Tensor) -> torch.Tensor:
        """Проверяет, достигнута ли цель: расстояние < 0.5 и ошибка по углу < π/9."""
        distance_condition = distance_to_goal < 1.1
        return distance_condition

    def get_contact(self):
        force_matrix = self.scene["contact_sensor"].data.net_forces_w
        if force_matrix is not None and force_matrix.numel() > 0:
            contact_forces = torch.norm(force_matrix, dim=-1)
            if contact_forces.dim() >= 3:
                has_contact = torch.any(contact_forces > 1.0, dim=(1, 2))
            else:
                has_contact = torch.any(contact_forces > 1.0, dim=1) if contact_forces.dim() == 2 else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            num_contacts_per_env = torch.sum(contact_forces > 1.0, dim=1)
            print("get_contact ", num_contacts_per_env)
            high_contact_envs = num_contacts_per_env > 5
        else:
            print("force_matrix_w is None or empty")

        
        return high_contact_envs

    def get_success_rate(self) -> torch.Tensor:
        # Сохраняем историю завершенных эпизодов за последние 100 шагов
        self.episode_completion_history = getattr(self, 'episode_completion_history', torch.zeros((100, self.num_envs), dtype=torch.bool, device=self.device))
        self.success_history = getattr(self, 'success_history', torch.zeros((100, self.num_envs), dtype=torch.bool, device=self.device))
        self.history_index = getattr(self, 'history_index', 0)
        
        # Обновляем историю при завершении эпизодов
        died, time_out = self._get_dones()
        completed = died | time_out
        if torch.any(completed):
            self.episode_completion_history[self.history_index] = completed
            self.success_history[self.history_index] = self.goal_reached(torch.linalg.norm(self._desired_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=1))
            self.history_index = (self.history_index + 1) % 100
        
        # Считаем общее количество завершенных эпизодов и успешных за последние 100 шагов
        total_completed = self.episode_completion_history.float().sum(dim=0).clamp(min=1)  # Избегаем деления на 0
        total_success = self.success_history.float().sum(dim=0)
        
        # Процент успеха для каждой среды
        success_rate = (total_success / total_completed) * 100.0
        self.success_rate = torch.mean(success_rate).item()
        return success_rate

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        
        has_contact = self.get_contact()
        
        died = torch.logical_or(
            torch.logical_or(self.goal_reached(distance_to_goal), has_contact),
            time_out,
        )
        
        if torch.any(died):
            goal_reached = self.goal_reached(distance_to_goal)
            self.episode_counter += died.long()
            self.success_counter += goal_reached.long()
            self.event_update_counter += torch.sum(died).item()

        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        # root_pos_w = self._robot.data.root_pos_w[:, :2]
        # distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        # died = torch.logical_or(self.goal_reached(distance_to_goal), distance_to_goal > 10.0)
        # if self.prev_max_dist != min(self.radius):
        #     self.prev_max_dist = min(self.radius)
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        self._update_chairs(env_ids)
        change_level = self.update_from_counters(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs: # or change_level:
            env_ids = self._robot._ALL_INDICES
        env_ids = env_ids.to(dtype=torch.long)

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids, :2] - self._robot.data.root_pos_w[env_ids, :2], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        contact_forces = torch.norm(self._contact_sensor.data.net_forces_w[env_ids], dim=-1)
        has_contact = (contact_forces > 0.1).any(dim=-1)
        extras["Episode_Termination/contact"] = torch.count_nonzero(has_contact).item()
        self.extras["log"]["Metrics/level"] = self.level
        self.extras["log"]["Metrics/mean_radius"] = self.mean_radius
        self.extras["log"]["Metrics/success_rate"] = torch.mean(self.get_success_rate()).item()
        self.extras["log"]["Metrics/event_update_counter"] = self.event_update_counter
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._desired_pos_w[env_ids, :2] = self._terrain.env_origins[env_ids, :2]
        
        min_radius_x = torch.tensor([((self.level + 1.5 + 0.2) * int(self.level > 0) + 1)**2 + 3**2])
        min_radius = torch.sqrt(min_radius_x)[0]
        robot_pos, quaternion, goal_pos = self.control_manager.reset(env_ids, self._terrain.env_origins, self.mean_radius, min_radius)

        self._desired_pos_w[env_ids, :2] = goal_pos
        self._desired_pos_w[env_ids, 2] = 0.6
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :2] = robot_pos
        default_root_state[:, 2] = 0.1
        default_root_state[:, 3:7] = quaternion
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        if torch.any(torch.isnan(robot_pos)) or torch.any(torch.isinf(robot_pos)):
            raise ValueError(f"Invalid robot_pos: {robot_pos}")
        
        # for _ in range(50):
        #     self.sim.step(render=False)

    def update_from_counters(self, env_ids: torch.Tensor):
        if self.success_rate > 80 and self._step_update_counter > 200:
            print("success_rate, self._step_update_counter: ", self.success_rate, self._step_update_counter)
            self.mean_radius += 0.2
            self._step_update_counter = 0
            print("mean_radius: ",self.mean_radius)
        
        change_level = False
        min_radius_x = torch.tensor([((self.level + 1.5 + 0.2) * int(self.level > 0) + 1)**2 + 3**2])
        min_radius = torch.sqrt(min_radius_x)[0] + 2
        print(min_radius, self.mean_radius)
        if self.mean_radius > min_radius:
            self.level = self.level + 1
            print("change level to : ",self.level)
            change_level = True

        self._obstacle_update_counter += 1
        self._step_update_counter += 1

        return change_level

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

    def close(self):
        super().close()

    def generate_obstacle_positions(self, key=None):
        """
        Генерирует позиции препятствий (стульев) на основе ключа для всех сред.
        Args:
            key: List[List[int]] или None, список ключей [[k1, k2], ...] для каждой среды.
                 Если None, генерируется случайный ключ для каждой среды.
        Returns:
            List[List[tuple]]: Список позиций стульев для каждой среды.
        """
        grid_x = [-2.0, -1.0]  # Уровни по X
        grid_y = [-1.0, 0.0, 1.0]  # Уровни по Y
        # if key is None:
        #     # Генерируем уникальный ключ [k1, k2] для каждой среды
        #     if self.level == 0:
        #         # Все ключи нулевые: [[0, 0], [0, 0], ...]
        #         key = [[0, 0] for _ in range(self.num_envs)]
        #     elif self.level == 1:
        #         # Зануляем линию x = -3 (k1 = 0), k2 случайное
        #         key = [[0, random.randint(0, 7)] for _ in range(self.num_envs)]
        #     else:  # self.level == 2
        #         # Случайные ключи: [[rand, rand], ...]
        #         key = [[random.randint(0, 7), random.randint(0, 7)] for _ in range(self.num_envs)]
        # else:
        #     if len(key) != self.num_envs or any(not (0 <= k[0] <= 8 and 0 <= k[1] <= 8) for k in key):
        #         raise ValueError(f"Key must be a list of {self.num_envs} pairs [k1, k2] with k1, k2 in [0, 8], got {key}")
        key = [[random.randint(0, 7), random.randint(0, 7)] for _ in range(self.num_envs)]
        # Преобразуем ключи в двоичную форму
        binary_keys = [[format(k[0], '03b'), format(k[1], '03b')] for k in key]
        # print(f"Generating obstacle positions with keys: {key}, binary: {binary_keys}")

        positions = [[] for _ in range(self.num_envs)]
        for env_id in range(self.num_envs):
            # Собираем все возможные позиции из сетки на основе ключа
            env_positions = []
            for x_idx, x_pos in enumerate(grid_x):
                binary = binary_keys[env_id][x_idx]
                for y_idx, bit in enumerate(binary):
                    if bit == '1':
                        y_pos = grid_y[y_idx]
                        env_positions.append((x_pos, y_pos, 0.0))

            positions[env_id] = env_positions

        return positions

    def _update_chairs(self, env_ids: torch.Tensor=None):
        from isaacsim.core.utils.prims import delete_prim
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        if env_ids is None:
            env_ids = range(self.cfg.scene.num_envs)
        # Удаление существующих стульев для указанных сред
        for env_id in env_ids:
            # env_id = env_id.item()  # Преобразуем тензорный индекс в int
            if len(self.chair_prims[env_id]) > 0:
                for prim_path in self.chair_prims[env_id]:
                    delete_prim(prim_path)
                self.chair_prims[env_id].clear()

        # Спавн новых стульев для указанных сред
        obstacle_positions = self.generate_obstacle_positions()
        self.obstacle_positions = obstacle_positions
        for env_id in env_ids:
            # env_id = env_id.item()  # Преобразуем тензорный индекс в int
            i = 0
            for pos in obstacle_positions[env_id]:
                chair_cfg = random.choice(self.cfg.chairs)
                prim_path = f"/World/envs/env_{env_id}/Chair_{i}"
                spawn_from_usd(
                    prim_path=prim_path,
                    cfg=chair_cfg,
                    translation=pos,
                    orientation=(0.0, 0.0, 0.0, 1.0),
                )
                self.chair_prims[env_id].append(prim_path)
                i += 1

