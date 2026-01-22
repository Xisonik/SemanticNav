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
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from .scene_manager import Scene_manager
import omni.kit.commands
import omni.usd
import datetime
from torch.utils.tensorboard import SummaryWriter
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
    episode_length_s = 10.0
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
        dt=1/60,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=20, replicate_physics=True)
    robot: ArticulationCfg = ALOHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheel_radius = 0.068
    wheel_distance = 0.34
    lin_vel_reward_scale = 0.1
    ang_vel_reward_scale = 0.05
    distance_to_goal_reward_scale = 15.0
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/fl_link6/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.01, 0.0, 0.05), rot=(1.0, 0.0, 0.0, -0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=80, clipping_range=(0.1, 20.0)
        ),
        width=224,  # ResNet18 expects 224x224 images
        height=224,
    )
    kitchen = sim_utils.UsdFileCfg(
        usd_path="/home/mipt/Downloads/assets/assets/scenes/scenes_sber_kitchen_for_BBQ/kitchen_new_simple.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
            rigid_body_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
        filter_prim_paths_expr=["/World/envs/env_.*"],
    )

    table = sim_utils.UsdFileCfg(
        usd_path="/home/mipt/Downloads/assets/scenes/scenes_sber_kitchen_for_BBQ/table/table2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,  # Стол неподвижен
            rigid_body_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,  # Отключаем коллизии для безопасности
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
            collision_enabled=True,  # Отключаем коллизии
        ),
    )

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
            for key in ["moves", "contact_penalty"]
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
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.scene_manager = Scene_manager(self.num_envs, self.device, num_obstacles=6)
        self.count = 0
        self._debug_log_enabled = True
        self._debug_envs_to_log = list(range(min(5, self.num_envs)))
        self._inconsistencies = []
        self._debug_step_counter = 0
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
        self.mean_radius = 0
        self.max_angle = 0 #torch.pi/6
        self._obstacle_update_counter = 0
        self.has_contact = torch.full((self.num_envs,), True, dtype=torch.bool, device=self.device)
        self.level = 0
        self.num_update_level = 0
        self.sim = SimulationContext.instance()
        self.obstacle_positions = None
        self.key = None
        self.sucess_ep_num = 0
        self.first_ep = True
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
        name = "simple"
        self.tensorboard_writer = SummaryWriter(log_dir=f"/home/mipt/Downloads/IsaacLab-main/logs/tensorboard/navigation_rl_{name}_{timestamp}")
        self.tensorboard_step = 0
        

    def _setup_scene(self):
        from isaaclab.sensors import ContactSensor
        import time
        from pxr import Usd
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=True)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.set_env()
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=700.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def set_env(self):
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        self.obstacle_positions = None
        self.chair_prims = [[] for _ in range(self.cfg.scene.num_envs)]
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
        goal_pos = (-4.5, 0, 0.65)  # z=0.8 для поверхности стола
        spawn_from_usd(
            prim_path="/World/envs/env_.*/Bowl",
            cfg=self.cfg.bowl,
            translation=goal_pos,
            orientation=(0.0, 0.0, 0.7071, 0.7071),
        )

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

        self.chair_objects = [[] for _ in range(6)]
        grid_x = [-2.0, -1.0]
        grid_y = [-1.0, 0.0, 1.0]
        initial_positions = [(-2.0, -1.0, 0.0), (-2.0, 0.0, 0.0), (-2.0, 1.0, 0.0),
                            (-1.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 1.0, 0.0)]  # Позиции для ключа [7, 7]
        # for env_id in range(self.cfg.scene.num_envs):
        for i in range(6):
            chair_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Chair_{i}",  # Уникальный путь для каждого стула в каждой среде
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/home/mipt/Downloads/assets/scenes/obstacles/chair2.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0 + initial_positions[i][0], initial_positions[i][1], 0.0)),
            )
            chair_object = RigidObject(cfg=chair_cfg)
            self.chair_objects[i] = chair_object  # Добавляем в список для конкретной среды


    def _get_observations(self) -> dict:
        self.tensorboard_step += 1
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
        
        # if self.use_controller:
        #     control_actions = self.scene_manager.compute_control(
        #         self._robot.data.root_pos_w,
        #         self._robot.data.root_state_w[:, 3:7]
        #     )
        #     linear_speed = control_actions[:, 0]
        #     angular_speed = control_actions[:, 1]
        #     self._actions[:, 0] = (linear_speed * 2.0) - 1
        #     self._actions[:, 1] = angular_speed
        # else:
        linear_speed = 0.6*(self._actions[:, 0] + 1.0) # [num_envs], всегда > 0
        angular_speed = 0.5*self._actions[:, 1]  # [num_envs], оставляем как есть от RL
        
        self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
        self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r
        # self._left_wheel_vel = torch.zeros(1, device=self.device)
        # self._right_wheel_vel = torch.zeros(1, device=self.device)
        # print("action: ", actions)
        return self._actions

    def _apply_action(self):
        wheel_velocities = torch.stack([self._left_wheel_vel, self._right_wheel_vel], dim=1).unsqueeze(-1)
        self._robot.set_joint_velocity_target(wheel_velocities, joint_ids=[self._left_wheel_id, self._right_wheel_id])

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        lin_vel_reward = lin_vel * 0.005
        ang_vel = torch.abs(self._robot.data.root_ang_vel_w[:, 2])
        ang_vel_reward = ang_vel * 0.005
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        # print("root_pos_w ", root_pos_w)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        out_of_bounds_penalty_scale = -10.0
        out_of_bounds = distance_to_goal > 10.0
        out_of_bounds_penalty = out_of_bounds.float() * out_of_bounds_penalty_scale
        goal_reached, num_subs = self.goal_reached(distance_to_goal, get_num_subs=True)
        goal_reached_reward_scale = 10.0 + 3*self.mean_radius
        num_subs_scale = 0.03
        goal_reached_reward = goal_reached.float() * goal_reached_reward_scale
        moves = 100*(self.previous_distance_to_goal-distance_to_goal)
        self.previous_distance_to_goal = distance_to_goal

        quat = self._robot.data.root_quat_w
        siny_cosp = 2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
        cosy_cosp = 1 - 2 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3])
        psi = torch.atan2(siny_cosp, cosy_cosp)
        # Compute angle to goal (theta)
        delta_pos = self._desired_pos_w[:, :2] - root_pos_w
        theta = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])
        # Compute heading error (alpha = theta - psi)
        alpha = theta - psi
        # Normalize alpha to [-pi, pi]
        alpha = torch.atan2(torch.sin(alpha), torch.cos(alpha))

        # Lyapunov derivative components
        rho = distance_to_goal
        u = lin_vel
        omega = ang_vel
        # Compute V_dot = -rho * u * cos(alpha) + alpha * (-omega + u * sin(alpha) / rho)
        term1 = -rho * u * torch.cos(alpha)
        term2 = alpha * (-omega + torch.where(rho > 0.01, u * torch.sin(alpha) / rho, torch.zeros_like(rho)))
        V_dot = term1 + term2

        # Lyapunov reward: Encourage negative V_dot for stability (scale to balance with other rewards)
        lyapunov_reward_scale = 1 # Adjustable scale
        lyapunov_reward = torch.max(-V_dot * lyapunov_reward_scale, torch.tensor(0.0))  # Negative V_dot gives positive reward


        #reward = (-1 + goal_reached_reward + lin_vel_reward + ang_vel_reward + out_of_bounds_penalty + moves)
        has_contact = self.get_contact()
        # print("contact ", has_contact.long())
        contact_penalty = -7 * has_contact.long()
        num_subs_reward = num_subs * num_subs_scale
        reward = (-0.2 + goal_reached_reward + num_subs_reward + lin_vel_reward + ang_vel_reward + contact_penalty)
        check = {
            "moves":moves,
            "contact_penalty": contact_penalty,
        }
        for key, value in check.items():
            self._episode_sums[key] += value
        if self.tensorboard_step % 100 == 0:
            self.tensorboard_writer.add_scalar("Metrics/reward", torch.mean(reward), self.tensorboard_step)
        return reward
    
    def quat_rotate(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Вращение вектора vec кватернионом quat.
        quat: [N, 4] (w, x, y, z)
        vec: [N, 3]
        Возвращает: [N, 3] - вектор vec, повернутый кватернионом quat
        """
        w, x, y, z = quat.unbind(dim=1)
        vx, vy, vz = vec.unbind(dim=1)

        # Кватернионное умножение q * v
        qw = -x*vx - y*vy - z*vz
        qx = w*vx + y*vz - z*vy
        qy = w*vy + z*vx - x*vz
        qz = w*vz + x*vy - y*vx

        # Обратный кватернион q*
        rw = w
        rx = -x
        ry = -y
        rz = -z

        # Результат (q * v) * q*
        rx_new = qw*rx + qx*rw + qy*rz - qz*ry
        ry_new = qw*ry - qx*rz + qy*rw + qz*rx
        rz_new = qw*rz + qx*ry - qy*rx + qz*rw

        return torch.stack([rx_new, ry_new, rz_new], dim=1)


    def goal_reached(self, distance_to_goal: torch.Tensor, angle_threshold: float = 15, get_num_subs=False) -> torch.Tensor:
        """
        Проверяет достижение цели с учётом расстояния и направления взгляда робота.
        distance_to_goal: [N] расстояния до цели
        angle_threshold: максимально допустимый угол в радианах между направлением взгляда и вектором на цель
        Возвращает: [N] булев тензор, True если цель достигнута
        """

        # Проверка по расстоянию (например, радиус достижения stored в self.radius)
        close_enough = distance_to_goal <= 1.2

        # Получаем ориентацию робота в виде кватерниона (w, x, y, z)
        root_quat_w = self._robot.data.root_quat_w  # shape [N, 4]

        # Локальный вектор взгляда робота (вперёд по оси X)
        local_forward = torch.tensor([1.0, 0.0, 0.0], device=root_quat_w.device, dtype=root_quat_w.dtype)
        local_forward = local_forward.unsqueeze(0).repeat(root_quat_w.shape[0], 1)  # [N, 3]

        # Вектор взгляда в мировых координатах
        forward_w = self.quat_rotate(root_quat_w, local_forward)  # [N, 3]

        # Вектор от робота к цели
        root_pos_w = self._robot.data.root_pos_w  # [N, 3]
        to_goal = self._desired_pos_w - root_pos_w  # [N, 3]

        # Нормализуем векторы
        forward_w_norm = torch.nn.functional.normalize(forward_w[:, :2] , dim=1)
        to_goal_norm = torch.nn.functional.normalize(to_goal[:, :2] , dim=1)

        # Косинус угла между векторами взгляда и направления на цель
        cos_angle = torch.sum(forward_w_norm * to_goal_norm, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # для безопасности
        # direction_to_goal = to_goal
        # yaw_g = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])

        # Вычисляем угол между векторами
        angle = torch.acos(cos_angle)
        angle_degrees = angle * 180.0 / 3.141592653589793
        # Проверяем, что угол меньше порога
        facing_goal = angle_degrees < angle_threshold

        # Итоговое условие: близко к цели и смотрит в её сторону
        # print(distance_to_goal, angle_degrees)
        
        conditions = torch.stack([close_enough, facing_goal], dim=1)  # shape [N, M]
        num_conditions_met = conditions.sum(dim=1)  # shape [N], количество True в каждой строк

        # self.step_counter += torch.ones_like(self.step_counter) #torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # enouth_steps = self.step_counter > 4
        # returns = torch.logical_and(torch.logical_and(close_enough, facing_goal), enouth_steps)
        # self.step_counter = torch.where(returns, torch.zeros_like(self.step_counter), self.step_counter)
        returns = torch.logical_and(close_enough, facing_goal)
        # if torch.any(returns):
        #     print(close_enough, facing_goal)
        # print("returns", returns)
        if get_num_subs == False:
            return returns
        return returns, num_conditions_met

    def get_contact(self):
        force_matrix = self.scene["contact_sensor"].data.net_forces_w
        force_matrix[..., 2] = 0
        forces_magnitude = torch.norm(torch.norm(force_matrix, dim=2), dim=1)  # shape: [batch_size, num_contacts]
        # вычисляем модуль силы для каждого контакта
        if force_matrix is not None and force_matrix.numel() > 0:
            contact_forces = torch.norm(force_matrix, dim=-1)
            if contact_forces.dim() >= 3:
                has_contact = torch.any(contact_forces > 0.1, dim=(1, 2))
            else:
                has_contact = torch.any(contact_forces > 0.1, dim=1) if contact_forces.dim() == 2 else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # print("c ", has_contact)
            num_contacts_per_env = torch.sum(contact_forces > 1.0, dim=1)
            high_contact_envs = num_contacts_per_env >= 1
        else:
            print("force_matrix_w is None or empty")
        # if torch.any(high_contact_envs):
        #     print("high_contact_envs ", high_contact_envs)
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
        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = self.episode_length_buf >= 256 - 1
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        
        has_contact = self.get_contact()
        self.has_contact = has_contact
        died = torch.logical_or(
            torch.logical_or(self.goal_reached(distance_to_goal, get_num_subs=False), has_contact),
            time_out,
        )
        
        if torch.any(died):
            goal_reached = self.goal_reached(distance_to_goal)
            self.episode_counter += died.long()
            self.success_counter += goal_reached.long()
            self.event_update_counter += torch.sum(died).item()
            # print("died ", died, goal_reached, time_out)
        self.episode_length_buf[died] = 0
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # base_x = 12.0
        # robot_pos = torch.stack([torch.full((self.num_envs,), base_x, device=self.device),
        #                       torch.zeros(self.num_envs, device=self.device)], dim=1)
        # joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        # joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        # default_root_state = self._robot.data.default_root_state[env_ids].clone()
        # default_root_state[:, :2] = robot_pos
        # default_root_state[:, 2] = 0.1
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.get_success_rate()
        self._update_chairs(env_ids)
        self.update_from_counters(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        env_ids = env_ids.to(dtype=torch.long)

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids, :2] - self._robot.data.root_pos_w[env_ids, :2], dim=1
        ).mean()
        self.tensorboard_writer.add_scalar("Metrics/Success_Rate", self.success_rate, self.tensorboard_step)
        self.tensorboard_writer.add_scalar("Metrics/max_angle", self.max_angle, self.tensorboard_step)
        self.tensorboard_writer.add_scalar("Metrics/mean_radius", self.mean_radius, self.tensorboard_step)
        self.tensorboard_writer.add_scalar("Metrics/contact", torch.count_nonzero(self.has_contact).item(), self.tensorboard_step)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids) #maybe this shuld be the first
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.zeros_like(self.episode_length_buf) #, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._desired_pos_w[env_ids, :2] = self._terrain.env_origins[env_ids, :2]
        
        min_radius_x = torch.tensor([1.1]) #((self.level + 1.5 + 0.2) * int(self.level > 0) + 1)**2 + 3**2])
        min_radius = torch.sqrt(min_radius_x)[0]
        robot_pos, quaternion, goal_pos = self.scene_manager.reset(env_ids, self._terrain.env_origins, self.mean_radius, min_radius, self.max_angle)
        # print("in reset robot pose ", robot_pos, goal_pos)
        self._desired_pos_w[env_ids, :2] = goal_pos
        self._desired_pos_w[env_ids, 2] = 0.6
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :2] = robot_pos
        default_root_state[:, 2] = 0.1
        default_root_state[:, 3:7] = quaternion
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def update_from_counters(self, env_ids: torch.Tensor):
        if self.success_rate >= 90 and self._step_update_counter >= 50:
            self.sucess_ep_num += 1
            if self.sucess_ep_num > 8:
                self.sucess_ep_num = 0
                print("success_rate, self._step_update_counter: ", self.success_rate, self._step_update_counter)
                self.num_update_level += 0.1
                self.max_angle += torch.pi / 18
                if self.max_angle > torch.pi / 6:
                    self.max_angle = 0
                    self.mean_radius += 0.2
                self._step_update_counter = 0
                print("udate r,a: ",self.mean_radius, self.max_angle )

        self._obstacle_update_counter += 1
        self._step_update_counter += 1
        return None

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

    def close(self):
        super().close()

    def _update_chairs(self, env_ids: torch.Tensor = None):
        """
        Args:
            env_ids: torch.Tensor, индексы сред для обновления. Если None, обновляются все среды.
        """
        if self.first_ep:
            self.first_ep = False
            min_num_active = 0
            max_num_active = 0
        else:
            min_num_active=0
            max_num_active=0
        self.scene_manager.generate_obstacle_positions(env_ids=env_ids,terrain_origins=self._terrain.env_origins,
                                                        min_num_active=min_num_active,max_num_active=max_num_active)
        # Обновляем позиции стульев для всех сред
        for i in range(self.scene_manager.num_obstacles):  # Для каждого стула
            root_poses = torch.zeros((len(env_ids), 7), device=self.device)  # [num_envs, 7] (x, y, z, qw, qx, qy, qz)
            for j, env_id in enumerate(env_ids):
                # Получаем информацию о препятствии из obstacle_manager
                node = self.scene_manager.graphs[env_id.item()].graph.nodes[i]
                pos = node['position']
                # Заполняем позицию и ориентацию
                root_poses[j, 0] = pos[0]  # x
                root_poses[j, 1] = pos[1]  # y
                root_poses[j, 2] = pos[2]  # z
                root_poses[j, 3] = 1.0  # qw
                root_poses[j, 4:7] = 0.0  # qx, qy, qz

            # Добавляем смещение среды
            root_poses[:, :2] += self._terrain.env_origins[env_ids, :2]
            self.chair_objects[i].write_root_pose_to_sim(root_poses, env_ids=env_ids)