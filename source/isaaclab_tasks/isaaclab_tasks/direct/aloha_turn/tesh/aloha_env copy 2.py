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

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from .control_manager import ControlManager  # Импорт ControlManager
import omni.kit.commands  # Для работы с камерой
import omni.usd  # Для доступа к сцене
from omni.kit.viewport.utility import capture_viewport_to_file  # Для захвата скриншотов

##
# Pre-defined configs
##
from isaaclab_assets.robots.aloha import ALOHA_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


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
        shape=(2,))
    # observation_space = 7
    observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,), dtype="float32")
    state_space = 0
    debug_vis = True
    use_controller = True  # Флаг для использования ControlManager

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=9, replicate_physics=True)
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
        width=128,
        height=128,
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
        history_length=3,  # Store recent contacts for stability
        debug_vis=True,  # Enable for debugging
        filter_prim_paths_expr=["/World/envs/env_.*"],  # Only detect kitchen collisions
    )
    write_image_to_file = False


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
        self.prev_dist = 5
        self.event_history = torch.zeros((self.num_envs, 50), dtype=torch.float, device=self.device)
        self.event_history_index = 0
        self.event_history_filled = False
        self.event_update_counter = 0
        self.episode_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.control_manager = ControlManager(self.num_envs, self.device)
        self.count = 0
        self._debug_log_enabled = True  # Флаг для включения/выключения отладки
        self._debug_envs_to_log = list(range(min(5, self.num_envs)))  # Логируем первые 5 сред
        self._inconsistencies = []
        self._debug_step_counter = 0
        self._debug_log_frequency = 10
        self._screenshot_dir = "/home/mipt/Downloads/IsaacLab-main/logs/camera_images/screenshots"
        os.makedirs(self._screenshot_dir, exist_ok=True)

        self.previous_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Настройка камеры для вида сверху
        # self._setup_top_view_camera()

    def _setup_top_view_camera(self):
        """Настраивает камеру для вида сверху на все среды."""
        env_spacing = self.scene.cfg.env_spacing
        num_envs = self.num_envs
        grid_size = int(math.ceil(math.sqrt(num_envs)))
        scene_center = torch.tensor([env_spacing * (grid_size - 1) / 2, env_spacing * (grid_size - 1) / 2, 0.0])
        scene_extent = env_spacing * grid_size

        camera_pos = torch.tensor([0.0, 0.0, 10])
        camera_target = scene_center

        # Создаем камеру с помощью omni.kit.commands
        camera_path = "/World/TopCamera"
        omni.kit.commands.execute(
            "CreatePrim",
            prim_path=camera_path,
            prim_type="Camera",
            attributes={
                "focusDistance": torch.norm(camera_pos - camera_target).item(),
                "focalLength": 24.0,
                "clippingRange": (0.1, 1000.0)
            }
        )

        # Устанавливаем позицию и ориентацию камеры
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        camera_prim.GetAttribute("xformOp:translate").Set(tuple(camera_pos.tolist()))


        # # Устанавливаем камеру для основного viewport
        # viewport_api = omni.kit.viewport.utility.get_active_viewport()
        # if viewport_api:
        #     viewport_api.set_active_camera(str(camera_path))

    def _save_top_view_screenshot(self, step: int):
        """Сохраняет скриншот всей сцены сверху."""
        try:
            viewport_api = omni.kit.viewport.utility.get_active_viewport()
            if viewport_api is None:
                raise RuntimeError("No active viewport found")
            filename = os.path.join(self._screenshot_dir, f"top_view_screenshot_step{step}.png")
            capture_viewport_to_file(viewport_api, filename)
            print(f"  Top view screenshot saved: {filename}")
        except Exception as e:
            self._inconsistencies.append(f"Failed to save top view screenshot: {str(e)}")

    def _check_consistency(self):
        """Проверяет соответствие действий и наблюдений с данными из _log_debug_info."""
        self._inconsistencies = []

        root_pos_w = self._robot.data.root_pos_w[:, :2]
        env_origins = self._terrain.env_origins[:, :2]
        local_pos = root_pos_w - env_origins
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        theta = torch.atan2(root_quat_w[:, 3], root_quat_w[:, 0])
        root_lin_vel_w = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        root_ang_vel_w = self._robot.data.root_ang_vel_w[:, 2]
        goal_pos_w = self._desired_pos_w[:, :2]
        goal_pos_local = goal_pos_w - env_origins

        for env_id in self._debug_envs_to_log:
            actions = self._actions[env_id].cpu().numpy()
            if not (np.all(actions >= -1.0) and np.all(actions <= 1.0)):
                self._inconsistencies.append(
                    f"Env {env_id}: Actions out of range [-1, 1]: {actions.tolist()}"
                )

            linear_speed = (actions[0] + 1.0) / 2
            angular_speed = actions[1]
            expected_lin_vel = linear_speed
            expected_ang_vel = angular_speed
            actual_lin_vel = root_lin_vel_w[env_id].item()
            actual_ang_vel = root_ang_vel_w[env_id].item()
            lin_vel_diff = abs(actual_lin_vel - expected_lin_vel)
            ang_vel_diff = abs(actual_ang_vel - expected_ang_vel)
            if lin_vel_diff > 0.5:
                self._inconsistencies.append(
                    f"Env {env_id}: Linear velocity mismatch. Expected: {expected_lin_vel:.3f}, Actual: {actual_lin_vel:.3f}"
                )
            if ang_vel_diff > 0.5:
                self._inconsistencies.append(
                    f"Env {env_id}: Angular velocity mismatch. Expected: {expected_ang_vel:.3f}, Actual: {actual_ang_vel:.3f}"
                )

        obs = self._get_observations()["policy"]
        if obs.shape != (self.num_envs, 7):
            self._inconsistencies.append(
                f"Observation shape mismatch. Expected: ({self.num_envs}, 7), Actual: {obs.shape}"
            )

        for env_id in self._debug_envs_to_log:
            obs_env = obs[env_id].cpu().numpy()
            if not np.isclose(obs_env[0], root_lin_vel_w[env_id].item(), atol=1e-5):
                self._inconsistencies.append(
                    f"Env {env_id}: Observation lin_vel mismatch. Obs: {obs_env[0]:.3f}, Debug: {root_lin_vel_w[env_id].item():.3f}"
                )
            if not np.isclose(obs_env[1], root_ang_vel_w[env_id].item(), atol=1e-5):
                self._inconsistencies.append(
                    f"Env {env_id}: Observation ang_vel mismatch. Obs: {obs_env[1]:.3f}, Debug: {root_lin_vel_w[env_id].item():.3f}"
                )
            if not np.isclose(obs_env[2], root_pos_w[env_id][0].item(), atol=1e-5):
                self._inconsistencies.append(
                    f"Env {env_id}: Observation pos_x mismatch. Obs: {obs_env[2]:.3f}, Debug: {root_pos_w[env_id][0].item():.3f}"
                )
            if not np.isclose(obs_env[3], root_pos_w[env_id][1].item(), atol=1e-5):
                self._inconsistencies.append(
                    f"Env {env_id}: Observation pos_y mismatch. Obs: {obs_env[3]:.3f}, Debug: {root_pos_w[env_id].item():.3f}"
                )
            if not np.isclose(obs_env[4], goal_pos_w[env_id][0].item(), atol=1e-5):
                self._inconsistencies.append(
                    f"Env {env_id}: Observation goal_x mismatch. Obs: {obs_env[4]:.3f}, Debug: {goal_pos_w[env_id][0].item():.3f}"
                )
            if not np.isclose(obs_env[5], goal_pos_w[env_id][1].item(), atol=1e-5):
                self._inconsistencies.append(
                    f"Env {env_id}: Observation goal_y mismatch. Obs: {obs_env[5]:.3f}, Debug: {goal_pos_w[env_id][1].item():.3f}"
                )
            if not np.isclose(obs_env[6], theta[env_id].item(), atol=1e-5):
                self._inconsistencies.append(
                    f"Env {env_id}: Observation theta mismatch. Obs: {obs_env[6]:.3f}, Debug: {theta[env_id].item():.3f}"
                )

        if obs.dtype != torch.float32:
            self._inconsistencies.append(
                f"Observation dtype mismatch. Expected: float32, Actual: {obs.dtype}"
            )

    def _log_debug_info(self):
        """Логирует отладочную информацию в файл, делает скриншот сверху и проверяет соответствие."""
        if not self._debug_log_enabled:
            return
        self._debug_step_counter += 1
        if self._debug_step_counter % self._debug_log_frequency != 0:
            return

        root_pos_w = self._robot.data.root_pos_w[:, :2]
        env_origins = self._terrain.env_origins[:, :2]
        local_pos = root_pos_w - env_origins
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        theta = torch.atan2(root_quat_w[:, 3], root_quat_w[:, 0])
        theta_deg = theta * 180.0 / math.pi
        root_lin_vel_w = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        root_ang_vel_w = self._robot.data.root_ang_vel_w[:, 2]
        goal_pos_w = self._desired_pos_w[:, :2]
        goal_pos_local = goal_pos_w - env_origins

        self._check_consistency()

        log_lines = [f"\n=== Debug Info {self._debug_step_counter} ==="]
        for env_id in self._debug_envs_to_log:
            global_pos = root_pos_w[env_id].cpu().numpy()
            local_pos_env = local_pos[env_id].cpu().numpy()
            goal_global = goal_pos_w[env_id].cpu().numpy()
            goal_local = goal_pos_local[env_id].cpu().numpy()
            log_lines.append(f"Environment {env_id}:")
            log_lines.append(f"  Robot Global Pos (x, y): [{global_pos[0]:.3f}, {global_pos[1]:.3f}]")
            log_lines.append(f"  Robot Local Pos (x, y): [{local_pos_env[0]:.3f}, {local_pos_env[1]:.3f}]")
            log_lines.append(f"  Robot Angle (rad): {theta[env_id].item():.3f}")
            log_lines.append(f"  Robot Angle (deg): {theta_deg[env_id].item():.3f}")
            log_lines.append(f"  Linear Velocity: {root_lin_vel_w[env_id].item():.3f}")
            log_lines.append(f"  Angular Velocity: {root_ang_vel_w[env_id].item():.3f}")
            log_lines.append(f"  Goal Global Pos (x, y): [{goal_global[0]:.3f}, {goal_global[1]:.3f}]")
            log_lines.append(f"  Goal Local Pos (x, y): [{goal_local[0]:.3f}, {goal_local[1]:.3f}]")
            log_lines.append(f"  Actions (linear, angular): [{self._actions[env_id][0]:.3f}, {self._actions[env_id][1]:.3f}]")

        self._save_top_view_screenshot(self._debug_step_counter)

        if self._inconsistencies:
            log_lines.append("\n=== Inconsistencies ===")
            for inconsistency in self._inconsistencies:
                log_lines.append(f"- {inconsistency}")
            log_lines.append("======================")
        else:
            log_lines.append("\n=== No Inconsistencies Found ===")
        log_lines.append("==================\n")

        with open(os.path.join(self._screenshot_dir, "debug_log.txt"), "a") as f:
            f.write("\n".join(log_lines))

        print("\n".join(log_lines))

    def _setup_scene(self):
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        from isaaclab.sensors import ContactSensor

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        spawn_from_usd(
            prim_path="/World/envs/env_.*/Kitchen",
            cfg=self.cfg.kitchen,
            translation=(3.0, 4.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        # print("actions: ", self._actions)
        r = self.cfg.wheel_radius
        L = self.cfg.wheel_distance
        
        if self.cfg.use_controller:
            control_actions = self.control_manager.compute_control(
                self._robot.data.root_pos_w,
                self._robot.data.root_state_w[:, 3:7]
            )
            linear_speed = control_actions[:, 0]
            angular_speed = control_actions[:, 1]
            self._actions[:, 0] = (linear_speed * 2.0) - 1
            self._actions[:, 1] = angular_speed
        else:
            linear_speed = (self._actions[:, 0] + 1.0)/2
            angular_speed = self._actions[:, 1] 
        
        # print("angular_speed ", angular_speed, linear_speed)
        self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
        self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r

        return self._actions

    def _apply_action(self):
        wheel_velocities = torch.stack([self._left_wheel_vel, self._right_wheel_vel], dim=1).unsqueeze(-1)
        self._robot.set_joint_velocity_target(wheel_velocities, joint_ids=[self._left_wheel_id, self._right_wheel_id])

    def _get_observations(self) -> dict:
        # self._log_debug_info()
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        root_lin_vel_w = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1).unsqueeze(-1)
        root_ang_vel_w = self._robot.data.root_ang_vel_w[:, 2].unsqueeze(-1)
        theta = torch.atan2(root_quat_w[:, 3], root_quat_w[:, 0])
        env_origins = self._terrain.env_origins[:, :2]
        goal_pos_w = self._desired_pos_w[:, :2]
        local_pos = root_pos_w - env_origins
        goal_pos_local = goal_pos_w - env_origins

        obs = torch.cat([
            root_lin_vel_w,
            root_ang_vel_w,
            local_pos[:, 0].unsqueeze(-1),
            local_pos[:, 1].unsqueeze(-1),
            goal_pos_local[:, 0].unsqueeze(-1),
            goal_pos_local[:, 1].unsqueeze(-1),
            theta.unsqueeze(-1)
        ], dim=-1)
        camera_data = self._tiled_camera.data.output["rgb"].clone() / 255.0
        if False:
            from isaaclab.sensors import save_images_to_file
            save_images_to_file(camera_data, "wheeled_robot_rgb.png")
        # print("obs", obs)
        observations = {"policy": obs}#, "rgb": camera_data}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # lin_vel = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        # lin_vel_reward = lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
        # ang_vel = torch.abs(self._robot.data.root_ang_vel_w[:, 2])
        # ang_vel_reward = ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        out_of_bounds_penalty_scale = -5.0
        out_of_bounds = distance_to_goal > 5.0
        out_of_bounds_penalty = out_of_bounds.float() * out_of_bounds_penalty_scale
        goal_reached = self.goal_reached(distance_to_goal)
        goal_reached_reward_scale = 15.0
        goal_reached_reward = goal_reached.float() * goal_reached_reward_scale
        
        moves = 100*(self.previous_distance_to_goal-distance_to_goal)
        self.previous_distance_to_goal = distance_to_goal
        contact_penalty = torch.zeros(self.num_envs, device=self.device)
        contact_penalty_scale = -15.0
        high_contact_envs = self.get_contact()
        contact_penalty[high_contact_envs] = contact_penalty_scale

        base_penalty_scale = -0.1
        base_penalty = torch.ones(self.num_envs, device=self.device) * base_penalty_scale

        rewards = {
            "base_penalty": base_penalty,
            "moves": moves,
            "goal_reached_reward": goal_reached_reward,
            "out_of_bounds_penalty": out_of_bounds_penalty,
            "contact_penalty": contact_penalty,
        }
        reward = (base_penalty + goal_reached_reward + contact_penalty + moves + out_of_bounds_penalty)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
    
    def get_contact(self):
        force_matrix = self.scene["contact_sensor"].data.net_forces_w
        if force_matrix is not None and force_matrix.numel() > 0:
            # print(f"force_matrix_w shape: {force_matrix.shape}")
            contact_forces = torch.norm(force_matrix, dim=-1)
            if contact_forces.dim() >= 3:
                has_contact = torch.any(contact_forces > 1.0, dim=(1, 2))
            else:
                has_contact = torch.any(contact_forces > 1.0, dim=1) if contact_forces.dim() == 2 else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            num_contacts_per_env = torch.sum(contact_forces > 1.0, dim=1)
            # print(f"Number of contacted bodies per env: {num_contacts_per_env.cpu().numpy().tolist()}")
            high_contact_envs = num_contacts_per_env > 5
            # print("high_contact_envs", high_contact_envs)
            if high_contact_envs.any():
                print(f"Environments with high contact penalty (>5 contacts): {torch.where(high_contact_envs)[0].cpu().numpy().tolist()}")
        else:
            print("force_matrix_w is None or empty")
        return high_contact_envs

    def get_success_rate(self) -> torch.Tensor:
        success_rate = torch.where(
            self.episode_counter > 0,
            (self.success_counter / self.episode_counter) * 100.0,
            torch.zeros_like(self.success_counter, dtype=torch.float)
        )
        return success_rate

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        
        has_contact = self.get_contact()
        
        died = torch.logical_or(
            torch.logical_or(self.goal_reached(distance_to_goal), distance_to_goal > 5.0),
            has_contact
        )
        
        if torch.any(died):
            goal_reached = self.goal_reached(distance_to_goal)
            self.episode_counter += died.long()
            self.success_counter += (died & goal_reached).long()
            self.event_update_counter += torch.sum(died).item()
        
        return died, time_out

    def goal_reached(self, distance_to_goal: torch.Tensor) -> torch.Tensor:
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        theta_robot = torch.atan2(root_quat_w[:, 3], root_quat_w[:, 0])
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        direction_to_goal = self._desired_pos_w[:, :2] - root_pos_w
        theta_goal = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
        distance_condition = distance_to_goal < 0.7
        angle_condition = torch.abs(theta_robot - theta_goal) < (math.pi / 3)
        return distance_condition

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
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
        extras["Metrics/success_rate"] = torch.mean(self.get_success_rate()).item()
        extras["Metrics/event_update_counter"] = self.event_update_counter
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        contact_forces = torch.norm(self._contact_sensor.data.net_forces_w[env_ids], dim=-1)
        has_contact = (contact_forces > 0.1).any(dim=-1)
        extras["Episode_Termination/contact"] = torch.count_nonzero(has_contact).item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._desired_pos_w[env_ids, :2] = self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = 0.0

        robot_pos, quaternion = self.control_manager.reset(env_ids, self._terrain.env_origins)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :2] = robot_pos
        default_root_state[:, 2] = 0.1
        default_root_state[:, 3:7] = quaternion
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.5, 0.5, 1.5)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def close(self):
        super().close()