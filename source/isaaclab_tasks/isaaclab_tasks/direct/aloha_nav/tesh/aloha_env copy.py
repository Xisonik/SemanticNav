# env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math

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
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from .control_manager import ControlManager  # Импорт ControlManager

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
    action_space = 2
    observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,), dtype="float32")
    state_space = 0
    debug_vis = True
    use_controller = False  # Флаг для использования ControlManager

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=5, replicate_physics=True)
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
            for key in ["goal_reached_reward", "out_of_bounds_penalty"]
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
        # Удаляем старые переменные, которые теперь в ControlManager
        # self.radius, self.start_angle_error, self.prev_max_dist

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        print(self._actions)
        r = self.cfg.wheel_radius
        L = self.cfg.wheel_distance
        
        if self.cfg.use_controller:
            # Используем ControlManager
            control_actions = self.control_manager.compute_control(
                self._robot.data.root_pos_w,
                self._robot.data.root_state_w[:, 3:7]
            )
            linear_speed = control_actions[:, 0]
            angular_speed = control_actions[:, 1]
            self._actions[:, 0] = (linear_speed * 2.0) - 1 # Обратное масштабирование из [0, 2] в [-1, 1]
            self._actions[:, 1] = angular_speed
        else:
            # Используем actions от RL
            linear_speed = (self._actions[:, 0] + 1.0)/2
            angular_speed = self._actions[:, 1] 
        
        print("angular_speed ", angular_speed, linear_speed)
        self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
        self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r

        return self._actions

    def _apply_action(self):
        wheel_velocities = torch.stack([self._left_wheel_vel, self._right_wheel_vel], dim=1).unsqueeze(-1)
        self._robot.set_joint_velocity_target(wheel_velocities, joint_ids=[self._left_wheel_id, self._right_wheel_id])

    def _get_observations(self) -> dict:
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        root_lin_vel_w = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1).unsqueeze(-1)
        root_ang_vel_w = self._robot.data.root_ang_vel_w[:, 2].unsqueeze(-1)
        theta = torch.atan2(root_quat_w[:, 3], root_quat_w[:, 0])
        obs = torch.cat([
            root_lin_vel_w,
            root_ang_vel_w,
            root_pos_w[:, 0].unsqueeze(-1),
            root_pos_w[:, 1].unsqueeze(-1),
            self._desired_pos_w[:, 0].unsqueeze(-1),
            self._desired_pos_w[:, 1].unsqueeze(-1),
            theta.unsqueeze(-1)
        ], dim=-1)
        camera_data = self._tiled_camera.data.output["rgb"] / 255.0
        if False: #self.cfg.write_image_to_file:
            from isaaclab.sensors import save_images_to_file
            save_images_to_file(camera_data, "wheeled_robot_rgb.png")
        observations = {"policy": obs, "camera_rgb": camera_data}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        lin_vel_reward = lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
        ang_vel = torch.abs(self._robot.data.root_ang_vel_w[:, 2])
        ang_vel_reward = ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        out_of_bounds_penalty_scale = -100.0
        out_of_bounds = distance_to_goal > 5.0
        out_of_bounds_penalty = out_of_bounds.float() * out_of_bounds_penalty_scale
        goal_reached = self.goal_reached(distance_to_goal)
        goal_reached_reward_scale = 15.0
        goal_reached_reward = goal_reached.float() * goal_reached_reward_scale
        rewards = {"goal_reached_reward": goal_reached_reward, "out_of_bounds_penalty": out_of_bounds_penalty}
        reward = (-1 + goal_reached_reward + out_of_bounds_penalty)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

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
        died = torch.logical_or(self.goal_reached(distance_to_goal), distance_to_goal > 5.0)
        event_occurred = torch.logical_or(died, time_out)
        if torch.any(event_occurred):
            goal_reached = self.goal_reached(distance_to_goal)
            self.episode_counter += event_occurred.long()
            self.success_counter += (event_occurred & goal_reached).long()
            self.event_update_counter += torch.sum(event_occurred).item()
        return died, time_out

    def goal_reached(self, distance_to_goal: torch.Tensor) -> torch.Tensor:
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        theta_robot = torch.atan2(root_quat_w[:, 3], root_quat_w[:, 0])
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        direction_to_goal = self._desired_pos_w[:, :2] - root_pos_w
        theta_goal = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
        distance_condition = distance_to_goal < 0.7
        angle_condition = torch.abs(theta_robot - theta_goal) < (math.pi / 3)
        return distance_condition & angle_condition

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
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._desired_pos_w[env_ids, :2] = self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = 0.0

        # Используем ControlManager для сброса
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
                marker_cfg.markers["cuboid"].size = (0.5, 0.5, 1)
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