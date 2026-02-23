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

##
# Pre-defined configs
##
from isaaclab_assets.robots.aloha import ALOHA_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class WheeledRobotEnvWindow(BaseEnvWindow):
    """Window manager for the Wheeled Robot environment."""

    def __init__(self, env: 'WheeledRobotEnv', window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class WheeledRobotEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 60.0
    decimation = 4
    action_space = 2  # Linear speed (scalar) and angular speed (scalar)
    observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,), dtype="float32")
    state_space = 0
    debug_vis = True

    ui_window_class_type = WheeledRobotEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
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
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=15, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ALOHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheel_radius = 0.068
    wheel_distance = 0.34

    # reward scales
    lin_vel_reward_scale = 0.5
    ang_vel_reward_scale = -0.1
    distance_to_goal_reward_scale = 15.0

    write_image_to_file = False


class WheeledRobotEnv(DirectRLEnv):
    cfg: WheeledRobotEnvCfg

    def __init__(self, cfg: WheeledRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Actions: [linear_speed, angular_speed]
        self._actions = torch.ones((self.num_envs, 2), device=self.device)  # [num_envs, 2]
        self._actions[:, 1] = 0.0  # Угловая скорость = 0
        self._left_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._right_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["goal_reached_reward", "out_of_bounds_penalty"]
        }
        self._left_wheel_id = self._robot.find_joints("left_wheel")[0]
        self._right_wheel_id = self._robot.find_joints("right_wheel")[0]

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        self.Debug = True
        self.prev_dist = 5
        self.event_history = torch.zeros((self.num_envs, 50), dtype=torch.float, device=self.device)
        self.event_history_index = 0
        self.event_history_filled = False
        self.event_update_counter = 0
        self.episode_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Camera
        # self._tiled_camera = None  # Initialized in _setup_scene

        # Инициализация радиусов и углов
        num_envs = self.num_envs
        radius_values = torch.arange(0.3, 4, 0.1, device=self.device)  # [0.2, 0.3, ..., 4.0]
        num_radiuses = len(radius_values)
        angle_values = torch.tensor([-math.pi/3, -math.pi/4, -math.pi/8, 0, math.pi/8, math.pi/4, math.pi/3],
                                   device=self.device)
        num_angles = len(angle_values)
        total_combinations = num_radiuses * num_angles
        radius_base = radius_values.repeat_interleave(num_angles)
        angle_base = angle_values.repeat(num_radiuses)
        repeat_times = (num_envs + total_combinations - 1) // total_combinations
        self.radius = radius_base.repeat(repeat_times)[:num_envs]
        self.start_angle_error = angle_base.repeat(repeat_times)[:num_envs]
        self.prev_max_dist = 0
        self.previous_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        
        # Add camera
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # def _pre_physics_step(self, actions: torch.Tensor):
    #     self._actions = actions.clone().clamp(-1.0, 1.0)
    #     print(self._actions)
    #     linear_speed = self._actions[:, 0]
    #     angular_speed = self._actions[:, 1]
    #     coeff = 7
    #     self._left_wheel_vel = linear_speed * coeff
    #     self._right_wheel_vel = angular_speed * coeff

    def _pre_physics_step(self, actions: torch.Tensor):
        # Игнорируем actions и задаем фиксированное движение вперед
        self._actions = actions.clone().clamp(-1.0, 1.0)
        # print(self._actions)
        r = self.cfg.wheel_radius  # 0.068
        L = self.cfg.wheel_distance  # 0.34
        
        # # Фиксированная линейная скорость для всех роботов, угловая скорость = 0
        # linear_speed = torch.full((self.num_envs,), 5, device=self.device)  # [num_envs], 0.5 м/с
        # angular_speed = torch.zeros(self.num_envs, device=self.device)  # [num_envs]
        # Прибавляем 1 к линейной скорости и ограничиваем диапазон [0, 1]
        linear_speed = 0.7*(self._actions[:, 0] + 1.0) # [num_envs], всегда > 0
        angular_speed = 0.8*self._actions[:, 1]  # [num_envs], оставляем как есть от RL
        # Преобразование для дифференциального привода
        self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r  # [num_envs], рад/с
        self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r  # [num_envs], рад/с
        

    def _apply_action(self):
        wheel_velocities = torch.stack([self._left_wheel_vel, self._right_wheel_vel], dim=1).unsqueeze(-1)
        self._robot.set_joint_velocity_target(wheel_velocities, joint_ids=[self._left_wheel_id, self._right_wheel_id])

    def _get_observations(self) -> dict:
        root_pos_w = self._robot.data.root_pos_w[:, :2]-self._terrain.env_origins[:, :2]
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        root_lin_vel_w = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1).unsqueeze(-1)
        root_ang_vel_w = self._robot.data.root_ang_vel_w[:, 2].unsqueeze(-1)
        theta = torch.atan2(root_quat_w[:, 3], root_quat_w[:, 0])
        desired_pos_w = self._desired_pos_w[:, :2]-self._terrain.env_origins[:, :2]
        obs = torch.cat([
            root_lin_vel_w,
            root_ang_vel_w,
            root_pos_w[:, 0].unsqueeze(-1),
            root_pos_w[:, 1].unsqueeze(-1),
            desired_pos_w[:, 0].unsqueeze(-1),
            desired_pos_w[:, 1].unsqueeze(-1),
            theta.unsqueeze(-1)
        ], dim=-1)

        observations = {
            "policy": obs,
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        lin_vel_reward = lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
        ang_vel = torch.abs(self._robot.data.root_ang_vel_w[:, 2])
        ang_vel_reward = ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        out_of_bounds_penalty_scale = -10.0
        out_of_bounds = distance_to_goal > 5.0
        out_of_bounds_penalty = out_of_bounds.float() * out_of_bounds_penalty_scale
        goal_reached = self.goal_reached(distance_to_goal)
        goal_reached_reward_scale = 15.0
        goal_reached_reward = goal_reached.float() * goal_reached_reward_scale
        moves = 100*(self.previous_distance_to_goal-distance_to_goal)
        self.previous_distance_to_goal = distance_to_goal
        rewards = {
            "goal_reached_reward": goal_reached_reward,
            "out_of_bounds_penalty": out_of_bounds_penalty,
        }
        reward = (-1 + goal_reached_reward + lin_vel_reward + ang_vel_reward + out_of_bounds_penalty + moves)
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
        died = torch.logical_or(self.goal_reached(distance_to_goal), distance_to_goal > 10.0)
        if self.prev_max_dist != min(self.radius):
            self.prev_max_dist = min(self.radius)
        return died, time_out

    def goal_reached(self, distance_to_goal: torch.Tensor) -> torch.Tensor:
        """Проверяет, достигнута ли цель: расстояние < 0.5 и ошибка по углу < π/9."""
        distance_condition = distance_to_goal < 0.7
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
        extras["Metrics/average_start_angle_error"] = torch.mean(self.start_angle_error).item()
        extras["Metrics/average_radius"] = torch.mean(self.radius).item()
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

        num_envs = len(env_ids)
        angle_steps = torch.tensor([0.0, torch.pi / 12, torch.pi / 6, torch.pi / 3], device=self.device)
        num_radii = num_envs // 4
        if num_envs % 4 != 0:
            num_radii += 1
        radius_values = torch.linspace(3, 4.2, num_radii, device=self.device)
        radii = radius_values.repeat_interleave(4)[:num_envs]
        # angle_errors = angle_steps.repeat(num_radii)[:num_envs]
        angle_errors = torch.rand(num_envs, device=self.device) * torch.pi/2
        random_sign = torch.sign(torch.rand(num_envs, device=self.device) - 0.5)
        self.radius[env_ids] = radii
        self.start_angle_error[env_ids] = angle_errors * random_sign

        theta = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        x = self.radius[env_ids] * torch.cos(theta)
        y = self.radius[env_ids] * torch.sin(theta)
        robot_pos = torch.stack([x, y], dim=1)
        robot_pos += self._desired_pos_w[env_ids, :2]
        direction_to_goal = self._desired_pos_w[env_ids, :2] - robot_pos
        yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
        yaw += self.start_angle_error[env_ids]
        cos_yaw_half = torch.cos(yaw / 2.0)
        sin_yaw_half = torch.sin(yaw / 2.0)
        quaternion = torch.zeros((len(env_ids), 4), device=self.device)
        quaternion[:, 0] = cos_yaw_half
        quaternion[:, 3] = sin_yaw_half

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
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def close(self):
        """Cleanup for the environment."""
        super().close()