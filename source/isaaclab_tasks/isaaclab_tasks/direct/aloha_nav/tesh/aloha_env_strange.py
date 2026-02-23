# env.py
from __future__ import annotations

import gymnasium as gym
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from .control_manager import ControlManager

from isaaclab_assets.robots.aloha import ALOHA_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # Оставляем только CUBOID_MARKER_CFG

class WheeledRobotEnvWindow(BaseEnvWindow):
    def __init__(self, env: 'WheeledRobotEnv', window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)
                    self._create_debug_vis_ui_element("obstacles", self.env)

@configclass
class WheeledRobotEnvCfg(DirectRLEnvCfg):
    episode_length_s = 60.0
    decimation = 4
    action_space = 2
    observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,), dtype="float32")
    state_space = 0
    debug_vis = True
    use_controller = True

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=5, replicate_physics=True)
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
        self._left_wheel_id = self._robot.find_joints("left_wheel")[0]
        self._right_wheel_id = self._robot.find_joints("right_wheel")[0]
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["goal_reached_reward", "out_of_bounds_penalty"]
        }
        self.control_manager = ControlManager(self.num_envs, self.device)
        
        # Добавляем препятствия
        self.obstacles = [[]]
        self.obstacle_markers = {}
        self.episode_counter = 0
        self.set_debug_vis(self.cfg.debug_vis)
        self.Debug = True
        self.prev_dist = 5
        self.event_history = torch.zeros((self.num_envs, 50), dtype=torch.float, device=self.device)
        self.event_history_index = 0
        self.event_history_filled = False
        self.event_update_counter = 0
        self.episode_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.count = 0

    def _setup_scene(self):
        self.control_manager = ControlManager(self.num_envs, self.device)
        self.obstacles = [[]]
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

        # Создаем препятствия только для базовой среды (env_0) один раз
        if False:
            print("Creating obstacles for base environment (env_0)")
            # Получаем данные только для одной среды (env_0)
            robot_pos, _, obstacle_data = self.control_manager.reset(torch.tensor([0], device=self.device), self._terrain.env_origins[:1])
            n_obstacles = obstacle_data["mask"][0].sum().item()
            print(f"Base env: Creating {n_obstacles} obstacles")
            
            for j in range(n_obstacles):
                pos = obstacle_data["positions"][0, j].cpu().numpy()
                size = obstacle_data["sizes"][0, j].cpu().numpy()
                type_idx = obstacle_data["types_idx"][0, j].item()
                color_idx = obstacle_data["colors_idx"][0, j].item()
                obs_type = obstacle_data["types"][type_idx]
                color = obstacle_data["colors"][color_idx].cpu().numpy()
                
                prim_path = f"/World/envs/env_0/Obstacles/obs_{j}"
                if obs_type == "cube":
                    cfg = RigidObjectCfg(
                        prim_path=prim_path,
                         init_state=RigidObjectCfg.InitialStateCfg(pos=(float(pos[0]), float(pos[1]), float(size[2])/2)),
                     spawn=sim_utils.CuboidCfg(
                            size=(float(size[0]), float(size[1]), float(size[2])),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(float(color[0]), float(color[1]), float(color[2]))),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                            collision_props=sim_utils.CollisionPropertiesCfg()
                        ),
                      )
                elif obs_type == "wall":
                    cfg = RigidObjectCfg(
                        prim_path=prim_path,
                        spawn=sim_utils.CuboidCfg(
                            size=(float(size[0]), float(size[1]), float(size[2])),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(float(color[0]), float(color[1]), float(color[2]))),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                            collision_props=sim_utils.CollisionPropertiesCfg()
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(pos=(float(pos[0]), float(pos[1]), float(size[2])/2))
                    )
                elif obs_type == "cylinder":
                    cfg = RigidObjectCfg(
                        prim_path=prim_path,
                        spawn=sim_utils.CylinderCfg(
                            radius=float(size[0]/2),
                            height=float(size[2]),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(float(color[0]), float(color[1]), float(color[2]))),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                            collision_props=sim_utils.CollisionPropertiesCfg()
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(pos=(float(pos[0]), float(pos[1]), float(size[2])/2))
                    )
                obj = RigidObject(cfg)
                self.obstacles[0].append(obj)  # Добавляем только в список для env_0
                self.scene.rigid_objects[f"env_0_obs_{j}"] = obj  # Уникальный ключ для базовой среды
            
            self._obstacles_created = True
            print("Obstacles created and will be cloned to all environments")
        else:
            print("Obstacles already created, skipping creation")

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
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
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        env_ids = env_ids.to(dtype=torch.long)

        # Добавим отладочную информацию
        print(f"Resetting env_ids: {env_ids}")
        print(f"Number of envs: {self.num_envs}")
        print(f"env_ids shape: {env_ids.shape}")
        print(f"Max env_id: {env_ids.max().item()}, Min env_id: {env_ids.min().item()}")

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

        # Получаем только позиции и ориентацию робота, не трогаем препятствия
        robot_pos, quaternion, _ = self.control_manager.reset(env_ids, self._terrain.env_origins)

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
            
            if not hasattr(self, "obstacle_visualizers"):
                self.obstacle_visualizers = {}
                cube_cfg = CUBOID_MARKER_CFG.copy()
                cube_cfg.prim_path = "/Visuals/Obstacles/cube"
                self.obstacle_visualizers["cube"] = VisualizationMarkers(cube_cfg)
                
                wall_cfg = CUBOID_MARKER_CFG.copy()
                wall_cfg.prim_path = "/Visuals/Obstacles/wall"
                self.obstacle_visualizers["wall"] = VisualizationMarkers(wall_cfg)
                
                # Создаем конфигурацию для цилиндра вручную
                cylinder_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Obstacles/cylinder",
                    markers={"cylinder": sim_utils.CylinderCfg(radius=0.1, height=0.1)}
                )
                self.obstacle_visualizers["cylinder"] = VisualizationMarkers(cylinder_cfg)
            
            self.goal_pos_visualizer.set_visibility(True)
            for vis in self.obstacle_visualizers.values():
                vis.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "obstacle_visualizers"):
                for vis in self.obstacle_visualizers.values():
                    vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        for env_id in range(self.num_envs):
            for j, obj in enumerate(self.obstacles[env_id]):
                pos = obj.data.root_pos_w[0].cpu().numpy()
                size = obj.cfg.spawn.size if hasattr(obj.cfg.spawn, "size") else (obj.cfg.spawn.radius*2, obj.cfg.spawn.radius*2, obj.cfg.spawn.height)
                color = obj.cfg.spawn.visual_material.diffuse_color
                obs_type = "cube" if isinstance(obj.cfg.spawn, sim_utils.CuboidCfg) and size[0] > 0.05 else "wall" if isinstance(obj.cfg.spawn, sim_utils.CuboidCfg) else "cylinder"
                if obs_type == "cylinder":
                    self.obstacle_visualizers[obs_type].visualize(
                        translations=torch.tensor([[pos[0], pos[1], size[2]/2]], device=self.device),
                        sizes=torch.tensor([[size[0]/2, size[2]]], device=self.device),
                        colors=torch.tensor([color], device=self.device)
                    )
                else:
                    self.obstacle_visualizers[obs_type].visualize(
                        translations=torch.tensor([[pos[0], pos[1], size[2]/2]], device=self.device),
                        sizes=torch.tensor([[size[0], size[1], size[2]]], device=self.device),
                        colors=torch.tensor([color], device=self.device)
                    )

    def close(self):
        super().close()

    # Оставляем остальные методы без изменений
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