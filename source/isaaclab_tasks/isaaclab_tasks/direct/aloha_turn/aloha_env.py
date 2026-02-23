from ..aloha_nav.aloha_env import WheeledRobotEnvCfg, WheeledRobotEnv, WheeledRobotEnvWindow
from ..aloha_nav.aloha_env import WheeledRobotEnv as WheeledRobotEnvBase

from .scene_manager import SceneManager
from .evaluation_manager import EvaluationManager
from .control_manager import VectorizedPurePursuit
from .path_manager import Path_manager
from .memory_manager import MemoryManager
from .asset_manager import AssetManager

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
from .scene_manager import SceneManager
from .evaluation_manager import EvaluationManager
from .control_manager import VectorizedPurePursuit
from .path_manager import Path_manager
from .memory_manager import MemoryManager, PathTracker
from .asset_manager import AssetManager
import omni.kit.commands
import omni.usd
import datetime
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
##
# Pre-defined configs
##

from isaaclab_assets.robots.aloha import ALOHA_CFG
from isaaclab.markers import CUBOID_MARKER_CFG
from transformers import CLIPProcessor, CLIPModel
import omni.kit.commands  # Уже импортировано в вашем коде
from omni.usd import get_context  # Для доступа к stage
from pxr import Gf
import json
import time

num_total_objects = 17
@configclass
class WheeledRobotEnvCfg(DirectRLEnvCfg):
    episode_length_s = 512.0
    decimation = 8
    action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        shape=(2,)
    )
    # Observation space is now the ResNet18 embedding size (512)
    m = 1  # Например, 3 эмбеддинга и действия
    # TODO automat compute num_total_objects
    num_total_objects = num_total_objects #36 12 num_total_objects * 5

    observation_space = gym.spaces.Dict({
        "img": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(512,), dtype=np.float32),  #518 512*4+4+2
        "memory": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(2056,)), 
        "goal": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(2,), dtype=np.float32),
        "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(1,), dtype=np.float32),
        "graph": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(24*num_total_objects,), dtype=np.float32)
    })
    state_space = 0
    debug_vis = False

    ui_window_class_type = WheeledRobotEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=0.2,
            dynamic_friction=0.15,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=18, replicate_physics=True)
    robot: ArticulationCfg = ALOHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheel_radius = 0.068
    wheel_distance = 0.34
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/box2_Link/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.35, 0, 1.1), rot=(0.99619469809,0,0.08715574274,0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0, focus_distance=2.0, horizontal_aperture=36, clipping_range=(0.2, 10.0)
        ),
        width=224,
        height=224,
    )
    current_dir = os.getcwd()
    room = sim_utils.UsdFileCfg(
        usd_path=os.path.join(current_dir, "source/isaaclab_assets/data/aloha_assets", "scenes/scenes_sber_kitchen_for_BBQ/room.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=False,
            rigid_body_enabled=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.1,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*"], #/obstacles/.*
    )


class WheeledRobotEnv(WheeledRobotEnvBase): 
    cfg: WheeledRobotEnvCfg

    def __init__(self, cfg: WheeledRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._super_init = True
        self.current_dir = os.getcwd()
        self.config_path=os.path.join(self.current_dir, "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_turn/scene_items.json")
        self.scene_objects = {}
        self.CAMERA = True
        self.memory_on = True
        # self.tracker = PathTracker(num_envs=self.num_envs, device=self.device)
        
        
        self.history_length_for_memory = 4
        self.orientation_history_length = 4
        self.orientation_history = torch.zeros(
            (self.num_envs, self.orientation_history_length),
            device=self.device,
            dtype=torch.float32
        )
        if self.memory_on:
            self.memory_manager = MemoryManager(
                num_envs=self.num_envs,
                embedding_size=512,  # Размер эмбеддинга ResNet18
                action_size=2,      # Размер действия (линейная и угловая скорость)
                history_length=self.history_length_for_memory,  # n = 10, можно настроить
                device=self.device
            )
        self._super_init = False
        self.eval = False
        self.eval_name = "CI_prelast"
        self.eval_printed = False
        self.scene_manager = SceneManager(self.num_envs, self.config_path, self.device)
        self.eval_manager = EvaluationManager(self.num_envs)
        self.eval_manager.set_task_lists(
            robot_positions=[[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                             [1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                             [2.0, -1.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]],  # список стартовых позиций
            angle_errors=[torch.pi, torch.pi*0.9, torch.pi*0.8]                         # список ошибок угла
        )
        self.use_controller = False
        self.imitation = False
        if self.imitation:
            self.use_controller = True
        if self.use_controller:
            self.path_manager = Path_manager(scene_manager=self.scene_manager, ratio=4.0, shift=[5, 5], device=self.device)
            self.control_module = VectorizedPurePursuit(num_envs=self.num_envs, device=self.device)
        self.scene_embeddings = torch.zeros(self.num_envs, 24*num_total_objects, device=self.device)


        self._actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._actions[:, 1] = 0.0
        self._left_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._right_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["moves"]
        }
        self._left_wheel_id = self._robot.find_joints("left_wheel")[0]
        self._right_wheel_id = self._robot.find_joints("right_wheel")[0]

        self.set_debug_vis(self.cfg.debug_vis)
        self.Debug = True
        self.event_update_counter = 0
        self.episode_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.possible_goal_position = []
        
        self.delete = 1
        self.count = 0
        self._debug_log_enabled = True
        self._debug_envs_to_log = list(range(min(5, self.num_envs)))
        self._inconsistencies = []
        self._debug_step_counter = 0
        self._debug_log_frequency = 10
        self.turn_on_controller = self.use_controller #it is not use or not use controller, it is flag for the first step
        self.turn_on_controller_step = 0
        self.my_episode_lenght = 256
        self.turn_off_controller_step = 0
        self.use_obstacles = True
        self.turn_on_obstacles = False
        self.turn_on_obstacles_always = False
        if self.turn_on_obstacles_always:
            self.use_obstacles = True
        self.previous_distance_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_angle_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_lin_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_ang_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.angular_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Initialize ResNet18 for image embeddings
        # self.resnet18 = models.resnet18(pretrained=True).to(self.device)
        # self.resnet18.eval()  # Set to evaluation mode
        # # Remove the final fully connected layer to get embeddings
        # self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        # # Image preprocessing for ResNet18
        # transforms.ToTensor()
        # self.transform = transforms.Compose([
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.success_rate = 0
        self.sr_stack_capacity = 0
        self.episode_completion_history = torch.zeros((self.num_envs*4, self.num_envs), dtype=torch.bool, device=self.device)
        self.success_history = torch.zeros((self.num_envs*4, self.num_envs), dtype=torch.bool, device=self.device)
        self.history_index = 0
        self.history_len = torch.zeros(self.num_envs, device=self.device)
        self._step_update_counter = 0
        self.mean_radius = 3.5
        self.max_angle_error = 0.3 * torch.pi
        self.cur_angle_error = torch.pi * 0.3
        self.warm = True
        self.warm_len = 1000
        self.without_imitation = self.warm_len / 2
        self.without_imitation_log = False
        self._obstacle_update_counter = 0
        self.has_contact = torch.full((self.num_envs,), True, dtype=torch.bool, device=self.device)
        self.sim = SimulationContext.instance()
        self.obstacle_positions = None
        self.key = None
        self.success_ep_num = 0
        # self.run = wandb.init(project="aloha_direct")
        self.first_ep = [True, True]
        self.first_ep_step = 0
        self.second_ep = True
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
        name = "dev"
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device)
        self.episode_count = 0
        self.total_episode_length = 0.0
        # self.tensorboard_writer = SummaryWriter(log_dir=f"/home/xiso/IsaacLab/logs/tensorboard/navigation_rl_{name}_{timestamp}")
        self.tensorboard_step = 0
        self.cur_step = 0
        self.velocities = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        if self.CAMERA:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()  # Установить в режим оценки
        self.second_try = 0
        self.foult_ep_num = 0
        # Инициализация стеков для хранения успехов (1 - успех, 0 - неуспех)
        self.success_stacks = [[] for _ in range(self.num_envs)]  # Список списков для каждой среды
        self.max_stack_size = 10  # Максимальный размер стека
        self.sr_stack_full = False
        self.start_mean_radius = 0
        self.min_level_radius = 0
        self.sr_treshhold = 85
        self.LOG = False
        import json
        from tabulate import tabulate
        self.debug_log_dir = os.path.join(os.getcwd(), "debug_logs")
        os.makedirs(self.debug_log_dir, exist_ok=True)
        self.last_log_step = 0
        self.last_log_step_t = 0
        self.EMERGANCY_STEP = 0
        self.text_embeddings = torch.zeros((self.num_envs, 512), device=self.device)
        if self.LOG:
            from comet_ml import start
            from comet_ml.integration.pytorch import log_model
            self.experiment = start(
                api_key="DRYfW6B6VtUQr9llvf3jup57R",
                project_name="general",
                workspace="xisonik"
            )
        self.print_config_info()
        # сразу после создания scene_manager
        self._material_cache = {}        # key -> material prim path (строка), key = "r_g_b"
        self._applied_color_map = {}     # obj_index (int) -> color_key (str), чтобы не биндим повторно
        self.DEBUG_TIME = False
        self.prev_root_pos = torch.zeros_like(self._robot.data.root_pos_w)
        self.prev_root_quat = torch.zeros_like(self._robot.data.root_quat_w)
        if self.DEBUG_TIME:
            self.operations_times = {}


    def goal_reached(self, angle_threshold: float = 20, radius_threshold: float = 1.3, get_num_subs=False):
        """
        Проверяет достижение цели с учётом расстояния и направления взгляда робота.
        distance_to_goal: [N] расстояния до цели
        angle_threshold: максимально допустимый угол в радианах между направлением взгляда и вектором на цель
        Возвращает: [N] булев тензор, True если цель достигнута
        """
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        # print("root: ", self.to_local(root_pos_w))
        # print("root_pos_w ", root_pos_w)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        # Проверка по расстоянию (например, радиус достижения stored в self.radius)
        close_enough = distance_to_goal <= radius_threshold

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

        # Вычисляем угол между векторами
        angle = torch.acos(cos_angle)
        angle_degrees = torch.abs(angle) * 180.0 / 3.141592653589793
        # Проверяем, что угол меньше порога
        facing_goal = angle_degrees < angle_threshold


        conditions = torch.stack([close_enough, facing_goal], dim=1)  # shape [N, M]
        num_conditions_met = conditions.sum(dim=1)  # shape [N], количество True в каждой строк

        returns = facing_goal
        if get_num_subs == False:
            return returns
        return returns, num_conditions_met, distance_to_goal+0.1-radius_threshold, angle_degrees

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        
        lin_vel_reward = torch.clamp(lin_vel*0.02, min=0, max=0.15)
        ang_vel_reward = torch.abs(self.angular_speed) * 0.1

        goal_reached, num_subs, r_error, a_error = self.goal_reached(get_num_subs=True)

        moves = torch.clamp(5 * (self.previous_distance_error - r_error), min=0, max=1) + \
                    torch.clamp(5 * (self.previous_angle_error - a_error), min=0, max=1)
        turnes =  torch.clamp(2 * math.pi * (self.previous_angle_error - a_error) / 180 , min=-1, max=1)

        F_s = -self.previous_angle_error 
        F_s_next = -a_error
        gamma = 0.99
        turnes += gamma * math.pi * (F_s_next - F_s)/ 180
        
        self.previous_angle_error = a_error

        has_contact = self.get_contact()

        time_out = self.is_time_out(self.my_episode_lenght-1)
        time_out_penalty = -1 * time_out.float()

        vel_penalty = -1 * (ang_vel_reward + lin_vel_reward)
        mask = ~goal_reached
        vel_penalty[mask] = 0
        lin_vel_reward[goal_reached] = 0

        collision_penalty = -1.0 * has_contact.float()
        timeout_penalty = -3.0 * time_out.float()

        goal_bonus = 4.0 * goal_reached.float()
        out = self.out_of_bounds()
        reward = -0.05 + 2*turnes + collision_penalty + timeout_penalty + goal_bonus - 3 * out.float()

        died, _ = self._get_dones(self.my_episode_lenght - 1, inner=True)
        if torch.any(died):
            sr = self.update_success_rate(goal_reached)
        check = {
            "moves":moves,
        }
        for key, value in check.items():
            self._episode_sums[key] += value

        self.previous_distance_error = r_error
        return reward
    

    def _pre_physics_step(self, actions: torch.Tensor):
        r = self.cfg.wheel_radius
        L = self.cfg.wheel_distance
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self.turn_off_controller_step += 1
        linear_speed = 0.6*(self._actions[:, 0] + 1.0) # [num_envs], всегда > 0
        angular_speed = 2*self._actions[:, 1]  # [num_envs], оставляем как есть от RL
        linear_speed = torch.zeros_like(self._actions[:, 0])
        angular_speed = torch.full_like(self._actions[:, 1], -2.0)
        self.angular_speed = angular_speed
        self.velocities = torch.stack([linear_speed, angular_speed], dim=1)
        self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
        self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r