"""
Упрощенные сети для PPO
---------------------------------------------------------
Архитектура:
  - Policy: img + gt_orientation → действия
  - Value: img + gt_orientation → V(s)
  
Без GraphEncoder и OrientationModule - только обучение на GT данных
"""
import torch
import torch.nn as nn
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler


# =====================================================================
# Preprocessor
# =====================================================================
class DictRunningStandardScaler(nn.Module):
    """Нормализует только img, остальное оставляет как есть."""
    def __init__(self, size, img_space, memory_space, goal_space=None, device=None, epsilon=1e-8, clip_threshold=5.0):
        super().__init__()
        self.full_space = size
        self.img_scaler = RunningStandardScaler(
            size=img_space, epsilon=epsilon,
            clip_threshold=clip_threshold, device=device,
        )
        self.memory_scaler = RunningStandardScaler(
            size=memory_space, epsilon=epsilon,
            clip_threshold=clip_threshold, device=device,
        )
        if goal_space is not None:
            self.goal_scaler = RunningStandardScaler(
                size=goal_space, epsilon=epsilon,
                clip_threshold=clip_threshold, device=device,
            )
        else:
            self.goal_scaler = None

    def forward(self, x, train=False, inverse=False, no_grad=True):
        s = unflatten_tensorized_space(self.full_space, x)
        s["img"] = self.img_scaler(s["img"], train=train, inverse=inverse, no_grad=no_grad)
        s["memory"] = self.memory_scaler(s["memory"], train=train, inverse=inverse, no_grad=no_grad)
        if self.goal_scaler is not None and "goal" in s:
            s["goal"] = self.goal_scaler(s["goal"], train=train, inverse=inverse, no_grad=no_grad)

        return flatten_tensorized_space(s)


# =====================================================================
# PPO Policy (Actor)
# =====================================================================
class Policy(GaussianMixin, Model):
    """Policy для PPO: использует img + gt_orientation
    
    На вход: img + gt_orientation
    На выход: среднее и std для гауссова распределения действий
    """
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        img_dim = int(observation_space["img"].shape[0])
        mlp_in = img_dim + 1  # img + gt_orientation

        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        gt_orientation = states["orientation"].to(self.device)

        # Конкатенируем img и gt_orientation
        x = torch.cat([img, gt_orientation], dim=-1)
        mu = self.net(x)
        return mu, self.log_std_parameter, {}


# =====================================================================
# PPO Value Function
# =====================================================================
class Value(DeterministicMixin, Model):
    """Value function для PPO: использует img + gt_orientation
    
    На вход: img + gt_orientation
    На выход: скалярное значение V(s)
    """
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        img_dim = int(observation_space["img"].shape[0])
        mlp_in = img_dim + 1  # img + gt_orientation

        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        gt_orientation = states["orientation"].to(self.device)

        # Конкатенируем img и gt_orientation
        x = torch.cat([img, gt_orientation], dim=-1)
        v = self.net(x)
        return v, {}