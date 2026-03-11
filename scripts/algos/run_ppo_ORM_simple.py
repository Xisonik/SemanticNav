# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from networks.networks_orm_ppo_simple import Policy, Value, DictRunningStandardScaler

set_seed(42)

"""
Упрощенное обучение PPO:
  - Policy: img + gt_orientation → действия
  - Value: img + gt_orientation → V(s)
  
Используются ТОЛЬКО наземные правда ориентация (GT)
"""

from comet_ml import start
from comet_ml.integration.pytorch import log_model
experiment = start(
    api_key="DRYfW6B6VtUQr9llvf3jup57R",
    project_name="general",
    workspace="xisonik"
)

# =====================================================================
# Конфиг
# =====================================================================
TASK_NAME = "Aloha_nav"
EVAL = False
VIDEO = False
LIVESTREAM = False

num_envs = 64
timestepslen = 100000
headless = True

if EVAL or VIDEO:
    timestepslen = 1000

if VIDEO:
    cli_args = ["--enable_cameras", "--video", "--livestream", "2"]
    from gymnasium.wrappers import RecordVideo
    num_envs = 2
    headless = True
elif LIVESTREAM:
    cli_args = ["--enable_cameras", "--livestream", "2"]
    num_envs = 4
    headless = True
else:
    cli_args = ["--enable_cameras"]

if headless == False:
    num_envs = 1

# =====================================================================
# Environment
# =====================================================================
if headless:
    env = load_isaaclab_env(
        task_name=TASK_NAME, 
        headless=headless, 
        num_envs=num_envs,
        cli_args=cli_args
    )
else:
    env = load_isaaclab_env(
        task_name=TASK_NAME, 
        num_envs=num_envs,
        cli_args=cli_args
    )

if VIDEO:
    env = RecordVideo(
        env,
        video_folder="logs/skrl/videos",
        name_prefix="aloha_eval",
        episode_trigger=lambda ep: True,
    )

env = wrap_env(env)
device = env.device

# =====================================================================
# Models
# =====================================================================
models = {
    "policy": Policy(
        env.observation_space, env.action_space, device
    ),
    "value": Value(
        env.observation_space, env.action_space, device
    ),
}

# =====================================================================
# Memory
# =====================================================================
memory = RandomMemory(memory_size=48, num_envs=env.num_envs, device=device)

# =====================================================================
# PPO Config
# =====================================================================
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 48
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 8

cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95

cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = None

cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True

cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 0.5
cfg["grad_norm_clip"] = 0.5

cfg["state_preprocessor"] = DictRunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space,
    "img_space": env.observation_space["img"],
    "memory_space": env.observation_space["memory"],
    "device": device,
}

cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo_simple"

# =====================================================================
# Agent
# =====================================================================
agent = PPO(
    models=models, memory=memory, cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space, device=device,
    comet_experiment = experiment)

# =====================================================================
# Callback для логирования метрик
# =====================================================================
_original_post = agent.post_interaction

def _post_with_logging(timestep, timesteps):
    _original_post(timestep, timesteps)
    
    if timestep % 50 == 0:
        metrics = env.unwrapped.get_metrics()
        if timestep % 2000 == 0:
            print(metrics)
        print(experiment)
        experiment.log_metric("success_rate", metrics["success_rate"], step=timestep)
        experiment.log_metric("mean_radius", metrics["mean_radius"], step=timestep)
        experiment.log_metric("angle_error", metrics["cur_angle_error"], step=timestep)
        experiment.log_metric("stage", metrics["stage"], step=timestep)
        experiment.log_metric("avg_episode_length", metrics["avg_episode_length"], step=timestep)
        experiment.log_metric("assistance_ratio", metrics["assistance_ratio"], step=timestep)
        experiment.log_metric("assistance_num_envs", metrics["assistance_num_envs"], step=timestep)

agent.post_interaction = _post_with_logging

# =====================================================================
# Training
# =====================================================================
if not EVAL:
    trainer = SequentialTrainer(cfg={"timesteps": timestepslen}, env=env, agents=agent)
    trainer.train()
else:
    checkpoint_path = "logs/skrl/aloha_ppo_simple"
    agent_path = f"{checkpoint_path}/agent_50000.pt"
    agent.load(agent_path)
    
    trainer = SequentialTrainer(cfg={"timesteps": timestepslen}, env=env, agents=agent)
    trainer.eval()

    metrics = env.unwrapped.get_metrics()
    print(metrics)