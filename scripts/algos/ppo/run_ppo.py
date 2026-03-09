# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from scripts.algos.ppo.helper import RolloutVideoWrapper
from scripts.algos.ppo.networks import Policy, Value, GraphEncoder
from scripts.algos.networks.networks_gt import DictRunningStandardScaler
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
set_seed(42)

"""
- Пайплайны:
    1. навигации - Aloha_nav
    2. поворота - Isaac-Aloha-Direct-v0
    2. поворота - Aloha_turn

- Для лайвстрима:
    headless=True,
    cli_args=["--enable_cameras", "--video", "--livestream", "2",],
"""
TASK_NAME = "Aloha_nav" 
EVAL = False
VIDEO = False
num_envs = 20
timestepslen = 300000
headless = True

if EVAL:
    timestepslen = 800

cli_args = ["--enable_cameras"]

from comet_ml import start
from comet_ml.integration.pytorch import log_model
experiment = start(
    api_key="bbCMVUhDwSJsEqwcmhZ2MXdfE",
    project_name="robo",
    workspace="denmanorwat"
)

if EVAL:
    # from gymnasium.wrappers import RecordVideo
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
        headless=headless, 
        cli_args=cli_args
    )
    env = RolloutVideoWrapper(env, experiment, episode_frequency=10)

env = wrap_env(env)
device = env.device

graph_encoder = GraphEncoder(
    embeddings_path="source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
    image_dim=env.observation_space['img'].shape[0]
).to(device)

# Должно быть соразмерно с временем, за которое можно в теории добраться до целевого объекта.
memory = RandomMemory(memory_size = 200, num_envs = env.num_envs)

models = {
    "policy": Policy(
        env.observation_space, env.action_space, device,
        shared_graph = graph_encoder
    ),
    "value": Value(
        env.observation_space, env.action_space, device,
        shared_graph = graph_encoder
    )
}

# TODO change at least learning rate. It looks to big for standard PPO implementation.
# Maybe change some other hyperparameters.
cfg = PPO_DEFAULT_CONFIG.copy()
mini_batch_size = 256

cfg['rollouts'] = 200
cfg['orientation_loss_weight'] = 0.
cfg['learning_epochs'] = 10
cfg['learning_rate'] = 3e-4
cfg['entropy_loss_scale'] = 0.
cfg['mini_batches'] = env.num_envs * cfg['rollouts'] // mini_batch_size

cfg["state_preprocessor"] = DictRunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space,
    "img_space": env.observation_space["img"],
    "device": device,
}
# cfg["state_preprocessor"] = None  
# cfg["state_preprocessor_kwargs"] = {}
# TODO: delete state_preprocessor and make image net image statistics

cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo"

agent = PPO(
    models=models, memory=memory, cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space, device=device,
    comet_experiment = experiment)

trainer = SequentialTrainer(cfg={"timesteps": timestepslen}, env=env, agents=agent)
trainer.train()
#memory.save(directory="logs/skrl/memory")
#torch.save(agent._state_preprocessor.state_dict(), "logs/skrl/memory/preprocessor.pt")