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
num_envs = 1
headless = True

cli_args = ["--enable_cameras", "--video"]

from comet_ml import start
from comet_ml.integration.pytorch import log_model
experiment = start(
    api_key="bbCMVUhDwSJsEqwcmhZ2MXdfE",
    project_name="robo",
    workspace="denmanorwat"
)

env = load_isaaclab_env(
    task_name=TASK_NAME, 
    num_envs=num_envs,
    headless=headless, 
    cli_args=cli_args
)
env = RolloutVideoWrapper(env, experiment, episode_frequency=1)

env = wrap_env(env)
env.reset()
episode_length = []
cur_episode_length = 0
for i in range(100_000):
    act = env.action_space.sample()
    act[0] = 1
    act = torch.from_numpy(act)
    obs, reward, terminated, truncated, info = env.step(act)
    cur_episode_length += 1
    if terminated[0] or truncated[0]:
        episode_length.append(cur_episode_length)
        if cur_episode_length == 1:
            print(f"episode {len(episode_length) - 1} of length 1")
        cur_episode_length = 0

experiment.log_metric("average_episode_length", sum(episode_length) / len(episode_length))
experiment.log_metric("max_episode_length", max(episode_length))
experiment.log_metric('episodes of length 1', sum(1 for length in episode_length if length == 1))