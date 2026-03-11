# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space
from ppo.helper import RolloutVideoWrapper

from networks.networks_orm_ppo import (
    GraphEncoder, OrientationModule, Policy, Value,
    DictRunningStandardScaler, AuxModuleTrainer,
    collect_orientation_data, print_orientation_accuracy
)
set_seed(42)


# DictRunningStandardScaler импортируется из networks_orm_ppo


"""
- Пайплайны:
    1. навигации - Aloha_nav
    2. поворота - Isaac-Aloha-Direct-v0
    2. поворота - Aloha_turn

- Для лайвстрима:
    headless=True,
    cli_args=["--enable_cameras", "--video", "--livestream", "2",],
"""

from comet_ml import start
from comet_ml.integration.pytorch import log_model
experiment = start(
    api_key="bbCMVUhDwSJsEqwcmhZ2MXdfE",
    project_name="robo",
    workspace="denmanorwat"
)


TASK_NAME = "Aloha_nav"
EVAL = False
VIDEO = False
LIVESTREAM = False
USE_PRETRAINED = False

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
# env = RolloutVideoWrapper(env, experiment, episode_frequency=10)

if VIDEO:
    env = RecordVideo(
        env,
        video_folder="logs/skrl/videos",
        name_prefix="aloha_eval",
        episode_trigger=lambda ep: True,
    )

env = wrap_env(env)
device = env.device

graph_encoder = GraphEncoder(
    embeddings_path="source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
).to(device)

orient_module = OrientationModule(
    img_dim=env.observation_space["img"].shape[0],
).to(device)

graph_encoder.eval()
orient_module.eval() # custom trainer turn it to train in train steps

# PPO Memory
memory = RandomMemory(memory_size=100, num_envs=env.num_envs, device=device)

models = {
    "policy": Policy(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
    "value": Value(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
}

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 100  # должно совпадать с memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 8

cfg["discount_factor"] = 0.99           # gamma
cfg["lambda"] = 0.95                    # GAE lambda

# Learning rates
cfg["learning_rate"] = 3e-4             # learning rate для обеих сетей
cfg["learning_rate_scheduler"] = None   # можно добавить scheduler

# PPO-specific
cfg["ratio_clip"] = 0.2                 # clipping parameter для PPO
cfg["value_clip"] = 0.                  # clipping для value loss
cfg["clip_predicted_values"] = False     # включить value clipping

# Regularization
cfg["entropy_loss_scale"] = 0.01        # коэффициент entropy bonus
cfg["value_loss_scale"] = 0.5           # коэффициент value loss
cfg["grad_norm_clip"] = 0.5             # gradient clipping

# Advantages
cfg["value_preprocessor"] = None        # можно добавить нормализацию advantages
cfg["advantages_clip"] = None           # можно добавить clipping для advantages

cfg["state_preprocessor"] = DictRunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space,
    "img_space": env.observation_space["img"],
    "memory_space": env.observation_space["memory"],
    "goal_space": env.observation_space["goal"],
    "device": device,
}

cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo"

agent = PPO(
    models=models, memory=memory, cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space, device=device,
    comet_experiment=experiment
)

# Auxiliary trainer + callback
aux_trainer = AuxModuleTrainer(
    graph_encoder=graph_encoder,
    orient_module=orient_module,
    agent=agent,
    obs_space=env.observation_space,
    device=device,
    lr_graph=3e-3,
    lr_orient=3e-3,
    batch_size=1024,
    train_steps_per_call=1,
    log_interval=1000,
)

_original_post = agent.post_interaction
mode_1 = False
def _post_with_aux(timestep, timesteps):
    _original_post(timestep, timesteps)
    if not mode_1:
        aux_trainer.step(timestep)

    if timestep % 2000 == 0:
        save_dir = cfg["experiment"]["directory"]
        #torch.save(graph_encoder.state_dict(), f"{save_dir}/added/graph_encoder_{timestep}.pt")
        #torch.save(orient_module.state_dict(), f"{save_dir}/added/orient_module_{timestep}.pt")
    
    if timestep % 50 == 0:
        metrics = env.unwrapped.get_metrics()
        if timestep % 2000 == 0:
            print(metrics)
        acc_10, acc_20, acc_30 = print_orientation_accuracy(True)
        experiment.log_metric("success_rate", metrics["success_rate"], step=timestep)
        experiment.log_metric("mean_radius", metrics["mean_radius"], step=timestep)
        experiment.log_metric("angle_error", metrics["cur_angle_error"], step=timestep)
        experiment.log_metric("stage", metrics["stage"], step=timestep)
        experiment.log_metric("avg_episode_length", metrics["avg_episode_length"], step=timestep)
        experiment.log_metric("assistance_ratio", metrics["assistance_ratio"], step=timestep)
        experiment.log_metric("assistance_num_envs", metrics["assistance_num_envs"], step=timestep)
        experiment.log_metric("accuracy orientation module 10 grad", acc_10, step=timestep)
        experiment.log_metric("accuracy orientation module 20 grad", acc_20, step=timestep)
        experiment.log_metric("accuracy orientation module 30 grad", acc_30, step=timestep)

agent.post_interaction = _post_with_aux

if mode_1:
    USE_PRETRAINED = True

if not EVAL:
    if USE_PRETRAINED:
        checkpoint_path = "logs/skrl/aloha_ppo"
        graph_encoder.load_state_dict(
            torch.load(f"{checkpoint_path}/added/graph_encoder_42000.pt")
        )
        orient_module.load_state_dict(
            torch.load(f"{checkpoint_path}/added/orient_module_42000.pt")
        )
        if mode_1:
            graph_encoder.eval()
            orient_module.eval()

            for param in graph_encoder.parameters():
                param.requires_grad = False

            for param in orient_module.parameters():
                param.requires_grad = False
    trainer = SequentialTrainer(cfg={"timesteps": timestepslen}, env=env, agents=agent)
    trainer.train()
else:
    checkpoint_path = "logs/skrl/aloha_ppo"
    agent_path = f"{checkpoint_path}/agent_50000.pt"
    agent.load(agent_path)
    graph_encoder.load_state_dict(
        torch.load(f"{checkpoint_path}/added/graph_encoder_30000.pt")
    )
    orient_module.load_state_dict(
        torch.load(f"{checkpoint_path}/added/orient_module_30000.pt")
    )
    trainer = SequentialTrainer(cfg={"timesteps": timestepslen}, env=env, agents=agent)
    trainer.eval()

    print_orientation_accuracy(True)
    metrics = env.unwrapped.get_metrics()
    print(metrics)