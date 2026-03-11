# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from ppo.helper import RolloutVideoWrapper

from networks.networks_orm import *
set_seed(42)

from comet_ml import start
from comet_ml.integration.pytorch import log_model
experiment = start(
    api_key="bbCMVUhDwSJsEqwcmhZ2MXdfE",
    project_name="robo",
    workspace="denmanorwat"
)

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
    api_key="DRYfW6B6VtUQr9llvf3jup57R",
    project_name="general",
    workspace="xisonik"
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


env = RolloutVideoWrapper(env, experiment, episode_frequency=1)
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

memory_size = 1500
if num_envs == 128:
    memory_size = 1500
elif num_envs == 64:
    memory_size = 2000
elif num_envs == 32:
    memory_size = 5000

print("memory size: ", memory_size)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

models = {
    "policy": StochasticActor(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
    "critic_1": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
    "critic_2": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
    "target_critic_1": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
    "target_critic_2": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
}

cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 512
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 3e-4 # 3e-4
cfg["critic_learning_rate"] = 3e-4 # 3e-4
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 100
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3 # 5e-3
cfg["initial_entropy_value"] = 1.0

cfg["state_preprocessor"] = DictRunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space,
    "img_space": env.observation_space["img"],
    "memory_space": env.observation_space["memory"],
    "goal_space": env.observation_space["goal"],
    "device": device,
}
# cfg["state_preprocessor"] = None  
# cfg["state_preprocessor_kwargs"] = {}
# TODO: delete state_preprocessor and make image net image statistics

cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "logs/skrl/aloha_sac"

agent = SAC(
    models=models, memory=memory, cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space, device=device,
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
        if timestep > cfg["learning_starts"]:
            aux_trainer.step(timestep)

    if timestep % 2000 == 0:
        save_dir = cfg["experiment"]["directory"]
        #torch.save(graph_encoder.state_dict(), f"{save_dir}/added/graph_encoder_{timestep}.pt")
        #torch.save(orient_module.state_dict(), f"{save_dir}/added/orient_module_{timestep}.pt")
    # if timestep % 2000 == 0:
    #     memory.save(directory="logs/skrl/memory")
    if timestep % 50 == 0:
        metrics = env.unwrapped.get_metrics()
        # print(metrics)
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
        checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_sac"
        # graph_encoder.load_state_dict(
        #     torch.load(f"{checkpoint_path}/added/graph_encoder_80000.pt")
        # )
        # orient_module.load_state_dict(
        #     torch.load(f"{checkpoint_path}/added/orient_module_80000.pt")
        # # )
        # checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_sac"
        # agent_path = f"{checkpoint_path}/26-02-26_16-35-01-718674_SAC/checkpoints/agent_42000.pt"
        # agent.load(agent_path)
        graph_encoder.load_state_dict(
            torch.load(f"logs/skrl/aloha_sac/added/graph_encoder_42000.pt")
        )
        orient_module.load_state_dict(
            torch.load(f"logs/skrl/aloha_sac/added/orient_module_42000.pt")
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
    memory.save(directory="logs/skrl/memory/4img_128")
    torch.save(agent._state_preprocessor.state_dict(), "logs/skrl/memory/4img_128/preprocessor.pt")
else:
    # memory.load(directory="logs/skrl/aloha_sac/memory")
    # agent._state_preprocessor.load_state_dict(
    #     torch.load("logs/skrl/aloha_sac/memory/preprocessor.pt") archive/memory
    # )
    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_sac"
    agent_path = f"{checkpoint_path}/26-03-10_23-34-50-145979_SAC/checkpoints/agent_50000.pt"
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