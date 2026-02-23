# -*- coding: utf-8 -*-
"""
SAC + отдельно обучаемые GraphEncoder и OrientationModule
---------------------------------------------------------
Архитектура:
  - GraphEncoder: CLIP text lookup + GATv2 → graph_emb (128)
  - OrientationModule: img + graph_emb → orientation angle
  - Actor/Critic: используют graph_emb и orientation через no_grad
  - GraphEncoder и OrientationModule имеют СВОИ оптимизаторы,
    обучаются из replay buffer по orientation loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space

from torch_geometric.nn import GATv2Conv, global_mean_pool

# =====================================================================
# Константы
# =====================================================================
NUM_GRAPH_NODES = 17
PER_OBJECT_DIM = 24
TEXT_EMB_DIM = 16
GRAPH_EMB_DIM = 128
NUM_ORIENT_BINS = 36
GOAL_NODE_INDEX = 0

set_seed(42)

# =====================================================================
# Edge builder
# =====================================================================


# =====================================================================
# Environment
# =====================================================================
EVAL = False

if EVAL:
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0", num_envs=4,
        cli_args=["--enable_cameras"],
    )
else:
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0", num_envs=4,
        headless=True, cli_args=["--enable_cameras"],
    )

env = wrap_env(env)
device = env.device

print(f"Device: {device} | Envs: {env.num_envs}")
print(f"Obs space: {env.observation_space}")
print(f"Act space: {env.action_space}")

# =====================================================================
# Shared modules (один экземпляр каждого)
# =====================================================================
graph_encoder = GraphEncoder(
    embeddings_path="source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
).to(device)

orient_module = OrientationModule(
    img_dim=env.observation_space["img"].shape[0],
).to(device)

# Начинаем в eval (actor/critic используют no_grad)
graph_encoder.eval()
orient_module.eval()

print(f"GraphEncoder params: {sum(p.numel() for p in graph_encoder.parameters()):,}")
print(f"OrientModule params: {sum(p.numel() for p in orient_module.parameters()):,}")

# =====================================================================
# Models
# =====================================================================
memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=device)

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

# =====================================================================
# SAC config
# =====================================================================
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 4
cfg["batch_size"] = 512
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 3e-4
cfg["critic_learning_rate"] = 3e-4
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 100
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0

cfg["state_preprocessor"] = DictRunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space,
    "img_space": env.observation_space["img"],
    "device": device,
}

cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "logs/skrl/aloha_sac"

agent = SAC(
    models=models, memory=memory, cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space, device=device,
)

# =====================================================================
# Auxiliary trainer + callback
# =====================================================================
aux_trainer = AuxModuleTrainer(
    graph_encoder=graph_encoder,
    orient_module=orient_module,
    agent=agent,
    obs_space=env.observation_space,
    device=device,
    lr_graph=3e-4,
    lr_orient=1e-3,
    batch_size=512,
    train_steps_per_call=2,
    log_interval=50,
)

# Подключаем к post_interaction
_original_post = agent.post_interaction

def _post_with_aux(timestep, timesteps):
    _original_post(timestep, timesteps)
    if timestep > cfg["learning_starts"]:
        aux_trainer.step(timestep)

agent.post_interaction = _post_with_aux

# =====================================================================
# Checkpoint loading (опционально)
# =====================================================================
# checkpoint_path = "logs/skrl/.../checkpoints/agent_XXXX.pt"
# agent.load(checkpoint_path)

# =====================================================================
# Train / Eval
# =====================================================================
if not EVAL:
    trainer = SequentialTrainer(cfg={"timesteps": 330000}, env=env, agents=agent)
    trainer.train()
else:
    checkpoint_path = "..."
    agent.load(checkpoint_path)
    trainer = SequentialTrainer(cfg={"timesteps": 1500}, env=env, agents=agent)
    trainer.eval()