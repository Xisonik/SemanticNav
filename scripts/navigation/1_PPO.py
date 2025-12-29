import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# ./isaaclab.sh -p scripts/demos/torch_ant_ppo.py --enable_cameras

# seed for reproducibility
set_seed()  # e.g. set_seed(42)


# define models (stochastic policy and value function) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        )
        # global log-std as in SAC version
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # PPO expects mean + log_std
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # value depends only on state
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# load and wrap the Isaac Lab environment
env = load_isaaclab_env(
    task_name="Isaac-Aloha-Direct-v0",
    num_envs=32,
    headless=False,
    cli_args=[
        "--enable_cameras",
    ],
    # show_cfg=True
)

env = wrap_env(env)
device = env.device

# PPO is on-policy → memory = rollout buffer
# memory_size = rollouts * num_envs (ниже rollouts=32)
memory = RandomMemory(memory_size=32 * env.num_envs, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators)
# PPO uses 1 policy and 1 value model
models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
models["value"] = Value(env.observation_space, env.action_space, device)

# configure and instantiate the PPO agent
cfg = PPO_DEFAULT_CONFIG.copy()

# how many environment steps per update (rollout length)
cfg["rollouts"] = 32          # 32 * 32 envs = 1024 шагов перед обновлением
cfg["learning_epochs"] = 5    # сколько эпох оптимизации на один rollout
cfg["mini_batches"] = 4       # батчей на epoch (=> 20 градиентных шагов на rollout)

cfg["discount_factor"] = 0.99
cfg["lambda_"] = 0.95         # GAE(lambda)

# learning rate: можно одним числом для policy+value
cfg["learning_rate"] = 3e-4

cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2       # PPO clipping epsilon
cfg["value_clip"] = 0.2
cfg["entropy_loss_scale"] = 0.0   # можно поднять до 0.001–0.01 если нужно больше exploration
cfg["value_loss_scale"] = 2.5
cfg["kl_threshold"] = 0.0     # можно включить early stopping по KL при желании

cfg["random_timesteps"] = 0   # без random policy прогрева
cfg["learning_starts"] = 0    # PPO сразу учится с первого rollout

# state/observation normalization
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

# logging to TensorBoard and checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 10000
cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1_000_000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if True:
    # start training
    trainer.train()
else:
    agent.load("/home/xiso/IsaacLab/skrl/aloha_ppo/checkpoints/agent_76000.pt")
    # start evaluation
    trainer.eval()
