import argparse
import torch
import torch.nn as nn

# skrl / Isaac Lab imports
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space

# GNN
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ---------------------------------------------------------------------
# Глобальные настройки сцены / графа
# ---------------------------------------------------------------------

# Должно совпадать с количеством объектов в SceneManager / encode_scene_graph
NUM_GRAPH_NODES = 17          # M
PER_OBJECT_DIM = 24           # столько фич на объект из encode_scene_graph
TEXT_EMB_DIM = 16             # размер текстового эмбеддинга (имя+цвет)
GRAPH_EMB_DIM = 128           # выход графового энкодера

# В графе "цель ↔ объекты" нужно знать индекс узла цели.
# Если цель у тебя всегда первая в encode_scene_graph — оставляй 0.
GOAL_NODE_INDEX = 0

# Evaluation mode flag
EVAL = False

# ---------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------
def build_star_chain_edge_index(
    num_nodes: int,
    batch_size: int,
    device: torch.device,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Рёбра:
      - звезда: 0 <-> i для i=1..N-1
      - цепочка: i <-> i+1 для i=0..N-2
      - (опц.) self-loops
    """
    N = num_nodes
    src, dst = [], []

    # star 0 <-> i
    for i in range(1, N):
        src += [0, i]
        dst += [i, 0]

    # chain i <-> i+1
    for i in range(N - 1):
        src += [i, i + 1]
        dst += [i + 1, i]

    if add_self_loops:
        for i in range(N):
            src.append(i)
            dst.append(i)

    edge_index_single = torch.tensor([src, dst], device=device, dtype=torch.long)  # [2, E_single]

    edge_indices = [edge_index_single + b * N for b in range(batch_size)]
    return torch.cat(edge_indices, dim=1)  # [2, B*E_single]


def build_goal_star_edge_index(num_nodes: int, batch_size: int, device: torch.device, goal_index: int = 0,
                               add_self_loops: bool = True) -> torch.Tensor:
    """Граф 'звезда': двунаправленные рёбра между goal и каждым узлом.

    Рёбра:
      goal -> i
      i -> goal
    + (опционально) self-loop i->i для всех i.

    Возвращает edge_index формы [2, E_total] для батча из B графов.
    """
    assert 0 <= goal_index < num_nodes
    N = num_nodes
    g = goal_index

    # edges within single graph
    src = []
    dst = []
    for i in range(N):
        if i == g:
            continue
        src += [g, i]
        dst += [i, g]
    if add_self_loops:
        for i in range(N):
            src.append(i)
            dst.append(i)

    edge_index_single = torch.tensor([src, dst], device=device, dtype=torch.long)  # [2, E_single]

    # batch shift
    edge_indices = []
    for b in range(batch_size):
        edge_indices.append(edge_index_single + b * N)
    return torch.cat(edge_indices, dim=1)  # [2, B*E_single]

# ---------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------

class SceneGraphGATEncoder(nn.Module):
    """GATv2 encoder для сценового графа без edge_attr.

    ВНИМАНИЕ: edge_index строится как 'цель ↔ объекты' (звезда), поэтому память/время намного меньше,
    чем у полносвязного графа.
    """
    def __init__(
        self,
        num_nodes: int,
        node_in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = GRAPH_EMB_DIM,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
        goal_index: int = GOAL_NODE_INDEX,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        self.dropout = dropout
        self.goal_index = goal_index

        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        convs = []
        norms = []
        in_ch = hidden_dim
        for _ in range(num_layers):
            convs.append(
                GATv2Conv(
                    in_channels=in_ch,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    edge_dim=None,
                    dropout=dropout,
                    concat=True,
                )
            )
            norms.append(nn.LayerNorm(hidden_dim))
            in_ch = hidden_dim

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        # маленький кэш edge_index по (B, device)
        self._edge_cache = {}

    def prune_edges_by_distance(self, edge_index: torch.Tensor, node_pos_flat: torch.Tensor, max_dist: float) -> torch.Tensor:
        """
        edge_index: [2, E]
        node_pos_flat: [B*N, 3]
        """
        src, dst = edge_index[0], edge_index[1]
        d = node_pos_flat[src] - node_pos_flat[dst]
        dist = torch.norm(d, dim=-1)
        mask = dist <= max_dist
        return edge_index[:, mask]


    def _get_edge_index(self, B: int, device: torch.device) -> torch.Tensor:
        key = (B, device.index if device.type == "cuda" else -1)
        ei = self._edge_cache.get(key, None)
        if ei is None or ei.device != device:
            ei = build_star_chain_edge_index(self.num_nodes, B, device, add_self_loops=True)

            self._edge_cache[key] = ei
        return ei

    def forward(self, node_feats: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        node_feats: [B * N, node_in_dim]
        return:    [B, out_dim]
        """
        device = node_feats.device
        B = int(batch_size)
        N = self.num_nodes

        x = self.node_mlp(node_feats)  # [B*N, hidden]

        edge_index = self._get_edge_index(B, device)
        batch = torch.repeat_interleave(torch.arange(B, device=device), repeats=N)  # [B*N]

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)      # [B*N, hidden]
            x = norm(x)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch)   # [B, hidden]
        return self.head(g)              # [B, out_dim]


class FrozenCLIPNameColorEncoder(nn.Module):
    """Lookup оффлайн CLIP-эмбеддингов и общий обучаемый проектор 512 -> TEXT_EMB_DIM.

    embeddings_path должен содержать:
      - name_embs:  [N_names, 512]
      - color_embs: [N_colors, 512]
    """
    def __init__(self, embeddings_path: str, text_dim: int = TEXT_EMB_DIM):
        super().__init__()
        self.text_dim = text_dim

        payload = torch.load(embeddings_path, map_location="cpu")
        name_embs = payload.get("name_embs", None)
        color_embs = payload.get("color_embs", None)

        if name_embs is None or color_embs is None:
            raise ValueError(f"embeddings_path='{embeddings_path}' must contain 'name_embs' and 'color_embs'.")

        # Замороженные эмбеддинги
        self.register_buffer("name_embs", name_embs)   # [N_names, 512]
        self.register_buffer("color_embs", color_embs) # [N_colors, 512]

        # Обучаемый проектор 512->TEXT_EMB_DIM (общий для имени и цвета)
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, text_dim),
        )

    def forward(self, name_ids: torch.Tensor, color_ids: torch.Tensor) -> torch.Tensor:
        """
        name_ids:  [B*N] int
        color_ids: [B*N] int
        return:   [B*N, text_dim]
        """
        ne = self.name_embs[name_ids]   # [B*N, 512]
        ce = self.color_embs[color_ids] # [B*N, 512]
        concat = (ne + ce) / 2.0        # усредняем (можно cat, если хочешь concat)
        return self.projector(concat)   # [B*N, text_dim]


class SharedGraphModule(nn.Module):
    """Общий графовый энкодер, который обучается только от градиентов Value функции (критика) в PPO.

    Принимает graph_flat [B, N*24], возвращает [B, GRAPH_EMB_DIM].
    """
    def __init__(
        self,
        embeddings_path: str,
        num_nodes: int = NUM_GRAPH_NODES,
        per_object_dim: int = PER_OBJECT_DIM,
        text_dim: int = TEXT_EMB_DIM,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.per_object_dim = per_object_dim
        self.text_dim = text_dim

        # Замороженный CLIP-энкодер для имени/цвета
        self.text_encoder = FrozenCLIPNameColorEncoder(embeddings_path, text_dim=text_dim)

        # Графовый энкодер
        # На вход узла: per_object_dim + text_dim
        node_in = per_object_dim + text_dim
        self.graph_encoder = SceneGraphGATEncoder(
            num_nodes=num_nodes,
            node_in_dim=node_in,
            hidden_dim=128,
            out_dim=GRAPH_EMB_DIM,
            num_layers=2,
            heads=2,
            dropout=0.1,
        )

    def forward(self, graph_flat: torch.Tensor) -> torch.Tensor:
        """
        graph_flat: [B, N * per_object_dim]
        return:    [B, GRAPH_EMB_DIM]
        """
        B = graph_flat.shape[0]
        N = self.num_nodes
        D = self.per_object_dim

        # reshape -> [B, N, D]
        graph_3d = graph_flat.view(B, N, D)

        # Извлекаем name_id, color_id (первые 2 канала, int)
        name_ids = graph_3d[:, :, 0].long()   # [B, N]
        color_ids = graph_3d[:, :, 1].long()  # [B, N]

        # Остальные фичи (pos, ori, vel, ...)
        other_feats = graph_3d[:, :, 2:]      # [B, N, D-2]

        # Получаем текстовые эмбеддинги
        name_ids_flat = name_ids.reshape(-1)    # [B*N]
        color_ids_flat = color_ids.reshape(-1)  # [B*N]
        text_emb_flat = self.text_encoder(name_ids_flat, color_ids_flat)  # [B*N, text_dim]

        # Объединяем
        other_flat = other_feats.reshape(B * N, -1)  # [B*N, D-2]
        node_feats = torch.cat([text_emb_flat, other_flat], dim=-1)  # [B*N, text_dim + D-2]

        # Энкодер
        graph_emb = self.graph_encoder(node_feats, batch_size=B)  # [B, GRAPH_EMB_DIM]
        return graph_emb


# ---------------------------------------------------------------------
# PPO Models: Policy (Actor) и Value (Critic)
# ---------------------------------------------------------------------

class Policy(GaussianMixin, Model):
    """PPO Policy (Actor): π(a|s). 
    
    Графовый энкодер используется в режиме no_grad, т.к. обучается только от критика.
    """
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.device = device

        # ВАЖНО: shared_graph не регистрируем как submodule в политике,
        # чтобы оптимизатор политики не трогал его параметры.
        self.__dict__["shared_graph"] = shared_graph

        # img — это "нормализуемая" часть
        self.img_dim = int(observation_space["img"].shape[0])

        mlp_in = self.img_dim + GRAPH_EMB_DIM
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        ).to(device)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)          # [B, img_dim]
        graph_flat = states["graph"].to(self.device) # [B, N*24]

        # Энкодер обучается только от критика -> для политики no_grad
        with torch.no_grad():
            graph_emb = self.shared_graph(graph_flat)  # [B, 128]

        x = torch.cat([img, graph_emb], dim=-1)
        mu = self.net(x)
        return mu, self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """PPO Value function: V(s). 
    
    Здесь shared_graph обучается (градиенты идут в него).
    """
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.device = device

        # Регистрируем shared_graph как submodule, чтобы его параметры обучались
        # вместе с параметрами value function
        self.shared_graph = shared_graph

        self.img_dim = int(observation_space["img"].shape[0])

        mlp_in = self.img_dim + GRAPH_EMB_DIM
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)            # [B, img_dim]
        graph_flat = states["graph"].to(self.device)   # [B, N*24]

        # Графовый энкодер обучается здесь
        graph_emb = self.shared_graph(graph_flat)  # [B, GRAPH_EMB_DIM]

        x = torch.cat([img, graph_emb], dim=-1)
        value = self.net(x)
        return value, {}


# ---------------------------------------------------------------------
# Custom State Preprocessor
# ---------------------------------------------------------------------

class DictRunningStandardScaler(nn.Module):
    """
    Нормализует только states["img"], states["graph"] оставляет как есть.
    Работает независимо от порядка flatten'а.
    """
    def __init__(self, size, img_space, device=None, epsilon=1e-8, clip_threshold=5.0):
        super().__init__()
        self.full_space = size

        self.img_scaler = RunningStandardScaler(
            size=img_space,
            epsilon=epsilon,
            clip_threshold=clip_threshold,
            device=device,
        )

    def forward(self, x: torch.Tensor, train: bool = False, inverse: bool = False, no_grad: bool = True) -> torch.Tensor:
        # 1) развернуть в dict
        s = unflatten_tensorized_space(self.full_space, x)

        # 2) нормализовать только img
        s["img"] = self.img_scaler(s["img"], train=train, inverse=inverse, no_grad=no_grad)

        # 3) свернуть обратно
        return flatten_tensorized_space(s)


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
if EVAL:
    from gymnasium.wrappers import RecordVideo
    print("[INFO] Running evaluation...")
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0",
        num_envs=1,
        headless=True,
        cli_args=["--enable_cameras", "--video"],
    )
    env = RecordVideo(
        env,
        video_folder="logs/skrl/aloha/videos",
        name_prefix="aloha_eval",
        episode_trigger=lambda ep: True,
    )
else:
    print("[INFO] Running training...")
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0",
        num_envs=32,
        headless=True,
        cli_args=["--enable_cameras"],
    )

env = wrap_env(env)
device = env.device

# ---------------------------------------------------------------------
# Memory (PPO использует RandomMemory, но размер определяется rollouts)
# ---------------------------------------------------------------------
memory = RandomMemory(memory_size=32 * 16, num_envs=env.num_envs, device=device)
# 32 envs * 16 rollout steps = 512 samples per iteration

# ---------------------------------------------------------------------
# Shared graph module (one instance)
# ---------------------------------------------------------------------
shared_graph = SharedGraphModule(
    embeddings_path="/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
    num_nodes=NUM_GRAPH_NODES,
    per_object_dim=PER_OBJECT_DIM,
    text_dim=TEXT_EMB_DIM,
).to(device)

# ---------------------------------------------------------------------
# Models for PPO
# ---------------------------------------------------------------------
models = {
    "policy": Policy(env.observation_space, env.action_space, device, shared_graph=shared_graph),
    "value": Value(env.observation_space, env.action_space, device, shared_graph=shared_graph),
}

# ---------------------------------------------------------------------
# PPO config
# ---------------------------------------------------------------------
cfg = PPO_DEFAULT_CONFIG.copy()

# Training parameters
cfg["rollouts"] = 16                    # количество шагов для сбора данных
cfg["learning_epochs"] = 8              # количество эпох обучения на одном rollout
cfg["mini_batches"] = 4                 # количество мини-батчей
cfg["discount_factor"] = 0.99           # gamma
cfg["lambda"] = 0.95                    # GAE lambda

# Learning rates
cfg["learning_rate"] = 3e-4             # learning rate для обеих сетей
cfg["learning_rate_scheduler"] = None   # можно добавить scheduler

# PPO-specific
cfg["ratio_clip"] = 0.2                 # clipping parameter для PPO
cfg["value_clip"] = 0.2                 # clipping для value loss
cfg["clip_predicted_values"] = True     # включить value clipping

# Regularization
cfg["entropy_loss_scale"] = 0.01        # коэффициент entropy bonus
cfg["value_loss_scale"] = 0.5           # коэффициент value loss
cfg["grad_norm_clip"] = 0.5             # gradient clipping

# Advantages
cfg["value_preprocessor"] = None        # можно добавить нормализацию advantages
cfg["advantages_clip"] = None           # можно добавить clipping для advantages

# State preprocessor
cfg["state_preprocessor"] = DictRunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space,              # ВАЖНО: полный dict-space
    "img_space": env.observation_space["img"],  # Нормализуем только img
    "device": device,
}

# Logging and checkpointing
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo_graph"

# Random timesteps (обычно 0 для PPO)
cfg["random_timesteps"] = 0

# Create PPO agent
agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
if not EVAL:
    cfg_trainer = {"timesteps": 330000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.train()
else:
    cfg_trainer = {"timesteps": 1000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_graph/checkpoints/best_agent.pt"
    agent.load(checkpoint_path)

    trainer.eval()