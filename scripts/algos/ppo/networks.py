import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from torch_geometric.nn import GATv2Conv, global_mean_pool

# Константы. TODO Лучше бы они все подтягивались из env, graph_encoder, но пока так.
NUM_GRAPH_NODES = 21
PER_OBJECT_DIM = 24
TEXT_EMB_DIM = 16
GRAPH_EMB_DIM = 128
NUM_ORIENT_BINS = 36
GOAL_NODE_INDEX = 0

# CleanRL PPO initialisation hack
def layer_init(layer, std=sqrt(2), bias_const=0.0):
    # TODO что-то странное. Почему почему подчёркивается красным? Дока пишет, что std должно быть int'ом, хотя это не так.
    torch.nn.init.orthogonal_(layer.weight, std) 
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_star_chain_edge_index(num_nodes, batch_size, device, add_self_loops=True):
    N = num_nodes
    src, dst = [], []
    for i in range(1, N):
        src += [0, i]; dst += [i, 0]
    for i in range(N - 1):
        src += [i, i + 1]; dst += [i + 1, i]
    if add_self_loops:
        for i in range(N):
            src.append(i); dst.append(i)
    ei = torch.tensor([src, dst], device=device, dtype=torch.long)
    return torch.cat([ei + b * N for b in range(batch_size)], dim=1)


class GraphEncoder(nn.Module):
    """
    graph_flat [B, N*24] → graph_emb [B, 128]

    Внутри:
      1. CLIP text lookup (frozen embeddings) → text_emb per node
      2. Node MLP: (24 + text_dim) → hidden
      3. GATv2 × num_layers
      4. Global mean pool → head → graph_emb
    """
    def __init__(
        self,
        embeddings_path: str,
        image_dim: int,
        num_nodes: int = NUM_GRAPH_NODES,
        per_object_dim: int = PER_OBJECT_DIM,
        text_dim: int = TEXT_EMB_DIM,
        hidden_dim: int = 128,
        out_dim: int = GRAPH_EMB_DIM,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.per_object_dim = per_object_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # --- Frozen CLIP text lookup ---
        payload = torch.load(embeddings_path, map_location="cpu")
        self.register_buffer("name_embs", payload["name_embs"].float(), persistent=False)
        self.register_buffer("color_embs", payload["color_embs"].float(), persistent=False)
        clip_dim = self.name_embs.shape[-1]  # 512

        self.text_proj = nn.Sequential(
            nn.Linear(clip_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, text_dim),
        )

        # --- GATv2 ---
        node_in = per_object_dim + text_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        in_layer_dim, out_layer_dim = hidden_dim + image_dim, hidden_dim
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                in_layer_dim, out_layer_dim // heads,
                heads=heads, edge_dim=None, dropout=dropout, concat=True,
            ))
            self.norms.append(nn.LayerNorm(out_layer_dim))
            in_layer_dim, out_layer_dim = hidden_dim, hidden_dim

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        self._edge_cache = {}

    # --- text encoding helpers ---
    def _encode_text(self, name_idx, color_bits_or_idx):
        """name_idx: [B,N], color_bits_or_idx: [B,N] or [B,N,3] → [B,N,text_dim]"""
        if name_idx.dim() == 3:
            name_idx = name_idx.argmax(-1)
        name_idx = name_idx.long().clamp(0, self.name_embs.shape[0] - 1)

        if color_bits_or_idx.dim() == 3 and color_bits_or_idx.size(-1) == 3:
            bits = color_bits_or_idx.round().long().clamp(0, 1)
            color_idx = (bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]) - 1
        else:
            color_idx = color_bits_or_idx.round().long()
        color_idx = color_idx.clamp(0, self.color_embs.shape[0] - 1)

        emb = 0.5 * (self.name_embs[name_idx] + self.color_embs[color_idx])
        return self.text_proj(emb)


    # --- edge index with cache ---
    def _get_edge_index(self, B, device):
        key = (B, device.index if device.type == "cuda" else -1)
        ei = self._edge_cache.get(key)
        if ei is None or ei.device != device:
            ei = build_star_chain_edge_index(self.num_nodes, B, device)
            self._edge_cache[key] = ei
        return ei


    def forward(self, graph_flat: torch.Tensor, image_emb) -> torch.Tensor:
        """graph_flat: [B, N*24] → [B, out_dim]"""
        B = graph_flat.shape[0]
        N = self.num_nodes

        node_raw = graph_flat.view(B, N, self.per_object_dim)
        text_emb = self._encode_text(node_raw[..., 20], node_raw[..., 21:24])
        # TODO don't we need to omit node_raw dimensions from 20-24?
        x = torch.cat([node_raw, text_emb], dim=-1)#.view(B * N, -1)

        x = self.node_mlp(x)
        x = torch.cat(tensors = [x, image_emb.unsqueeze(1).repeat(1, N, 1)], dim = 2).view(B * N, -1)
        edge_index = self._get_edge_index(B, x.device)
        batch_vec = torch.repeat_interleave(torch.arange(B, device=x.device), N)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch_vec)
        return self.head(g)


class Policy(GaussianMixin, Model):
    """Policy (Actor): принимает dict-obs: {img, graph}. 
    В PPO графовый энкодер ОБУЧАЕТСЯ вместе с policy."""
    def __init__(self, observation_space, action_space, device, shared_graph, 
                 starting_std, clip_actions = True, clip_log_std = False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std)
        self.device = device

        # В PPO shared_graph регистрируем как submodule policy - он будет обучаться!
        self.shared_graph = shared_graph

        # img — это "нормализуемая" часть
        self.img_dim = int(observation_space["img"].shape[0])

        mlp_in = self.img_dim + GRAPH_EMB_DIM
        self.preprocessor = nn.Sequential(layer_init(nn.Linear(2056, self.img_dim)), nn.ReLU())
        self.net = nn.Sequential(
            layer_init(nn.Linear(mlp_in, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, self.num_actions), std=0.01),
            nn.Tanh()
        ).to(device)

        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), device = device, fill_value = starting_std))

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        emb = self.preprocessor(states["memory"].to(self.device)) # [B, img_dim]
        graph_flat = states["graph"].to(self.device) # [B, N*24]

        # В PPO графовый энкодер обучается через policy
        graph_emb = self.shared_graph(graph_flat, img)  # [B, 128]

        x = torch.cat([emb, graph_emb], dim=-1)
        mu = self.net(x)
        return mu, self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Value function: V(s). Использует тот же shared_graph через no_grad."""
    def __init__(self, observation_space, action_space, device, shared_graph,
                 clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.device = device

        self.shared_graph = shared_graph

        self.img_dim = int(observation_space["img"].shape[0])

        mlp_in = self.img_dim + GRAPH_EMB_DIM
        self.preprocessor = nn.Sequential(layer_init(nn.Linear(2056, self.img_dim)), nn.ReLU())
        self.net = nn.Sequential(
            layer_init(nn.Linear(mlp_in, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.)
        ).to(device)


    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)  
        emb = self.preprocessor(states["memory"].to(self.device))            # [B, img_dim]
        graph_flat = states["graph"].to(self.device)   # [B, N*24]

        graph_emb = self.shared_graph(graph_flat, img)

        x = torch.cat([emb, graph_emb], dim=-1)
        v = self.net(x)
        return v, {}
