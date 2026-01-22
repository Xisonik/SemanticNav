import torch
from typing import Optional

class MemoryManager:
    def __init__(self, num_envs, embedding_size, action_size, history_length, device,
                 dtype=torch.float32):
        self.num_envs = num_envs
        self.embedding_size = embedding_size
        self.action_size = action_size
        self.history_length = history_length
        self.device = device
        self.dtype = dtype

        self.embedding_history = torch.zeros(
            (num_envs, history_length, embedding_size), device=device, dtype=dtype
        )
        self.action_history = torch.zeros(
            (num_envs, history_length, action_size), device=device, dtype=dtype
        )

        # per-env флаг наличия истории
        self.initialized = torch.zeros((num_envs,), device=device, dtype=torch.bool)

        # нулевое действие
        self.zero_action = torch.zeros((action_size,), device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Сбрасывает историю для env_ids (или для всех)."""
        if env_ids is None:
            self.embedding_history.zero_()
            self.action_history.zero_()
            self.initialized.fill_(False)
            return

        env_ids = env_ids.to(self.device, dtype=torch.long)
        self.embedding_history[env_ids].zero_()
        self.action_history[env_ids].zero_()
        self.initialized[env_ids] = False

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, actions: torch.Tensor):
        """
        Алгоритм:
        1) найти неинициализированные env (initialized=False)
        2) для них: продублировать текущий embedding на всю историю
           и заполнить actions нулями на всю историю
        3) затем сделать общий тензорный push-front для ВСЕХ env:
           history[:, 0] = текущие данные, history сдвигается вправо
        """
        embeddings = embeddings.to(self.device, dtype=self.dtype)
        actions = actions.to(self.device, dtype=self.dtype)

        # 1) найти env без истории
        new_mask = ~self.initialized  # [N]

        # 2) инициализировать только их
        if new_mask.any():
            # embeddings: дублируем по всей истории
            self.embedding_history[new_mask] = embeddings[new_mask].unsqueeze(1).expand(-1, self.history_length, -1)

            # actions: по всей истории нули
            self.action_history[new_mask] = self.zero_action.view(1, 1, -1).expand(
                int(new_mask.sum().item()), self.history_length, -1
            )

            self.initialized[new_mask] = True

        # 3) общий push-front для всех env (одним тензорным обновлением)
        self.embedding_history = torch.cat([embeddings.unsqueeze(1), self.embedding_history[:, :-1]], dim=1)
        self.action_history = torch.cat([actions.unsqueeze(1), self.action_history[:, :-1]], dim=1)

    @torch.no_grad()
    def get_observations(self, m: int = 4, k: int = 4) -> torch.Tensor:
        """
        Возвращает [N, m*(E+A)] = concat(emb_0, act_0, emb_k, act_k, ...)
        """
        indices = torch.arange(0, m * k, k, device=self.device, dtype=torch.long)
        indices = torch.clamp(indices, 0, self.history_length - 1)

        sel_emb = self.embedding_history[:, indices]  # [N,m,E]
        sel_act = self.action_history[:, indices]     # [N,m,A]
        return torch.cat([sel_emb, sel_act], dim=-1).reshape(self.num_envs, -1)



class PathTracker:
    def __init__(self, num_envs: int, T_max: int = 256, device: str = "cuda", pos_dim: int = 2):
        """
        Батчевый менеджер траекторий и управляющих воздействий.

        Args:
            num_envs (int): количество сред
            T_max (int): максимальная длина траектории
            device (str): устройство
            pos_dim (int): размерность позиции (обычно 2 или 3)
        """
        self.num_envs = num_envs
        self.T_max = T_max
        self.device = device
        self.pos_dim = pos_dim

        # [num_envs, T_max, pos_dim]
        self.positions = torch.zeros((num_envs, T_max, pos_dim), device=device, dtype=torch.float32)
        # [num_envs, T_max, 2] (lin, ang)
        self.velocities = torch.zeros((num_envs, T_max, 2), device=device, dtype=torch.float32)
        # Счётчик длины траектории для каждой среды
        self.lengths = torch.zeros(num_envs, device=device, dtype=torch.long)

    @torch.no_grad()
    def add_step(self, env_ids: torch.Tensor, positions: torch.Tensor, velocities: torch.Tensor):
        """
        Добавить позиции и управляющие воздействия в батчевом режиме.
        Args:
            env_ids (torch.Tensor): [K]
            positions (torch.Tensor): [K, pos_dim]
            velocities (torch.Tensor): [K, 2]
        """
        env_ids = env_ids.to(self.device)
        idxs = self.lengths[env_ids]  # текущие индексы вставки
        for i, env_id in enumerate(env_ids):
            if idxs[i] < self.T_max:
                self.positions[env_id, idxs[i]] = positions[i].to(self.device)
                self.velocities[env_id, idxs[i]] = velocities[i].to(self.device)
                self.lengths[env_id] += 1  # увеличиваем счётчик

    def reset(self, env_ids: torch.Tensor):
        """
        Очистить траектории и управляющие воздействия для указанных сред.
        """
        env_ids = env_ids.to(self.device)
        self.positions[env_ids] = 0.0
        self.velocities[env_ids] = 0.0
        self.lengths[env_ids] = 0

    @torch.no_grad()
    def compute_path_lengths(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Подсчитать длину пути для агентов (евклидова сумма).
        Returns: [K]
        """
        env_ids = env_ids.to(self.device)
        pos = self.positions[env_ids]  # [K, T_max, pos_dim]
        L = self.lengths[env_ids]      # [K]

        # Считаем диффы вдоль оси T
        diffs = pos[:, 1:] - pos[:, :-1]       # [K, T_max-1, pos_dim]
        dist = torch.norm(diffs, dim=-1)       # [K, T_max-1]

        # Маска по длине
        mask = torch.arange(self.T_max-1, device=self.device).unsqueeze(0) < (L.unsqueeze(1)-1)
        dist = dist * mask

        return dist.sum(dim=1)

    def get_paths(self, env_ids: torch.Tensor):
        """
        Вернуть пути агентов (с обрезкой до длины).
        """
        env_ids = env_ids.to(self.device)
        out = {}
        for i, env_id in enumerate(env_ids):
            L = self.lengths[env_id].item()
            out[env_id.item()] = self.positions[env_id, :L]
        return out

    def get_velocities(self, env_ids: torch.Tensor):
        """
        Вернуть последовательности управляющих воздействий.
        """
        env_ids = env_ids.to(self.device)
        out = {}
        for i, env_id in enumerate(env_ids):
            L = self.lengths[env_id].item()
            out[env_id.item()] = self.velocities[env_id, :L]
        return out

    @torch.no_grad()
    def compute_jerk(self, env_ids: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """
        Подсчитать количество резких скачков скоростей.
        Args:
            threshold (float): порог
        Returns: [K] количество скачков
        """
        env_ids = env_ids.to(self.device)
        vels = self.velocities[env_ids]  # [K, T_max, 2]
        L = self.lengths[env_ids]

        diffs = torch.norm(vels[:, 1:] - vels[:, :-1], dim=-1)  # [K, T_max-1]

        mask = torch.arange(self.T_max-1, device=self.device).unsqueeze(0) < (L.unsqueeze(1)-1)
        diffs = diffs * mask

        return (diffs > threshold).sum(dim=1).float()
