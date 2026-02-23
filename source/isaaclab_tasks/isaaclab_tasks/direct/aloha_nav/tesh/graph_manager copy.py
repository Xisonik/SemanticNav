import random
import torch
import networkx as nx

class ObstacleGraph:
    def __init__(self, num_chairs, device):
        """
        Args:
            num_chairs (int): Количество препятствий (узлов) в графе.
            device (str or torch.device): Устройство для тензоров (cuda или cpu).
        """
        self.num_chairs = num_chairs
        self.device = device
        self.graph = nx.Graph()
        self._initialize_nodes()

    def _initialize_nodes(self):
        """Инициализирует узлы с дефолтными позициями за сценой в тензорах."""
        base_x = 10.0
        y_pos = torch.linspace(-self.num_chairs / 2.0, self.num_chairs / 2.0, self.num_chairs, device=self.device)
        self.default_positions = torch.stack([torch.full((self.num_chairs,), base_x, device=self.device),
                                             y_pos,
                                             torch.zeros(self.num_chairs, device=self.device)], dim=1)  # [num_chairs, 3]
        self.positions = self.default_positions.clone()  # [num_chairs, 3]
        self.radii = torch.full((self.num_chairs,), 0.5, device=self.device)  # [num_chairs]
        self.active = torch.zeros(self.num_chairs, dtype=torch.bool, device=self.device)  # [num_chairs]
        self.names = ['chair'] * self.num_chairs  # Список имён (не тензор, так как строки)

        # Инициализируем NetworkX граф для рёбер
        for i in range(self.num_chairs):
            self.graph.add_node(i, name=self.names[i], radius=self.radii[i].item(),
                               position=tuple(self.positions[i].tolist()),
                               default_position=tuple(self.default_positions[i].tolist()),
                               active=self.active[i].item())
        self._update_edges()

    def _update_edges(self):
        """Обновляет рёбра на основе текущих позиций узлов."""
        self.graph.remove_edges_from(list(self.graph.edges))  # Очищаем старые рёбра
        dists = torch.cdist(self.positions[:, :2], self.positions[:, :2])  # [num_chairs, num_chairs]
        active_pairs = self.active[:, None] & self.active[None, :]  # [num_chairs, num_chairs]
        dists = torch.where(active_pairs, dists, torch.tensor(float('inf'), device=self.device))
        
        for i in range(self.num_chairs):
            for j in range(i + 1, self.num_chairs):
                self.graph.add_edge(i, j, weight=dists[i, j].item())
                # Синхронизируем NetworkX узлы с тензорами
                self.graph.nodes[i]['position'] = tuple(self.positions[i].tolist())
                self.graph.nodes[i]['active'] = self.active[i].item()
                self.graph.nodes[i]['radius'] = self.radii[i].item()
                self.graph.nodes[i]['name'] = self.names[i]
                self.graph.nodes[j]['position'] = tuple(self.positions[j].tolist())
                self.graph.nodes[j]['active'] = self.active[j].item()
                self.graph.nodes[j]['radius'] = self.radii[j].item()
                self.graph.nodes[j]['name'] = self.names[j]

    def set_radii(self, radii):
        """Устанавливает радиусы для узлов."""
        if len(radii) != self.num_chairs:
            raise ValueError(f"Expected {self.num_chairs} radii, got {len(radii)}")
        self.radii = torch.tensor(radii, device=self.device, dtype=torch.float32)
        for i in range(self.num_chairs):
            self.graph.nodes[i]['radius'] = self.radii[i].item()

    def update_positions(self, active_indices, positions):
        """Обновляет позиции активных узлов и рёбра."""
        self.active[:] = False
        if active_indices:
            self.active[active_indices] = True
            self.positions[active_indices] = torch.tensor(positions, device=self.device, dtype=torch.float32)
        self.positions[~self.active] = self.default_positions[~self.active]
        self._update_edges()

    def get_active_nodes(self):
        """Возвращает тензор с данными активных узлов."""
        active_mask = self.active
        active_positions = self.positions[active_mask]  # [num_active, 3]
        active_radii = self.radii[active_mask]  # [num_active]
        active_names = [self.names[i] for i in range(self.num_chairs) if self.active[i]]  # Список строк
        active_default_positions = self.default_positions[active_mask]  # [num_active, 3]
        
        # Формируем тензор с данными активных узлов
        active_nodes = [{
            'name': name,
            'radius': radius.item(),
            'position': tuple(position.tolist()),
            'active': True,
            'default_position': tuple(default_pos.tolist())
        } for name, radius, position, default_pos in zip(active_names, active_radii, active_positions, active_default_positions)]
        return active_nodes

    def get_graph_info(self):
        """
        Возвращает отформатированную информацию о графе: узлы и рёбра.

        Returns:
            str: Текстовая строка с информацией о графе.
        """
        info = f"ObstacleGraph with {self.num_chairs} nodes:\n"
        info += "Nodes:\n"
        for i in range(self.num_chairs):
            info += f"  Node {i}:\n"
            info += f"    Name: {self.names[i]}\n"
            info += f"    Radius: {self.radii[i]:.2f}\n"
            info += f"    Position: ({self.positions[i, 0]:.2f}, {self.positions[i, 1]:.2f}, {self.positions[i, 2]:.2f})\n"
            info += f"    Active: {self.active[i]}\n"
            info += f"    Default Position: ({self.default_positions[i, 0]:.2f}, {self.default_positions[i, 1]:.2f}, {self.default_positions[i, 2]:.2f})\n"
        
        active_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True) if d['weight'] != float('inf')]
        info += f"\nEdges ({len(active_edges)} active):\n"
        if not active_edges:
            info += "  No active edges (all nodes inactive or too far apart).\n"
        else:
            for u, v, data in active_edges:
                info += f"  Edge ({u}, {v}): Weight = {data['weight']:.2f}\n"
        
        return info