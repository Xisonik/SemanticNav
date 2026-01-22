import torch
import networkx as nx
import os
from pathlib import Path
import json
from itertools import combinations

class ControlModule:
    def __init__(self, num_envs, device, scene_manager, kp_linear=0.5, ki_linear=0.01, kd_linear=0.02,
                 kp_angular=1.0, ki_angular=0.01, kd_angular=0.05):
        """
        Args:
            num_envs (int): Number of environments.
            device (torch.device): Device for tensor operations (cuda or cpu).
            scene_manager (SceneManager): Instance of SceneManager for obstacle data.
            kp_linear, ki_linear, kd_linear (float): PID coefficients for linear velocity.
            kp_angular, ki_angular, kd_angular (float): PID coefficients for angular velocity.
        """
        self.num_envs = num_envs
        self.device = device
        self.scene_manager = scene_manager
        self.ratio = 10
        self.room_len_x = 6
        self.room_len_y = 8
        self.ratio_x = self.ratio * self.room_len_x
        self.ratio_y = self.ratio * self.room_len_y
        self.shift = torch.tensor([0.0, 0.0], device=self.device)  # [2]

        # PID coefficients
        self.kp_linear = kp_linear
        self.ki_linear = ki_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.ki_angular = ki_angular
        self.kd_angular = kd_angular

        # Initialize PID state
        self.integral_linear = torch.zeros(num_envs, device=device)
        self.prev_linear_error = torch.zeros(num_envs, device=device)
        self.integral_angular = torch.zeros(num_envs, device=device)
        self.prev_angular_error = torch.zeros(num_envs, device=device)

        # Path and state tensors
        self.max_path_length = 100
        self.paths = torch.zeros((num_envs, self.max_path_length, 2), device=device)  # [num_envs, max_path_length, 2]
        self.path_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.current_pos = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.start = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.end = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.first_ep = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Track current start node index for each environment
        self.current_start_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Cache for scene graphs, paths, and start nodes
        self.scene_graphs_cache = {}
        self.all_paths = {}
        self.available_start_nodes = {}  # {config_key: list of (x, y) nodes}
        self.dijkstra_initialized = False

        # Log directory
        self.log_dir = Path().resolve() / "standalone_examples/Aloha_graph/Aloha/logs"
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize default robot positions (x=12.0)
        y_pos = torch.linspace(-num_envs / 2.0, num_envs / 2.0, num_envs, device=device)
        self.robot_pos = torch.stack([
            torch.full((num_envs,), 12.0, device=device),
            y_pos,
            torch.zeros(num_envs, device=device)
        ], dim=1)  # [num_envs, 3]

        # Initialize configurations and start nodes
        self.initialize_configs_and_start_nodes()

    def _create_grid_with_diagonals(self):
        """Creates a grid graph with diagonals."""
        graph = nx.grid_2d_graph(self.ratio_x, self.ratio_y)
        for x in range(self.ratio_x):
            for y in range(self.ratio_y):
                if x + 1 < self.ratio_x and y + 1 < self.ratio_y:
                    graph.add_edge((x, y), (x + 1, y + 1), weight=1.414)
                if x + 1 < self.ratio_x and y - 1 >= 0:
                    graph.add_edge((x, y), (x + 1, y - 1), weight=1.414)
        return graph

    def find_boundary_nodes(self, graph):
        """Finds boundary nodes with fewer neighbors."""
        max_degree = max(dict(graph.degree()).values())
        return {node for node in graph.nodes() if graph.degree(node) < max_degree}

    def find_expanded_boundary(self, graph, boundary_nodes):
        """Finds nodes adjacent to boundary nodes."""
        expanded_boundary = set()
        for node in boundary_nodes:
            expanded_boundary.update(graph.neighbors(node))
        return expanded_boundary - boundary_nodes

    def assign_edge_weights(self, graph, boundary_nodes, expanded_boundary):
        """Assigns weights to edges based on proximity to boundaries."""
        for u, v in graph.edges():
            if u in boundary_nodes or v in boundary_nodes:
                graph[u][v]['weight'] = 3
            elif u in expanded_boundary or v in expanded_boundary:
                graph[u][v]['weight'] = 2
            else:
                graph[u][v]['weight'] = 1

    def get_scene_grid(self, config_key, positions_for_obstacles):
        """
        Creates or retrieves a cached scene graph for a given obstacle configuration.

        Args:
            config_key (str): Configuration key (sorted obstacle IDs).
            positions_for_obstacles (list): Indices of active obstacles.

        Returns:
            nx.Graph: Scene graph for the configuration.
            list: Available start nodes.
        """
        if config_key in self.scene_graphs_cache:
            return self.scene_graphs_cache[config_key], self.available_start_nodes[config_key]

        # Create new graph
        G = self._create_grid_with_diagonals()
        # Set obstacle positions in SceneManager for env_id=0
        self.scene_manager.generate_obstacle_positions(
            env_ids=torch.tensor([0], device=self.device),
            terrain_origins=torch.zeros((self.num_envs, 3), device=self.device)
        )
        # Update obstacle positions for the configuration
        obstacle_pos = torch.tensor(self.scene_manager.possible_positions, device=self.device, dtype=torch.float32)[positions_for_obstacles]  # [num_active, 3]
        self.scene_manager.obstacle_manager.graphs[0].update_positions(positions_for_obstacles, obstacle_pos)

        # Convert grid nodes to real-world coordinates
        nodes = torch.tensor(list(G.nodes()), device=self.device, dtype=torch.float32) / self.ratio + self.shift
        # Check intersections with obstacles
        intersect = self.scene_manager.get_checked_for_obstacles(nodes, torch.tensor([0], device=self.device))
        # Check intersections with walls
        intersect |= (nodes[:, 0] < self.scene_manager.room_bounds['x_min']) | \
                    (nodes[:, 0] > self.scene_manager.room_bounds['x_max']) | \
                    (nodes[:, 1] < self.scene_manager.room_bounds['y_min']) | \
                    (nodes[:, 1] > self.scene_manager.room_bounds['y_max'])

        # Store available start nodes
        available_nodes = [node for i, node in enumerate(G.nodes()) if not intersect[i]]
        self.available_start_nodes[config_key] = available_nodes

        # Remove edges intersecting obstacles or walls
        edges_to_remove = []
        for u, v in G.edges():
            node_1 = torch.tensor(u, device=self.device, dtype=torch.float32) / self.ratio + self.shift
            node_2 = torch.tensor(v, device=self.device, dtype=torch.float32) / self.ratio + self.shift
            u_idx = list(G.nodes).index(u)
            v_idx = list(G.nodes).index(v)
            if intersect[u_idx] or intersect[v_idx]:
                edges_to_remove.append((u, v))
            elif (node_1[0] < self.scene_manager.room_bounds['x_min'] or \
                  node_1[0] > self.scene_manager.room_bounds['x_max'] or \
                  node_1[1] < self.scene_manager.room_bounds['y_min'] or \
                  node_1[1] > self.scene_manager.room_bounds['y_max']) or \
                 (node_2[0] < self.scene_manager.room_bounds['x_min'] or \
                  node_2[0] > self.scene_manager.room_bounds['x_max'] or \
                  node_2[1] < self.scene_manager.room_bounds['y_min'] or \
                  node_2[1] > self.scene_manager.room_bounds['y_max']):
                edges_to_remove.append((u, v))
        G.remove_edges_from(edges_to_remove)

        # Assign weights
        boundary_nodes = self.find_boundary_nodes(G)
        expanded_boundary = self.find_expanded_boundary(G, boundary_nodes)
        self.assign_edge_weights(G, boundary_nodes, expanded_boundary)

        # Cache and save graph
        self.scene_graphs_cache[config_key] = G
        graph_file = self.log_dir / f"graph_{config_key}.json"
        graph_data = {
            "edges": [{"u": list(u), "v": list(v), "weight": G[u][v]["weight"]} for u, v in G.edges()]
        }
        try:
            with open(graph_file, 'w') as f:
                json.dump(graph_data, f, indent=4)
        except Exception as e:
            print(f"Error saving graph file {graph_file}: {e}")
        
        return G, available_nodes

    def initialize_configs_and_start_nodes(self):
        """Precomputes configurations, graphs, and Dijkstra paths."""
        if self.dijkstra_initialized:
            return
        
        paths_file = self.log_dir / "all_paths.json"
        if paths_file.exists():
            try:
                with open(paths_file, 'r') as f:
                    loaded_paths = json.load(f)
                for config_key, targets in loaded_paths.items():
                    self.all_paths[config_key] = {}
                    self.available_start_nodes[config_key] = []
                    for target_str, nodes in targets.items():
                        target = tuple(map(int, target_str.split(',')))
                        self.all_paths[config_key][target] = {}
                        for node_str, path in nodes.items():
                            node = tuple(map(int, node_str.split(',')))
                            self.all_paths[config_key][target][node] = [tuple(p) for p in path]
                            if node not in self.available_start_nodes[config_key]:
                                self.available_start_nodes[config_key].append(node)
                print(f"Loaded {len(self.all_paths)} configurations")
                self.dijkstra_initialized = True
                return
            except Exception as e:
                print(f"Error loading paths file: {e}")

        # Generate all possible obstacle configurations
        num_obstacles = self.scene_manager.obstacle_manager.num_obstacles
        arr = range(num_obstacles)
        all_configs = [''.join(str(x) for x in sorted(combo)) for r in range(5, len(arr) + 1) for combo in combinations(arr, r)]
        
        for config_key in all_configs:
            positions_for_obstacles = [int(ch) for ch in config_key]
            config_graph, available_nodes = self.get_scene_grid(config_key, positions_for_obstacles)
            
            targets = []
            for i in range(self.num_envs):
                target_node = self.find_nearest_reachable_node(
                    config_graph,
                    (int((self.scene_manager.goal_pos[i, 0] - self.shift[0]) * self.ratio),
                     int((self.scene_manager.goal_pos[i, 1] - self.shift[1]) * self.ratio))
                )
                if target_node and target_node not in targets:
                    targets.append(target_node)
            
            self.all_paths[config_key] = {target: {} for target in targets}
            for target in targets:
                for node in available_nodes:
                    if nx.has_path(config_graph, node, target):
                        path = nx.shortest_path(config_graph, source=node, target=target, weight='weight')
                        self.all_paths[config_key][target][node] = path

        # Save paths to JSON
        json_paths = {}
        for config_key, targets in self.all_paths.items():
            json_paths[config_key] = {}
            for target, nodes in targets.items():
                target_str = f"{target[0]},{target[1]}"
                json_paths[config_key][target_str] = {}
                for node, path in nodes.items():
                    node_str = f"{node[0]},{node[1]}"
                    json_paths[config_key][target_str][node_str] = [list(p) for p in path]
        
        try:
            with open(paths_file, 'w') as f:
                json.dump(json_paths, f, indent=4)
            print(f"Saved {len(self.all_paths)} configurations")
        except Exception as e:
            print(f"Error saving paths file: {e}")
        
        self.dijkstra_initialized = True

    def find_nearest_reachable_node(self, graph, target):
        """Finds the nearest reachable node in the graph to the target position."""
        if target in graph and len(list(graph.neighbors(target))) > 0:
            return target
        nodes = torch.tensor(list(graph.nodes()), device=self.device, dtype=torch.float32)
        target_tensor = torch.tensor(target, device=self.device, dtype=torch.float32)
        distances = torch.abs(nodes - target_tensor).sum(dim=1)  # Manhattan distance
        valid_nodes = torch.tensor([len(list(graph.neighbors(node))) > 0 for node in graph.nodes()], device=self.device)
        distances[~valid_nodes] = float('inf')
        min_idx = torch.argmin(distances)
        return tuple(nodes[min_idx].int().tolist())

    def remove_zigzags(self, paths, path_lengths):
        """
        Removes zigzags from paths for all environments.

        Args:
            paths (torch.Tensor): Paths [num_envs, max_path_length, 2].
            path_lengths (torch.Tensor): Length of each path [num_envs].

        Returns:
            torch.Tensor: Simplified paths.
            torch.Tensor: Updated path lengths.
        """
        simplified_paths = torch.zeros_like(paths)
        simplified_lengths = torch.ones_like(path_lengths)
        
        for env_idx in range(self.num_envs):
            path = paths[env_idx, :path_lengths[env_idx]]
            if path.shape[0] < 3:
                simplified_paths[env_idx, :path.shape[0]] = path
                simplified_lengths[env_idx] = path.shape[0]
                continue
            
            simplified = [path[0]]
            for i in range(1, path.shape[0] - 1):
                x1, y1 = path[i - 1]
                x2, y2 = path[i]
                x3, y3 = path[i + 1]
                if not (x1 == x3 or y1 == y3 or abs(x1 - x3) == abs(y1 - y3)):
                    simplified.append(path[i])
            simplified.append(path[-1])
            
            simplified_tensor = torch.tensor(simplified, device=self.device)
            simplified_paths[env_idx, :len(simplified)] = simplified_tensor
            simplified_lengths[env_idx] = len(simplified)
        
        return simplified_paths, simplified_lengths

    def update(self, current_positions, target_positions, env_ids=None, config_key=None):
        """
        Updates paths for all environments using the next available start node.

        Args:
            current_positions (torch.Tensor): Robot positions [num_envs, 3].
            target_positions (torch.Tensor): Goal positions [num_envs, 2 or 3].
            env_ids (torch.Tensor, optional): Environment indices to update.
            config_key (str, optional): Obstacle configuration key.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        self.robot_pos[env_ids] = current_positions[env_ids]
        self.scene_manager.goal_pos[env_ids] = target_positions[env_ids, :2]

        # Initialize Dijkstra paths if not done
        if not self.dijkstra_initialized:
            self.initialize_configs_and_start_nodes()

        # Use provided config_key or default to a single configuration
        if config_key is None:
            config_key = ''.join(str(i) for i in range(self.scene_manager.obstacle_manager.num_obstacles))

        # Get scene graph for the configuration
        graph, available_nodes = self.get_scene_grid(config_key, [int(ch) for ch in config_key])

        # Update paths for all environments
        for env_idx, env_id in enumerate(env_ids):
            # Select next start node
            start_idx = self.current_start_idx[env_idx].item() % len(available_nodes)
            start_node = available_nodes[start_idx]
            goal = self.find_nearest_reachable_node(
                graph,
                (int((target_positions[env_idx, 0] - self.shift[0]) * self.ratio),
                 int((target_positions[env_idx, 1] - self.shift[1]) * self.ratio))
            )
            path = self.all_paths.get(config_key, {}).get(goal, {}).get(start_node, [])
            if not path:
                print(f"No path found for env {env_id}, config {config_key}, start {start_node}, goal {goal}")
                self.paths[env_idx] = 0
                self.path_lengths[env_idx] = 0
                continue
            
            path_tensor = torch.tensor(path, device=self.device, dtype=torch.float32) / self.ratio + self.shift
            self.paths[env_idx, :len(path)] = path_tensor[:, :2]
            self.path_lengths[env_idx] = len(path)
            
            # Increment start index for next update
            self.current_start_idx[env_idx] += 1

        # Remove zigzags
        self.paths, self.path_lengths = self.remove_zigzags(self.paths, self.path_lengths)

    def pure_pursuit_controller(self, current_positions, current_orientations, linear_velocity=0.3, lookahead_distance=0.35):
        """
        Computes linear and angular velocities for all environments using Pure Pursuit.

        Args:
            current_positions (torch.Tensor): Robot positions [num_envs, 3].
            current_orientations (torch.Tensor): Robot orientations [num_envs, 4] (quaternions).
            linear_velocity (float): Desired linear velocity.
            lookahead_distance (float): Lookahead distance for Pure Pursuit.

        Returns:
            torch.Tensor: Linear velocities [num_envs].
            torch.Tensor: Angular velocities [num_envs].
        """
        # Update robot positions
        self.robot_pos = current_positions

        # Convert quaternions to Euler angles (yaw only)
        roll, pitch, yaw = self.euler_from_quaternion(current_orientations)
        current_heading = torch.where(yaw < 0, -torch.pi - yaw, torch.pi - yaw)  # [num_envs]

        # Check if robots are close to targets
        distances = torch.norm(self.scene_manager.goal_pos[:, :2] - current_positions[:, :2], dim=1)
        path_end_distances = torch.norm(self.paths[torch.arange(self.num_envs), self.path_lengths - 1], dim=1)
        self.end = (distances < 1.0) | (path_end_distances < torch.max(torch.tensor(0.2, device=self.device), 1.0 / self.ratio))

        # Initialize outputs
        linear_velocities = torch.zeros(self.num_envs, device=self.device)
        angular_velocities = torch.zeros(self.num_envs, device=self.device)

        # Handle first episode or end state
        mask = self.first_ep | self.end
        linear_velocities[mask] = 0.0
        angular_velocities[mask] = 0.0
        self.first_ep[mask] = False

        # Active environments (not in start or end state)
        active_mask = ~self.start & ~self.end
        if torch.any(active_mask):
            lookahead_points = self.get_lookahead_point(current_positions[active_mask], lookahead_distance)
            to_target = lookahead_points - current_positions[active_mask, :2]
            target_angle = torch.atan2(to_target[:, 1], to_target[:, 0])
            alpha = self.normalize_angle(target_angle - current_heading[active_mask])
            curvature = 2 * torch.sin(alpha) / lookahead_distance
            angular_velocities[active_mask] = curvature * linear_velocity
            max_angular_velocity = torch.pi * 0.4
            angular_velocities[active_mask] = torch.clamp(angular_velocities[active_mask], -max_angular_velocity, max_angular_velocity)
            linear_velocities[active_mask] = linear_velocity * (max_angular_velocity - torch.abs(angular_velocities[active_mask])) / max_angular_velocity

        # Handle start state (align to path)
        start_mask = self.start & ~self.end
        if torch.any(start_mask):
            nx = torch.tensor([-1.0, 0.0], device=self.device).repeat(self.num_envs, 1)  # [num_envs, 2]
            ny = torch.tensor([0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
            to_goal_vec = torch.where(
                self.end[:, None],
                self.scene_manager.goal_pos[:, :2] - current_positions[:, :2],
                self.paths[torch.arange(self.num_envs), 1] - current_positions[:, :2]
            )  # [num_envs, 2]
            cos_angle = torch.sum(to_goal_vec * nx, dim=1) / torch.norm(to_goal_vec, dim=1) / torch.norm(nx, dim=1)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            LR = to_goal_vec[:, 0] * nx[:, 1] - to_goal_vec[:, 1] * nx[:, 0]
            quadrant = torch.where(LR >= 0, 1.0, -1.0)
            true_angle = quadrant * torch.acos(cos_angle) + torch.pi
            true_angle = torch.where(true_angle >= 2 * torch.pi, true_angle - 2 * torch.pi, true_angle)
            true_angle = torch.where(true_angle == 2 * torch.pi, torch.tensor(0.0, device=self.device), true_angle)
            angle_diff = torch.abs(true_angle - (2 * torch.pi - current_heading))
            angular_velocities[start_mask] = torch.where(
                angle_diff[start_mask] < torch.pi / 80,
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0, device=self.device) * torch.sign(true_angle[start_mask] - (2 * torch.pi - current_heading[start_mask]))
            )
            self.start[start_mask & (angle_diff < torch.pi / 80)] = False

        return linear_velocities, angular_velocities

    def get_lookahead_point(self, current_positions, lookahead_distance):
        """
        Computes lookahead points for active environments.

        Args:
            current_positions (torch.Tensor): Robot positions [num_active, 3].
            lookahead_distance (float): Lookahead distance.

        Returns:
            torch.Tensor: Lookahead points [num_active, 2].
        """
        num_active = current_positions.shape[0]
        lookahead_points = torch.zeros((num_active, 2), device=self.device)
        
        for env_idx in range(num_active):
            path = self.paths[env_idx, :self.path_lengths[env_idx]]
            for i in range(path.shape[0] - 1, 0, -1):
                segment_start = path[i - 1]
                segment_end = path[i]
                segment_vector = segment_end - segment_start
                segment_length = torch.norm(segment_vector)
                to_segment_start = current_positions[env_idx, :2] - segment_start
                projection = torch.dot(to_segment_start, segment_vector) / segment_length
                if projection < 0:
                    closest_point = segment_start
                elif projection > segment_length:
                    closest_point = segment_end
                else:
                    closest_point = segment_start + (segment_vector / segment_length) * projection
                distance_to_closest = torch.norm(current_positions[env_idx, :2] - closest_point)
                if distance_to_closest <= lookahead_distance:
                    remaining_distance = lookahead_distance - distance_to_closest
                    lookahead_points[env_idx] = closest_point + (segment_vector / segment_length) * remaining_distance
                    break
            else:
                lookahead_points[env_idx] = path[-1]
        
        return lookahead_points

    def normalize_angle(self, angle):
        """Normalizes angles to [-pi, pi]."""
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def euler_from_quaternion(self, quaternions):
        """
        Converts quaternions to Euler angles (roll, pitch, yaw).

        Args:
            quaternions (torch.Tensor): Quaternions [num_envs, 4].

        Returns:
            tuple: (roll, pitch, yaw) as tensors [num_envs].
        """
        x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(t0, t1)
        
        t2 = 2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)
        
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(t3, t4)
        
        return roll, pitch, yaw 