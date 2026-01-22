import torch
import math

class Control_module:
    def __init__(self, num_envs: int, device: str = 'cuda:0'):
        """
        Инициализирует модуль управления для векторизованной среды.

        Args:
            num_envs (int): Количество сред.
            ratio (float): Масштабный коэффициент для преобразования координат.
            device (str): Устройство для тензоров ('cuda:0' или 'cpu').
        """
        self.num_envs = num_envs
        self.device = device
        self.paths = None  # [num_envs, max_path_length, 2]
        self.current_pos = torch.zeros((num_envs, 2), device=device)
        self.target_positions = torch.zeros((num_envs, 2), device=device)
        self.end = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.start = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.first_ep = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.lookahead_distance = 0.35
        self.linear_velocity = 0.3
        self.max_angular_velocity = math.pi * 0.4

    def update(self, current_positions: torch.Tensor, target_positions: torch.Tensor, paths: torch.Tensor):
        """
        Обновляет текущие позиции, цели и пути для следования.

        Args:
            current_positions (torch.Tensor): Текущие позиции роботов [num_envs, 2].
            target_positions (torch.Tensor): Целевые позиции [num_envs, 2].
            paths (torch.Tensor): Пути для каждой среды [num_envs, max_path_length, 2].
        """
        self.current_pos = current_positions.clone()
        self.target_positions = target_positions.clone()
        self.paths = paths.clone()
        self.end[:] = False
        self.start[:] = True
        self.first_ep[:] = True

    def normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Нормализует углы в диапазон [-pi, pi].

        Args:
            angle (torch.Tensor): Углы [num_envs].

        Returns:
            torch.Tensor: Нормализованные углы [num_envs].
        """
        while torch.any(angle > math.pi):
            angle[angle > math.pi] -= 2 * math.pi
        while torch.any(angle < -math.pi):
            angle[angle < -math.pi] += 2 * math.pi
        return angle

    def get_lookahead_point(self, current_positions: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет точки следования (lookahead points) для всех сред.

        Args:
            current_positions (torch.Tensor): Текущие позиции роботов [num_envs, 2].

        Returns:
            torch.Tensor: Точки следования [num_envs, 2].
        """
        lookahead_points = torch.zeros((self.num_envs, 2), device=self.device)
        path_lengths = torch.sum(self.paths.abs().sum(dim=2) > 0, dim=1)

        for i in range(self.num_envs):
            if path_lengths[i] == 0:
                lookahead_points[i] = self.paths[i, 0]
                continue

            for j in range(path_lengths[i] - 1, -1, -1):
                segment_start = self.paths[i, j]
                segment_end = self.paths[i, min(j + 1, path_lengths[i] - 1)]
                segment_vector = segment_end - segment_start
                segment_length = torch.norm(segment_vector)

                if segment_length < 1e-6:
                    continue

                to_segment_start = current_positions[i] - segment_start
                projection = torch.dot(to_segment_start, segment_vector) / segment_length

                if projection < 0:
                    closest_point = segment_start
                elif projection > segment_length:
                    closest_point = segment_end
                else:
                    closest_point = segment_start + (segment_vector / segment_length) * projection

                distance_to_closest = torch.norm(current_positions[i] - closest_point)
                if distance_to_closest <= self.lookahead_distance:
                    remaining_distance = self.lookahead_distance - distance_to_closest
                    lookahead_points[i] = closest_point + (segment_vector / segment_length) * remaining_distance
                    break
                else:
                    lookahead_points[i] = self.paths[i, path_lengths[i] - 1]

            if torch.norm(current_positions[i] - self.paths[i, path_lengths[i] - 1]) < 0.3:
                self.end[i] = True
                lookahead_points[i] = self.paths[i, path_lengths[i] - 1]

        return lookahead_points

    def get_quadrant(self, nx: torch.Tensor, ny: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """
        Определяет квадрант для вектора относительно осей nx, ny.

        Args:
            nx (torch.Tensor): Ось X [num_envs, 2].
            ny (torch.Tensor): Ось Y [num_envs, 2].
            vector (torch.Tensor): Вектор к цели [num_envs, 2].

        Returns:
            torch.Tensor: Знак квадранта [num_envs].
        """
        LR = vector[:, 0] * nx[:, 1] - vector[:, 1] * nx[:, 0]
        return torch.sign(LR)

    def pure_pursuit_controller(self, current_positions: torch.Tensor, current_orientations: torch.Tensor):
        """
        Реализует Pure Pursuit контроллер для всех сред.

        Args:
            current_positions (torch.Tensor): Текущие позиции роботов [num_envs, 2].
            current_orientations (torch.Tensor): Текущие ориентации (yaw) [num_envs].

        Returns:
            tuple: (linear_velocity [num_envs], angular_velocity [num_envs])
        """
        linear_velocity = torch.full((self.num_envs,), self.linear_velocity, device=self.device)
        angular_velocity = torch.zeros(self.num_envs, device=self.device)

        current_heading = torch.where(
            current_orientations < 0,
            -math.pi - current_orientations,
            math.pi - current_orientations
        )

        distance_to_target = torch.norm(self.target_positions - current_positions, dim=1)
        distance_to_path_end = torch.norm(self.paths[:, -1] - current_positions, dim=1)
        close_to_target = (distance_to_target < 1.0) | (distance_to_path_end < 0.3)
        self.end |= close_to_target

        mask_start_or_end = self.start | self.end
        mask_normal = ~mask_start_or_end

        if torch.any(mask_normal):
            lookahead_points = self.get_lookahead_point(current_positions)
            to_target = lookahead_points - current_positions
            target_angle = torch.atan2(to_target[:, 1], to_target[:, 0])
            alpha = self.normalize_angle(target_angle - current_heading)
            curvature = 2 * torch.sin(alpha) / self.lookahead_distance
            angular_velocity[mask_normal] = curvature[mask_normal] * self.linear_velocity
            angular_velocity[mask_normal] = torch.clamp(
                angular_velocity[mask_normal],
                -self.max_angular_velocity,
                self.max_angular_velocity
            )
            linear_velocity[mask_normal] *= (self.max_angular_velocity - angular_velocity[mask_normal].abs()) / self.max_angular_velocity

        if torch.any(mask_start_or_end):
            if torch.any(self.first_ep):
                linear_velocity[self.first_ep] = 0.0
                angular_velocity[self.first_ep] = 0.0
                self.first_ep[:] = False
            else:
                nx = torch.tensor([[-1.0, 0.0]], device=self.device).repeat(self.num_envs, 1)
                ny = torch.tensor([[0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
                to_goal_vec = torch.where(
                    self.end[:, None],
                    self.target_positions - current_positions,
                    self.paths[:, 1] - current_positions
                )
                cos_angle = torch.sum(to_goal_vec * nx, dim=1) / torch.norm(to_goal_vec, dim=1) / torch.norm(nx, dim=1)
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                true_angle = self.get_quadrant(nx, ny, to_goal_vec) * torch.acos(cos_angle)
                
                angle_1 = torch.where(true_angle >= 0, true_angle, true_angle + 2 * math.pi) + math.pi
                angle_1 = torch.where(angle_1 >= 2 * math.pi, angle_1 - 2 * math.pi, angle_1)
                angle_1 = torch.where(angle_1 == 2 * math.pi, torch.zeros_like(angle_1), angle_1)
                
                angle_2 = 2 * math.pi - torch.where(current_heading >= 0, current_heading, current_heading + 2 * math.pi)
                angle_2 = torch.where(angle_2 == 2 * math.pi, torch.zeros_like(angle_2), angle_2)
                
                sign = torch.ones(self.num_envs, device=self.device)
                angle_diff = angle_2 - angle_1
                sign = torch.where(
                    angle_2 > angle_1,
                    torch.where(angle_diff < 2 * math.pi - angle_diff, 1.0, -1.0),
                    torch.where(angle_1 - angle_2 < 2 * math.pi - angle_1 + angle_2, -1.0, 1.0)
                )
                
                angular_velocity[mask_start_or_end] = sign[mask_start_or_end] * 1.0
                linear_velocity[mask_start_or_end] = 0.0
                
                self.start &= ~(torch.abs(angle_1 - angle_2) < math.pi / 80)

        return linear_velocity, angular_velocity