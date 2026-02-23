import torch
import math

class ControlManager:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        
        # Тензоры для хранения позиций и ориентаций робота
        self.robot_pos = torch.zeros((num_envs, 2), device=device)  # [x, y]
        self.robot_yaw = torch.zeros(num_envs, device=device)      # угол yaw
        self.goal_pos = torch.zeros((num_envs, 2), device=device)  # [x, y] цели
        
        # Параметры управления
        self.max_linear_speed = 1.0   # максимальная линейная скорость (м/с)
        self.max_angular_speed = 0.5  # максимальная угловая скорость (рад/с)
        self.angle_threshold = math.pi / 30  # порог угла (10 градусов)
        
        # Параметры для генерации стартовых позиций
        self.radius_values = torch.arange(3, 4, 0.1, device=device)
        self.angle_values = torch.tensor([-math.pi/3, -math.pi/4, -math.pi/8, 0, math.pi/8, math.pi/4, math.pi/3], device=device)

        # Параметры препятствий
        self.obstacle_types = ["cube", "wall", "cylinder"]
        self.obstacle_colors = [
            torch.tensor([0.0, 0.0, 1.0], device=device),  # синий
            torch.tensor([1.0, 0.5, 0.0], device=device),  # оранжевый
            torch.tensor([1.0, 0.0, 0.0], device=device),  # красный
            torch.tensor([0.5, 0.0, 1.0], device=device)   # фиолетовый
        ]
        self.obstacle_sizes = torch.arange(0.1, 0.31, 0.05, device=device)  # от 0.1 до 0.3 с шагом 0.05

    

    def reset(self, env_ids, terrain_origins):
        num_envs = len(env_ids)

        # Генерация радиусов и угловых ошибок
        num_radiuses = len(self.radius_values)
        num_angles = len(self.angle_values)
        total_combinations = num_radiuses * num_angles
        radius_base = self.radius_values.repeat_interleave(num_angles)
        angle_base = self.angle_values.repeat(num_radiuses)
        repeat_times = (num_envs + total_combinations - 1) // total_combinations
        radii = radius_base.repeat(repeat_times)[:num_envs]
        angle_errors = angle_base.repeat(repeat_times)[:num_envs]
        random_sign = torch.sign(torch.rand(num_envs, device=self.device) - 0.5)
        
        # Позиции целей в центре среды
        self.goal_pos[env_ids] = terrain_origins[env_ids, :2]
        
        # Позиции робота на радиусе от цели
        theta = torch.rand(num_envs, device=self.device) * 2 * torch.pi
        x = radii * torch.cos(theta)
        y = radii * torch.sin(theta)
        robot_pos = torch.stack([x, y], dim=1) + self.goal_pos[env_ids]
        # Ограничение координат робота
        robot_pos[:, 0] = torch.clamp(robot_pos[:, 0], terrain_origins[env_ids, 0] - 4.5, terrain_origins[env_ids, 0] + 4.5)
        robot_pos[:, 1] = torch.clamp(robot_pos[:, 1], terrain_origins[env_ids, 1] - 4.5, terrain_origins[env_ids, 1] + 4.5)
        self.robot_pos[env_ids] = robot_pos
        
        # Ориентация робота
        direction_to_goal = self.goal_pos[env_ids] - self.robot_pos[env_ids]
        yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
        self.robot_yaw[env_ids] = yaw + angle_errors * random_sign
        
        # Генерация препятствий
        num_obstacles = torch.randint(2, 4, (num_envs,), device=self.device)  # от 2 до 3 препятствий
        max_obstacles = num_obstacles.max().item()
        
        # Тензоры для хранения данных препятствий
        obstacle_positions = torch.zeros((num_envs, max_obstacles, 2), device=self.device)
        obstacle_sizes = torch.zeros((num_envs, max_obstacles, 3), device=self.device)
        obstacle_types_idx = torch.zeros((num_envs, max_obstacles), dtype=torch.long, device=self.device)
        obstacle_colors_idx = torch.zeros((num_envs, max_obstacles), dtype=torch.long, device=self.device)
        obstacle_mask = torch.zeros((num_envs, max_obstacles), dtype=torch.bool, device=self.device)
        
        for i in range(num_envs):
            n = num_obstacles[i]
            # Случайные позиции препятствий между роботом и целью
            t = torch.rand(n, device=self.device)
            obs_x = self.robot_pos[env_ids[i]][0] + t * (self.goal_pos[env_ids[i]][0] - self.robot_pos[env_ids[i]][0])
            obs_y = self.robot_pos[env_ids[i]][1] + t * (self.goal_pos[env_ids[i]][1] - self.robot_pos[env_ids[i]][1])
            obstacle_positions[i, :n] = torch.stack([obs_x, obs_y], dim=1)
            
            # Случайные размеры и типы
            sizes = self.obstacle_sizes[torch.randint(0, len(self.obstacle_sizes), (n,), device=self.device)]
            types_idx = torch.randint(0, len(self.obstacle_types), (n,), device=self.device)
            colors_idx = torch.randint(0, len(self.obstacle_colors), (n,), device=self.device)
            
            for j in range(n):
                if self.obstacle_types[types_idx[j]] == "wall":
                    obstacle_sizes[i, j] = torch.tensor([0.05, sizes[j], sizes[j]], device=self.device)  # тонкая стена
                else:  # куб или цилиндр
                    obstacle_sizes[i, j] = torch.tensor([sizes[j]] * 3, device=self.device)
            
            obstacle_types_idx[i, :n] = types_idx
            obstacle_colors_idx[i, :n] = colors_idx
            obstacle_mask[i, :n] = True

        # Возвращаем данные для симуляции
        quaternion = torch.zeros((num_envs, 4), device=self.device)
        quaternion[:, 0] = torch.cos(self.robot_yaw[env_ids] / 2.0)  # w
        quaternion[:, 3] = torch.sin(self.robot_yaw[env_ids] / 2.0)  # z
        
        return (
            self.robot_pos[env_ids],
            quaternion,
            {
                "positions": obstacle_positions,
                "sizes": obstacle_sizes,
                "types_idx": obstacle_types_idx,
                "colors_idx": obstacle_colors_idx,
                "mask": obstacle_mask,
                "types": self.obstacle_types,
                "colors": self.obstacle_colors
            }
        )

    def compute_control(self, current_pos, current_quat):
        # Оставляем без изменений
        current_yaw = torch.atan2(current_quat[:, 3], current_quat[:, 0]) * 2
        direction_to_goal = self.goal_pos - current_pos[:, :2]
        desired_yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
        angle_error = (desired_yaw - current_yaw)
        
        linear_speed = torch.zeros(self.num_envs, device=self.device)
        angular_speed = torch.zeros(self.num_envs, device=self.device)
        
        needs_rotation = torch.abs(angle_error) > self.angle_threshold
        angular_speed[needs_rotation] = self.max_angular_speed * torch.sign(angle_error[needs_rotation])
        can_move_forward = ~needs_rotation
        linear_speed[can_move_forward] = self.max_linear_speed
        
        return torch.stack([linear_speed, angular_speed], dim=1)