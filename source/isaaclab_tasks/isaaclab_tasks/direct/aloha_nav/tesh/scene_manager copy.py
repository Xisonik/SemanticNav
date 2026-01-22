import torch
import math
from scipy.spatial import ConvexHull
import random
from .graph_manager import ObstacleGraph

class SceneManager:
    def __init__(self, num_envs, device, num_obstacles=6):
        self.num_envs = num_envs
        self.device = device
        
        # Тензоры для хранения позиций и ориентаций
        self.robot_pos = torch.zeros((num_envs, 2), device=device)  # [x, y]
        self.robot_yaw = torch.zeros(num_envs, device=device)      # угол yaw
        self.goal_local_pos = torch.zeros((num_envs, 2), device=device)  # [x, y] цели
        
        # Параметры управления
        self.max_linear_speed = 1.0   # максимальная линейная скорость (м/с)
        self.max_angular_speed = 0.5  # максимальная угловая скорость (рад/с)
        self.angle_threshold = math.pi / 30  # порог угла (10 градусов) для начала движения вперед
        
        # Размеры объектов
        self.robot_radius = 0.5  # радиус робота (м)
        self.goal_radius = 0.3   # радиус цели (м)
        
        # Границы комнаты в локальной системе координат
        self.room_bounds = {
            'x_min': -5.0,
            'x_max': 3.5,
            'y_min': -3,
            'y_max': 3
        }
        
        # Параметры для генерации расстояний цель-робот
        self.max_radius_values = 8
        self.radius_values = torch.zeros(num_envs, device=device)  
        self.angle_values = torch.tensor([0], device=device)
        
        # Расчёт максимального радиуса (расстояние цель-робот)
        room_diagonal = math.sqrt(self.room_bounds['x_max']**2 + self.room_bounds['y_max']**2)  # 5 м
        self.max_radius = room_diagonal - (self.robot_radius + self.goal_radius + 0.2)  # 5 - (0.5 + 0.3) = 4.2 м
        
        # Границы для проверки позиций с учётом радиусов
        self.robot_bounds = {
            'x_min': self.room_bounds['x_min'] + self.robot_radius + 1,  # -3.5
            'x_max': self.room_bounds['x_max'] - self.robot_radius,  # 3.5
            'y_min': self.room_bounds['y_min'] + self.robot_radius,  # -2.5
            'y_max': self.room_bounds['y_max'] - self.robot_radius   # 2.5
        }
        self.goal_bounds = {
            'x_min': self.room_bounds['x_min'] + self.goal_radius,  # -3.7
            'x_max': self.room_bounds['x_max'] - self.goal_radius,  # 3.7
            'y_min': self.room_bounds['y_min'] + self.goal_radius,  # -2.7
            'y_max': self.room_bounds['y_max'] - self.goal_radius   # 2.7
        }

        self.room_vertices = torch.tensor([
            [self.room_bounds['x_min'] + self.robot_radius + self.goal_radius + 0.1, self.room_bounds['y_min'] + self.robot_radius + self.goal_radius + 0.1],
            [self.room_bounds['x_min'] + self.robot_radius + self.goal_radius + 0.1, self.room_bounds['y_max'] - self.robot_radius - self.goal_radius - 0.1],
            [self.room_bounds['x_max'] - self.robot_radius - self.goal_radius - 0.1, self.room_bounds['y_min'] + self.robot_radius + self.goal_radius + 0.1],
            [self.room_bounds['x_max'] - self.robot_radius - self.goal_radius - 0.1, self.room_bounds['y_max'] - self.robot_radius - self.goal_radius - 0.1] 
        ], device=self.device)

        # Дискретизация углов
        self.n_angles = 36  # Шаг дискретизации: pi/n
        self.num_angle_steps = 2 * self.n_angles  # 72 угла (0 до 2pi)
        self.angle_step = math.pi / self.n_angles
        self.discrete_angles = torch.arange(0, 2 * math.pi, self.angle_step, device=device)

        self.num_envs = num_envs
        self.device = device
        self.num_obstacles = num_obstacles
        grid_x = [-2.0, -1.0]  # Уровни по X для активных позиций
        grid_y = [-1.0, 0.0, 1.0]  # Уровни по Y для активных позиций
        self.possible_positions = [(x, y, 0.0) for x in grid_x for y in grid_y]  # 6 возможных позиций
        self.graphs = [ObstacleGraph(num_obstacles, device) for _ in range(num_envs)]
        self.obstacle_radii = [0.5] * num_obstacles
        for graph in self.graphs:
            graph.set_radii(self.obstacle_radii)

    def compute_max_radii(self, goal_pos):
        """Вычисляет максимальный радиус для размещения робота относительно каждой цели."""
        # Расчёт расстояний от каждой цели до всех вершин
        distances = torch.norm(goal_pos.unsqueeze(1) - self.room_vertices.unsqueeze(0), dim=2)
        # Максимальное расстояние до вершин для каждой цели
        max_distances = torch.max(distances, dim=1)[0]  # [num_envs]
        # Максимальный радиус = максимальное расстояние - радиусы робота и цели
        max_radii = max_distances - (self.robot_radius + self.goal_radius + 0.2) # [num_envs]
        # Убедимся, что радиус не отрицательный
        max_radii = torch.clamp(max_radii, min=0.5)
        max_radii = 8*torch.ones(self.num_envs,device=self.device)
        return max_radii

    def _compute_valid_angles(self, goal_pos, radii):
        """Вычисляет булев тензор допустимых углов для каждой среды."""
        num_envs = goal_pos.shape[0]
        # Тензор [num_envs, num_angle_steps], изначально все углы допустимы
        valid_angles = torch.ones((num_envs, self.num_angle_steps), dtype=torch.bool, device=self.device)
        # Проверяем каждый угол
        for k in range(self.num_angle_steps):
            theta = self.discrete_angles[k]
            # Вычисляем позиции робота для всех сред
            rx = goal_pos[:, 0] + radii * torch.cos(theta)
            ry = goal_pos[:, 1] + radii * torch.sin(theta)
            
            # Проверяем, выходит ли робот за границы
            invalid = (
                (rx < self.robot_bounds['x_min']) |
                (rx > self.robot_bounds['x_max']) |
                (ry < self.robot_bounds['y_min']) |
                (ry > self.robot_bounds['y_max'])
            )
            valid_angles[:, k] = ~invalid
        return valid_angles

    def initialize_radii(self, num_envs, mean_radius, std_radius, max_radii=8, min_radius=1.2, max_radius=None, device='cuda'):
        """
        Инициализация радиусов для num_envs сред с нормальным распределением.
        
        Args:
            num_envs (int): Количество сред.
            mean_radius (float): Центр распределения радиусов (среднее).
            std_radius (float): Стандартное отклонение (дисперсия).
            min_radius (float): Минимальный радиус (по умолчанию 1.8).
            max_radius (float, optional): Глобальный максимальный радиус. Если None, используется self.max_radius_values.
            device (str): Устройство для тензоров (cuda или cpu).
        
        Returns:
            torch.Tensor: Тензор радиусов размером [num_envs].
        """
        if max_radius is None:
            max_radius = self.max_radius_values

        # Генерация радиусов из нормального распределения
        radii = torch.normal(mean=mean_radius, std=std_radius, size=(num_envs,), device=device)

        # Ограничение радиусов минимальным и глобальным максимальным значениями
        radii = torch.clamp(radii, min=min_radius, max=max_radius)

        # Учет локальных максимальных радиусов для каждой среды
        return radii
   
    def generate_goal_positions_local(self, env_ids):
        """Генерирует случайные позиции для целей в пределах комнаты."""
        num_envs = len(env_ids)
        
        sigma_x = (self.goal_bounds['x_max'] - self.goal_bounds['x_min']) / 4  # ~1/4 ширины
        sigma_y = (self.goal_bounds['y_max'] - self.goal_bounds['y_min']) / 4  # ~1/4 высоты
        x = torch.randn(num_envs, device=self.device) * sigma_x
        y = torch.randn(num_envs, device=self.device) * sigma_y

        # Ограничение координат в пределах goal_bounds
        x = torch.clamp(x, self.goal_bounds['x_min'] + 0.5, self.goal_bounds['x_max'] - 0.5)
        y = torch.clamp(y, self.goal_bounds['y_min'] + 0.5, self.goal_bounds['y_max'] - 0.5)
        
        # positions = torch.zeros((num_envs,2),  device=self.device) #+ torch.stack([x, y], dim=1)
        positions = torch.tensor([[-4.5, 0.0]], device=self.device).repeat(num_envs, 1)
        # # print("goal pos ", positions)
        return positions
    
    def shift_pos(self, env_ids, obj_pos, terrain_origins):
        return obj_pos[env_ids] + terrain_origins[env_ids, :2]

    def _generate_obj_positions(self, env_ids, goal_pos, mean_radius, min_radius, std_radius=0.8):
        """Генерирует позиции робота на заданном расстоянии от цели."""
        num_envs = len(env_ids)
        
        # Генерация радиусов
        max_radii = self.compute_max_radii(goal_pos)
        # Инициализация радиусов
        delta = 0.2
        radii = self.initialize_radii(
            num_envs=num_envs,
            mean_radius=mean_radius,
            std_radius=std_radius,
            max_radii=8,
            min_radius=min_radius,
            max_radius=self.max_radius_values,
            device=self.device
        )
        # Вычисление допустимых углов
        valid_angles = self._compute_valid_angles(goal_pos, radii)
        # Выбор случайного допустимого угла для каждой среды
        theta = torch.zeros(num_envs, device=self.device)
        for i in range(num_envs):
            valid_indices = torch.where(valid_angles[i])[0]
            if valid_indices.numel() > 0:
                # Случайно выбираем индекс из допустимых
                idx = valid_indices[torch.randint(0, valid_indices.numel(), (1,), device=self.device)]
                theta[i] = self.discrete_angles[idx]
            else:
                theta[i] = torch.rand(1, device=self.device) * 2 * math.pi
        # Вычисление координат робота
        x = radii * torch.cos(theta)
        y = radii * torch.sin(theta)
        robot_offsets = torch.stack([x, y], dim=1)
        positions = goal_pos + robot_offsets
        return positions

    def check_bouds(self, env_ids, new_obj_pos):
        robot_valid = (
                (new_obj_pos[:, 0] >= self.robot_bounds['x_min']) &
                (new_obj_pos[:, 0] <= self.robot_bounds['x_max']) &
                (new_obj_pos[:, 1] >= self.robot_bounds['y_min']) &
                (new_obj_pos[:, 1] <= self.robot_bounds['y_max'])
            )
        # print("check_bouds ", robot_valid)
        if torch.any(~robot_valid):
            return False
        return True
    
    def get_checked_for_room_bounds_pos(self, env_ids, mean_radius, min_radius):
        invalid_env_ids = torch.tensor(range(len(env_ids)), device=self.device)
        max_iterations = 100
        iteration = 0
        obj_pos = self._generate_obj_positions(invalid_env_ids, self.goal_local_pos[invalid_env_ids], mean_radius, min_radius)
        # Проверяем коллизии
        while len(invalid_env_ids) > 0:
            num_invalid = len(invalid_env_ids)
            new_obj_pos = self._generate_obj_positions(invalid_env_ids, self.goal_local_pos[invalid_env_ids], mean_radius, min_radius)
            robot_valid = (
                (new_obj_pos[:, 0] >= self.robot_bounds['x_min']) &
                (new_obj_pos[:, 0] <= self.robot_bounds['x_max']) &
                (new_obj_pos[:, 1] >= self.robot_bounds['y_min']) &
                (new_obj_pos[:, 1] <= self.robot_bounds['y_max'])
            )
            valid_new = robot_valid
            valid_new_ids = invalid_env_ids[valid_new]
            # print("valid_new_ids", valid_new_ids)
            # print(valid_new)
            # print(invalid_env_ids)
            # print(obj_pos)
            # print(new_obj_pos)
            obj_pos[valid_new_ids] = new_obj_pos[valid_new]
            # print(valid_new, invalid_env_ids)
            invalid_env_ids = invalid_env_ids[~valid_new]
            iteration += 1
            if iteration > max_iterations:
                for i in len(self.graphs):
                    print(i, self.graphs[i].get_graph_info())
                print(invalid_env_ids)
                print(self.goal_local_pos[invalid_env_ids])
                print(obj_pos)
                print("error in getting correction position")

        if len(invalid_env_ids) > 0:
            print(f"Warning: Could not generate valid positions for {len(invalid_env_ids)} environments")
        
        return obj_pos

    def get_checked_for_obstacles_rp(self, env_ids, obj_pos):
        """
        Проверяет позиции робота на коллизии с препятствиями и корректирует их векторно.
        
        Args:
            env_ids (torch.Tensor): Индексы сред для проверки.
        
        Returns:
            None: Обновляет obj_pos для сред с коллизиями.
        """
        num_envs = len(env_ids)
        env_ids = torch.tensor(range(num_envs), device=self.device)
        safety_margin = 0.2
        max_iterations = 10  # Ограничение на итерации
        # print("i ", obj_pos)
        # Собираем активные препятствия
        obstacles_per_env = [self.graphs[env_id.item()].get_active_nodes() for env_id in env_ids]
        max_obstacles = max(len(obs) for obs in obstacles_per_env) if obstacles_per_env else 0
        if max_obstacles == 0:
            return obj_pos

        # Инициализируем тензоры
        obstacle_positions = torch.full((num_envs, max_obstacles, 2), 6666.0, device=self.device)
        obstacle_radii = torch.full((num_envs, max_obstacles), 0.001, device=self.device)
        valid_mask = torch.zeros((num_envs, max_obstacles), dtype=torch.bool, device=self.device)

        # Заполняем тензоры позиций и радиусов
        num_obstacles = torch.tensor([len(obs) for obs in obstacles_per_env], device=self.device)
        env_indices = torch.arange(num_envs, device=self.device).repeat_interleave(num_obstacles)
        obstacle_indices = torch.cat([torch.arange(n, device=self.device) for n in num_obstacles])
        positions = torch.tensor([obs['position'][:2] for env_id, obs_list in enumerate(obstacles_per_env) 
                                for obs in obs_list], device=self.device)
        radii = torch.tensor([obs['radius'] for env_id, obs_list in enumerate(obstacles_per_env) 
                            for obs in obs_list], device=self.device)
        
        if len(env_indices) > 0:
            obstacle_positions[env_indices, obstacle_indices] = positions
            obstacle_radii[env_indices, obstacle_indices] = radii
            valid_mask[env_indices, obstacle_indices] = True

        # Генерируем случайное направление для каждой среды один раз
        directions = torch.randn((num_envs, 2), device=self.device)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)  # Нормализуем

        # Итерации для устранения коллизий
        while True:
            # Проверяем коллизии
            # print("obstacles_per_env", obstacles_per_env)
            # print("directions", directions)
            # print("obstacle_positions", obstacle_positions)

            distances = torch.norm(obj_pos[env_ids, None, :] - obstacle_positions, dim=2)
            # print("2 ", obj_pos)
            # print("distances", distances)
            required_dists = self.robot_radius + 0.5# obstacle_radii
            # print("required_dists", required_dists)
            collisions = (distances < required_dists) & valid_mask
            # print("collisions", collisions)
            # Если нет коллизий, выходим
            has_collisions = torch.any(collisions, dim=1)
            # print("has_collisions", has_collisions)
            if not torch.any(has_collisions):
                break
            # print("aloha dance with obstacles")
            # Находим ближайшее препятствие для сред с коллизиями
            distances_masked = distances.clone()
            distances_masked[~valid_mask] = float('inf')
            # print("distances_masked", distances_masked)
            closest_indices = torch.argmin(distances_masked, dim=1)
            # print("closest_indices", closest_indices)
            env_ids_collisions = torch.where(has_collisions)[0]
            # print("env_ids_collisions", env_ids_collisions)
            abs_env_ids = env_ids[env_ids_collisions]
            # print("abs_env_ids", abs_env_ids)
            # Векторно вычисляем смещение
            closest_positions = obstacle_positions[env_ids_collisions, closest_indices[env_ids_collisions]]
            # print("closest_positions", closest_positions)
            closest_radii = obstacle_radii[env_ids_collisions, closest_indices[env_ids_collisions]]
            # print("closest_radii", closest_radii)
            vecs = obj_pos[abs_env_ids] - closest_positions
            # print("3 ", obj_pos)
            # print("vecs", vecs)
            dists = torch.norm(vecs, dim=1)
            # print("dists", dists)
            # Используем заранее сгенерированное направление для сред с коллизиями
            used_directions = directions[env_ids_collisions]
            # print("used_directions", used_directions)
            near_zero = dists < 1e-6
            # print("near_zero", near_zero)
            if torch.any(near_zero):
                used_directions[near_zero] = torch.randn((near_zero.sum(), 2), device=self.device)
                used_directions[near_zero] = used_directions[near_zero] / torch.norm(used_directions[near_zero], dim=1, keepdim=True)

            # Смещаем робота на границу препятствия в направлении used_directions
            displacements = used_directions * (closest_radii + self.robot_radius + safety_margin)[:, None]
            # print("displacements", displacements)
            obj_pos[abs_env_ids] = closest_positions + displacements
            # print("4 ", obj_pos, abs_env_ids)
            # print(f"self.robot_pos[{abs_env_ids}]", self.robot_pos[abs_env_ids])
        return obj_pos

    def get_pos(self, env_ids, terrain_origins,mean_radius,min_radius):
        """Проверяет позиции роботов и целей, перегенерирует для недопустимых."""
        while True:
            obj_pos = self.get_checked_for_room_bounds_pos(env_ids, mean_radius, min_radius)
            # print("check 1: ", obj_pos, env_ids)
            obj_pos = self.get_checked_for_obstacles_rp(env_ids, obj_pos)
            # print("check 2: ", obj_pos, env_ids)
            if self.check_bouds(env_ids, obj_pos):
                break
        return obj_pos + terrain_origins[env_ids, :2]

    def reset(self, env_ids, terrain_origins,mean_radius=1.3,min_radius=1.2,max_angle=0):
        """Инициализация стартовых позиций и целей при сбросе."""
        num_envs = len(env_ids)       
        # Проверка и корректировка позиций
        self.goal_local_pos[env_ids] = self.generate_goal_positions_local(
            env_ids)
        self.robot_pos[env_ids] = self.get_pos(
            env_ids, terrain_origins, mean_radius, min_radius)
        
        # Генерация угловых ошибок
        random_sign = torch.sign(torch.rand(num_envs, device=self.device) - 0.5)
        angle_errors = torch.rand(num_envs, device=self.device) * max_angle
        
        # Начальная ориентация робота
        direction_to_goal = self.shift_pos(env_ids, self.goal_local_pos, terrain_origins) - self.robot_pos[env_ids]
        yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
        self.robot_yaw[env_ids] = yaw + angle_errors * random_sign
        
        # Возвращаем данные для симуляции
        quaternion = torch.zeros((num_envs, 4), device=self.device)
        quaternion[:, 0] = torch.cos(self.robot_yaw[env_ids] / 2.0)  # w
        quaternion[:, 3] = torch.sin(self.robot_yaw[env_ids] / 2.0)  # z
        
        return self.robot_pos[env_ids], quaternion, self.shift_pos(env_ids, self.goal_local_pos, terrain_origins)

    def compute_control(self, current_pos, current_quat):
        """Вычисление управляющих сигналов: разворот с фиксированной скоростью, затем движение к цели."""
        # Текущий угол робота
        current_yaw = torch.atan2(current_quat[:, 3], current_quat[:, 0]) * 2
        # Направление к цели
        direction_to_goal = self.goal_local_pos - current_pos[:, :2]
        desired_yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
        
        # Ошибка по углу (нормализуем в диапазон [-pi, pi])
        angle_error = (desired_yaw - current_yaw)
        
        # Линейная и угловая скорости
        linear_speed = torch.zeros(self.num_envs, device=self.device)
        angular_speed = torch.zeros(self.num_envs, device=self.device)
        
        # Если угол больше порога, поворачиваем с фиксированной скоростью
        needs_rotation = torch.abs(angle_error) > self.angle_threshold
        angular_speed[needs_rotation] = self.max_angular_speed * torch.sign(angle_error[needs_rotation])
        
        # Если угол в пределах порога, двигаемся вперед
        can_move_forward = ~needs_rotation
        linear_speed[can_move_forward] = self.max_linear_speed
        
        return torch.stack([linear_speed, angular_speed], dim=1)

    def generate_obstacle_positions(self, mess=False, env_ids=None, terrain_origins=None, min_num_active=0, max_num_active=None):
        """
        Выбирает активные стулья и распределяет им случайные позиции из сетки.
        Неактивные стулья остаются на дефолтных позициях.
        
        Args:
            env_ids (torch.Tensor, optional): Индексы сред для обновления. Если None, обновляются все среды.
            terrain_origins (torch.Tensor, optional): Смещения локальных координат для каждой среды [num_envs, 3].
        
        Returns:
            List[ObstacleGraph]: Список графов для каждой среды.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        if terrain_origins is None:
            terrain_origins = torch.zeros((self.num_envs, 3), device=self.device)
        
        num_envs = len(env_ids)
        max_attempts = 10
        max_obstacles = self.num_obstacles
        max_positions = len(self.possible_positions)
        
        # Инициализируем позиции для всех сред
        positions = torch.stack([self.graphs[env_id.item()].default_positions 
                                for env_id in env_ids], dim=0)  # [num_envs, num_obstacles, 3]
        
        valid_envs = None
        for attempt in range(max_attempts):
            # Генерируем количество активных узлов для каждой среды
            if max_num_active == None:
                max_num_active = min(max_obstacles, max_positions)
            else:
                max_num_active = min(max_obstacles, max_positions, max_num_active)
            num_active = torch.randint(min_num_active, max_num_active + 1, 
                                    size=(num_envs,), device=self.device)  # [num_envs]
            
            # Маска для сред с num_active == 0
            zero_active_mask = num_active == 0
            
            # Обрабатываем среды с num_active == 0
            if torch.any(zero_active_mask):
                terrain_offset = terrain_origins[env_ids[zero_active_mask], :2]  # [num_zero_active, 2]
                positions[zero_active_mask, :, :2] += terrain_offset[:, None, :]  # [num_zero_active, num_obstacles, 2]
                for env_idx, env_id in enumerate(env_ids[zero_active_mask]):
                    self.graphs[env_id.item()].update_positions([], positions[env_idx])
            
            # Обрабатываем среды с num_active > 0
            active_mask = ~zero_active_mask
            if not torch.any(active_mask):
                continue
            
            active_env_ids = env_ids[active_mask]  # [num_active_envs]
            active_num_active = num_active[active_mask]  # [num_active_envs]
            
            # Генерируем случайные индексы для активных узлов
            indices = torch.randperm(max_obstacles, device=self.device)[:active_num_active.max()]  # [max(num_active)]
            active_indices = [indices[:n].tolist() for n in active_num_active]  # Список списков для каждой среды
            
            if mess:
                max_num_env = active_num_active.max()
                # Initialize selected_positions as a list of empty lists
                selected_positions = [[] for _ in range(len(active_env_ids))]
                active_indices = [random.sample(range(max_obstacles), n.item()) for n in active_num_active]
                
                # Sequentially add obstacles up to max(active_num_active)
                for step in range(active_num_active.max().item()):
                    # Identify environments that need another obstacle
                    need_obstacle = active_num_active > step  # [num_active_envs]
                    if not torch.any(need_obstacle):
                        break
                    
                    # Select environments that still need obstacles
                    current_env_ids = active_env_ids[need_obstacle]  # [num_current_envs]
                    
                    # Generate one position per environment using get_pos
                    new_positions = self.get_pos(
                        env_ids=current_env_ids,
                        terrain_origins=terrain_origins,
                        mean_radius=1.3,
                        min_radius=1.2
                    )  # [num_current_envs, 2]
                    new_positions_3d = torch.cat([new_positions, torch.zeros((len(current_env_ids), 1), device=self.device)], dim=1)  # [num_current_envs, 3]
                    
                    # Update selected_positions and graphs for these environments
                    for env_idx, env_id in enumerate(current_env_ids):
                        local_env_idx = (active_env_ids == env_id).nonzero(as_tuple=True)[0].item()
                        selected_positions[local_env_idx].append(new_positions_3d[env_idx:env_idx+1])
                        # Temporarily update graph to include this obstacle
                        self.graphs[env_id.item()].update_positions([active_indices[local_env_idx][step]], new_positions_3d[env_idx:env_idx+1])
                
                # Convert selected_positions to list of tensors
                selected_positions = [torch.cat(pos_list, dim=0) if pos_list else torch.tensor([], device=self.device).reshape(0, 3) 
                                    for pos_list in selected_positions]
            else:
                # Выбираем случайные позиции из possible_positions
                possible_positions = torch.tensor(self.possible_positions, 
                                                device=self.device, dtype=torch.float32)  # [num_positions, 3]
                selected_indices = torch.randperm(max_positions, device=self.device)[:active_num_active.max()]  # [max(num_active)]
                selected_positions = [possible_positions[indices[:n]] for n in active_num_active]  # Список тензоров [num_active_i, 3]
            
            # Обновляем позиции для активных узлов
            for env_idx, (env_id, indices, sel_pos) in enumerate(zip(active_env_ids, active_indices, selected_positions)):
                # terrain_offset = terrain_origins[env_id, :2]  # [2]
                # sel_pos[:, :2] += terrain_offset  # [num_active, 2]
                positions[env_idx, indices] = sel_pos  # Обновляем активные узлы
                self.graphs[env_id.item()].update_positions(indices, sel_pos)
            
            # Проверяем минимальное расстояние между активными узлами
            valid_envs = torch.ones(num_envs, dtype=torch.bool, device=self.device)
            for env_idx, env_id in enumerate(env_ids):
                if zero_active_mask[env_idx]:
                    continue
                active_edges = [(u, v) for u, v, d in self.graphs[env_id.item()].graph.edges(data=True) 
                                if d['weight'] != float('inf')]
                if not active_edges:
                    valid_envs[env_idx] = True
                    continue
                min_dist = min(self.graphs[env_id.item()].graph.edges[u, v]['weight'] for u, v in active_edges)
                min_required = min(node['radius'] for node in self.graphs[env_id.item()].get_active_nodes()) * 2
                valid_envs[env_idx] = min_dist >= min_required
            
            # Если все среды валидны, выходим
            if torch.all(valid_envs):
                break
        
        # Для невалидных сред устанавливаем дефолтные позиции
        if valid_envs is not None:
            invalid_envs = ~valid_envs

            if torch.any(invalid_envs):
                invalid_env_ids = env_ids[invalid_envs]
                positions[invalid_envs] = torch.stack([self.graphs[env_id.item()].default_positions 
                                                    for env_id in invalid_env_ids], dim=0)
                # terrain_offset = terrain_origins[invalid_env_ids, :2]
                # positions[invalid_envs, :, :2] += terrain_offset[:, None, :]
                for env_idx, env_id in enumerate(invalid_env_ids):
                    self.graphs[env_id.item()].update_positions([], positions[env_idx])
            
        # Обновляем active_nodes и выводим отладочную информацию
        # for env_id in env_ids:
        #     print(env_id, self.graphs[env_id.item()].get_graph_info())
        
        return self.graphs