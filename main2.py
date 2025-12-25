try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("Pygame не установлен. Установите его командой: pip install pygame")
    exit(1)
import pygame
import sys
import math
import json
import os
import random
import numpy as np

pygame.init()

SCREEN_WIDTH = 1550
SCREEN_HEIGHT = 900
FPS = 60

BACKGROUND = (40, 44, 52)
PANEL_BG = (30, 34, 42)
ACCENT = (86, 98, 246)
SECONDARY = (56, 66, 84)
TEXT_COLOR = (220, 220, 220)
GRID_COLOR = (60, 70, 90)
TRACK_COLOR = (80, 90, 110)
ROAD_COLOR = (60, 70, 80)
WHITE = (255, 255, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
BLUE = (100, 150, 255)
YELLOW = (255, 255, 100)
PURPLE = (180, 100, 220)
ORANGE = (255, 150, 50)
CYAN = (100, 220, 220)
DGREEN = (3,203,0)
class Button:
    def __init__(self, x, y, width, height, text, color=ACCENT, hover_color=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color or self._adjust_color(color, 30)
        self.is_hovered = False
        self.font = pygame.font.SysFont('Arial', 20)
        self.border_radius = 8

    def _adjust_color(self, color, adjustment):
        return tuple(max(0, min(255, c + adjustment)) for c in color)

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, SECONDARY, self.rect, 2, border_radius=self.border_radius)

        text_surf = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def is_clicked(self, pos, click):
        return self.rect.collidepoint(pos) and click

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.weights_ih = np.random.randn(hidden_nodes, input_nodes) * np.sqrt(2.0 / input_nodes)
        self.weights_ho = np.random.randn(output_nodes, hidden_nodes) * np.sqrt(2.0 / hidden_nodes)
        self.bias_h = np.zeros((hidden_nodes, 1))
        self.bias_o = np.zeros((output_nodes, 1))
        
        self.fitness = 0
        self.best_lap_time = float('inf')
        self.lap_count = 0
        self.total_time = 0
        self.consistency = 0 

    def copy(self):
        """Создание копии нейронной сети"""
        copy = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        copy.weights_ih = self.weights_ih.copy()
        copy.weights_ho = self.weights_ho.copy()
        copy.bias_h = self.bias_h.copy()
        copy.bias_o = self.bias_o.copy()
        copy.best_lap_time = self.best_lap_time
        copy.lap_count = self.lap_count
        return copy

    def predict(self, input_array):
        """Прямое распространение сигнала"""
        # Преобразуем вход в столбец
        inputs = np.array(input_array, ndmin=2).T
        
        # Скрытый слой
        hidden = np.dot(self.weights_ih, inputs)
        hidden += self.bias_h
        hidden = self.relu(hidden)
        
        # Выходной слой
        output = np.dot(self.weights_ho, hidden)
        output += self.bias_o
        output = self.sigmoid(output)
        
        return output.flatten()

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def mutate(self, mutation_rate=0.1):
        """Мутация весов и смещений"""
        def mutate_array(arr):
            mask = np.random.random(arr.shape) < mutation_rate
            arr[mask] += np.random.randn(*arr[mask].shape) * 0.5
        
        mutate_array(self.weights_ih)
        mutate_array(self.weights_ho)
        mutate_array(self.bias_h)
        mutate_array(self.bias_o)

    def crossover(self, partner):
        """Скрещивание с другой нейронной сетью"""
        child = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        
        # Одноточечное скрещивание
        crossover_point = np.random.randint(0, self.weights_ih.size)
        
        # Преобразуем матрицы в плоские массивы для скрещивания
        ih_flat_self = self.weights_ih.flatten()
        ih_flat_partner = partner.weights_ih.flatten()
        ih_flat_child = np.concatenate([ih_flat_self[:crossover_point], 
                                        ih_flat_partner[crossover_point:]])
        child.weights_ih = ih_flat_child.reshape(self.weights_ih.shape)
        
        # Скрещивание для второй матрицы весов
        crossover_point = np.random.randint(0, self.weights_ho.size)
        ho_flat_self = self.weights_ho.flatten()
        ho_flat_partner = partner.weights_ho.flatten()
        ho_flat_child = np.concatenate([ho_flat_self[:crossover_point], 
                                        ho_flat_partner[crossover_point:]])
        child.weights_ho = ho_flat_child.reshape(self.weights_ho.shape)
        
        return child

    def save(self, filename):
        """Сохранение нейронной сети в файл"""
        data = {
            'weights_ih': self.weights_ih.tolist(),
            'weights_ho': self.weights_ho.tolist(),
            'bias_h': self.bias_h.tolist(),
            'bias_o': self.bias_o.tolist(),
            'fitness': self.fitness,
            'best_lap_time': self.best_lap_time,
            'lap_count': self.lap_count
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load(self, filename):
        """Загрузка нейронной сети из файла"""
        with open(filename, 'r') as f:
            data = json.load(f)
        self.weights_ih = np.array(data['weights_ih'])
        self.weights_ho = np.array(data['weights_ho'])
        self.bias_h = np.array(data['bias_h'])
        self.bias_o = np.array(data['bias_o'])
        self.fitness = data['fitness']
        self.best_lap_time = data.get('best_lap_time', float('inf'))
        self.lap_count = data.get('lap_count', 0)

class GeneticAlgorithm:
    def __init__(self, population_size, input_nodes, hidden_nodes, output_nodes):
        self.population_size = population_size
        self.generation = 0
        self.best_fitness = 0
        self.best_network = None
        self.population = []
        
        # Параметры алгоритма
        self.mutation_rate = 0.2
        self.elitism = 0.2  #лучшие особи сохраняются без изменений
        self.crossover_rate = 0.7
        
        # Инициализация популяции
        for _ in range(population_size):
            self.population.append(NeuralNetwork(input_nodes, hidden_nodes, output_nodes))
    
    def evolve(self):
        """Создание нового поколения"""
        self.generation += 1
        
        # Сортировка по приспособленности
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Сохраняем лучшую сеть
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_network = self.population[0].copy()
        
        # Выбор лучших для следующего поколения
        next_generation = []
        
        # Элитизм(сохраняем лучшие особи)
        elitism_count = int(self.elitism * self.population_size)
        for i in range(elitism_count):
            next_generation.append(self.population[i].copy())
        
        # Заполняем остаток популяции
        while len(next_generation) < self.population_size:
            #выбор родителей с учетом их приспособленности
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            # Скрещивание
            if random.random() < self.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1.copy()
            
            # Мутация
            child.mutate(self.mutation_rate)
            next_generation.append(child)
        
        self.population = next_generation
        
        #сброс fitness для нового поколения
        for network in self.population:
            network.fitness = 0
            network.total_time = 0
            network.consistency = 0

    def select_parent(self):
        """Рулеточный выбор родителя"""
        total_fitness = sum(max(net.fitness, 0) for net in self.population)
        if total_fitness == 0:
            return random.choice(self.population)
        
        spin = random.random() * total_fitness
        running_sum = 0
        
        for network in self.population:
            running_sum += max(network.fitness, 0)
            if running_sum > spin:
                return network
        
        return self.population[-1]
    
    def get_stats(self):
        """Получение статистики генетического алгоритма"""
        fitnesses = [net.fitness for net in self.population]
        best_lap_times = [net.best_lap_time for net in self.population if net.best_lap_time < float('inf')]
        lap_counts = [net.lap_count for net in self.population]
        
        best_lap_seconds = min(best_lap_times) / FPS if best_lap_times else float('inf')
        
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'best_lap_time': best_lap_seconds,
            'population_size': len(self.population)
        }

class TrackEditor:
    def __init__(self, screen):
        self.screen = screen
        self.points = []
        self.track_width = 60
        self.min_width = 20
        self.max_width = 200
        self.is_closed = False
        self.dragging_index = None
        self.current_track = None
        self.start_line = None

        #папка для треков
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tracks_folder = os.path.join(script_dir, "saved_tracks")
        if not os.path.exists(self.tracks_folder):
            os.makedirs(self.tracks_folder)

        # кнопки
        self.buttons = [
            Button(50, 720, 120, 50, "ОЧИСТИТЬ", RED),
            Button(190, 720, 120, 50, "СОХРАНИТЬ", DGREEN),
            Button(330, 720, 120, 50, "ЗАГРУЗИТЬ", BLUE),
            Button(470, 720, 120, 50, "УДАЛИТЬ", ORANGE),
            Button(610, 720, 120, 50, "НАЗАД", SECONDARY),
            Button(750, 720, 50, 50, "+", DGREEN),
            Button(810, 720, 50, 50, "-", RED)
        ]

        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)
        
        self.message = ""
        self.message_timer = 0

    def show_message(self, text, duration=120):
        self.message = text
        self.message_timer = duration

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()

                    # Проверка кнопок
                    for i, button in enumerate(self.buttons):
                        if button.is_clicked(mouse_pos, True):
                            if i == 0: self.clear_track()
                            elif i == 1: self.save_track()
                            elif i == 2: self.load_track()
                            elif i == 3: self.delete_track()
                            elif i == 4: return "menu"
                            elif i == 5: self.increase_width()
                            elif i == 6: self.decrease_width()

                    # Работа с треком
                    if 200 <= mouse_pos[0] <= 1000 and 100 <= mouse_pos[1] <= 700:
                        self.handle_track_click(mouse_pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_index = None

            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                for button in self.buttons:
                    button.check_hover(mouse_pos)

                if self.dragging_index is not None and 200 <= mouse_pos[0] <= 1000 and 100 <= mouse_pos[1] <= 700:
                    self.points[self.dragging_index] = (mouse_pos[0] - 400, 400 - mouse_pos[1])
                    if self.is_closed and self.dragging_index in [0, len(self.points) - 1]:
                        self.points[0] = self.points[-1] = (mouse_pos[0] - 400, 400 - mouse_pos[1])
                    self.rebuild_track()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.increase_width()
                elif event.key == pygame.K_MINUS:
                    self.decrease_width()

        if self.message_timer > 0:
            self.message_timer -= 1
            if self.message_timer == 0:
                self.message = ""

        return "editor"

    def increase_width(self):
        if self.track_width < self.max_width:
            self.track_width += 5
            self.rebuild_track()
            self.show_message(f"Ширина: {self.track_width}", 60)

    def decrease_width(self):
        if self.track_width > self.min_width:
            self.track_width -= 5
            self.rebuild_track()
            self.show_message(f"Ширина: {self.track_width}", 60)

    def handle_track_click(self, mouse_pos):
        x, y = mouse_pos[0] - 400, 400 - mouse_pos[1]
        new_point = (x, y)

        for i, point in enumerate(self.points):
            dist = math.hypot(x - point[0], y - point[1])
            if dist < 15:
                if i == 0 and not self.is_closed and len(self.points) >= 3:
                    self.is_closed = True
                    self.show_message("Трек замкнут! Можно сохранять.", 180)
                    self.rebuild_track()
                    return
                self.dragging_index = i
                return

        if self.is_closed:
            self.show_message("Трек замкнут. Очистите для нового.", 120)
            return

        if self.points:
            min_dist = min(math.hypot(new_point[0] - p[0], new_point[1] - p[1]) for p in self.points)
            if min_dist < 20:
                self.show_message("Слишком близко!", 90)
                return

        self.points.append(new_point)
        self.show_message(f"Точка добавлена. Всего: {len(self.points)}", 90)

        #автозамыкание
        if len(self.points) >= 3 and not self.is_closed:
            first = self.points[0]
            if math.hypot(new_point[0] - first[0], new_point[1] - first[1]) < 30:
                self.points.pop()
                self.is_closed = True
                self.show_message("Трек замкнут! Можно сохранять.", 180)

        self.rebuild_track()

    def rebuild_track(self):
        if len(self.points) < 2:
            return

        self.current_track = {
            'points': self.points.copy(),
            'width': self.track_width,
            'closed': self.is_closed
        }
        self.create_start_line()

    def create_start_line(self):
        if len(self.points) < 2 or not self.is_closed:
            self.start_line = None
            return
            
        start_point = self.points[0]
        next_point = self.points[1]
        
        dx, dy = next_point[0] - start_point[0], next_point[1] - start_point[1]
        length = math.hypot(dx, dy)
        if length == 0:
            self.start_line = None
            return
            
        dx, dy = dx/length, dy/length
        perp_dx, perp_dy = -dy, dx
        
        line_length = self.track_width * 0.8
        start_x = start_point[0] - perp_dx * line_length / 2
        start_y = start_point[1] - perp_dy * line_length / 2
        end_x = start_point[0] + perp_dx * line_length / 2
        end_y = start_point[1] + perp_dy * line_length / 2
        
        start_angle = math.degrees(math.atan2(-dy, dx))
        
        self.start_line = {
            'start': (start_x, start_y),
            'end': (end_x, end_y),
            'direction': (dx, dy),
            'angle': start_angle,
            'position': start_point
        }
        
        if self.current_track:
            self.current_track['start_line'] = self.start_line

    def clear_track(self):
        self.points = []
        self.is_closed = False
        self.current_track = None
        self.start_line = None
        self.show_message("Трек очищен", 90)

    def save_track(self):
        if not self.current_track or not self.is_closed:
            self.show_message("Ошибка: трек должен быть замкнут!", 120)
            return

        track_count = len([f for f in os.listdir(self.tracks_folder) if f.endswith('.json')])
        name = f"track_{track_count + 1}"
        filename = os.path.join(self.tracks_folder, f"{name}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_track, f, indent=2, ensure_ascii=False)
            self.show_message(f"Трек сохранен: {name}.json", 180)
        except Exception as e:
            self.show_message(f"Ошибка: {str(e)}", 180)

    def load_track(self):
        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]
        if not tracks:
            self.show_message("Нет сохраненных треков", 120)
            return

        filename = os.path.join(self.tracks_folder, tracks[0])
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.points = [tuple(p) for p in data['points']]
            self.track_width = data['width']
            self.is_closed = data.get('closed', True)
            self.current_track = data
            self.start_line = data.get('start_line')
            self.rebuild_track()
            self.show_message(f"Загружен: {tracks[0]}", 180)
        except Exception as e:
            self.show_message(f"Ошибка: {str(e)}", 180)

    def delete_track(self):
        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]
        if tracks:
            filename = os.path.join(self.tracks_folder, tracks[0])
            try:
                os.remove(filename)
                self.show_message(f"Удален: {tracks[0]}", 180)
            except Exception as e:
                self.show_message(f"Ошибка: {str(e)}", 180)
        else:
            self.show_message("Нет треков для удаления", 120)

    def calculate_smooth_track(self, points):
        if len(points) < 2:
            return [], []

        smooth_inner, smooth_outer = [], []

        for i in range(len(points)):
            if self.is_closed:
                p_prev, p_curr, p_next = points[i - 1], points[i], points[(i + 1) % len(points)]
            else:
                if i == 0:
                    p_prev, p_curr, p_next = points[0], points[0], points[1]
                elif i == len(points) - 1:
                    p_prev, p_curr, p_next = points[-2], points[-1], points[-1]
                else:
                    p_prev, p_curr, p_next = points[i - 1], points[i], points[i + 1]

            dir1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            dir2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            len1, len2 = math.hypot(*dir1), math.hypot(*dir2)

            if len1 > 0 and len2 > 0:
                dir1 = (dir1[0]/len1, dir1[1]/len1)
                dir2 = (dir2[0]/len2, dir2[1]/len2)

                norm1, norm2 = (-dir1[1], dir1[0]), (-dir2[1], dir2[0])
                avg_norm = ((norm1[0] + norm2[0])/2, (norm1[1] + norm2[1])/2)
                avg_len = math.hypot(*avg_norm)

                if avg_len > 0:
                    avg_norm = (avg_norm[0]/avg_len, avg_norm[1]/avg_len)

                    inner_point = (p_curr[0] - avg_norm[0] * self.track_width/2,
                                 p_curr[1] - avg_norm[1] * self.track_width/2)
                    outer_point = (p_curr[0] + avg_norm[0] * self.track_width/2,
                                 p_curr[1] + avg_norm[1] * self.track_width/2)

                    smooth_inner.append(inner_point)
                    smooth_outer.append(outer_point)

        return smooth_inner, smooth_outer

    def draw(self):
        self.screen.fill(BACKGROUND)
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, SCREEN_WIDTH, 80))
        pygame.draw.rect(self.screen, PANEL_BG, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))
        title = self.font.render("РЕДАКТОР ТРЕКОВ", True, TEXT_COLOR)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 20))

        if not self.is_closed:
            instruction = "Добавляйте точки щелчками. Замкните трек, кликнув на первую точку."
        else:
            instruction = "Трек замкнут. Перетаскивайте точки для редактирования."
        
        instr_text = self.small_font.render(instruction, True, YELLOW if self.is_closed else TEXT_COLOR)
        self.screen.blit(instr_text, (SCREEN_WIDTH // 2 - instr_text.get_width() // 2, 50))

        pygame.draw.rect(self.screen, SECONDARY, (190, 90, 820, 620), 2)

        #сетка
        for x in range(200, 1000, 40):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 100), (x, 700), 1)
        for y in range(100, 700, 40):
            pygame.draw.line(self.screen, GRID_COLOR, (200, y), (1000, y), 1)

        # Оси
        pygame.draw.line(self.screen, GRID_COLOR, (600, 100), (600, 700), 2)
        pygame.draw.line(self.screen, GRID_COLOR, (200, 400), (1000, 400), 2)

        # Рисуем трек
        if self.points:
            screen_points = [(p[0] + 400, 400 - p[1]) for p in self.points]
            if len(screen_points) > 1:
                pygame.draw.lines(self.screen, TRACK_COLOR, self.is_closed, screen_points, 2)

        # Границы трека
        if len(self.points) >= 2:
            inner_points, outer_points = self.calculate_smooth_track(self.points)

            if inner_points and outer_points:
                inner_screen = [(p[0] + 400, 400 - p[1]) for p in inner_points]
                outer_screen = [(p[0] + 400, 400 - p[1]) for p in outer_points]

                if len(inner_screen) >= 2 and len(outer_screen) >= 2:
                    road_points = inner_screen + list(reversed(outer_screen))
                    if len(road_points) >= 3:
                        pygame.draw.polygon(self.screen, ROAD_COLOR, road_points)

                if self.is_closed:
                    if len(inner_screen) >= 3:
                        pygame.draw.lines(self.screen, ACCENT, True, inner_screen, 3)
                    if len(outer_screen) >= 3:
                        pygame.draw.lines(self.screen, ACCENT, True, outer_screen, 3)
                else:
                    if len(inner_screen) >= 2:
                        pygame.draw.lines(self.screen, ACCENT, False, inner_screen, 3)
                    if len(outer_screen) >= 2:
                        pygame.draw.lines(self.screen, ACCENT, False, outer_screen, 3)

        #Стартовая линия
        if self.start_line and self.is_closed:
            start_screen = (self.start_line['start'][0] + 400, 400 - self.start_line['start'][1])
            end_screen = (self.start_line['end'][0] + 400, 400 - self.start_line['end'][1])
            
            pygame.draw.line(self.screen, WHITE, start_screen, end_screen, 4)
            
            #Стрелка направления
            mid_x, mid_y = (start_screen[0] + end_screen[0])/2, (start_screen[1] + end_screen[1])/2
            dir_x, dir_y = self.start_line['direction']
            arrow_length = 25
            arrow_end_x, arrow_end_y = mid_x + dir_x * arrow_length, mid_y - dir_y * arrow_length
            
            pygame.draw.line(self.screen, GREEN, (mid_x, mid_y), (arrow_end_x, arrow_end_y), 4)
            
            #боковые стороны стрелки
            arrow_side_length, perp_dx, perp_dy = 12, -dir_y * 0.7, dir_x * 0.7
            
            left_x = arrow_end_x - dir_x * arrow_side_length + perp_dx * arrow_side_length
            left_y = arrow_end_y + dir_y * arrow_side_length + perp_dy * arrow_side_length
            right_x = arrow_end_x - dir_x * arrow_side_length - perp_dx * arrow_side_length
            right_y = arrow_end_y + dir_y * arrow_side_length - perp_dy * arrow_side_length
            
            pygame.draw.line(self.screen, GREEN, (arrow_end_x, arrow_end_y), (left_x, left_y), 3)
            pygame.draw.line(self.screen, GREEN, (arrow_end_x, arrow_end_y), (right_x, right_y), 3)

        #точки
        if self.points:
            screen_points = [(p[0] + 400, 400 - p[1]) for p in self.points]

            for i, point in enumerate(screen_points):
                if self.is_closed:
                    color = YELLOW
                else:
                    color = GREEN if i == 0 else RED if i == len(screen_points) - 1 else BLUE

                pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), 10)
                pygame.draw.circle(self.screen, WHITE, (int(point[0]), int(point[1])), 10, 2)

                if self.is_closed:
                    number_text = self.small_font.render(str(i + 1), True, WHITE)
                    text_rect = number_text.get_rect(center=(int(point[0]), int(point[1])))
                    self.screen.blit(number_text, text_rect)

        #информация
        status = f"Точек: {len(self.points)}"
        if self.is_closed:
            status += " (ЗАМКНУТ)"
            status_color = YELLOW
        else:
            status += " (строится)"
            status_color = TEXT_COLOR
        status_text = self.small_font.render(status, True, status_color)
        self.screen.blit(status_text, (1020, 120))

        width_text = self.small_font.render(f"Ширина: {self.track_width}", True, TEXT_COLOR)
        self.screen.blit(width_text, (1020, 150))

        #сообщение
        if self.message:
            msg_color = YELLOW if "Ошибка" not in self.message else RED
            msg_text = self.small_font.render(self.message, True, msg_color)
            self.screen.blit(msg_text, (1020, 320))

        # Кол-во треков
        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]
        tracks_info = self.small_font.render(f"Сохранено треков: {len(tracks)}", True, TEXT_COLOR)
        self.screen.blit(tracks_info, (1020, 350))

        #кнопки
        for button in self.buttons:
            button.draw(self.screen)

class TrackSelector:
    def __init__(self, screen):
        self.screen = screen
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tracks_folder = os.path.join(script_dir, "saved_tracks")
        if not os.path.exists(self.tracks_folder):
            os.makedirs(self.tracks_folder)
            
        self.selected_track = None
        self.track_buttons = []
        self.back_button = Button(SCREEN_WIDTH // 2 - 75, 700, 150, 50, "НАЗАД", SECONDARY)

        self.font = pygame.font.SysFont('Arial', 36)
        self.small_font = pygame.font.SysFont('Arial', 20)

        self.load_tracks()

    def load_tracks(self):
        self.track_buttons = []
        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]
        tracks.sort()

        for i, track in enumerate(tracks):
            track_name = track.replace('.json', '')
            y_pos = 200 + i * 70
            if y_pos < 650:
                button = Button(SCREEN_WIDTH // 2 - 200, y_pos, 400, 60, track_name, BLUE)
                button.track_file = track
                self.track_buttons.append(button)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()

                    for button in self.track_buttons:
                        if button.is_clicked(mouse_pos, True):
                            self.selected_track = button.track_file
                            return "game"

                    if self.back_button.is_clicked(mouse_pos, True):
                        return "menu"

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"

        mouse_pos = pygame.mouse.get_pos()
        for button in self.track_buttons:
            button.check_hover(mouse_pos)
        self.back_button.check_hover(mouse_pos)

        return "track_select"

    def draw(self):
        self.screen.fill(BACKGROUND)

        title = self.font.render("ВЫБЕРИТЕ ТРАССУ", True, TEXT_COLOR)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 100))

        track_count = len(self.track_buttons)
        if track_count == 0:
            info_text = self.small_font.render("Нет сохраненных треков. Создайте трек в редакторе.", True, YELLOW)
            self.screen.blit(info_text, (SCREEN_WIDTH // 2 - info_text.get_width() // 2, 160))
        else:
            info_text = self.small_font.render(f"Доступно треков: {track_count}", True, TEXT_COLOR)
            self.screen.blit(info_text, (SCREEN_WIDTH // 2 - info_text.get_width() // 2, 160))

        for button in self.track_buttons:
            button.draw(self.screen)

        self.back_button.draw(self.screen)

class Car:
    def __init__(self, x, y, angle, network=None, car_id=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.car_id = car_id
        
        #физ параметры
        self.velocity_x = 0
        self.velocity_y = 0
        self.angular_velocity = 0
        self.wheel_angle = 0
        self.engine_power = 0
        
        #физика
        self.max_speed = 8
        self.acceleration = 0.6
        self.brake_power = 0.4
        self.steering_speed = 4
        self.max_wheel_angle = 35
        
        #физика заноса
        self.friction = 0.93
        self.traction_fast = 0.08
        self.traction_slow = 0.02
        self.drift_factor = 0.08
        self.slide_factor = 0.95
        self.angular_friction = 0.9
        
        self.handbrake_on = False
        
        #Нейронная сеть
        self.network = network
        self.sensor_angles = [-90, -45, 0, 45, 90]
        self.sensor_distances = [0] * len(self.sensor_angles)
        self.max_sensor_distance = 300
        
        # Для расчета фитнеса и отслеживания кругов
        self.lap_count = 0
        self.current_lap_time = 0
        self.last_lap_time = 0
        self.best_lap_time = float('inf')
        self.total_time = 0
        self.time_alive = 0
        self.is_alive = True
        self.crash_timer = 0
        self.fitness = 0
        
        self.WINNING_LAPS = 10
        
        # Для отслеживания прохождения стартовой линии
        self.last_crossed_start_time = 0
        self.can_cross_start = False  # Можно ли пересекать стартовую линию
        self.start_crossing_cooldown = 30  #задержка перед повторным пересечением
        
        # Чекпоинты
        self.checkpoints_passed = set()
        self.next_checkpoint_index = 1
        self.checkpoint_progress = 0
        
        # Для отслеживания бездействия
        self.inactive_timer = 0
        self.last_position = (x, y)
        self.last_distance_traveled = 0
        
        #графика
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.sprite = self.create_sprite()

    def create_sprite(self):
        sprite = pygame.Surface((40, 20), pygame.SRCALPHA)
        pygame.draw.rect(sprite, self.color, (0, 5, 40, 10), border_radius=3)
        pygame.draw.rect(sprite, (40, 40, 40), (5, 7, 30, 6), border_radius=2)

        #колеса
        pygame.draw.rect(sprite, (30, 30, 30), (5, 3, 6, 3))
        pygame.draw.rect(sprite, (30, 30, 30), (29, 3, 6, 3))
        pygame.draw.rect(sprite, (30, 30, 30), (5, 14, 6, 3))
        pygame.draw.rect(sprite, (30, 30, 30), (29, 14, 6, 3))

        return sprite

    def update_sensors(self, track_data):
        if not track_data or not track_data.get('points'):
            self.sensor_distances = [self.max_sensor_distance] * len(self.sensor_angles)
            return

        for i, angle_offset in enumerate(self.sensor_angles):
            sensor_angle = math.radians(self.angle + angle_offset)
            start_x, start_y = self.x, self.y
            end_x = start_x + math.cos(sensor_angle) * self.max_sensor_distance
            end_y = start_y - math.sin(sensor_angle) * self.max_sensor_distance
            
            min_distance = self.max_sensor_distance
            inner_points, outer_points = self.get_track_boundaries(track_data)
            
            if inner_points and outer_points:
                inner_screen = [(p[0] + 400, 400 - p[1]) for p in inner_points]
                outer_screen = [(p[0] + 400, 400 - p[1]) for p in outer_points]
                
                for boundary in [inner_screen, outer_screen]:
                    for j in range(len(boundary) - 1):
                        p1, p2 = boundary[j], boundary[j + 1]
                        intersection = self.line_intersection(
                            (start_x, start_y), (end_x, end_y), p1, p2
                        )
                        if intersection:
                            distance = math.hypot(intersection[0] - start_x, intersection[1] - start_y)
                            if distance < min_distance:
                                min_distance = distance
            
            self.sensor_distances[i] = min_distance

    def line_intersection(self, a, b, c, d):
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        x4, y4 = d
        
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None

    def get_track_boundaries(self, track_data):
        if not track_data:
            return None, None
            
        points = track_data['points']
        track_width = track_data.get('width', 40)
        is_closed = track_data.get('closed', True)
        
        inner_points, outer_points = [], []

        for i in range(len(points)):
            if is_closed:
                p_prev, p_curr, p_next = points[i - 1], points[i], points[(i + 1) % len(points)]
            else:
                if i == 0:
                    p_prev, p_curr, p_next = points[0], points[0], points[1]
                elif i == len(points) - 1:
                    p_prev, p_curr, p_next = points[-2], points[-1], points[-1]
                else:
                    p_prev, p_curr, p_next = points[i - 1], points[i], points[i + 1]

            dir1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            dir2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            len1, len2 = math.hypot(*dir1), math.hypot(*dir2)

            if len1 > 0 and len2 > 0:
                dir1 = (dir1[0]/len1, dir1[1]/len1)
                dir2 = (dir2[0]/len2, dir2[1]/len2)

                norm1, norm2 = (-dir1[1], dir1[0]), (-dir2[1], dir2[0])
                avg_norm = ((norm1[0] + norm2[0])/2, (norm1[1] + norm2[1])/2)
                avg_len = math.hypot(*avg_norm)

                if avg_len > 0:
                    avg_norm = (avg_norm[0]/avg_len, avg_norm[1]/avg_len)

                    inner_points.append((p_curr[0] - avg_norm[0] * track_width/2,
                                       p_curr[1] - avg_norm[1] * track_width/2))
                    outer_points.append((p_curr[0] + avg_norm[0] * track_width/2,
                                       p_curr[1] + avg_norm[1] * track_width/2))

        if is_closed and inner_points and outer_points:
            if len(inner_points) >= 3:
                inner_points.append(inner_points[0])
            if len(outer_points) >= 3:
                outer_points.append(outer_points[0])

        return inner_points, outer_points

    def is_outside_track(self, track_data):
        """Проверяет, находится ли машинка за пределами трека"""
        if not track_data or not track_data.get('points'):
            return False
            
        #получаем границы трека
        inner_points, outer_points = self.get_track_boundaries(track_data)
        if not inner_points or not outer_points:
            return False
            
        #преобразуем координаты машинки в систему координат трека
        car_pos = (self.x - 400, 400 - self.y)
        
        #проверяем находится ли точка внутри полигона дороги
        all_points = inner_points + list(reversed(outer_points))
        
        inside = False
        n = len(all_points)
        
        for i in range(n):
            j = (i + 1) % n
            if ((all_points[i][1] > car_pos[1]) != (all_points[j][1] > car_pos[1])):
                intersect_x = (all_points[j][0] - all_points[i][0]) * (car_pos[1] - all_points[i][1]) / (all_points[j][1] - all_points[i][1]) + all_points[i][0]
                if car_pos[0] < intersect_x:
                    inside = not inside
        
        return not inside

    def check_start_line_crossing(self, track_data):
        """Проверяет пересечение стартовой линии"""
        if not track_data or 'start_line' not in track_data:
            return False
            
        start_line = track_data['start_line']
        start = (start_line['start'][0] + 400, 400 - start_line['start'][1])
        end = (start_line['end'][0] + 400, 400 - start_line['end'][1])
        
        car_pos = (self.x, self.y)
        last_pos = self.last_position
        
        #вектор движения машинки
        dx = car_pos[0] - last_pos[0]
        dy = car_pos[1] - last_pos[1]
        
        if dx == 0 and dy == 0:
            return False
            
        intersection = self.line_intersection(last_pos, car_pos, start, end)
        
        if intersection:
            line_dir = start_line['direction']
            car_dir = (dx, dy)
            
            # Угол между направлением трека и движением машинки
            dot_product = line_dir[0] * car_dir[0] + line_dir[1] * car_dir[1]
            
            if dot_product > 0:
                current_time = self.time_alive
                if current_time - self.last_crossed_start_time > self.start_crossing_cooldown:
                    self.last_crossed_start_time = current_time
                    return True
                    
        return False

    def update_checkpoints(self, track_data):
        """Обновляет состояние чекпоинтов"""
        if not track_data or not track_data.get('points'):
            return
            
        points = track_data['points']
        if not points:
            return
            
        # Находим ближайшую точку трека
        car_pos = (self.x - 400, 400 - self.y)
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, point in enumerate(points):
            dist = math.hypot(car_pos[0] - point[0], car_pos[1] - point[1])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                
        # Если это следующий чекпоинт по порядку, отмечаем его
        if nearest_idx == self.next_checkpoint_index:
            self.checkpoints_passed.add(nearest_idx)
            self.next_checkpoint_index = (nearest_idx + 1) % len(points)
            self.checkpoint_progress = len(self.checkpoints_passed) / len(points)
            
            # Если прошли все чекпоинты, разрешаем пересечение стартовой линии
            if len(self.checkpoints_passed) == len(points):
                self.can_cross_start = True

    def check_inactivity(self):
        """Проверяет, не двигалась ли машинка слишком долго"""
        speed = math.hypot(self.velocity_x, self.velocity_y)
        
        if speed < 0.1:
            self.inactive_timer += 1
        else:
            self.inactive_timer = 0
            
        if self.inactive_timer > 120:
            return True
        return False

    def get_inputs(self):
        speed = math.hypot(self.velocity_x, self.velocity_y)
        normalized_sensors = [d / self.max_sensor_distance for d in self.sensor_distances]
        normalized_speed = speed / self.max_speed
        normalized_angle = self.angle % 360 / 360
        
        return normalized_sensors + [normalized_speed, normalized_angle]

    def think(self):
        if not self.network:
            return
            
        inputs = self.get_inputs()
        outputs = self.network.predict(inputs)
        
        self.engine_power = 0
        self.wheel_angle = 0
        self.handbrake_on = False
        
        if outputs[0] > 0.7:
            self.engine_power = self.acceleration
        elif outputs[0] < 0.3:
            self.engine_power = -self.acceleration
        
        if outputs[1] > 0.7:
            self.wheel_angle = self.max_wheel_angle
        elif outputs[2] > 0.7:
            self.wheel_angle = -self.max_wheel_angle
            
        if outputs[3] > 0.7:
            self.handbrake_on = True

    def update(self, track_data):
        if not self.is_alive:
            return
            
        self.time_alive += 1
        self.current_lap_time += 1
        self.total_time += 1
        
        #Проверка
        if self.lap_count >= self.WINNING_LAPS:
            self.is_alive = False
            return
            
        self.update_sensors(track_data)
        
        if min(self.sensor_distances) < 15:
            self.crash_timer += 1
            if self.crash_timer > 10:
                self.is_alive = False
                return
        else:
            self.crash_timer = 0
            
        if self.is_outside_track(track_data):
            self.is_alive = False
            return
            
        # Проверка бездействия
        if self.check_inactivity():
            self.is_alive = False
            return
        
        self.think()
        self.update_physics()
        
        if self.check_start_line_crossing(track_data) and self.can_cross_start:
            self.lap_count += 1
            self.last_lap_time = self.current_lap_time
            
            if self.current_lap_time < self.best_lap_time:
                self.best_lap_time = self.current_lap_time
                
            # Сбрасываем для следующего круга
            self.current_lap_time = 0
            self.checkpoints_passed.clear()
            self.next_checkpoint_index = 1
            self.can_cross_start = False
            self.checkpoint_progress = 0
            
        self.update_checkpoints(track_data)
        
        self.last_position = (self.x, self.y)
        
        self.calculate_fitness()

    def update_physics(self):
        angle_rad = math.radians(self.angle)
        
        # Ускорение
        if self.engine_power != 0:
            self.velocity_x += self.engine_power * math.cos(angle_rad)
            self.velocity_y -= self.engine_power * math.sin(angle_rad)
        else:
            self.velocity_x *= self.friction
            self.velocity_y *= self.friction

        if self.engine_power < 0 and not self.handbrake_on:
            speed = math.hypot(self.velocity_x, self.velocity_y)
            if speed > 0.1:
                brake_x = -self.velocity_x / speed * self.brake_power
                brake_y = -self.velocity_y / speed * self.brake_power
                self.velocity_x += brake_x
                self.velocity_y += brake_y

        # Физика заноса
        speed = math.hypot(self.velocity_x, self.velocity_y)
        forward_vector = (math.cos(angle_rad), -math.sin(angle_rad))
        dot_product = (self.velocity_x * forward_vector[0] + 
                      self.velocity_y * forward_vector[1])
        is_moving_backward = dot_product < 0
        
        if self.handbrake_on and speed > 1:
            current_traction = self.traction_fast
            self.angular_velocity += self.wheel_angle * self.drift_factor * 0.07
            self.velocity_x *= self.slide_factor
            self.velocity_y *= self.slide_factor
        else:
            current_traction = self.traction_slow if speed < 2 else self.traction_fast

        # Поворот
        if abs(self.wheel_angle) > 1 and speed > 0.5:
            turn_force = math.tan(math.radians(self.wheel_angle)) * current_traction
            if is_moving_backward:
                turn_force = -turn_force
            self.angular_velocity += turn_force * speed

        self.angle += self.angular_velocity
        self.angle %= 360
        self.angular_velocity *= self.angular_friction

        self.x += self.velocity_x
        self.y += self.velocity_y

        # Огр. скорости
        speed = math.hypot(self.velocity_x, self.velocity_y)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity_x *= scale
            self.velocity_y *= scale

    def calculate_fitness(self):
        """Расчет фитнес-функции - основное внимание на время кругов"""
        fitness = 0
        
        # Основной компонент - количество кругов (большой бонус)
        fitness += self.lap_count * 1000
        
        # Бонус за победу (10 кругов)
        if self.lap_count >= self.WINNING_LAPS:
            fitness += 10000  #бонус за победу
        
        # Бонус за лучшее время круга (чем меньше время, тем больше бонус)
        if self.best_lap_time < float('inf'):
            # меньшее время = больший бонус
            time_bonus = max(0, 5000 / (self.best_lap_time + 1))  
            fitness += time_bonus
            
        # Бонус за стабильность (если есть несколько кругов с похожим временем)
        if self.lap_count > 1 and self.last_lap_time > 0 and self.best_lap_time < float('inf'):
            time_diff = abs(self.last_lap_time - self.best_lap_time)
            if time_diff < 60:  
                consistency_bonus = 100 * (60 - time_diff) / 60
                fitness += consistency_bonus
        
        # Бонус за прогресс по чекпоинтам
        fitness += self.checkpoint_progress * 100
        
        # Штраф за столкновения
        if min(self.sensor_distances) < 20:
            fitness *= 0.9
        
        # Штраф за слишком медленное движение
        speed = math.hypot(self.velocity_x, self.velocity_y)
        if speed < 0.5:
            fitness *= 0.95
            
        self.fitness = fitness
        return fitness

    def draw(self, surface, show_sensors=False):
        if not self.is_alive:
            return
            
        rotated_car = pygame.transform.rotate(self.sprite, self.angle)
        car_rect = rotated_car.get_rect(center=(self.x, self.y))
        surface.blit(rotated_car, car_rect)
        
        #сенсоры
        if show_sensors:
            for i, angle_offset in enumerate(self.sensor_angles):
                sensor_angle = math.radians(self.angle + angle_offset)
                start_x, start_y = self.x, self.y
                distance = self.sensor_distances[i]
                end_x = start_x + math.cos(sensor_angle) * distance
                end_y = start_y - math.sin(sensor_angle) * distance
                
                color_ratio = distance / self.max_sensor_distance
                color = (
                    int(255 * (1 - color_ratio)),
                    int(255 * color_ratio),
                    0
                )
                
                pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 1)

class CarGame:
    def __init__(self, screen):
        self.screen = screen
        self.selected_track = None
        self.track_data = None
        self.start_position = None
        self.start_angle = 0
        
        self.population_size = 50
        self.ga = GeneticAlgorithm(
            population_size=self.population_size,
            input_nodes=7,  # 5 сенсоров + скорость + угол
            hidden_nodes=10,
            output_nodes=4   # ускорение, влево, вправо, ручной тормоз
        )
        
        self.cars = []
        self.best_car_index = 0
        self.generation_time = 0
        
        self.training_mode = True
        self.show_sensors = True
        self.show_all_cars = True
        self.fast_mode = False
        self.fast_mode_multiplier = 5
        
        self.current_car_index = 0
        self.manual_control = False
        
        self.buttons = [
            Button(50, 720, 120, 50, "МЕНЮ", SECONDARY),
            Button(200, 720, 120, 50, "СЕНСОРЫ", BLUE),
            Button(350, 720, 120, 50, "СБРОС", RED),
            Button(500, 720, 120, 50, "УСКОРИТЬ", PURPLE),
            Button(650, 720, 220, 50, "СЛЕД. ПОПУЛЯЦИЯ", TRACK_COLOR)
        ]

        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_folder = os.path.join(script_dir, "saved_models")
        if not os.path.exists(self.models_folder):
            os.makedirs(self.models_folder)

    def set_track(self, track_file):
        self.selected_track = track_file
        self.load_track()
        self.initialize_cars()

    def load_track(self):
        if not self.selected_track:
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        tracks_folder = os.path.join(script_dir, "saved_tracks")
        filename = os.path.join(tracks_folder, self.selected_track)

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.track_data = json.load(f)
            
            if self.track_data and 'start_line' in self.track_data:
                start_line = self.track_data['start_line']
                mid_x = (start_line['start'][0] + start_line['end'][0]) / 2
                mid_y = (start_line['start'][1] + start_line['end'][1]) / 2
                self.start_position = (mid_x + 400, 400 - mid_y)
                self.start_angle = start_line['angle']
        except Exception as e:
            print(f"Ошибка загрузки трека: {e}")
            self.track_data = None

    def initialize_cars(self):
        self.cars = []
        if not self.start_position:
            return
            
        for i, network in enumerate(self.ga.population):
            car = Car(
                x=self.start_position[0] + random.uniform(-10, 10),
                y=self.start_position[1] + random.uniform(-10, 10),
                angle=self.start_angle + random.uniform(-10, 10),
                network=network,
                car_id=i
            )
            self.cars.append(car)

    def update(self):
        iterations = self.fast_mode_multiplier if self.fast_mode else 1
        
        for _ in range(iterations):
            self.generation_time += 1
            
            # Обновление всех автомобилей
            alive_count = 0
            for car in self.cars:
                if car.is_alive:
                    car.update(self.track_data)
                    if car.is_alive:
                        alive_count += 1
            
            for i, car in enumerate(self.cars):
                if car.is_alive:
                    self.ga.population[i].fitness = car.fitness
                    self.ga.population[i].best_lap_time = car.best_lap_time
                    self.ga.population[i].lap_count = car.lap_count
                    self.ga.population[i].total_time = car.total_time
            
            if self.cars:
                alive_cars = [i for i, car in enumerate(self.cars) if car.is_alive]
                if alive_cars:
                    self.best_car_index = max(alive_cars, 
                                            key=lambda i: self.cars[i].fitness)
            
            # Проверка завершения поколения
            if alive_count == 0 or self.generation_time > 3000:  
                self.next_generation()

    def next_generation(self):
        for i, car in enumerate(self.cars):
            self.ga.population[i].fitness = car.fitness
        
        self.ga.evolve()
        self.initialize_cars()
        self.generation_time = 0

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"
                elif event.key == pygame.K_s:
                    self.show_sensors = not self.show_sensors
                    self.buttons[1].color = YELLOW if self.show_sensors else BLUE
                elif event.key == pygame.K_f:
                    self.fast_mode = not self.fast_mode
                    self.buttons[3].color = YELLOW if self.fast_mode else PURPLE
                elif event.key == pygame.K_r:
                    self.ga = GeneticAlgorithm(
                        population_size=self.population_size,
                        input_nodes=7,
                        hidden_nodes=10,
                        output_nodes=4
                    )
                    self.initialize_cars()
                elif event.key == pygame.K_n:
                    self.force_next_generation()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    for i, button in enumerate(self.buttons):
                        if button.is_clicked(mouse_pos, True):
                            if i == 0: return "menu"
                            elif i == 1: 
                                self.show_sensors = not self.show_sensors
                                button.color = YELLOW if self.show_sensors else BLUE
                            elif i == 2: 
                                self.ga = GeneticAlgorithm(
                                    population_size=self.population_size,
                                    input_nodes=7,
                                    hidden_nodes=10,
                                    output_nodes=4
                                )
                                self.initialize_cars()
                            elif i == 3: 
                                self.fast_mode = not self.fast_mode
                                button.color = YELLOW if self.fast_mode else PURPLE
                            elif i == 4: 
                                self.force_next_generation()

        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.check_hover(mouse_pos)

        return "game"

    def force_next_generation(self):
        """Принудительный переход к следующей популяции"""
        self.next_generation()

    def draw(self):
        self.screen.fill(BACKGROUND)

        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, SCREEN_WIDTH, 80))
        pygame.draw.rect(self.screen, PANEL_BG, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))

        if self.selected_track:
            track_name = self.selected_track.replace('.json', '')
            title = self.font.render(f"ГЕНЕТИЧЕСКИЙ АЛГОРИТМ - {track_name}", True, TEXT_COLOR)
        else:
            title = self.font.render("ГЕНЕТИЧЕСКИЙ АЛГОРИТМ", True, TEXT_COLOR)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 20))


        if self.track_data and self.track_data.get('points'):
            self.draw_custom_track()
        else:
            self.draw_default_track()

        if self.cars:
            if self.show_all_cars:
                for i, car in enumerate(self.cars):
                    if car.is_alive:
                        car.draw(self.screen, self.show_sensors and i == self.best_car_index)
            else:
                if self.best_car_index < len(self.cars) and self.cars[self.best_car_index].is_alive:
                    self.cars[self.best_car_index].draw(self.screen, self.show_sensors)

        stats = self.ga.get_stats()
        alive_cars = [car for car in self.cars if car.is_alive]
        
        generation_time_seconds = self.generation_time / FPS
        
        if alive_cars:
            best_car = self.cars[self.best_car_index]
            best_car_lap_time = best_car.best_lap_time / FPS if best_car.best_lap_time < float('inf') else float('inf')
            best_car_current_lap_time = best_car.current_lap_time / FPS
            
            car_info = [
                f"Поколение: {stats['generation']}",
                f"Лучший фитнес: {stats['best_fitness']:.1f}",
                f"Средний фитнес: {stats['avg_fitness']:.1f}",
                f"Лучшее время круга: {stats['best_lap_time']:.2f} сек",
                f"Время поколения: {generation_time_seconds:.1f} сек",
                f"Популяция: {len(self.cars)}",
                f"Живые: {len(alive_cars)}",
                f"Лучший авто - круги: {best_car.lap_count}/10",
                f"Лучший авто - лучшее время: {best_car_lap_time:.2f} сек",
                f"Лучший авто - текущий круг: {best_car_current_lap_time:.1f} сек"
            ]
        else:
            car_info = [
                f"Поколение: {stats['generation']}",
                f"Лучший фитнес: {stats['best_fitness']:.1f}",
                f"Средний фитнес: {stats['avg_fitness']:.1f}",
                f"Лучшее время круга: {stats['best_lap_time']:.2f} сек",
                f"Время поколения: {generation_time_seconds:.1f} сек",
                f"Популяция: {len(self.cars)}",
                f"Живые: {len(alive_cars)}",
                "Все машинки исчезли. Следующее поколение скоро начнется..."
            ]

        stat_width = 580
        stat_height = len(car_info) * 25 + 40
        stat_x = 1000
        stat_y = 100
        
        pygame.draw.rect(self.screen, PANEL_BG, (stat_x, stat_y, stat_width, stat_height), border_radius=8)
        
        pygame.draw.rect(self.screen, ACCENT, (stat_x, stat_y, stat_width, stat_height), 3, border_radius=8)
        
        pygame.draw.rect(self.screen, SECONDARY, (stat_x + 3, stat_y + 3, stat_width - 6, stat_height - 6), 1, border_radius=6)
        
        stat_title = self.small_font.render("СТАТИСТИКА", True, YELLOW)
        self.screen.blit(stat_title, (stat_x + stat_width // 2 - stat_title.get_width() // 2, stat_y + 10))

        for i, text in enumerate(car_info):
            text_surf = self.small_font.render(text, True, TEXT_COLOR)
            self.screen.blit(text_surf, (stat_x + 20, stat_y + 40 + i * 25))

        if self.fast_mode:
            fast_text = self.small_font.render("УСКОРЕННЫЙ РЕЖИМ ВКЛ", True, YELLOW)
            self.screen.blit(fast_text, (1020, stat_y + stat_height + 20))
            multiplier_text = self.small_font.render(f"Скорость: x{self.fast_mode_multiplier}", True, YELLOW)
            self.screen.blit(multiplier_text, (1020, stat_y + stat_height + 40))

        hints = [
            "S - показать/скрыть сенсоры",
            "F - ускоренный режим",
            "R - сбросить обучение",
            "N - следующая популяция"
        ]

        for i, hint in enumerate(hints):
            hint_surf = self.small_font.render(hint, True, YELLOW)
            self.screen.blit(hint_surf, (1020, stat_y + stat_height + 70 + i * 25))

        # Кнопки
        for button in self.buttons:
            button.draw(self.screen)

    def draw_custom_track(self):
        if not self.track_data:
            return

        points = self.track_data['points']
        track_width = self.track_data.get('width', 40)
        is_closed = self.track_data.get('closed', True)

        screen_points = [(p[0] + 400, 400 - p[1]) for p in points]

        inner_points, outer_points = [], []

        for i in range(len(points)):
            if is_closed:
                p_prev, p_curr, p_next = points[i - 1], points[i], points[(i + 1) % len(points)]
            else:
                if i == 0:
                    p_prev, p_curr, p_next = points[0], points[0], points[1]
                elif i == len(points) - 1:
                    p_prev, p_curr, p_next = points[-2], points[-1], points[-1]
                else:
                    p_prev, p_curr, p_next = points[i - 1], points[i], points[i + 1]

            dir1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            dir2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            len1, len2 = math.hypot(*dir1), math.hypot(*dir2)

            if len1 > 0 and len2 > 0:
                dir1 = (dir1[0]/len1, dir1[1]/len1)
                dir2 = (dir2[0]/len2, dir2[1]/len2)

                norm1, norm2 = (-dir1[1], dir1[0]), (-dir2[1], dir2[0])
                avg_norm = ((norm1[0] + norm2[0])/2, (norm1[1] + norm2[1])/2)
                avg_len = math.hypot(*avg_norm)

                if avg_len > 0:
                    avg_norm = (avg_norm[0]/avg_len, avg_norm[1]/avg_len)

                    inner_points.append((p_curr[0] - avg_norm[0] * track_width/2,
                                       p_curr[1] - avg_norm[1] * track_width/2))
                    outer_points.append((p_curr[0] + avg_norm[0] * track_width/2,
                                       p_curr[1] + avg_norm[1] * track_width/2))

        if inner_points and outer_points:
            inner_screen = [(p[0] + 400, 400 - p[1]) for p in inner_points]
            outer_screen = [(p[0] + 400, 400 - p[1]) for p in outer_points]

            road_points = inner_screen + list(reversed(outer_screen))
            if len(road_points) >= 3:
                pygame.draw.polygon(self.screen, ROAD_COLOR, road_points)

            if is_closed:
                pygame.draw.lines(self.screen, ACCENT, True, inner_screen, 3)
                pygame.draw.lines(self.screen, ACCENT, True, outer_screen, 3)
            else:
                pygame.draw.lines(self.screen, ACCENT, False, inner_screen, 3)
                pygame.draw.lines(self.screen, ACCENT, False, outer_screen, 3)

        if 'start_line' in self.track_data:
            start_line = self.track_data['start_line']
            start_screen = (start_line['start'][0] + 400, 400 - start_line['start'][1])
            end_screen = (start_line['end'][0] + 400, 400 - start_line['end'][1])
            
            pygame.draw.line(self.screen, WHITE, start_screen, end_screen, 4)
            
            mid_x, mid_y = (start_screen[0] + end_screen[0])/2, (start_screen[1] + end_screen[1])/2
            dir_x, dir_y = start_line['direction']
            arrow_length = 25
            arrow_end_x, arrow_end_y = mid_x + dir_x * arrow_length, mid_y - dir_y * arrow_length
            
            pygame.draw.line(self.screen, GREEN, (mid_x, mid_y), (arrow_end_x, arrow_end_y), 4)
            
            arrow_side_length, perp_dx, perp_dy = 12, -dir_y * 0.7, dir_x * 0.7
            
            left_x = arrow_end_x - dir_x * arrow_side_length + perp_dx * arrow_side_length
            left_y = arrow_end_y + dir_y * arrow_side_length + perp_dy * arrow_side_length
            right_x = arrow_end_x - dir_x * arrow_side_length - perp_dx * arrow_side_length
            right_y = arrow_end_y + dir_y * arrow_side_length - perp_dy * arrow_side_length
            
            pygame.draw.line(self.screen, GREEN, (arrow_end_x, arrow_end_y), (left_x, left_y), 3)
            pygame.draw.line(self.screen, GREEN, (arrow_end_x, arrow_end_y), (right_x, right_y), 3)

    def draw_default_track(self):
        pygame.draw.rect(self.screen, ROAD_COLOR, (100, 100, SCREEN_WIDTH - 200, SCREEN_HEIGHT - 200), border_radius=20)
        pygame.draw.rect(self.screen, ACCENT, (100, 100, SCREEN_WIDTH - 200, SCREEN_HEIGHT - 200), 5, border_radius=20)

        for i in range(0, SCREEN_WIDTH - 200, 40):
            pygame.draw.rect(self.screen, WHITE, (120 + i, SCREEN_HEIGHT // 2 - 2, 20, 4))

class MainMenu:
    def __init__(self, screen):
        self.screen = screen
        self.buttons = [
            Button(SCREEN_WIDTH // 2 - 150, 300, 300, 60, "ИГРАТЬ", ACCENT),
            Button(SCREEN_WIDTH // 2 - 150, 380, 300, 60, "РЕДАКТОР ТРЕКОВ", BLUE),
            Button(SCREEN_WIDTH // 2 - 150, 460, 300, 60, "ВЫХОД", RED)
        ]
        self.font = pygame.font.SysFont('Arial', 48)
        self.small_font = pygame.font.SysFont('Arial', 24)
        self.title_font = pygame.font.SysFont('Arial', 64, bold=True)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    for i, button in enumerate(self.buttons):
                        if button.is_clicked(mouse_pos, True):
                            if i == 0: return "track_select"
                            elif i == 1: return "editor"
                            elif i == 2: return "quit"

        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.check_hover(mouse_pos)

        return "menu"

    def draw(self):
        DARK_GRAY = (70, 70, 70)
        self.screen.fill(DARK_GRAY)

        #градиент
        for y in range(0, SCREEN_HEIGHT, 5):
            alpha = int(100 * (y / SCREEN_HEIGHT))
            color = (
                DARK_GRAY[0] + alpha // 10,
                DARK_GRAY[1] + alpha // 10,
                DARK_GRAY[2] + alpha // 10
            )
            pygame.draw.line(self.screen, color, (0, y), (SCREEN_WIDTH, y), 5)

        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 80))
        self.screen.blit(overlay, (0, 0))

        title = self.title_font.render("AI RACING SIMULATOR", True, ACCENT)
        title_shadow = self.title_font.render("AI RACING SIMULATOR", True, (30, 30, 30))
        
        self.screen.blit(title_shadow, (SCREEN_WIDTH // 2 - title.get_width() // 2 + 3, 103))
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 100))

        subtitle = self.small_font.render("Q - обучение совмещенное с генетическим алгоритмом для обучения машинок", True, CYAN)
        self.screen.blit(subtitle, (SCREEN_WIDTH // 2 - subtitle.get_width() // 2, 180))

        for button in self.buttons:
            button.draw(self.screen)

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Гоночная Игра с Генетическим Алгоритмом")
    clock = pygame.time.Clock()

    current_screen = "menu"
    menu = MainMenu(screen)
    editor = TrackEditor(screen)
    game = CarGame(screen)
    track_selector = TrackSelector(screen)

    running = True
    while running:
        if current_screen == "menu":
            result = menu.handle_events()
            current_screen = result if result != "menu" else current_screen
            menu.draw()

        elif current_screen == "track_select":
            track_selector.load_tracks()
            result = track_selector.handle_events()
            if result == "game" and track_selector.selected_track:
                game.set_track(track_selector.selected_track)
                current_screen = "game"
            else:
                current_screen = result if result != "track_select" else current_screen
            track_selector.draw()

        elif current_screen == "editor":
            result = editor.handle_events()
            current_screen = result if result != "editor" else current_screen
            editor.draw()

        elif current_screen == "game":
            result = game.handle_events()
            current_screen = result if result != "game" else current_screen
            game.update()
            game.draw()

        elif current_screen == "quit":
            running = False

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
