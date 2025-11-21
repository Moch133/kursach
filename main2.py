try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("Pygame не установлен. Установите его командой: pip install pygame")
    exit(1)

import sys
import math
import json
import os

# Инициализация Pygame
pygame.init()

# Константы
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Цвета
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


class Button:
    def __init__(self, x, y, width, height, text, color=ACCENT, hover_color=(100, 110, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False
        self.font = pygame.font.SysFont('Arial', 20)

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, SECONDARY, self.rect, 2, border_radius=8)

        text_surf = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def is_clicked(self, pos, click):
        return self.rect.collidepoint(pos) and click


class TrackEditor:
    def __init__(self, screen):
        self.screen = screen
        self.points = []
        self.track_width = 40
        self.is_closed = False
        self.dragging_index = None
        self.drag_start = None
        self.current_track = None

        # Создаем папку для сохранения треков
        self.tracks_folder = "saved_tracks"
        if not os.path.exists(self.tracks_folder):
            os.makedirs(self.tracks_folder)

        # Кнопки
        self.buttons = [
            Button(50, 720, 120, 50, "ОЧИСТИТЬ", RED),
            Button(190, 720, 120, 50, "СОХРАНИТЬ", GREEN),
            Button(330, 720, 120, 50, "ЗАГРУЗИТЬ", BLUE),
            Button(470, 720, 120, 50, "УДАЛИТЬ", (255, 150, 50)),
            Button(610, 720, 120, 50, "НАЗАД", SECONDARY)
        ]

        # Шрифты
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Левая кнопка мыши
                    mouse_pos = pygame.mouse.get_pos()

                    # Проверка кликов по кнопкам
                    for i, button in enumerate(self.buttons):
                        if button.is_clicked(mouse_pos, True):
                            if i == 0:  # Очистить
                                self.clear_track()
                            elif i == 1:  # Сохранить
                                self.save_track()
                            elif i == 2:  # Загрузить
                                self.load_track()
                            elif i == 3:  # Удалить
                                self.delete_track()
                            elif i == 4:  # Назад
                                return "menu"

                    # Добавление/редактирование точек трека
                    if 200 <= mouse_pos[0] <= 1000 and 100 <= mouse_pos[1] <= 700:
                        self.handle_track_click(mouse_pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_index = None
                    self.drag_start = None

            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()

                # Обновление состояния кнопок
                for button in self.buttons:
                    button.check_hover(mouse_pos)

                # Перетаскивание точек
                if self.dragging_index is not None and 200 <= mouse_pos[0] <= 1000 and 100 <= mouse_pos[1] <= 700:
                    self.points[self.dragging_index] = (mouse_pos[0] - 400, 400 - mouse_pos[1])
                    # Если трек замкнут и перемещаем первую/последнюю точку
                    if self.is_closed and self.dragging_index in [0, len(self.points) - 1]:
                        self.points[0] = self.points[-1] = (mouse_pos[0] - 400, 400 - mouse_pos[1])
                    self.rebuild_track()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"

        return "editor"

    def handle_track_click(self, mouse_pos):
        """Обработка кликов для добавления и редактирования точек трека"""
        # Преобразование координат (центр в 0,0)
        x = mouse_pos[0] - 400
        y = 400 - mouse_pos[1]
        new_point = (x, y)

        # Проверка клика на существующие точки
        for i in range(len(self.points)):
            dist = math.hypot(x - self.points[i][0], y - self.points[i][1])
            if dist < 15:
                # Если кликнули на первую точку и трек еще не замкнут - замыкаем трек
                if i == 0 and not self.is_closed and len(self.points) >= 3:
                    self.is_closed = True
                    print("Трек замкнут! Нажата первая точка.")
                    self.rebuild_track()
                    return
                self.dragging_index = i
                self.drag_start = (x, y)
                return

        # Если трек уже замкнут, не добавляем новые точки
        if self.is_closed:
            return

        # Проверка минимального расстояния для новой точки
        if self.points:
            min_dist = min(math.hypot(new_point[0] - p[0], new_point[1] - p[1]) for p in self.points)
            if min_dist < 20:
                return

        # Добавляем новую точку
        self.points.append(new_point)

        # Автоматическое замыкание при приближении к первой точке
        if len(self.points) >= 3 and not self.is_closed:
            first = self.points[0]
            if math.hypot(new_point[0] - first[0], new_point[1] - first[1]) < 30:
                self.points.pop()  # Удаляем последнюю точку
                self.is_closed = True
                print("Трек замкнут! Первая и последняя точки объединены.")

        self.rebuild_track()

    def rebuild_track(self):
        if len(self.points) < 2:
            return

        self.current_track = {
            'points': self.points.copy(),
            'width': self.track_width,
            'closed': self.is_closed
        }

    def clear_track(self):
        self.points = []
        self.is_closed = False
        self.current_track = None

    def save_track(self):
        if not self.current_track or not self.is_closed:
            return

        # Создаем имя для трека
        name = "track_" + str(len([f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]) + 1)

        filename = os.path.join(self.tracks_folder, f"{name}.json")
        with open(filename, 'w') as f:
            json.dump(self.current_track, f, indent=2)

        print(f"Трек сохранен как {filename}")

    def load_track(self):
        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]
        if not tracks:
            print("Нет сохраненных треков")
            return

        # Загружаем первый найденный трек
        filename = os.path.join(self.tracks_folder, tracks[0])
        with open(filename, 'r') as f:
            data = json.load(f)

        self.points = [tuple(p) for p in data['points']]
        self.track_width = data['width']
        self.is_closed = data.get('closed', True)
        self.current_track = data
        self.rebuild_track()
        print(f"Трек {tracks[0]} загружен")

    def delete_track(self):
        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]
        if tracks:
            filename = os.path.join(self.tracks_folder, tracks[0])
            os.remove(filename)
            print(f"Трек {tracks[0]} удален")

    def calculate_smooth_track(self, points):
        """Создает сглаженную трассу с закругленными углами"""
        if len(points) < 2:
            return [], []

        smooth_inner = []
        smooth_outer = []

        for i in range(len(points)):
            if self.is_closed:
                # Для замкнутого трека - все точки соединены по кругу
                p_prev = points[i - 1]
                p_curr = points[i]
                p_next = points[(i + 1) % len(points)]
            else:
                # Для незамкнутого трека
                if i == 0:
                    p_prev = points[0]
                    p_curr = points[0]
                    p_next = points[1]
                elif i == len(points) - 1:
                    p_prev = points[-2]
                    p_curr = points[-1]
                    p_next = points[-1]
                else:
                    p_prev = points[i - 1]
                    p_curr = points[i]
                    p_next = points[i + 1]

            # Векторы направлений
            dir1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            dir2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            # Нормализуем векторы
            len1 = math.hypot(dir1[0], dir1[1])
            len2 = math.hypot(dir2[0], dir2[1])

            if len1 > 0 and len2 > 0:
                dir1 = (dir1[0] / len1, dir1[1] / len1)
                dir2 = (dir2[0] / len2, dir2[1] / len2)

                # Нормали (перпендикуляры)
                norm1 = (-dir1[1], dir1[0])
                norm2 = (-dir2[1], dir2[0])

                # Усредненная нормаль для плавного перехода
                avg_norm = ((norm1[0] + norm2[0]) / 2, (norm1[1] + norm2[1]) / 2)
                avg_len = math.hypot(avg_norm[0], avg_norm[1])

                if avg_len > 0:
                    avg_norm = (avg_norm[0] / avg_len, avg_norm[1] / avg_len)

                    # Внутренняя и внешняя точки
                    inner_point = (p_curr[0] - avg_norm[0] * self.track_width / 2,
                                   p_curr[1] - avg_norm[1] * self.track_width / 2)
                    outer_point = (p_curr[0] + avg_norm[0] * self.track_width / 2,
                                   p_curr[1] + avg_norm[1] * self.track_width / 2)

                    smooth_inner.append(inner_point)
                    smooth_outer.append(outer_point)
            else:
                # Если векторы нулевые, используем простой расчет
                if i > 0:
                    dir_vec = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                else:
                    dir_vec = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

                length = math.hypot(dir_vec[0], dir_vec[1])
                if length > 0:
                    dir_vec = (dir_vec[0] / length, dir_vec[1] / length)
                    norm = (-dir_vec[1], dir_vec[0])

                    inner_point = (p_curr[0] - norm[0] * self.track_width / 2,
                                   p_curr[1] - norm[1] * self.track_width / 2)
                    outer_point = (p_curr[0] + norm[0] * self.track_width / 2,
                                   p_curr[1] + norm[1] * self.track_width / 2)

                    smooth_inner.append(inner_point)
                    smooth_outer.append(outer_point)

        return smooth_inner, smooth_outer

    def create_track_boundaries(self, points):
        """Создает границы трека с закругленными углами"""
        if len(points) < 2:
            return None, None

        inner_points, outer_points = self.calculate_smooth_track(points)

        # Для замкнутого трека замыкаем границы
        if self.is_closed and inner_points and outer_points:
            if len(inner_points) >= 3:
                inner_points.append(inner_points[0])
            if len(outer_points) >= 3:
                outer_points.append(outer_points[0])

        return inner_points, outer_points

    def draw(self):
        self.screen.fill(BACKGROUND)

        # Рисуем панель
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, SCREEN_WIDTH, 80))
        pygame.draw.rect(self.screen, PANEL_BG, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))

        # Заголовок
        title = self.font.render("РЕДАКТОР ТРЕКОВ", True, TEXT_COLOR)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 20))

        # Инструкция
        if not self.is_closed:
            instruction = self.small_font.render(
                "Добавляйте точки щелчками мыши. Замкните трек, кликнув на первую точку.", True, TEXT_COLOR)
        else:
            instruction = self.small_font.render(
                "Трек замкнут. Редактируйте существующие точки перетаскиванием. Очистите для создания нового.", True,
                YELLOW)
        self.screen.blit(instruction, (SCREEN_WIDTH // 2 - instruction.get_width() // 2, 50))

        # Рисуем область трека
        pygame.draw.rect(self.screen, SECONDARY, (190, 90, 820, 620), 2)

        # Рисуем сетку
        for x in range(200, 1000, 40):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 100), (x, 700), 1)
        for y in range(100, 700, 40):
            pygame.draw.line(self.screen, GRID_COLOR, (200, y), (1000, y), 1)

        # Центральные оси
        pygame.draw.line(self.screen, GRID_COLOR, (600, 100), (600, 700), 2)
        pygame.draw.line(self.screen, GRID_COLOR, (200, 400), (1000, 400), 2)

        # Рисуем трек (сначала фон трека)
        if self.points:
            # Рисуем линии между точками
            screen_points = [(p[0] + 400, 400 - p[1]) for p in self.points]
            if len(screen_points) > 1:
                if self.is_closed:
                    pygame.draw.lines(self.screen, TRACK_COLOR, True, screen_points, 2)
                else:
                    pygame.draw.lines(self.screen, TRACK_COLOR, False, screen_points, 2)

        # Рисуем границы трека
        if len(self.points) >= 2:
            inner_points, outer_points = self.create_track_boundaries(self.points)

            if inner_points and outer_points:
                # Преобразуем координаты для отображения
                inner_screen = [(p[0] + 400, 400 - p[1]) for p in inner_points]
                outer_screen = [(p[0] + 400, 400 - p[1]) for p in outer_points]

                # Рисуем дорогу
                if len(inner_screen) >= 2 and len(outer_screen) >= 2:
                    road_points = inner_screen + list(reversed(outer_screen))
                    if len(road_points) >= 3:
                        pygame.draw.polygon(self.screen, ROAD_COLOR, road_points)

                # Рисуем границы
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

        # Рисуем точки поверх трассы
        if self.points:
            screen_points = [(p[0] + 400, 400 - p[1]) for p in self.points]

            for i, point in enumerate(screen_points):
                if self.is_closed:
                    color = YELLOW
                else:
                    if i == 0:
                        color = GREEN
                    elif i == len(screen_points) - 1:
                        color = RED
                    else:
                        color = BLUE

                # Рисуем точку
                pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), 10)
                pygame.draw.circle(self.screen, WHITE, (int(point[0]), int(point[1])), 10, 2)

                # Номера точек для замкнутого трека
                if self.is_closed:
                    number_text = self.small_font.render(str(i + 1), True, WHITE)
                    text_rect = number_text.get_rect(center=(int(point[0]), int(point[1])))
                    self.screen.blit(number_text, text_rect)

        # Статус
        status = f"Точек: {len(self.points)}"
        if self.is_closed:
            status += " (ЗАМКНУТ)"
            status_color = YELLOW
        else:
            status += " (строится)"
            status_color = TEXT_COLOR
        status_text = self.small_font.render(status, True, status_color)
        self.screen.blit(status_text, (1020, 120))

        # Подсказка
        if len(self.points) >= 3 and not self.is_closed:
            hint = self.small_font.render("Нажмите на первую точку (зеленую), чтобы замкнуть трек", True, GREEN)
            self.screen.blit(hint, (1020, 150))
        elif self.is_closed:
            hint = self.small_font.render("Трек замкнут. Перетаскивайте точки для редактирования", True, YELLOW)
            self.screen.blit(hint, (1020, 150))

        # Легенда цветов точек
        legend_y = 180
        legend_text = self.small_font.render("Цвета точек:", True, TEXT_COLOR)
        self.screen.blit(legend_text, (1020, legend_y))

        if not self.is_closed:
            colors_legend = [
                (GREEN, "Первая точка"),
                (RED, "Последняя точка"),
                (BLUE, "Промежуточные")
            ]
        else:
            colors_legend = [
                (YELLOW, "Все точки (трек замкнут)")
            ]

        for i, (color, text) in enumerate(colors_legend):
            y_pos = legend_y + 25 + i * 20
            pygame.draw.circle(self.screen, color, (1030, y_pos + 6), 6)
            pygame.draw.circle(self.screen, WHITE, (1030, y_pos + 6), 6, 1)
            legend_item = self.small_font.render(text, True, TEXT_COLOR)
            self.screen.blit(legend_item, (1045, y_pos))

        # Рисуем кнопки
        for button in self.buttons:
            button.draw(self.screen)


class TrackSelector:
    def __init__(self, screen):
        self.screen = screen
        self.tracks_folder = "saved_tracks"
        self.selected_track = None
        self.track_buttons = []
        self.back_button = Button(SCREEN_WIDTH // 2 - 75, 700, 150, 50, "НАЗАД", SECONDARY)

        # Шрифты
        self.font = pygame.font.SysFont('Arial', 36)
        self.small_font = pygame.font.SysFont('Arial', 20)

        self.load_tracks()

    def load_tracks(self):
        """Загружает список доступных треков"""
        self.track_buttons = []

        if not os.path.exists(self.tracks_folder):
            os.makedirs(self.tracks_folder)

        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]

        # Создаем кнопки для каждого трека
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

                    # Проверка кликов по кнопкам треков
                    for button in self.track_buttons:
                        if button.is_clicked(mouse_pos, True):
                            self.selected_track = button.track_file
                            return "game"

                    # Проверка клика по кнопке "Назад"
                    if self.back_button.is_clicked(mouse_pos, True):
                        return "menu"

        # Обновление состояния кнопок
        mouse_pos = pygame.mouse.get_pos()
        for button in self.track_buttons:
            button.check_hover(mouse_pos)
        self.back_button.check_hover(mouse_pos)

        return "track_select"

    def draw(self):
        self.screen.fill(BACKGROUND)

        # Заголовок
        title = self.font.render("ВЫБЕРИТЕ ТРАССУ", True, TEXT_COLOR)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 100))

        # Информация о количестве треков
        track_count = len(self.track_buttons)
        if track_count == 0:
            info_text = self.small_font.render("Нет сохраненных треков. Сначала создайте трек в редакторе.", True,
                                               YELLOW)
            self.screen.blit(info_text, (SCREEN_WIDTH // 2 - info_text.get_width() // 2, 160))
        else:
            info_text = self.small_font.render(f"Доступно треков: {track_count}", True, TEXT_COLOR)
            self.screen.blit(info_text, (SCREEN_WIDTH // 2 - info_text.get_width() // 2, 160))

        # Рисуем кнопки треков
        for button in self.track_buttons:
            button.draw(self.screen)

        # Рисуем кнопку "Назад"
        self.back_button.draw(self.screen)


class CarGame:
    def __init__(self, screen):
        self.screen = screen
        self.car_x = SCREEN_WIDTH // 2
        self.car_y = SCREEN_HEIGHT // 2
        self.car_angle = 0
        self.car_speed = 0
        self.wheel_angle = 0
        self.max_speed = 8
        self.acceleration = 0.2
        self.deceleration = 0.1
        self.steering_speed = 3
        self.max_wheel_angle = 30
        self.handbrake_on = False
        self.selected_track = None
        self.track_data = None

        # Загружаем спрайты машинки
        self.car_sprites = {}
        self.load_car_sprites()

        # Кнопки
        self.buttons = [
            Button(50, 720, 120, 50, "МЕНЮ", SECONDARY)
        ]

        # Шрифты
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)

    def set_track(self, track_file):
        """Устанавливает выбранную трассу"""
        self.selected_track = track_file
        self.load_track()

    def load_track(self):
        """Загружает данные трека"""
        if not self.selected_track:
            return

        tracks_folder = "saved_tracks"
        filename = os.path.join(tracks_folder, self.selected_track)

        try:
            with open(filename, 'r') as f:
                self.track_data = json.load(f)
            print(f"Трек '{self.selected_track}' загружен успешно!")
        except Exception as e:
            print(f"Ошибка загрузки трека: {e}")
            self.track_data = None

    def load_car_sprites(self):
        # Создаем простые спрайты
        for state in ['straight', 'left', 'right', 'reverse']:
            sprite = pygame.Surface((40, 20), pygame.SRCALPHA)

            if state == 'straight':
                color = (220, 60, 60)
            elif state == 'left':
                color = (60, 150, 220)
            elif state == 'right':
                color = (60, 220, 150)
            else:
                color = (220, 180, 60)

            # Кузов машинки
            pygame.draw.rect(sprite, color, (0, 5, 40, 10), border_radius=3)
            pygame.draw.rect(sprite, (40, 40, 40), (5, 7, 30, 6), border_radius=2)

            # Колеса
            pygame.draw.rect(sprite, (30, 30, 30), (5, 3, 6, 3))
            pygame.draw.rect(sprite, (30, 30, 30), (29, 3, 6, 3))
            pygame.draw.rect(sprite, (30, 30, 30), (5, 14, 6, 3))
            pygame.draw.rect(sprite, (30, 30, 30), (29, 14, 6, 3))

            self.car_sprites[state] = sprite

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    for button in self.buttons:
                        if button.is_clicked(mouse_pos, True):
                            return "menu"

        return "game"

    def update(self):
        keys = pygame.key.get_pressed()

        # Управление поворотом колес
        if keys[pygame.K_a]:
            self.wheel_angle += self.steering_speed
        if keys[pygame.K_d]:
            self.wheel_angle -= self.steering_speed

        # Ограничение угла поворота колес
        self.wheel_angle = max(-self.max_wheel_angle, min(self.max_wheel_angle, self.wheel_angle))

        # Плавный возврат колес в нейтральное положение
        if not keys[pygame.K_a] and not keys[pygame.K_d]:
            if self.wheel_angle > 0:
                self.wheel_angle -= self.steering_speed * 0.7
                if self.wheel_angle < 0:
                    self.wheel_angle = 0
            elif self.wheel_angle < 0:
                self.wheel_angle += self.steering_speed * 0.7
                if self.wheel_angle > 0:
                    self.wheel_angle = 0

        # Управление движением
        if keys[pygame.K_w]:
            self.car_speed += self.acceleration
        elif keys[pygame.K_s]:
            self.car_speed -= self.acceleration
        elif abs(self.car_speed) > 0:
            # Плавное замедление
            if self.car_speed > 0:
                self.car_speed -= self.deceleration
                if self.car_speed < 0:
                    self.car_speed = 0
            else:
                self.car_speed += self.deceleration
                if self.car_speed > 0:
                    self.car_speed = 0

        # Ручной тормоз
        self.handbrake_on = keys[pygame.K_SPACE]

        # Ограничение скорости
        self.car_speed = max(-self.max_speed / 2, min(self.max_speed, self.car_speed))

        # Поворот машины только при движении
        if abs(self.car_speed) > 0.1 and abs(self.wheel_angle) > 1:
            turn_factor = self.car_speed * math.tan(math.radians(self.wheel_angle)) / 50
            self.car_angle += math.degrees(turn_factor)
            self.car_angle %= 360

        # Обновление позиции
        angle_rad = math.radians(self.car_angle)
        self.car_x += self.car_speed * math.cos(angle_rad)
        self.car_y -= self.car_speed * math.sin(angle_rad)

        # Ограничение движения в пределах экрана
        self.car_x = max(20, min(SCREEN_WIDTH - 20, self.car_x))
        self.car_y = max(20, min(SCREEN_HEIGHT - 20, self.car_y))

        # Обновление состояния кнопок
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.check_hover(mouse_pos)

    def draw(self):
        self.screen.fill(BACKGROUND)

        # Рисуем панель
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, SCREEN_WIDTH, 80))
        pygame.draw.rect(self.screen, PANEL_BG, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))

        # Заголовок
        if self.selected_track:
            track_name = self.selected_track.replace('.json', '')
            title = self.font.render(f"ГОНОЧНАЯ МАШИНКА - {track_name}", True, TEXT_COLOR)
        else:
            title = self.font.render("ГОНОЧНАЯ МАШИНКА", True, TEXT_COLOR)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 20))

        # Информация об управлении
        controls = self.small_font.render("W/S - газ/тормоз, A/D - поворот, SPACE - ручной тормоз", True, TEXT_COLOR)
        self.screen.blit(controls, (SCREEN_WIDTH // 2 - controls.get_width() // 2, 50))

        # Рисуем загруженную трассу или стандартную дорогу
        if self.track_data and self.track_data.get('points'):
            self.draw_custom_track()
        else:
            self.draw_default_track()

        # Рисуем машинку
        car_sprite = self.get_car_sprite()
        rotated_car = pygame.transform.rotate(car_sprite, self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, car_rect)

        # Информация о машинке
        info_text = [
            f"Скорость: {abs(self.car_speed):.1f}",
            f"Угол: {self.car_angle:.1f}°",
            f"Колеса: {self.wheel_angle:.1f}°",
            f"Ручной тормоз: {'ВКЛ' if self.handbrake_on else 'выкл'}"
        ]

        for i, text in enumerate(info_text):
            text_surf = self.small_font.render(text, True, TEXT_COLOR)
            self.screen.blit(text_surf, (1020, 120 + i * 25))

        # Рисуем кнопки
        for button in self.buttons:
            button.draw(self.screen)

    def draw_custom_track(self):
        """Рисует загруженную пользовательскую трассу"""
        if not self.track_data:
            return

        points = self.track_data['points']
        track_width = self.track_data.get('width', 40)
        is_closed = self.track_data.get('closed', True)

        # Преобразуем координаты для отображения
        screen_points = [(p[0] + 400, 400 - p[1]) for p in points]

        # Создаем границы трека
        inner_points = []
        outer_points = []

        for i in range(len(points)):
            if is_closed:
                p_prev = points[i - 1]
                p_curr = points[i]
                p_next = points[(i + 1) % len(points)]
            else:
                if i == 0:
                    p_prev = points[0]
                    p_curr = points[0]
                    p_next = points[1]
                elif i == len(points) - 1:
                    p_prev = points[-2]
                    p_curr = points[-1]
                    p_next = points[-1]
                else:
                    p_prev = points[i - 1]
                    p_curr = points[i]
                    p_next = points[i + 1]

            dir1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            dir2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            len1 = math.hypot(dir1[0], dir1[1])
            len2 = math.hypot(dir2[0], dir2[1])

            if len1 > 0 and len2 > 0:
                dir1 = (dir1[0] / len1, dir1[1] / len1)
                dir2 = (dir2[0] / len2, dir2[1] / len2)

                norm1 = (-dir1[1], dir1[0])
                norm2 = (-dir2[1], dir2[0])

                avg_norm = ((norm1[0] + norm2[0]) / 2, (norm1[1] + norm2[1]) / 2)
                avg_len = math.hypot(avg_norm[0], avg_norm[1])

                if avg_len > 0:
                    avg_norm = (avg_norm[0] / avg_len, avg_norm[1] / avg_len)

                    inner_points.append((p_curr[0] - avg_norm[0] * track_width / 2,
                                         p_curr[1] - avg_norm[1] * track_width / 2))
                    outer_points.append((p_curr[0] + avg_norm[0] * track_width / 2,
                                         p_curr[1] + avg_norm[1] * track_width / 2))

        # Рисуем дорогу
        if inner_points and outer_points:
            inner_screen = [(p[0] + 400, 400 - p[1]) for p in inner_points]
            outer_screen = [(p[0] + 400, 400 - p[1]) for p in outer_points]

            road_points = inner_screen + list(reversed(outer_screen))
            if len(road_points) >= 3:
                pygame.draw.polygon(self.screen, ROAD_COLOR, road_points)

            # Рисуем границы
            if is_closed:
                pygame.draw.lines(self.screen, ACCENT, True, inner_screen, 3)
                pygame.draw.lines(self.screen, ACCENT, True, outer_screen, 3)
            else:
                pygame.draw.lines(self.screen, ACCENT, False, inner_screen, 3)
                pygame.draw.lines(self.screen, ACCENT, False, outer_screen, 3)

    def draw_default_track(self):
        """Рисует стандартную дорогу если нет загруженной трассы"""
        pygame.draw.rect(self.screen, ROAD_COLOR, (100, 100, SCREEN_WIDTH - 200, SCREEN_HEIGHT - 200), border_radius=20)
        pygame.draw.rect(self.screen, ACCENT, (100, 100, SCREEN_WIDTH - 200, SCREEN_HEIGHT - 200), 5, border_radius=20)

        # Разметка
        for i in range(0, SCREEN_WIDTH - 200, 40):
            pygame.draw.rect(self.screen, WHITE, (120 + i, SCREEN_HEIGHT // 2 - 2, 20, 4))

    def get_car_sprite(self):
        if self.handbrake_on and abs(self.car_speed) > 1:
            return self.car_sprites['reverse']
        elif self.wheel_angle < -5:
            return self.car_sprites['right']
        elif self.wheel_angle > 5:
            return self.car_sprites['left']
        else:
            return self.car_sprites['straight']


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

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    for i, button in enumerate(self.buttons):
                        if button.is_clicked(mouse_pos, True):
                            if i == 0:
                                return "track_select"
                            elif i == 1:
                                return "editor"
                            elif i == 2:
                                return "quit"

        # Обновление состояния кнопок
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.check_hover(mouse_pos)

        return "menu"

    def draw(self):
        self.screen.fill(BACKGROUND)

        # Заголовок
        title = self.font.render("ГОНОЧНАЯ ИГРА", True, TEXT_COLOR)
        subtitle = self.small_font.render("Управляй машинкой и создавай свои треки!", True, TEXT_COLOR)

        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 150))
        self.screen.blit(subtitle, (SCREEN_WIDTH // 2 - subtitle.get_width() // 2, 220))

        # Рисуем кнопки
        for button in self.buttons:
            button.draw(self.screen)

        # Информация
        info = self.small_font.render("WASD для движения, SPACE для ручного тормоза", True, TEXT_COLOR)
        self.screen.blit(info, (SCREEN_WIDTH // 2 - info.get_width() // 2, 550))


def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Гоночная Игра")
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
            if result != "menu":
                current_screen = result
            menu.draw()

        elif current_screen == "track_select":
            result = track_selector.handle_events()
            if result == "game" and track_selector.selected_track:
                game.set_track(track_selector.selected_track)
                current_screen = "game"
            elif result != "track_select":
                current_screen = result
            track_selector.draw()

        elif current_screen == "editor":
            result = editor.handle_events()
            if result != "editor":
                current_screen = result
            editor.draw()

        elif current_screen == "game":
            result = game.handle_events()
            if result != "game":
                current_screen = result
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