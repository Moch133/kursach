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
        self.min_width = 20
        self.max_width = 200
        self.is_closed = False
        self.dragging_index = None
        self.drag_start = None
        self.current_track = None
        self.start_line = None

        # Создаем папку для сохранения треков
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tracks_folder = os.path.join(script_dir, "saved_tracks")
        if not os.path.exists(self.tracks_folder):
            os.makedirs(self.tracks_folder)

        # Кнопки
        self.buttons = [
            Button(50, 720, 120, 50, "ОЧИСТИТЬ", RED),
            Button(190, 720, 120, 50, "СОХРАНИТЬ", GREEN),
            Button(330, 720, 120, 50, "ЗАГРУЗИТЬ", BLUE),
            Button(470, 720, 120, 50, "УДАЛИТЬ", (255, 150, 50)),
            Button(610, 720, 120, 50, "НАЗАД", SECONDARY),
            Button(750, 720, 50, 50, "+", GREEN),
            Button(810, 720, 50, 50, "-", RED)
        ]

        # Шрифты
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)
        
        # Сообщение для пользователя
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
                            elif i == 5:  # Увеличить ширину
                                self.increase_width()
                            elif i == 6:  # Уменьшить ширину
                                self.decrease_width()

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
                    if self.is_closed and self.dragging_index in [0, len(self.points) - 1]:
                        self.points[0] = self.points[-1] = (mouse_pos[0] - 400, 400 - mouse_pos[1])
                    self.rebuild_track()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.increase_width()
                elif event.key == pygame.K_MINUS:
                    self.decrease_width()

        # Обновление таймера сообщения
        if self.message_timer > 0:
            self.message_timer -= 1
            if self.message_timer == 0:
                self.message = ""

        return "editor"

    def increase_width(self):
        if self.track_width < self.max_width:
            self.track_width += 5
            self.rebuild_track()
            self.show_message(f"Ширина трека: {self.track_width}", 60)

    def decrease_width(self):
        if self.track_width > self.min_width:
            self.track_width -= 5
            self.rebuild_track()
            self.show_message(f"Ширина трека: {self.track_width}", 60)

    def handle_track_click(self, mouse_pos):
        x = mouse_pos[0] - 400
        y = 400 - mouse_pos[1]
        new_point = (x, y)

        # Проверка клика на существующие точки
        for i in range(len(self.points)):
            dist = math.hypot(x - self.points[i][0], y - self.points[i][1])
            if dist < 15:
                if i == 0 and not self.is_closed and len(self.points) >= 3:
                    self.is_closed = True
                    self.show_message("Трек замкнут! Теперь его можно сохранить.", 180)
                    self.rebuild_track()
                    return
                self.dragging_index = i
                self.drag_start = (x, y)
                return

        # Если трек уже замкнут, не добавляем новые точки
        if self.is_closed:
            self.show_message("Трек уже замкнут. Очистите для создания нового.", 120)
            return

        # Проверка минимального расстояния для новой точки
        if self.points:
            min_dist = min(math.hypot(new_point[0] - p[0], new_point[1] - p[1]) for p in self.points)
            if min_dist < 20:
                self.show_message("Слишком близко к существующей точке!", 90)
                return

        # Добавляем новую точку
        self.points.append(new_point)
        self.show_message(f"Точка добавлена. Всего точек: {len(self.points)}", 90)

        # Автоматическое замыкание при приближении к первой точке
        if len(self.points) >= 3 and not self.is_closed:
            first = self.points[0]
            if math.hypot(new_point[0] - first[0], new_point[1] - first[1]) < 30:
                self.points.pop()
                self.is_closed = True
                self.show_message("Трек замкнут! Теперь его можно сохранить.", 180)

        self.rebuild_track()

    def rebuild_track(self):
        if len(self.points) < 2:
            return

        self.current_track = {
            'points': self.points.copy(),
            'width': self.track_width,
            'closed': self.is_closed
        }
        
        # Создаем стартовую линию
        self.create_start_line()

    def create_start_line(self):
        """Создает стартовую линию с направлением для спавна машинки"""
        if len(self.points) < 2 or not self.is_closed:
            self.start_line = None
            return
            
        # Направление движения - от первой точки ко второй
        start_point = self.points[0]
        next_point = self.points[1]
        
        # Вектор направления движения
        dx = next_point[0] - start_point[0]
        dy = next_point[1] - start_point[1]
        
        # Нормализуем вектор
        length = math.hypot(dx, dy)
        if length == 0:
            self.start_line = None
            return
            
        dx /= length
        dy /= length
        
        # Перпендикулярный вектор для стартовой линии
        perp_dx = -dy
        perp_dy = dx
        
        # Создаем стартовую линию (перпендикулярно направлению движения)
        line_length = self.track_width * 0.8
        start_x = start_point[0] - perp_dx * line_length / 2
        start_y = start_point[1] - perp_dy * line_length / 2
        end_x = start_point[0] + perp_dx * line_length / 2
        end_y = start_point[1] + perp_dy * line_length / 2
        
        # Вычисляем угол для ориентации машинки
        start_angle = math.degrees(math.atan2(-dy, dx))
        
        self.start_line = {
            'start': (start_x, start_y),
            'end': (end_x, end_y),
            'direction': (dx, dy),
            'angle': start_angle,
            'position': start_point  # Позиция для спавна машинки
        }
        
        # Добавляем стартовую линию в данные трека
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
            self.show_message("Ошибка: трек должен быть замкнут для сохранения!", 120)
            return

        track_count = len([f for f in os.listdir(self.tracks_folder) if f.endswith('.json')])
        name = f"track_{track_count + 1}"
        filename = os.path.join(self.tracks_folder, f"{name}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_track, f, indent=2, ensure_ascii=False)
            self.show_message(f"Трек сохранен как {name}.json", 180)
        except Exception as e:
            self.show_message(f"Ошибка сохранения: {str(e)}", 180)

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
            self.show_message(f"Трек {tracks[0]} загружен", 180)
        except Exception as e:
            self.show_message(f"Ошибка загрузки: {str(e)}", 180)

    def delete_track(self):
        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]
        if tracks:
            filename = os.path.join(self.tracks_folder, tracks[0])
            try:
                os.remove(filename)
                self.show_message(f"Трек {tracks[0]} удален", 180)
            except Exception as e:
                self.show_message(f"Ошибка удаления: {str(e)}", 180)
        else:
            self.show_message("Нет треков для удаления", 120)

    def calculate_smooth_track(self, points):
        if len(points) < 2:
            return [], []

        smooth_inner = []
        smooth_outer = []

        for i in range(len(points)):
            if self.is_closed:
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

                # Усредненная нормаль
                avg_norm = ((norm1[0] + norm2[0]) / 2, (norm1[1] + norm2[1]) / 2)
                avg_len = math.hypot(avg_norm[0], avg_norm[1])

                if avg_len > 0:
                    avg_norm = (avg_norm[0] / avg_len, avg_norm[1] / avg_len)

                    inner_point = (p_curr[0] - avg_norm[0] * self.track_width / 2,
                                   p_curr[1] - avg_norm[1] * self.track_width / 2)
                    outer_point = (p_curr[0] + avg_norm[0] * self.track_width / 2,
                                   p_curr[1] + avg_norm[1] * self.track_width / 2)

                    smooth_inner.append(inner_point)
                    smooth_outer.append(outer_point)
            else:
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
        if len(points) < 2:
            return None, None

        inner_points, outer_points = self.calculate_smooth_track(points)

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
                "Трек замкнут. Редактируйте существующие точки перетаскиванием. Очистите для создания нового.", True, YELLOW)
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

        # Рисуем трек
        if self.points:
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

        # Рисуем стартовую линию
        if self.start_line and self.is_closed:
            start_screen = (self.start_line['start'][0] + 400, 400 - self.start_line['start'][1])
            end_screen = (self.start_line['end'][0] + 400, 400 - self.start_line['end'][1])
            
            # Рисуем линию
            pygame.draw.line(self.screen, WHITE, start_screen, end_screen, 4)
            
            # Рисуем стрелку направления
            mid_x = (start_screen[0] + end_screen[0]) / 2
            mid_y = (start_screen[1] + end_screen[1]) / 2
            
            # Направление стрелки
            dir_x, dir_y = self.start_line['direction']
            arrow_length = 25
            arrow_end_x = mid_x + dir_x * arrow_length
            arrow_end_y = mid_y - dir_y * arrow_length  # Инвертируем Y для экранных координат
            
            pygame.draw.line(self.screen, GREEN, (mid_x, mid_y), (arrow_end_x, arrow_end_y), 4)
            
            # Боковые стороны стрелки
            arrow_side_length = 12
            perp_dx = -dir_y * 0.7
            perp_dy = dir_x * 0.7
            
            left_arrow_x = arrow_end_x - dir_x * arrow_side_length + perp_dx * arrow_side_length
            left_arrow_y = arrow_end_y + dir_y * arrow_side_length + perp_dy * arrow_side_length
            
            right_arrow_x = arrow_end_x - dir_x * arrow_side_length - perp_dx * arrow_side_length
            right_arrow_y = arrow_end_y + dir_y * arrow_side_length - perp_dy * arrow_side_length
            
            pygame.draw.line(self.screen, GREEN, (arrow_end_x, arrow_end_y), (left_arrow_x, left_arrow_y), 3)
            pygame.draw.line(self.screen, GREEN, (arrow_end_x, arrow_end_y), (right_arrow_x, right_arrow_y), 3)

        # Рисуем точки
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

                pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), 10)
                pygame.draw.circle(self.screen, WHITE, (int(point[0]), int(point[1])), 10, 2)

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

        # Информация о ширине трека
        width_text = self.small_font.render(f"Ширина трека: {self.track_width}", True, TEXT_COLOR)
        self.screen.blit(width_text, (1020, 150))
        
        # Подсказка по управлению шириной
        width_hint = self.small_font.render("Используйте +/- или кнопки для изменения ширины", True, TEXT_COLOR)
        self.screen.blit(width_hint, (1020, 175))

        # Подсказка
        if len(self.points) >= 3 and not self.is_closed:
            hint = self.small_font.render("Нажмите на первую точку (зеленую), чтобы замкнуть трек", True, GREEN)
            self.screen.blit(hint, (1020, 200))
        elif self.is_closed:
            hint = self.small_font.render("Трек замкнут. Перетаскивайте точки для редактирования", True, YELLOW)
            self.screen.blit(hint, (1020, 200))
            
        # Информация о стартовой линии
        if self.is_closed:
            start_info = self.small_font.render("Стартовая линия создана автоматически", True, GREEN)
            self.screen.blit(start_info, (1020, 225))

        # Легенда цветов точек
        legend_y = 250
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

        # Сообщение для пользователя
        if self.message:
            msg_color = YELLOW if "Ошибка" not in self.message else RED
            msg_text = self.small_font.render(self.message, True, msg_color)
            self.screen.blit(msg_text, (1020, 320))

        # Информация о сохраненных треках
        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]
        tracks_info = self.small_font.render(f"Сохранено треков: {len(tracks)}", True, TEXT_COLOR)
        self.screen.blit(tracks_info, (1020, 350))

        # Рисуем кнопки
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

        if not os.path.exists(self.tracks_folder):
            os.makedirs(self.tracks_folder)

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
            info_text = self.small_font.render("Нет сохраненных треков. Сначала создайте трек в редакторе.", True, YELLOW)
            self.screen.blit(info_text, (SCREEN_WIDTH // 2 - info_text.get_width() // 2, 160))
        else:
            info_text = self.small_font.render(f"Доступно треков: {track_count}", True, TEXT_COLOR)
            self.screen.blit(info_text, (SCREEN_WIDTH // 2 - info_text.get_width() // 2, 160))

        for button in self.track_buttons:
            button.draw(self.screen)

        self.back_button.draw(self.screen)


class CarGame:
    def __init__(self, screen):
        self.screen = screen
        
        # Параметры машинки
        self.car_x = SCREEN_WIDTH // 2
        self.car_y = SCREEN_HEIGHT // 2
        self.car_angle = 0
        self.velocity_x = 0
        self.velocity_y = 0
        self.angular_velocity = 0
        
        # Управление
        self.wheel_angle = 0
        self.engine_power = 0
        
        # Физика
        self.max_speed = 8
        self.acceleration = 0.5
        self.brake_power = 0.3
        self.steering_speed = 3
        self.max_wheel_angle = 30
        
        # Физика заноса
        self.friction = 0.95
        self.traction_fast = 0.05
        self.traction_slow = 0.01
        self.drift_factor = 0.05
        self.slide_factor = 0.98
        self.angular_friction = 0.93
        
        self.handbrake_on = False
        self.selected_track = None
        self.track_data = None
        self.start_position = None
        self.start_angle = 0

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
        self.selected_track = track_file
        self.load_track()
        self.spawn_car()

    def load_track(self):
        if not self.selected_track:
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        tracks_folder = os.path.join(script_dir, "saved_tracks")
        filename = os.path.join(tracks_folder, self.selected_track)

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.track_data = json.load(f)
            
            # Устанавливаем позицию спавна из стартовой линии
            if self.track_data and 'start_line' in self.track_data:
                start_line = self.track_data['start_line']
                
                # Позиция спавна - середина стартовой линии
                mid_x = (start_line['start'][0] + start_line['end'][0]) / 2
                mid_y = (start_line['start'][1] + start_line['end'][1]) / 2
                
                # Преобразуем в экранные координаты
                self.start_position = (mid_x + 400, 400 - mid_y)
                
                # Угол спавна - направление зеленой стрелки
                self.start_angle = start_line['angle']
        except Exception as e:
            print(f"Ошибка загрузки трека: {e}")
            self.track_data = None

    def spawn_car(self):
        """Спавнит машинку на стартовой линии по направлению зеленой стрелки"""
        if self.start_position:
            self.car_x, self.car_y = self.start_position
            self.car_angle = self.start_angle
            
            # Сбрасываем физику
            self.velocity_x = 0
            self.velocity_y = 0
            self.angular_velocity = 0
            self.wheel_angle = 0
            self.engine_power = 0
            self.handbrake_on = False
        else:
            # Если нет стартовой позиции, спавним по центру
            self.car_x = SCREEN_WIDTH // 2
            self.car_y = SCREEN_HEIGHT // 2
            self.car_angle = 0

    def load_car_sprites(self):
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
                elif event.key == pygame.K_r:  # Респавн машинки
                    self.spawn_car()

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

        # Управление двигателем
        self.engine_power = 0
        if keys[pygame.K_w]:
            self.engine_power = self.acceleration
        elif keys[pygame.K_s]:
            self.engine_power = -self.acceleration

        # Ручной тормоз
        self.handbrake_on = keys[pygame.K_SPACE]

        # Расчет ускорения
        angle_rad = math.radians(self.car_angle)
        
        # Ускорение вперед/назад
        if self.engine_power != 0:
            self.velocity_x += self.engine_power * math.cos(angle_rad)
            self.velocity_y -= self.engine_power * math.sin(angle_rad)
        else:
            # Замедление при отпускании педали газа
            self.velocity_x *= self.friction
            self.velocity_y *= self.friction

        # Торможение
        if keys[pygame.K_s] and not keys[pygame.K_w]:
            speed = math.hypot(self.velocity_x, self.velocity_y)
            if speed > 0.1:
                brake_x = -self.velocity_x / speed * self.brake_power
                brake_y = -self.velocity_y / speed * self.brake_power
                self.velocity_x += brake_x
                self.velocity_y += brake_y

        # Физика заноса
        speed = math.hypot(self.velocity_x, self.velocity_y)
        
        # Определяем направление движения относительно ориентации машины
        forward_vector = (math.cos(angle_rad), -math.sin(angle_rad))
        dot_product = (self.velocity_x * forward_vector[0] + 
                      self.velocity_y * forward_vector[1])
        
        # Если dot_product отрицательный - движемся назад
        is_moving_backward = dot_product < 0
        
        # Ручной тормоз - усиливает занос
        if self.handbrake_on and speed > 1:
            current_traction = self.traction_fast
            self.angular_velocity += self.wheel_angle * self.drift_factor * 0.07
            self.velocity_x *= self.slide_factor
            self.velocity_y *= self.slide_factor
        else:
            current_traction = self.traction_slow if speed < 2 else self.traction_fast

        # Поворот машины с учетом заноса и направления движения
        if abs(self.wheel_angle) > 1 and speed > 0.5:
            turn_force = math.tan(math.radians(self.wheel_angle)) * current_traction
            
            # Инвертируем поворот при движении назад
            if is_moving_backward:
                turn_force = -turn_force
                
            self.angular_velocity += turn_force * speed

        # Применяем угловую скорость
        self.car_angle += self.angular_velocity
        self.car_angle %= 360

        # Затухание угловой скорости
        self.angular_velocity *= self.angular_friction

        # Обновление позиции
        self.car_x += self.velocity_x
        self.car_y += self.velocity_y

        # Ограничение скорости
        speed = math.hypot(self.velocity_x, self.velocity_y)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity_x *= scale
            self.velocity_y *= scale

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
        controls = self.small_font.render("W/S - газ/тормоз, A/D - поворот, SPACE - ручной тормоз, R - респавн", True, TEXT_COLOR)
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
        speed = math.hypot(self.velocity_x, self.velocity_y)
        
        # Определяем направление движения
        angle_rad = math.radians(self.car_angle)
        forward_vector = (math.cos(angle_rad), -math.sin(angle_rad))
        dot_product = (self.velocity_x * forward_vector[0] + 
                      self.velocity_y * forward_vector[1])
        direction = "ВПЕРЕД" if dot_product >= 0 else "НАЗАД"
        
        info_text = [
            f"Скорость: {speed:.1f}",
            f"Направление: {direction}",
            f"Угол: {self.car_angle:.1f}°",
            f"Колеса: {self.wheel_angle:.1f}°",
            f"Занос: {abs(self.angular_velocity):.2f}",
            f"Ручной тормоз: {'ВКЛ' if self.handbrake_on else 'выкл'}"
        ]

        for i, text in enumerate(info_text):
            text_surf = self.small_font.render(text, True, TEXT_COLOR)
            self.screen.blit(text_surf, (1020, 120 + i * 25))

        # Рисуем кнопки
        for button in self.buttons:
            button.draw(self.screen)

    def draw_custom_track(self):
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

        # Рисуем стартовую линию
        if 'start_line' in self.track_data:
            start_line = self.track_data['start_line']
            start_screen = (start_line['start'][0] + 400, 400 - start_line['start'][1])
            end_screen = (start_line['end'][0] + 400, 400 - start_line['end'][1])
            
            # Рисуем линию
            pygame.draw.line(self.screen, WHITE, start_screen, end_screen, 4)
            
            # Рисуем стрелку направления
            mid_x = (start_screen[0] + end_screen[0]) / 2
            mid_y = (start_screen[1] + end_screen[1]) / 2
            
            # Направление стрелки
            dir_x, dir_y = start_line['direction']
            arrow_length = 25
            arrow_end_x = mid_x + dir_x * arrow_length
            arrow_end_y = mid_y - dir_y * arrow_length
            
            pygame.draw.line(self.screen, GREEN, (mid_x, mid_y), (arrow_end_x, arrow_end_y), 4)
            
            # Боковые стороны стрелки
            arrow_side_length = 12
            perp_dx = -dir_y * 0.7
            perp_dy = dir_x * 0.7
            
            left_arrow_x = arrow_end_x - dir_x * arrow_side_length + perp_dx * arrow_side_length
            left_arrow_y = arrow_end_y + dir_y * arrow_side_length + perp_dy * arrow_side_length
            
            right_arrow_x = arrow_end_x - dir_x * arrow_side_length - perp_dx * arrow_side_length
            right_arrow_y = arrow_end_y + dir_y * arrow_side_length - perp_dy * arrow_side_length
            
            pygame.draw.line(self.screen, GREEN, (arrow_end_x, arrow_end_y), (left_arrow_x, left_arrow_y), 3)
            pygame.draw.line(self.screen, GREEN, (arrow_end_x, arrow_end_y), (right_arrow_x, right_arrow_y), 3)

    def draw_default_track(self):
        pygame.draw.rect(self.screen, ROAD_COLOR, (100, 100, SCREEN_WIDTH - 200, SCREEN_HEIGHT - 200), border_radius=20)
        pygame.draw.rect(self.screen, ACCENT, (100, 100, SCREEN_WIDTH - 200, SCREEN_HEIGHT - 200), 5, border_radius=20)

        # Разметка
        for i in range(0, SCREEN_WIDTH - 200, 40):
            pygame.draw.rect(self.screen, WHITE, (120 + i, SCREEN_HEIGHT // 2 - 2, 20, 4))

    def get_car_sprite(self):
        speed = math.hypot(self.velocity_x, self.velocity_y)
        
        # Определяем направление движения
        angle_rad = math.radians(self.car_angle)
        forward_vector = (math.cos(angle_rad), -math.sin(angle_rad))
        dot_product = (self.velocity_x * forward_vector[0] + 
                      self.velocity_y * forward_vector[1])
        is_moving_backward = dot_product < 0
        
        if is_moving_backward and speed > 0.5:
            return self.car_sprites['reverse']
        elif self.wheel_angle < -5 and speed > 0.5:
            return self.car_sprites['right']
        elif self.wheel_angle > 5 and speed > 0.5:
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

        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.check_hover(mouse_pos)

        return "menu"

    def draw(self):
        self.screen.fill(BACKGROUND)

        title = self.font.render("ГОНОЧНАЯ ИГРА", True, TEXT_COLOR)
        subtitle = self.small_font.render("Управляй машинкой и создавай свои треки!", True, TEXT_COLOR)

        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 150))
        self.screen.blit(subtitle, (SCREEN_WIDTH // 2 - subtitle.get_width() // 2, 220))

        for button in self.buttons:
            button.draw(self.screen)

        info = self.small_font.render("WASD для движения, SPACE для ручного тормоза, R для респавна", True, TEXT_COLOR)
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
            track_selector.load_tracks()
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
