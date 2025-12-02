import pygame
from pygame.locals import *
import sys
import math
import json
import os

# Инициализация Pygame
pygame.init()

# Получаем размеры экрана
info = pygame.display.Info()
SCREEN_WIDTH = info.current_w
SCREEN_HEIGHT = info.current_h

# Константы
FPS = 60
TRACK_WIDTH = 40  # Ширина трека в пикселях (базовая, до масштабирования)

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
        self.track_width = TRACK_WIDTH
        self.is_closed = False
        self.dragging_index = None
        self.drag_start = None
        self.current_track = None
        self.show_delete_dialog = False
        self.tracks_for_deletion = []
        self.delete_buttons = []

        # Создаем папку для сохранения треков
        self.tracks_folder = "saved_tracks"
        if not os.path.exists(self.tracks_folder):
            os.makedirs(self.tracks_folder)

        # Кнопки редактора
        self.buttons = [
            Button(50, SCREEN_HEIGHT - 80, 120, 50, "ОЧИСТИТЬ", RED),
            Button(190, SCREEN_HEIGHT - 80, 120, 50, "СОХРАНИТЬ", GREEN),
            Button(330, SCREEN_HEIGHT - 80, 120, 50, "ЗАГРУЗИТЬ", BLUE),
            Button(470, SCREEN_HEIGHT - 80, 120, 50, "УДАЛИТЬ", (255, 150, 50)),
            Button(610, SCREEN_HEIGHT - 80, 120, 50, "НАЗАД", SECONDARY)
        ]

        # Кнопки для диалога удаления
        self.cancel_delete_button = Button(SCREEN_WIDTH // 2 + 50, SCREEN_HEIGHT - 150, 120, 50, "ОТМЕНА", SECONDARY)

        # Шрифты
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 32, bold=True)

    def world_to_screen(self, point):
        """Преобразует мировые координаты в экранные для редактора"""
        x, y = point
        screen_x = x + SCREEN_WIDTH // 2
        screen_y = SCREEN_HEIGHT // 2 - y
        return screen_x, screen_y

    def screen_to_world(self, point):
        """Преобразует экранные координаты в мировые для редактора"""
        screen_x, screen_y = point
        world_x = screen_x - SCREEN_WIDTH // 2
        world_y = SCREEN_HEIGHT // 2 - screen_y
        return world_x, world_y

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Левая кнопка мыши
                    mouse_pos = pygame.mouse.get_pos()

                    if self.show_delete_dialog:
                        # Обработка кликов в диалоге удаления
                        for button in self.delete_buttons:
                            if button.is_clicked(mouse_pos, True):
                                self.delete_selected_track(button.track_file)
                                self.show_delete_dialog = False
                                return "editor"

                        if self.cancel_delete_button.is_clicked(mouse_pos, True):
                            self.show_delete_dialog = False
                            return "editor"

                        # Не даем кликать по основным кнопкам когда открыт диалог удаления
                        return "editor"

                    # Проверка кликов по основным кнопкам
                    for i, button in enumerate(self.buttons):
                        if button.is_clicked(mouse_pos, True):
                            if i == 0:  # Очистить
                                self.clear_track()
                            elif i == 1:  # Сохранить
                                self.save_track()
                            elif i == 2:  # Загрузить
                                self.load_track_dialog()
                            elif i == 3:  # Удалить
                                self.show_delete_dialog = True
                                self.load_tracks_for_deletion()
                            elif i == 4:  # Назад
                                return "menu"

                    # Добавление/редактирование точек трека
                    if not self.show_delete_dialog:
                        draw_area_left = SCREEN_WIDTH * 0.1
                        draw_area_right = SCREEN_WIDTH * 0.9
                        draw_area_top = SCREEN_HEIGHT * 0.15
                        draw_area_bottom = SCREEN_HEIGHT * 0.85

                        if (draw_area_left <= mouse_pos[0] <= draw_area_right and
                                draw_area_top <= mouse_pos[1] <= draw_area_bottom):
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

                if self.show_delete_dialog:
                    for button in self.delete_buttons:
                        button.check_hover(mouse_pos)
                    self.cancel_delete_button.check_hover(mouse_pos)

                # Перетаскивание точек
                if self.dragging_index is not None and not self.show_delete_dialog:
                    draw_area_left = SCREEN_WIDTH * 0.1
                    draw_area_right = SCREEN_WIDTH * 0.9
                    draw_area_top = SCREEN_HEIGHT * 0.15
                    draw_area_bottom = SCREEN_HEIGHT * 0.85

                    if (draw_area_left <= mouse_pos[0] <= draw_area_right and
                            draw_area_top <= mouse_pos[1] <= draw_area_bottom):

                        world_pos = self.screen_to_world(mouse_pos)
                        self.points[self.dragging_index] = world_pos

                        # Если трек замкнут и перемещаем первую/последнюю точку
                        if self.is_closed and self.dragging_index in [0, len(self.points) - 1]:
                            self.points[0] = self.points[-1] = world_pos

                        self.rebuild_track()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.show_delete_dialog:
                        self.show_delete_dialog = False
                    else:
                        return "menu"

        return "editor"

    def handle_track_click(self, mouse_pos):
        """Обработка кликов для добавления и редактирования точек трека"""
        # Преобразование координат в мировые
        world_pos = self.screen_to_world(mouse_pos)
        x, y = world_pos
        new_point = (x, y)

        # Проверка клика на существующие точки
        for i in range(len(self.points)):
            screen_point = self.world_to_screen(self.points[i])
            dist = math.hypot(mouse_pos[0] - screen_point[0], mouse_pos[1] - screen_point[1])
            if dist < 15:
                # Если кликнули на первую точку и трек еще не замкнут - замыкаем трек
                if i == 0 and not self.is_closed and len(self.points) >= 3:
                    self.is_closed = True
                    print("Трек замкнут! Нажата первая точка.")
                    self.rebuild_track()
                    return
                self.dragging_index = i
                self.drag_start = world_pos
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
        if not self.is_closed or len(self.points) < 3:
            print("Ошибка: Трек должен быть замкнут и содержать минимум 3 точки")
            return

        # Создаем данные трека
        self.current_track = {
            'points': self.points.copy(),
            'width': self.track_width,
            'closed': self.is_closed
        }

        # Запрашиваем имя для трека
        track_count = len([f for f in os.listdir(self.tracks_folder) if f.endswith('.json')])
        default_name = f"track_{track_count + 1}"

        # В реальном приложении здесь можно было бы добавить диалог ввода имени
        # Для простоты используем автоматическое имя
        name = default_name

        filename = os.path.join(self.tracks_folder, f"{name}.json")
        with open(filename, 'w') as f:
            json.dump(self.current_track, f, indent=2)

        print(f"Трек сохранен как {filename}")

    def load_track_dialog(self):
        """Показывает диалог выбора трека для загрузки"""
        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]
        if not tracks:
            print("Нет сохраненных треков")
            return

        # Для простоты загружаем первый трек
        # В реальном приложении можно добавить диалог выбора
        self.load_track(tracks[0])

    def load_track(self, track_file):
        """Загружает указанный трек"""
        filename = os.path.join(self.tracks_folder, track_file)
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.points = [tuple(p) for p in data['points']]
            self.track_width = data['width']
            self.is_closed = data.get('closed', True)
            self.current_track = data
            self.rebuild_track()
            print(f"Трек {track_file} загружен")
        except Exception as e:
            print(f"Ошибка загрузки трека {track_file}: {e}")

    def load_tracks_for_deletion(self):
        """Загружает список треков для удаления"""
        self.tracks_for_deletion = []
        self.delete_buttons = []

        if not os.path.exists(self.tracks_folder):
            return

        tracks = [f for f in os.listdir(self.tracks_folder) if f.endswith('.json')]

        for i, track in enumerate(tracks):
            track_name = track.replace('.json', '')
            y_pos = 200 + i * 60
            if y_pos < SCREEN_HEIGHT - 200:
                button = Button(SCREEN_WIDTH // 2 - 200, y_pos, 400, 50, track_name, RED)
                button.track_file = track
                self.delete_buttons.append(button)
                self.tracks_for_deletion.append(track)

    def delete_selected_track(self, track_file):
        """Удаляет выбранный трек"""
        filename = os.path.join(self.tracks_folder, track_file)
        try:
            os.remove(filename)
            print(f"Трек {track_file} удален")
            # Обновляем список после удаления
            self.load_tracks_for_deletion()
        except Exception as e:
            print(f"Ошибка удаления трека {track_file}: {e}")

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

        if self.show_delete_dialog:
            # Рисуем диалог удаления
            self.draw_delete_dialog()
        else:
            # Рисуем обычный интерфейс редактора
            self.draw_editor()

    def draw_editor(self):
        """Рисует интерфейс редактора треков"""
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
        draw_area_left = SCREEN_WIDTH * 0.1
        draw_area_top = SCREEN_HEIGHT * 0.15
        draw_area_width = SCREEN_WIDTH * 0.8
        draw_area_height = SCREEN_HEIGHT * 0.7

        pygame.draw.rect(self.screen, SECONDARY,
                         (draw_area_left, draw_area_top, draw_area_width, draw_area_height), 2)

        # Рисуем сетку
        grid_size = 40
        start_x = draw_area_left - ((draw_area_left - SCREEN_WIDTH // 2) % grid_size)
        start_y = draw_area_top - ((draw_area_top - SCREEN_HEIGHT // 2) % grid_size)

        for x in range(int(start_x), int(draw_area_left + draw_area_width), grid_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, draw_area_top),
                             (x, draw_area_top + draw_area_height), 1)
        for y in range(int(start_y), int(draw_area_top + draw_area_height), grid_size):
            pygame.draw.line(self.screen, GRID_COLOR, (draw_area_left, y),
                             (draw_area_left + draw_area_width, y), 1)

        # Центральные оси
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        pygame.draw.line(self.screen, GRID_COLOR, (center_x, draw_area_top),
                         (center_x, draw_area_top + draw_area_height), 2)
        pygame.draw.line(self.screen, GRID_COLOR, (draw_area_left, center_y),
                         (draw_area_left + draw_area_width, center_y), 2)

        # Рисуем трек (сначала фон трека)
        if self.points:
            # Рисуем линии между точками
            screen_points = [self.world_to_screen(p) for p in self.points]
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
                inner_screen = [self.world_to_screen(p) for p in inner_points]
                outer_screen = [self.world_to_screen(p) for p in outer_points]

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
            screen_points = [self.world_to_screen(p) for p in self.points]

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
        self.screen.blit(status_text, (SCREEN_WIDTH - 250, 120))

        # Подсказка
        if len(self.points) >= 3 and not self.is_closed:
            hint = self.small_font.render("Нажмите на первую точку (зеленую), чтобы замкнуть трек", True, GREEN)
            self.screen.blit(hint, (SCREEN_WIDTH - 450, 150))
        elif self.is_closed:
            hint = self.small_font.render("Трек замкнут. Перетаскивайте точки для редактирования", True, YELLOW)
            self.screen.blit(hint, (SCREEN_WIDTH - 450, 150))

        # Легенда цветов точек
        legend_y = 180
        legend_text = self.small_font.render("Цвета точек:", True, TEXT_COLOR)
        self.screen.blit(legend_text, (SCREEN_WIDTH - 200, legend_y))

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
            pygame.draw.circle(self.screen, color, (int(SCREEN_WIDTH - 190), y_pos + 6), 6)
            pygame.draw.circle(self.screen, WHITE, (int(SCREEN_WIDTH - 190), y_pos + 6), 6, 1)
            legend_item = self.small_font.render(text, True, TEXT_COLOR)
            self.screen.blit(legend_item, (int(SCREEN_WIDTH - 175), y_pos))

        # Рисуем кнопки
        for button in self.buttons:
            button.draw(self.screen)

    def draw_delete_dialog(self):
        """Рисует диалог удаления треков"""
        # Затемняем фон
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Рисуем диалоговое окно
        dialog_rect = pygame.Rect(SCREEN_WIDTH // 2 - 300, 100, 600, SCREEN_HEIGHT - 200)
        pygame.draw.rect(self.screen, PANEL_BG, dialog_rect, border_radius=15)
        pygame.draw.rect(self.screen, ACCENT, dialog_rect, 3, border_radius=15)

        # Заголовок
        title = self.title_font.render("ВЫБЕРИТЕ ТРЕК ДЛЯ УДАЛЕНИЯ", True, TEXT_COLOR)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 130))

        # Информация
        info = self.small_font.render("Кликните на трек, который хотите удалить", True, TEXT_COLOR)
        self.screen.blit(info, (SCREEN_WIDTH // 2 - info.get_width() // 2, 170))

        # Рисуем список треков для удаления
        if not self.delete_buttons:
            no_tracks = self.font.render("Нет сохраненных треков", True, YELLOW)
            self.screen.blit(no_tracks, (SCREEN_WIDTH // 2 - no_tracks.get_width() // 2, 250))
        else:
            for button in self.delete_buttons:
                button.draw(self.screen)

        # Кнопка отмены
        self.cancel_delete_button.draw(self.screen)


class TrackSelector:
    def __init__(self, screen):
        self.screen = screen
        self.tracks_folder = "saved_tracks"
        self.selected_track = None
        self.track_buttons = []
        self.back_button = Button(SCREEN_WIDTH // 2 - 75, SCREEN_HEIGHT - 100, 150, 50, "НАЗАД", SECONDARY)

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
            if y_pos < SCREEN_HEIGHT - 150:
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
        self.selected_track = None
        self.track_data = None
        self.track_scale = 1.0
        self.track_offset_x = 0
        self.track_offset_y = 0
        self.track_screen_points = []
        self.inner_screen_points = []
        self.outer_screen_points = []
        self.car_size = TRACK_WIDTH / 1.5  # Машинка в 1.5 раза меньше ширины трека (базовая пропорция)
        self.car_on_track = False
        self.car_world_x = 0
        self.car_world_y = 0

        # Загружаем спрайты машинки
        self.car_sprites = {}
        self.load_car_sprites()

        # Кнопки
        self.buttons = [
            Button(50, SCREEN_HEIGHT - 80, 120, 50, "МЕНЮ", SECONDARY)
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
            self.calculate_track_scale_and_offset()
        except Exception as e:
            print(f"Ошибка загрузки трека: {e}")
            self.track_data = None

    def calculate_track_scale_and_offset(self):
        """Рассчитывает масштаб и смещение для отображения трека на весь экран"""
        if not self.track_data or not self.track_data.get('points'):
            return

        points = self.track_data['points']
        track_width = self.track_data.get('width', TRACK_WIDTH)

        # Находим границы трека
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)

        # Добавляем отступ для ширины трека
        min_x -= track_width
        max_x += track_width
        min_y -= track_width
        max_y += track_width

        width = max_x - min_x
        height = max_y - min_y

        if width == 0 or height == 0:
            self.track_scale = 1.0
            self.track_offset_x = -min_x
            self.track_offset_y = -min_y
            return

        # Рассчитываем масштаб чтобы трек занимал весь экран
        screen_width = SCREEN_WIDTH
        screen_height = SCREEN_HEIGHT - 100  # Учитываем панель сверху

        # Рассчитываем масштаб для заполнения экрана с учетом сохранения пропорций
        scale_x = screen_width / width
        scale_y = screen_height / height

        # Используем минимальный масштаб чтобы трек помещался по обеим осям
        self.track_scale = min(scale_x, scale_y)

        # Проверяем, не слишком ли маленький масштаб
        min_scale = 0.1  # Минимальный масштаб для трека
        if self.track_scale < min_scale:
            self.track_scale = min_scale
            print(f"Трек слишком большой, установлен минимальный масштаб: {min_scale}")

        # Рассчитываем смещение для центрирования
        self.track_offset_x = -min_x + (screen_width / self.track_scale - width) / 2
        self.track_offset_y = -min_y + (screen_height / self.track_scale - height) / 2

        # Пересоздаем спрайты машинки с новым размером (масштабированным относительно трека)
        self.load_car_sprites()

    def world_to_screen(self, point):
        """Преобразует мировые координаты в экранные"""
        if not self.track_data:
            return point

        x, y = point
        screen_height = SCREEN_HEIGHT - 100
        screen_x = (x + self.track_offset_x) * self.track_scale
        screen_y = screen_height - (y + self.track_offset_y) * self.track_scale + 50
        return screen_x, screen_y

    def screen_to_world(self, point):
        """Преобразует экранные координаты в мировые"""
        if not self.track_data:
            return point

        screen_x, screen_y = point
        screen_height = SCREEN_HEIGHT - 100
        world_x = screen_x / self.track_scale - self.track_offset_x
        world_y = (screen_height - screen_y + 50) / self.track_scale - self.track_offset_y
        return world_x, world_y

    def create_track_boundaries(self):
        """Создает границы трека с закругленными углами"""
        if not self.track_data or not self.track_data.get('points'):
            return

        points = self.track_data['points']
        track_width = self.track_data.get('width', TRACK_WIDTH)
        is_closed = self.track_data.get('closed', True)

        self.inner_screen_points = []
        self.outer_screen_points = []

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

                    inner_point = (p_curr[0] - avg_norm[0] * track_width / 2,
                                   p_curr[1] - avg_norm[1] * track_width / 2)
                    outer_point = (p_curr[0] + avg_norm[0] * track_width / 2,
                                   p_curr[1] + avg_norm[1] * track_width / 2)

                    self.inner_screen_points.append(self.world_to_screen(inner_point))
                    self.outer_screen_points.append(self.world_to_screen(outer_point))

    def load_car_sprites(self):
        """Создает спрайты машинки с учетом текущего масштаба трека"""
        # Вычисляем размер машинки с учетом масштаба трека
        # Базовая ширина машинки = TRACK_WIDTH / 1.5, затем масштабируем
        scaled_car_width = (TRACK_WIDTH / 1.5) * self.track_scale
        scaled_car_height = scaled_car_width / 2

        # Минимальный размер машинки
        min_car_size = 10
        if scaled_car_width < min_car_size:
            scaled_car_width = min_car_size
            scaled_car_height = scaled_car_width / 2

        self.car_sprites = {}

        for state in ['straight', 'left', 'right']:
            sprite = pygame.Surface((int(scaled_car_width), int(scaled_car_height)), pygame.SRCALPHA)

            if state == 'straight':
                color = (220, 60, 60)
            elif state == 'left':
                color = (60, 150, 220)
            else:
                color = (60, 220, 150)

            # Кузов машинки
            pygame.draw.rect(sprite, color, (0, scaled_car_height * 0.25, scaled_car_width, scaled_car_height * 0.5),
                             border_radius=int(scaled_car_width * 0.1))
            pygame.draw.rect(sprite, (40, 40, 40),
                             (scaled_car_width * 0.125, scaled_car_height * 0.35, scaled_car_width * 0.75,
                              scaled_car_height * 0.3),
                             border_radius=int(scaled_car_width * 0.08))

            # Колеса
            wheel_width = scaled_car_width * 0.15
            wheel_height = scaled_car_height * 0.15

            # Передние колеса
            pygame.draw.rect(sprite, (30, 30, 30),
                             (scaled_car_width * 0.125, scaled_car_height * 0.15, wheel_width, wheel_height))
            pygame.draw.rect(sprite, (30, 30, 30),
                             (scaled_car_width * 0.725, scaled_car_height * 0.15, wheel_width, wheel_height))

            # Задние колеса
            pygame.draw.rect(sprite, (30, 30, 30),
                             (scaled_car_width * 0.125, scaled_car_height * 0.7, wheel_width, wheel_height))
            pygame.draw.rect(sprite, (30, 30, 30),
                             (scaled_car_width * 0.725, scaled_car_height * 0.7, wheel_width, wheel_height))

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

        # Ограничение скорости с учетом масштаба
        scaled_max_speed = self.max_speed / max(0.1, self.track_scale)
        self.car_speed = max(-scaled_max_speed / 2, min(scaled_max_speed, self.car_speed))

        # Поворот машины только при движении
        if abs(self.car_speed) > 0.1 and abs(self.wheel_angle) > 1:
            turn_factor = self.car_speed * math.tan(math.radians(self.wheel_angle)) / 50
            self.car_angle += math.degrees(turn_factor)
            self.car_angle %= 360

        # Обновление позиции машины в мировых координатах
        angle_rad = math.radians(self.car_angle)
        self.car_world_x += self.car_speed * math.cos(angle_rad) / self.track_scale
        self.car_world_y += self.car_speed * math.sin(angle_rad) / self.track_scale

        # Преобразуем мировые координаты в экранные
        screen_pos = self.world_to_screen((self.car_world_x, self.car_world_y))
        self.car_x, self.car_y = screen_pos

        # Обновление состояния кнопок
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.check_hover(mouse_pos)

    def draw(self):
        self.screen.fill(BACKGROUND)

        # Рисуем панель
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, SCREEN_WIDTH, 50))
        pygame.draw.rect(self.screen, PANEL_BG, (0, SCREEN_HEIGHT - 50, SCREEN_WIDTH, 50))

        # Заголовок
        if self.selected_track:
            track_name = self.selected_track.replace('.json', '')
            title = self.font.render(f"ГОНОЧНАЯ МАШИНКА - {track_name}", True, TEXT_COLOR)
        else:
            title = self.font.render("ГОНОЧНАЯ МАШИНКА", True, TEXT_COLOR)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 10))

        # Информация об управлении
        controls = self.small_font.render("W/S - газ/тормоз, A/D - поворот", True, TEXT_COLOR)
        self.screen.blit(controls, (SCREEN_WIDTH // 2 - controls.get_width() // 2, 35))

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
            f"Масштаб: {self.track_scale:.2f}"
        ]

        for i, text in enumerate(info_text):
            text_surf = self.small_font.render(text, True, TEXT_COLOR)
            self.screen.blit(text_surf, (SCREEN_WIDTH - 200, 60 + i * 25))

        # Рисуем кнопки
        for button in self.buttons:
            button.draw(self.screen)

    def draw_custom_track(self):
        """Рисует загруженную пользовательскую трассу"""
        if not self.track_data:
            return

        is_closed = self.track_data.get('closed', True)

        # Создаем границы трека если их еще нет
        if not self.inner_screen_points or not self.outer_screen_points:
            self.create_track_boundaries()

        # Рисуем дорогу
        if len(self.inner_screen_points) >= 2 and len(self.outer_screen_points) >= 2:
            road_points = self.inner_screen_points + list(reversed(self.outer_screen_points))
            if len(road_points) >= 3:
                pygame.draw.polygon(self.screen, ROAD_COLOR, road_points)

        # Рисуем границы
        if is_closed:
            if len(self.inner_screen_points) >= 3:
                pygame.draw.lines(self.screen, ACCENT, True, self.inner_screen_points, 3)
            if len(self.outer_screen_points) >= 3:
                pygame.draw.lines(self.screen, ACCENT, True, self.outer_screen_points, 3)
        else:
            if len(self.inner_screen_points) >= 2:
                pygame.draw.lines(self.screen, ACCENT, False, self.inner_screen_points, 3)
            if len(self.outer_screen_points) >= 2:
                pygame.draw.lines(self.screen, ACCENT, False, self.outer_screen_points, 3)

    def draw_default_track(self):
        """Рисует стандартную дорогу если нет загруженной трассы"""
        track_width = TRACK_WIDTH
        road_rect = pygame.Rect(track_width, 50 + track_width,
                                SCREEN_WIDTH - 2 * track_width,
                                SCREEN_HEIGHT - 100 - 2 * track_width)
        pygame.draw.rect(self.screen, ROAD_COLOR, road_rect)
        pygame.draw.rect(self.screen, ACCENT, road_rect, 5)

        # Разметка
        for i in range(0, road_rect.width, 40):
            mark_rect = pygame.Rect(road_rect.x + 10 + i, road_rect.centery - 2, 20, 4)
            pygame.draw.rect(self.screen, WHITE, mark_rect)

    def get_car_sprite(self):
        if self.wheel_angle < -5:
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
        info = self.small_font.render("WASD для движения", True, TEXT_COLOR)
        self.screen.blit(info, (SCREEN_WIDTH // 2 - info.get_width() // 2, 550))


def main():
    # Создаем окно во весь экран
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
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