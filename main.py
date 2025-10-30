import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib import patheffects
import os
import json
import tkinter as tk
from tkinter import simpledialog, messagebox
import time
import glob
import math
from PIL import Image
import io
import random

# КОНСТАНТЫ ДЛЯ НАСТРОЙКИ СИМУЛЯЦИИ
SIMULATION_CONFIG = {
    'CAR_COUNT': 1,  # ← ИЗМЕНИТЕ ЗДЕСЬ КОЛИЧЕСТВО МАШИНОК
    'MAX_GENERATION_TIME': 300,  # Увеличено время поколения (5 минут)
    'MIN_LAP_TIME': 10.0,  # Увеличено минимальное время круга
    'TRACK_WIDTH': 20,  # Ширина трека по умолчанию
    'MAX_SPEED': 60.0,  # Максимальная скорость машинок вперед
    'MAX_REVERSE_SPEED': 30.0,  # Максимальная скорость назад
    'TIME_SCALE': 0.5,  # Масштаб времени (0.5 = время идет в 2 раза медленнее)
}


class RacingGame:
    """Главный класс гоночного трекера"""

    def __init__(self):
        self.fig = None
        self.ax = None
        self.tracks_folder = "saved_tracks"
        self.cars_folder = "car_sprites"
        self._create_folders()

        # Сохраняем ссылки на кнопки
        self.btn_editor = None
        self.btn_sim = None
        self.btn_exit = None

    def _create_folders(self):
        """Создает необходимые папки"""
        for folder in [self.tracks_folder, self.cars_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Создана папка: {folder}")

    def show_main_menu(self):
        """Показывает главное меню"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.patch.set_facecolor('#1a1a2e')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor('#16213e')

        # Заголовок
        title = self.ax.text(0.5, 0.85, 'ГОНОЧНЫЙ ТРЕКЕР',
                             fontsize=32, fontweight='bold', ha='center',
                             color='#e94560')
        title.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])

        subtitle = self.ax.text(0.5, 0.78, 'Конструктор трасс и симулятор гонок',
                                fontsize=16, ha='center', color='#0f3460', style='italic')

        # Информация о количестве треков и машинок
        track_count = self._count_files(self.tracks_folder, '.json')
        car_count = self._count_files(self.cars_folder, ('.png', '.jpg', '.jpeg'))

        info_text = self.ax.text(0.5, 0.25,
                                 f'Сохранено треков: {track_count}\n'
                                 f'Доступно машинок: {car_count}\n'
                                 f'Машинок в симуляции: {SIMULATION_CONFIG["CAR_COUNT"]}',
                                 fontsize=14, ha='center', color='#1f4068',
                                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#e6e6e6', alpha=0.8))

        # Кнопки
        self._create_menu_buttons()

        # Убрал tight_layout() чтобы избежать предупреждения
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()

    def _count_files(self, folder, extensions):
        """Считает файлы в папке по расширениям"""
        try:
            if isinstance(extensions, str):
                return len([f for f in os.listdir(folder) if f.endswith(extensions)])
            else:
                return len([f for f in os.listdir(folder)
                            if any(f.lower().endswith(ext) for ext in extensions)])
        except:
            return 0

    def _create_menu_buttons(self):
        """Создает кнопки главного меню"""
        # Кнопка редактора треков
        btn_editor_ax = plt.axes([0.3, 0.55, 0.4, 0.08])
        self.btn_editor = Button(btn_editor_ax, 'РЕДАКТОР ТРЕКОВ',
                                 color='#e94560', hovercolor='#c73550')
        self.btn_editor.on_clicked(self._open_editor)

        # Кнопка симуляции гонки
        btn_sim_ax = plt.axes([0.3, 0.45, 0.4, 0.08])
        self.btn_sim = Button(btn_sim_ax, 'СИМУЛЯЦИЯ ГОНКИ',
                              color='#e94560', hovercolor='#c73550')
        self.btn_sim.on_clicked(self._open_simulation)

        # Кнопка выхода
        btn_exit_ax = plt.axes([0.3, 0.35, 0.4, 0.08])
        self.btn_exit = Button(btn_exit_ax, 'ВЫХОД',
                               color='#1f4068', hovercolor='#16213e')
        self.btn_exit.on_clicked(self._exit_app)

    def _open_editor(self, event=None):
        """Открывает редактор треков"""
        print("Открытие редактора треков...")
        plt.close(self.fig)
        editor = TrackEditor(self.tracks_folder)
        editor.run()

    def _open_simulation(self, event=None):
        """Открывает симуляцию гонки"""
        print("Открытие симуляции гонки...")
        plt.close(self.fig)
        simulation = RaceSimulation(self.tracks_folder, self.cars_folder)
        simulation.run()

    def _exit_app(self, event=None):
        """Выход из приложения"""
        print("Выход из приложения...")
        plt.close('all')


class TrackEditor:
    """Редактор гоночных треков"""

    COLORS = {
        'bg': '#1a1a2e',
        'panel': '#16213e',
        'accent': '#e94560',
        'secondary': '#0f3460',
        'text': '#ffffff',
        'grid': '#2d4059'
    }

    def __init__(self, tracks_folder):
        self.tracks_folder = tracks_folder
        self.fig = None
        self.ax_main = None
        self.ax_list = None

        self.points = []
        self.track_width = SIMULATION_CONFIG['TRACK_WIDTH']
        self.current_track = None
        self.selected_track = None
        self.is_closed = False

        self.dragging_index = None
        self.drag_start = None

        # Сохраняем ссылки на кнопки и слайдер
        self.btn_clear = None
        self.btn_save = None
        self.btn_load = None
        self.btn_delete = None
        self.width_slider = None

    def run(self):
        """Запускает редактор"""
        self._setup_ui()
        self._load_tracks_list()
        self._update_display()
        plt.show()

    def _setup_ui(self):
        """Настраивает пользовательский интерфейс"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(self.COLORS['bg'])

        # Основная область рисования - УВЕЛИЧЕНА
        self.ax_main = plt.axes([0.15, 0.1, 0.8, 0.8])  # было [0.25, 0.1, 0.7, 0.8]
        self.ax_main.set_xlim(-200, 200)
        self.ax_main.set_ylim(-200, 200)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3, color=self.COLORS['grid'])
        self.ax_main.set_facecolor(self.COLORS['bg'])
        self.ax_main.set_title("Редактор треков - добавляйте точки щелчками",
                               fontsize=12, color=self.COLORS['text'], pad=20)

        # Панель списка треков - УМЕНЬШЕНА
        self.ax_list = plt.axes([0.02, 0.1, 0.12, 0.8])  # было [0.02, 0.1, 0.2, 0.8]
        self.ax_list.set_facecolor(self.COLORS['panel'])
        self.ax_list.set_xlim(0, 1)
        self.ax_list.set_ylim(0, 1)
        self.ax_list.set_xticks([])
        self.ax_list.set_yticks([])

        # Кнопки управления
        self._create_control_buttons()

        # Обработчики событий
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_drag)
        self.fig.canvas.mpl_connect('pick_event', self._on_track_select)

    def _create_control_buttons(self):
        """Создает кнопки управления"""
        # Основные кнопки - перемещены из-за изменения размеров
        btn_clear_ax = plt.axes([0.15, 0.02, 0.1, 0.05])  # было [0.25, 0.02, 0.1, 0.05]
        self.btn_clear = Button(btn_clear_ax, 'ОЧИСТИТЬ',
                                color=self.COLORS['accent'], hovercolor='#c73550')
        self.btn_clear.on_clicked(self._clear_track)

        btn_save_ax = plt.axes([0.26, 0.02, 0.1, 0.05])  # было [0.36, 0.02, 0.1, 0.05]
        self.btn_save = Button(btn_save_ax, 'СОХРАНИТЬ',
                               color=self.COLORS['accent'], hovercolor='#c73550')
        self.btn_save.on_clicked(self._save_track)

        btn_load_ax = plt.axes([0.37, 0.02, 0.1, 0.05])  # было [0.47, 0.02, 0.1, 0.05]
        self.btn_load = Button(btn_load_ax, 'ЗАГРУЗИТЬ',
                               color=self.COLORS['accent'], hovercolor='#c73550')
        self.btn_load.on_clicked(self._load_track)

        btn_delete_ax = plt.axes([0.48, 0.02, 0.1, 0.05])  # было [0.58, 0.02, 0.1, 0.05]
        self.btn_delete = Button(btn_delete_ax, 'УДАЛИТЬ',
                                 color=self.COLORS['accent'], hovercolor='#c73550')
        self.btn_delete.on_clicked(self._delete_track)

        # Слайдер ширины трека - перемещен
        slider_ax = plt.axes([0.59, 0.02, 0.15, 0.05])  # было [0.69, 0.02, 0.15, 0.05]
        slider_ax.set_facecolor(self.COLORS['panel'])
        self.width_slider = Slider(slider_ax, 'Ширина:', 10, 40,
                                   valinit=self.track_width, valstep=2,
                                   color=self.COLORS['secondary'])
        self.width_slider.on_changed(self._on_width_change)

    def _load_tracks_list(self):
        """Загружает список сохраненных треков"""
        try:
            tracks = [f for f in os.listdir(self.tracks_folder)
                      if f.endswith('.json')]
        except:
            tracks = []

        self.ax_list.clear()
        self.ax_list.set_facecolor(self.COLORS['panel'])
        self.ax_list.set_xlim(0, 1)
        self.ax_list.set_ylim(0, 1)
        self.ax_list.set_xticks([])
        self.ax_list.set_yticks([])

        # Заголовок - уменьшен шрифт
        self.ax_list.text(0.5, 0.95, 'СОХРАНЕННЫЕ ТРЕКИ',
                          fontsize=10, fontweight='bold', ha='center',  # было 12
                          color=self.COLORS['text'])

        if not tracks:
            self.ax_list.text(0.5, 0.5, 'Нет сохраненных треков',
                              ha='center', color=self.COLORS['grid'], fontsize=9)  # было 10
            return

        # Список треков - уменьшен интервал
        for i, track in enumerate(tracks[::-1]):
            y_pos = 0.85 - i * 0.06  # было 0.07
            if y_pos < 0.05:
                break

            track_name = track.replace('.json', '')
            color = self.COLORS['accent'] if track == self.selected_track else self.COLORS['text']

            text = self.ax_list.text(0.1, y_pos, f"• {track_name}",
                                     fontsize=8, color=color)  # было 9
            text.set_picker(True)

    def _on_click(self, event):
        """Обработчик клика мыши"""
        if event.inaxes != self.ax_main:
            return

        # Проверка редактирования существующих точек
        if self.is_closed and self.points:
            for i in range(len(self.points)):
                dist = math.hypot(event.xdata - self.points[i][0],
                                  event.ydata - self.points[i][1])
                if dist < 8:
                    self.dragging_index = i
                    self.drag_start = (event.xdata, event.ydata)
                    return

        # Добавление новой точки
        new_point = (float(event.xdata), float(event.ydata))

        # Проверка минимального расстояния
        if self.points:
            min_dist = min(math.hypot(new_point[0] - p[0], new_point[1] - p[1])
                           for p in self.points)
            if min_dist < 10:
                return

        self.points.append(new_point)

        # Проверка замыкания трека
        if len(self.points) >= 3:
            first = self.points[0]
            if math.hypot(new_point[0] - first[0], new_point[1] - first[1]) < 20:
                self.points[-1] = first
                self.is_closed = True
                print("Трек замкнут!")

        self._rebuild_track()

    def _on_release(self, event):
        """Обработчик отпускания кнопки мыши"""
        self.dragging_index = None
        self.drag_start = None

    def _on_drag(self, event):
        """Обработчик перетаскивания"""
        if (self.dragging_index is not None and event.inaxes == self.ax_main and
                event.xdata is not None and event.ydata is not None):

            self.points[self.dragging_index] = (event.xdata, event.ydata)

            # Если перемещаем первую/последнюю точку в замкнутом треке
            if self.is_closed and self.dragging_index in [0, len(self.points) - 1]:
                self.points[0] = self.points[-1] = (event.xdata, event.ydata)

            self._rebuild_track()

    def _on_track_select(self, event):
        """Обработчик выбора трека из списка"""
        if hasattr(event, 'artist'):
            tracks = [f for f in os.listdir(self.tracks_folder)
                      if f.endswith('.json')]
            texts = [child for child in self.ax_list.get_children()
                     if hasattr(child, 'get_text') and child.get_text().startswith('•')]

            if event.artist in texts:
                idx = texts.index(event.artist)
                self.selected_track = tracks[::-1][idx]
                self._load_tracks_list()
                print(f"Выбран трек: {self.selected_track}")

    def _on_width_change(self, val):
        """Обработчик изменения ширины трека"""
        self.track_width = val
        if self.points:
            self._rebuild_track()

    def _rebuild_track(self):
        """Перестраивает трек"""
        if len(self.points) < 2:
            self._update_display()
            return

        points = self.points.copy()
        if self.is_closed and points[-1] != points[0]:
            points[-1] = points[0]

        # Создание сглаженной центральной линии
        n_points = min(100, len(points) * 10)
        t = np.arange(len(points))
        t_new = np.linspace(0, len(points) - 1, n_points)

        points_array = np.array(points)
        center_x = np.interp(t_new, t, points_array[:, 0])
        center_y = np.interp(t_new, t, points_array[:, 1])

        if self.is_closed:
            center_x[-1] = center_x[0]
            center_y[-1] = center_y[0]

        # Создание границ трека
        success = self._create_track_boundaries(center_x, center_y)

        if success:
            self.current_track = {
                'center_line': (center_x.tolist(), center_y.tolist()),
                'inner_boundary': (self.inner_x.tolist(), self.inner_y.tolist()),
                'outer_boundary': (self.outer_x.tolist(), self.outer_y.tolist()),
                'points': points_array.tolist(),
                'width': self.track_width,
                'closed': self.is_closed
            }

        self._update_display()

    def _create_track_boundaries(self, center_x, center_y):
        """Создает границы трека"""
        dx = np.gradient(center_x)
        dy = np.gradient(center_y)

        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        nx = -dy / norm
        ny = dx / norm

        half_width = self.track_width / 2

        self.inner_x = center_x - half_width * nx
        self.inner_y = center_y - half_width * ny
        self.outer_x = center_x + half_width * nx
        self.outer_y = center_y + half_width * ny

        return True

    def _update_display(self):
        """Обновляет отображение"""
        self.ax_main.clear()
        self.ax_main.set_xlim(-200, 200)
        self.ax_main.set_ylim(-200, 200)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3, color=self.COLORS['grid'])
        self.ax_main.set_facecolor(self.COLORS['bg'])

        # Отрисовка трека
        if hasattr(self, 'inner_x') and self.inner_x is not None:
            # Заполнение дороги
            road_x = np.concatenate([self.inner_x, self.outer_x[::-1]])
            road_y = np.concatenate([self.inner_y, self.outer_y[::-1]])
            self.ax_main.fill(road_x, road_y, color='#4a6572', alpha=0.8, zorder=1)

            # Границы
            self.ax_main.plot(self.inner_x, self.inner_y,
                              color=self.COLORS['accent'], linewidth=2, zorder=2)
            self.ax_main.plot(self.outer_x, self.outer_y,
                              color=self.COLORS['accent'], linewidth=2, zorder=2)

        # Отрисовка точек
        if self.points:
            points_to_show = self.points[1:] if self.is_closed else self.points
            if points_to_show:
                points_array = np.array(points_to_show)
                self.ax_main.scatter(points_array[:, 0], points_array[:, 1],
                                     c=self.COLORS['secondary'], s=60, zorder=3,
                                     edgecolors='white', linewidths=1.5)

            # Линии между точками
            all_points = np.array(self.points)
            if len(all_points) > 1:
                self.ax_main.plot(all_points[:, 0], all_points[:, 1],
                                  color=self.COLORS['secondary'], alpha=0.5,
                                  linestyle='--', linewidth=1, zorder=2)

        # Статус
        status = f"Точек: {len(self.points)}"
        if self.is_closed:
            status += " (ЗАМКНУТ)"
        self.ax_main.text(0.02, 0.98, status, transform=self.ax_main.transAxes,
                          fontsize=11, color=self.COLORS['text'], va='top',
                          bbox=dict(boxstyle='round', facecolor=self.COLORS['panel'], alpha=0.8))

        self.fig.canvas.draw_idle()

    def _clear_track(self, event=None):
        """Очищает текущий трек"""
        print("Очистка трека...")
        self.points.clear()
        self.current_track = None
        self.is_closed = False
        self.inner_x = self.inner_y = None
        self.outer_x = self.outer_y = None
        self._update_display()

    def _save_track(self, event=None):
        """Сохраняет трек"""
        print("Сохранение трека...")
        if not self.current_track or not self.is_closed:
            self._show_message("Сначала создайте и замкните трек!")
            return

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        name = simpledialog.askstring("Сохранение", "Название трека:", parent=root)
        root.destroy()

        if not name:
            return

        # Очистка имени файла
        name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not name:
            name = "track"

        filename = os.path.join(self.tracks_folder, f"{name}.json")

        if os.path.exists(filename):
            root = tk.Tk()
            root.withdraw()
            if not messagebox.askyesno("Перезапись", "Трек уже существует. Перезаписать?"):
                root.destroy()
                return
            root.destroy()

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_track, f, ensure_ascii=False, indent=2)
            self._load_tracks_list()
            self._show_message("Трек сохранен!")
        except Exception as e:
            self._show_message(f"Ошибка: {e}")

    def _load_track(self, event=None):
        """Загружает выбранный трек"""
        print("Загрузка трека...")
        if not self.selected_track:
            self._show_message("Выберите трек из списка!")
            return

        try:
            filename = os.path.join(self.tracks_folder, self.selected_track)
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.points = [tuple(p) for p in data['points']]
            self.track_width = data['width']
            self.is_closed = data.get('closed', True)
            self.width_slider.set_val(self.track_width)

            self.inner_x = np.array(data['inner_boundary'][0])
            self.inner_y = np.array(data['inner_boundary'][1])
            self.outer_x = np.array(data['outer_boundary'][0])
            self.outer_y = np.array(data['outer_boundary'][1])

            self.current_track = data
            self._update_display()
            self._show_message("Трек загружен!")

        except Exception as e:
            self._show_message(f"Ошибка загрузки: {e}")

    def _delete_track(self, event=None):
        """Удаляет выбранный трек"""
        print("Удаление трека...")
        if not self.selected_track:
            self._show_message("Выберите трек для удаления!")
            return

        root = tk.Tk()
        root.withdraw()
        name = self.selected_track.replace('.json', '')

        if messagebox.askyesno("Удаление", f"Удалить трек '{name}'?"):
            try:
                os.remove(os.path.join(self.tracks_folder, self.selected_track))
                self.selected_track = None
                self._load_tracks_list()
                self._show_message("Трек удален!")
            except Exception as e:
                self._show_message(f"Ошибка: {e}")

        root.destroy()

    def _show_message(self, text):
        """Показывает временное сообщение"""
        self.ax_main.set_title(text, fontsize=12, color=self.COLORS['accent'], pad=20)
        self.fig.canvas.draw_idle()
        plt.pause(2)  # Показываем сообщение 2 секунды
        self.ax_main.set_title("Редактор треков - добавляйте точки щелчками",
                               fontsize=12, color=self.COLORS['text'], pad=20)


class NeuralNetwork:
    """Улучшенная нейронная сеть для управления машинкой"""

    def __init__(self, input_size=8, hidden_size=12, output_size=3):  # Увеличено количество выходов
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.bias2 = np.zeros((1, output_size))

    def forward(self, x):
        """Прямой проход через сеть"""
        self.layer1 = np.tanh(np.dot(x, self.weights1) + self.bias1)
        output = np.tanh(np.dot(self.layer1, self.weights2) + self.bias2)
        return output

    def mutate(self, mutation_rate=0.1, mutation_strength=0.2):
        """Мутация весов сети"""
        if random.random() < mutation_rate:
            self.weights1 += np.random.randn(*self.weights1.shape) * mutation_strength
        if random.random() < mutation_rate:
            self.bias1 += np.random.randn(*self.bias1.shape) * mutation_strength
        if random.random() < mutation_rate:
            self.weights2 += np.random.randn(*self.weights2.shape) * mutation_strength
        if random.random() < mutation_rate:
            self.bias2 += np.random.randn(*self.bias2.shape) * mutation_strength

    def copy(self):
        """Создает копию сети"""
        new_nn = NeuralNetwork()
        new_nn.weights1 = self.weights1.copy()
        new_nn.bias1 = self.bias1.copy()
        new_nn.weights2 = self.weights2.copy()
        new_nn.bias2 = self.bias2.copy()
        return new_nn


class GeneticAlgorithm:
    """Генетический алгоритм для обучения машинок"""

    def __init__(self, population_size=SIMULATION_CONFIG['CAR_COUNT']):
        self.population_size = population_size
        self.generation = 1
        self.best_fitness = -float('inf')
        self.best_time = float('inf')
        self.population = [NeuralNetwork() for _ in range(population_size)]
        self.fitness_scores = [0] * population_size
        self.completion_times = [float('inf')] * population_size

    def select_parent(self):
        """Выбор родителя на основе фитнес-функции"""
        total_fitness = sum(max(0, f) for f in self.fitness_scores)
        if total_fitness == 0:
            return random.choice(self.population)

        r = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(self.fitness_scores):
            current += max(0, fitness)
            if current >= r:
                return self.population[i].copy()
        return self.population[0].copy()

    def evolve(self):
        """Создание нового поколения"""
        new_population = []

        # Элитизм - сохраняем лучшую машинку
        best_index = np.argmax(self.fitness_scores)
        new_population.append(self.population[best_index].copy())

        # Создаем остальных детей
        while len(new_population) < self.population_size:
            parent = self.select_parent()
            child = parent.copy()
            child.mutate(mutation_rate=0.3, mutation_strength=0.15)
            new_population.append(child)

        self.population = new_population
        self.fitness_scores = [0] * self.population_size
        self.completion_times = [float('inf')] * self.population_size
        self.generation += 1


class RaceSimulation:
    """Улучшенный симулятор гонки с расширенным управлением"""

    COLORS = {
        'bg': '#1a1a2e',
        'panel': '#16213e',
        'accent': '#e94560',
        'text': '#ffffff',
        'grid': '#2d4059'
    }

    def __init__(self, tracks_folder, cars_folder):
        self.tracks_folder = tracks_folder
        self.cars_folder = cars_folder
        self.fig = None
        self.ax_main = None
        self.ax_list = None

        self.selected_track = None
        self.track_data = None
        self.car_sprites = {}
        self.car_objects = []
        self.car_artists = []

        self.is_running = False
        self.simulation_step = 0
        self.start_time = 0
        self.current_time = 0
        self.best_time = float('inf')
        self.generation_time = 0

        # Генетический алгоритм
        self.ga = GeneticAlgorithm(population_size=SIMULATION_CONFIG['CAR_COUNT'])
        self.generation_start_time = 0
        self.start_line_position = None

        # Сохраняем ссылки на кнопки
        self.btn_start = None
        self.btn_stop = None
        self.btn_reset = None

        # Время для физики в реальном времени
        self.last_update_time = 0
        self.frame_count = 0
        self.fps = 0

    def run(self):
        """Запускает симуляцию"""
        self._setup_ui()
        self._load_tracks_list()
        self._load_car_sprites()
        self._update_display()
        plt.show()

    def _setup_ui(self):
        """Настраивает интерфейс симуляции"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(self.COLORS['bg'])

        # Основная область - УВЕЛИЧЕНА
        self.ax_main = plt.axes([0.15, 0.1, 0.8, 0.8])
        self.ax_main.set_xlim(-200, 200)
        self.ax_main.set_ylim(-200, 200)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3, color=self.COLORS['grid'])
        self.ax_main.set_facecolor(self.COLORS['bg'])
        self.ax_main.set_title(f"Симуляция гонки с ИИ - {SIMULATION_CONFIG['CAR_COUNT']} машинок",
                               fontsize=12, color=self.COLORS['text'], pad=20)

        # Панель списка - УМЕНЬШЕНА
        self.ax_list = plt.axes([0.02, 0.1, 0.12, 0.8])
        self.ax_list.set_facecolor(self.COLORS['panel'])
        self.ax_list.set_xlim(0, 1)
        self.ax_list.set_ylim(0, 1)
        self.ax_list.set_xticks([])
        self.ax_list.set_yticks([])

        # Кнопки управления
        self._create_control_buttons()

        # Обработчики
        self.fig.canvas.mpl_connect('pick_event', self._on_track_select)

    def _create_control_buttons(self):
        """Создает кнопки управления симуляцией"""
        btn_start_ax = plt.axes([0.15, 0.02, 0.12, 0.06])
        self.btn_start = Button(btn_start_ax, 'СТАРТ',
                                color='#27ae60', hovercolor='#229954')
        self.btn_start.on_clicked(self._start_simulation)

        btn_stop_ax = plt.axes([0.28, 0.02, 0.12, 0.06])
        self.btn_stop = Button(btn_stop_ax, 'СТОП',
                               color='#e74c3c', hovercolor='#c0392b')
        self.btn_stop.on_clicked(self._stop_simulation)

        btn_reset_ax = plt.axes([0.41, 0.02, 0.12, 0.06])
        self.btn_reset = Button(btn_reset_ax, 'СБРОС',
                                color='#f39c12', hovercolor='#d68910')
        self.btn_reset.on_clicked(self._reset_simulation)

    def _load_car_sprites(self):
        """Загружает спрайты машинок"""
        try:
            # Добавлены спрайты для движения назад
            sprite_files = {
                'straight': self._find_sprite('*straight*', '*forward*', '*center*'),
                'left': self._find_sprite('*left*', '*turn_left*'),
                'right': self._find_sprite('*right*', '*turn_right*'),
                'reverse': self._find_sprite('*reverse*', '*back*', '*backward*'),
                'reverse_left': self._find_sprite('*reverse_left*', '*back_left*'),
                'reverse_right': self._find_sprite('*reverse_right*', '*back_right*')
            }

            print(f"Найдены спрайты: {sprite_files}")

            for state, file_path in sprite_files.items():
                if file_path:
                    try:
                        img = mpimg.imread(file_path)
                        if img.dtype == np.float32 or img.dtype == np.float64:
                            img = np.clip(img, 0, 1)
                        elif img.dtype == np.uint8:
                            img = img.astype(np.float32) / 255.0

                        if img.shape[2] == 3:
                            rgba = np.ones((img.shape[0], img.shape[1], 4))
                            rgba[:, :, :3] = img
                            rgba[:, :, 3] = 1.0
                            img = rgba

                        self.car_sprites[state] = img
                        print(f"Загружен спрайт: {state} - {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Ошибка загрузки {file_path}: {e}")
                        self.car_sprites[state] = self._create_dummy_car_image(state)
                else:
                    self.car_sprites[state] = self._create_dummy_car_image(state)
                    print(f"Создана заглушка для: {state}")

            if not self.car_sprites:
                self._create_all_dummy_sprites()

        except Exception as e:
            print(f"Ошибка загрузки спрайтов: {e}")
            self._create_all_dummy_sprites()

    def _find_sprite(self, *patterns):
        """Ищет файл по паттернам"""
        for pattern in patterns:
            files = glob.glob(os.path.join(self.cars_folder, pattern))
            for ext in ['.png', '.jpg', '.jpeg']:
                ext_files = glob.glob(os.path.join(self.cars_folder, pattern + ext))
                files.extend(ext_files)

            if files:
                return files[0]
        return None

    def _create_all_dummy_sprites(self):
        """Создает все спрайты-заглушки"""
        for state in ['straight', 'left', 'right', 'reverse', 'reverse_left', 'reverse_right']:
            self.car_sprites[state] = self._create_dummy_car_image(state)
        print("Созданы все спрайты-заглушки")

    def _create_dummy_car_image(self, state='straight', size=24):
        """Создает изображение машинки-заглушки"""
        img = np.ones((size, size, 4))

        colors = {
            'straight': '#e74c3c',  # Красный - вперед прямо
            'left': '#3498db',  # Синий - вперед налево
            'right': '#27ae60',  # Зеленый - вперед направо
            'reverse': '#f39c12',  # Оранжевый - назад прямо
            'reverse_left': '#9b59b6',  # Фиолетовый - назад налево
            'reverse_right': '#34495e'  # Темно-синий - назад направо
        }

        color = colors.get(state, '#e74c3c')
        r, g, b = self._hex_to_rgb(color)

        center = size // 2

        for i in range(size):
            for j in range(size):
                dist_from_center = math.hypot(i - center, j - center)

                if (center - 4 <= i <= center + 4 and
                        center - 7 <= j <= center + 7):
                    img[i, j, :3] = [r, g, b]
                    img[i, j, 3] = 1.0
                elif dist_from_center < 8:
                    img[i, j, :3] = [r, g, b]
                    img[i, j, 3] = 1.0

        return img

    def _hex_to_rgb(self, hex_color):
        """Конвертирует hex в RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def _get_car_sprite(self, car):
        """Выбирает спрайт в зависимости от состояния машинки"""
        speed = car['speed']
        wheel_angle = car['wheel_angle']
        is_reversing = car.get('is_reversing', False)

        if is_reversing:
            if wheel_angle < -5:
                return 'reverse_right'
            elif wheel_angle > 5:
                return 'reverse_left'
            else:
                return 'reverse'
        else:
            if wheel_angle < -5:
                return 'right'
            elif wheel_angle > 5:
                return 'left'
            else:
                return 'straight'

    def _rotate_image(self, image, angle):
        """Вращает изображение на заданный угол"""
        try:
            corrected_angle = angle - 90

            if image.dtype == np.float32 or image.dtype == np.float64:
                img_array = (image * 255).astype(np.uint8)
            else:
                img_array = image

            if img_array.shape[2] == 4:
                pil_img = Image.fromarray(img_array, 'RGBA')
            else:
                pil_img = Image.fromarray(img_array, 'RGB')

            rotated_img = pil_img.rotate(corrected_angle, expand=True, resample=Image.BICUBIC)
            rotated_array = np.array(rotated_img)

            if image.dtype == np.float32 or image.dtype == np.float64:
                rotated_array = rotated_array.astype(np.float32) / 255.0

            return rotated_array

        except Exception as e:
            print(f"Ошибка вращения изображения: {e}")
            return image

    def _load_tracks_list(self):
        """Загружает список треков"""
        try:
            tracks = [f for f in os.listdir(self.tracks_folder)
                      if f.endswith('.json')]
        except:
            tracks = []

        self.ax_list.clear()
        self.ax_list.set_facecolor(self.COLORS['panel'])
        self.ax_list.set_xlim(0, 1)
        self.ax_list.set_ylim(0, 1)
        self.ax_list.set_xticks([])
        self.ax_list.set_yticks([])

        self.ax_list.text(0.5, 0.95, 'ВЫБЕРИТЕ ТРЕК',
                          fontsize=10, fontweight='bold', ha='center',
                          color=self.COLORS['text'])

        if not tracks:
            self.ax_list.text(0.5, 0.5, 'Нет треков',
                              ha='center', color=self.COLORS['grid'], fontsize=9)
            return

        for i, track in enumerate(tracks[::-1]):
            y_pos = 0.85 - i * 0.06
            if y_pos < 0.05:
                break

            track_name = track.replace('.json', '')
            color = self.COLORS['accent'] if track == self.selected_track else self.COLORS['text']

            text = self.ax_list.text(0.1, y_pos, f"• {track_name}",
                                     fontsize=8, color=color)
            text.set_picker(True)

    def _on_track_select(self, event):
        """Обработчик выбора трека"""
        if hasattr(event, 'artist'):
            tracks = [f for f in os.listdir(self.tracks_folder)
                      if f.endswith('.json')]
            texts = [child for child in self.ax_list.get_children()
                     if hasattr(child, 'get_text') and child.get_text().startswith('•')]

            if event.artist in texts:
                idx = texts.index(event.artist)
                self.selected_track = tracks[::-1][idx]
                self._load_track_data()
                self._load_tracks_list()

    def _load_track_data(self):
        """Загружает данные трека"""
        if not self.selected_track:
            return

        try:
            filename = os.path.join(self.tracks_folder, self.selected_track)
            with open(filename, 'r', encoding='utf-8') as f:
                self.track_data = json.load(f)

            # Определяем стартовую линию (первая точка трека)
            center_x = np.array(self.track_data['center_line'][0])
            center_y = np.array(self.track_data['center_line'][1])
            self.start_line_position = (center_x[0], center_y[0])

            self._initialize_cars()
            self._update_display()
            print(f"Трек '{self.selected_track}' загружен!")

        except Exception as e:
            print(f"Ошибка загрузки трека: {e}")

    def _get_car_colors(self):
        """Возвращает список цветов для машинок"""
        base_colors = [
            '#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6',
            '#e67e22', '#34495e', '#1abc9c', '#d35400', '#c0392b',
            '#16a085', '#8e44ad', '#2c3e50', '#f1c40f', '#e74c3c',
            '#2980b9', '#27ae60', '#d35400', '#8e44ad', '#2c3e50'
        ]

        if SIMULATION_CONFIG['CAR_COUNT'] > len(base_colors):
            additional_colors = []
            for i in range(SIMULATION_CONFIG['CAR_COUNT'] - len(base_colors)):
                r = random.randint(100, 255)
                g = random.randint(100, 255)
                b = random.randint(100, 255)
                additional_colors.append(f'#{r:02x}{g:02x}{b:02x}')
            return base_colors + additional_colors

        return base_colors[:SIMULATION_CONFIG['CAR_COUNT']]

    def _initialize_cars(self):
        """Инициализирует машинки на стартовой линии"""
        if not self.track_data:
            return

        center_x = np.array(self.track_data['center_line'][0])
        center_y = np.array(self.track_data['center_line'][1])

        self.car_objects = []
        self.car_artists = []

        # Получаем цвета для машинок
        colors = self._get_car_colors()

        for i in range(SIMULATION_CONFIG['CAR_COUNT']):
            # Случайная позиция на стартовой линии с небольшим разбросом
            start_x, start_y = self.start_line_position
            pos = (start_x + random.uniform(-5, 5), start_y + random.uniform(-5, 5))

            # Угол направления вдоль трека
            next_idx = 10 % len(center_x)
            dx = center_x[next_idx] - center_x[0]
            dy = center_y[next_idx] - center_y[0]
            track_angle = math.degrees(math.atan2(dy, dx))

            self.car_objects.append({
                'position': pos,
                'body_angle': track_angle,
                'wheel_angle': 0.0,
                'target_wheel_angle': 0.0,
                'speed': 0.0,  # Начинаем с нулевой скорости
                'max_speed': SIMULATION_CONFIG['MAX_SPEED'],
                'max_reverse_speed': SIMULATION_CONFIG['MAX_REVERSE_SPEED'],
                'acceleration': 80.0,  # Увеличено ускорение
                'braking_power': 100.0,  # Увеличено торможение
                'reverse_acceleration': 60.0,  # Ускорение назад
                'friction': 1.5,  # Уменьшено трение для плавности
                'drift_factor': 0.2,
                'progress': 0.0,
                'max_steering_angle': 40.0,  # Увеличен угол поворота
                'steering_speed': 120.0,  # Уменьшена скорость поворота для плавности
                'sprite_state': 'straight',
                'color': colors[i],
                'size_scale': 0.6,
                'lap_count': 0,
                'last_checkpoint': 0,
                'fitness': 0,
                'completed_lap': False,
                'off_track': False,
                'off_track_time': 0,
                'neural_network': self.ga.population[i],
                'checkpoints_passed': 0,
                'total_checkpoints': len(center_x) // 10,
                'lap_start_time': 0,
                'has_started_lap': True,
                'lap_times': [],
                'is_drifting': False,
                'is_reversing': False,  # Новый флаг для движения назад
                'drift_angle': 0.0,
                'stuck_time': 0,
                'last_position': pos,
                'smooth_steering': 0.0,  # Для плавного поворота
                'smooth_throttle': 0.0  # Для плавного ускорения
            })

        self.generation_start_time = time.time()
        self.last_update_time = time.time()

    def _get_car_inputs(self, car, center_x, center_y):
        """Получает входные данные для нейронной сети машинки"""
        # Находим ближайшую точку на треке
        current_pos = np.array(car['position'])
        center_points = np.column_stack((center_x, center_y))

        distances = np.linalg.norm(center_points - current_pos, axis=1)
        nearest_idx = np.argmin(distances)

        # Смотрим вперед по треку
        look_ahead = 25
        target_idx = (nearest_idx + look_ahead) % len(center_x)

        # Вектор до целевой точки
        target_dx = center_x[target_idx] - car['position'][0]
        target_dy = center_y[target_idx] - car['position'][1]
        target_angle = math.degrees(math.atan2(target_dy, target_dx))

        # Разница углов
        angle_diff = (target_angle - car['body_angle'] + 180) % 360 - 180

        # Дистанции до границ
        inner_x = np.array(self.track_data['inner_boundary'][0])
        inner_y = np.array(self.track_data['inner_boundary'][1])
        outer_x = np.array(self.track_data['outer_boundary'][0])
        outer_y = np.array(self.track_data['outer_boundary'][1])

        # Ближайшие точки на границах
        inner_distances = [math.hypot(car['position'][0] - inner_x[i],
                                      car['position'][1] - inner_y[i])
                           for i in range(len(inner_x))]
        outer_distances = [math.hypot(car['position'][0] - outer_x[i],
                                      car['position'][1] - outer_y[i])
                           for i in range(len(outer_x))]

        inner_dist = min(inner_distances)
        outer_dist = min(outer_distances)

        # Текущая скорость и состояние
        speed_ratio = car['speed'] / car['max_speed']
        is_reversing = 1.0 if car['is_reversing'] else -1.0
        is_drifting = 1.0 if car['is_drifting'] else -1.0

        return np.array([[
            inner_dist / 40.0,
            outer_dist / 40.0,
            angle_diff / 90.0,
            speed_ratio,
            math.sin(math.radians(car['body_angle'])),
            math.cos(math.radians(car['body_angle'])),
            is_drifting,
            is_reversing
        ]])

    def _check_start_line_crossing(self, car, center_x, center_y):
        """Проверяет пересечение стартовой линии"""
        if not self.start_line_position:
            return

        dist_to_start = math.hypot(car['position'][0] - self.start_line_position[0],
                                   car['position'][1] - self.start_line_position[1])

        if dist_to_start < 20 and abs(car['speed']) > 2.0:  # Учитываем движение назад
            if not car['has_started_lap']:
                car['has_started_lap'] = True
                car['lap_start_time'] = time.time()
                print(f"Машинка {self.car_objects.index(car)} начала круг!")
            else:
                # Проверяем, что машинка движется в правильном направлении
                current_pos = np.array(car['position'])
                center_points = np.column_stack((center_x, center_y))
                distances = np.linalg.norm(center_points - current_pos, axis=1)
                progress = np.argmin(distances)

                if progress > 50:
                    current_time = time.time()
                    lap_time = current_time - car['lap_start_time']

                    if lap_time > SIMULATION_CONFIG['MIN_LAP_TIME']:
                        car['lap_count'] += 1
                        car['lap_times'].append(lap_time)
                        car['lap_start_time'] = current_time

                        if lap_time < self.ga.best_time:
                            self.ga.best_time = lap_time
                            print(f"НОВОЕ ЛУЧШЕЕ ВРЕМЯ: {lap_time:.2f} сек!")

                        car['fitness'] += 2000 / lap_time
                        car['completed_lap'] = True

                        car_idx = self.car_objects.index(car)
                        self.ga.completion_times[car_idx] = lap_time
                        print(f"Машинка {car_idx} завершила круг за {lap_time:.2f} сек!")

    def _update_car_physics(self, car, center_x, center_y, dt):
        """Обновляет физику движения машинки с расширенным управлением"""
        if car['completed_lap']:
            return

        # Применяем масштаб времени
        dt *= SIMULATION_CONFIG['TIME_SCALE']

        # Получаем входные данные для нейронной сети
        inputs = self._get_car_inputs(car, center_x, center_y)
        outputs = car['neural_network'].forward(inputs)[0]

        # Управление: [ускорение/торможение, поворот, реверс]
        throttle = (outputs[0] + 1) / 2  # [0, 1] - газ
        steering = outputs[1]  # [-1, 1] - руль
        reverse = (outputs[2] + 1) / 2  # [0, 1] - реверс

        # Определяем направление движения
        should_reverse = reverse > 0.6
        car['is_reversing'] = should_reverse

        # Плавное управление ускорением
        target_throttle = throttle if not should_reverse else -reverse
        car['smooth_throttle'] += (target_throttle - car['smooth_throttle']) * 0.1

        # Управление в реальном времени с учетом направления
        if not should_reverse:
            # Движение вперед
            if car['smooth_throttle'] > 0.1:
                acceleration_power = (car['smooth_throttle'] - 0.1) * 1.5
                car['speed'] += car['acceleration'] * acceleration_power * dt
            else:
                # Торможение или замедление
                car['speed'] -= car['braking_power'] * 0.5 * dt
        else:
            # Движение назад
            if car['smooth_throttle'] < -0.1:
                reverse_power = (-car['smooth_throttle'] - 0.1) * 1.5
                car['speed'] -= car['reverse_acceleration'] * reverse_power * dt
            else:
                # Торможение или замедление
                car['speed'] += car['braking_power'] * 0.5 * dt

        # Ограничение скорости в зависимости от направления
        if car['speed'] > 0:
            car['speed'] = min(car['max_speed'], car['speed'])
        else:
            car['speed'] = max(-car['max_reverse_speed'], car['speed'])

        # Трение (пропорционально скорости)
        speed_sign = 1 if car['speed'] >= 0 else -1
        friction = car['friction'] * (1 + abs(car['speed']) / 10.0)  # Увеличиваем трение на высокой скорости
        car['speed'] -= car['speed'] * friction * dt

        # Плавное управление поворотом
        car['smooth_steering'] += (steering - car['smooth_steering']) * 0.2
        car['target_wheel_angle'] = car['smooth_steering'] * car['max_steering_angle']

        # Плавный поворот колес
        angle_diff_wheel = car['target_wheel_angle'] - car['wheel_angle']
        max_angle_change = car['steering_speed'] * dt
        car['wheel_angle'] += max(-max_angle_change, min(max_angle_change, angle_diff_wheel))

        # Физика заноса (только при движении вперед)
        current_speed = abs(car['speed'])
        if current_speed > 15.0 and abs(car['wheel_angle']) > 10 and not should_reverse:
            car['is_drifting'] = True
            drift_intensity = min(1.0, (current_speed - 15.0) / 30.0 * abs(car['wheel_angle']) / 30.0)
            car['drift_angle'] = car['wheel_angle'] * drift_intensity * 0.6
        else:
            car['is_drifting'] = False
            car['drift_angle'] *= 0.8  # Плавный выход из заноса

        # Обновляем угол кузова с учетом заноса и направления
        effective_angle = car['body_angle'] + car['drift_angle']

        # Поворот зависит от направления движения
        turn_direction = 1 if car['speed'] >= 0 else -1
        turn_rate = car['wheel_angle'] * abs(car['speed']) * 0.015 * dt * turn_direction
        car['body_angle'] = (car['body_angle'] + turn_rate) % 360

        # Обновляем позицию в реальном времени
        rad_angle = math.radians(effective_angle)
        dx = math.cos(rad_angle) * car['speed'] * dt
        dy = math.sin(rad_angle) * car['speed'] * dt

        new_x = car['position'][0] + dx
        new_y = car['position'][1] + dy

        # Сохраняем предыдущую позицию для определения движения
        car['last_position'] = car['position']
        car['position'] = (new_x, new_y)

        # Проверка нахождения на треке
        self._check_track_boundaries(car)

        # Обновляем прогресс
        current_pos = np.array(car['position'])
        center_points = np.column_stack((center_x, center_y))
        distances = np.linalg.norm(center_points - current_pos, axis=1)
        car['progress'] = np.argmin(distances)

        # Проверка прохождения чекпоинтов (только при движении вперед)
        if car['speed'] > 2.0:
            checkpoint_idx = int(car['progress'] / 20)
            if checkpoint_idx > car['last_checkpoint']:
                car['checkpoints_passed'] += 1
                car['last_checkpoint'] = checkpoint_idx
                car['fitness'] += 20

        # Проверка пересечения стартовой линии
        self._check_start_line_crossing(car, center_x, center_y)

        # Проверка застревания
        movement = math.hypot(car['position'][0] - car['last_position'][0],
                              car['position'][1] - car['last_position'][1])
        if abs(car['speed']) < 2.0 and movement < 0.5:
            car['stuck_time'] += dt
            if car['stuck_time'] > 5.0:  # Застряла более 5 секунд
                car['fitness'] -= 30
                car['completed_lap'] = True
        else:
            car['stuck_time'] = max(0, car['stuck_time'] - dt * 0.5)

        # Выбираем спрайт
        car['sprite_state'] = self._get_car_sprite(car)

        # Награда за скорость и штраф за нахождение вне трека
        if not car['off_track']:
            # Награда за движение вперед, штраф за движение назад
            if car['speed'] > 5.0:
                car['fitness'] += car['speed'] * 0.02 * dt
            elif car['speed'] < -2.0:
                car['fitness'] -= 0.1 * dt
        else:
            car['fitness'] -= 0.8 * dt
            # Сильнее замедление вне трека
            car['speed'] *= 0.6

    def _check_track_boundaries(self, car):
        """Проверяет, находится ли машинка на треке"""
        if not self.track_data:
            return

        inner_x = np.array(self.track_data['inner_boundary'][0])
        inner_y = np.array(self.track_data['inner_boundary'][1])
        outer_x = np.array(self.track_data['outer_boundary'][0])
        outer_y = np.array(self.track_data['outer_boundary'][1])

        # Проверяем расстояние до границ
        min_inner_dist = min(math.hypot(car['position'][0] - inner_x[i],
                                        car['position'][1] - inner_y[i])
                             for i in range(len(inner_x)))
        min_outer_dist = min(math.hypot(car['position'][0] - outer_x[i],
                                        car['position'][1] - outer_y[i])
                             for i in range(len(outer_x)))

        track_width = self.track_data.get('width', 20)
        boundary_margin = track_width * 0.4

        if min_inner_dist < boundary_margin or min_outer_dist < boundary_margin:
            car['off_track'] = True
            car['off_track_time'] += 0.1  # Учитываем время вне трека
        else:
            car['off_track'] = False

    def _calculate_fitness(self, car):
        """Вычисляет фитнес-функцию для машинки"""
        fitness = car['fitness']

        # Награда за пройденные чекпоинты
        fitness += car['checkpoints_passed'] * 15

        # Штраф за время вне трека
        fitness -= car['off_track_time'] * 3

        # Награда за скорость вперед, штраф за движение назад
        if car['speed'] > 0:
            fitness += car['speed'] * 0.2
        else:
            fitness -= abs(car['speed']) * 0.1

        # Большая награда за завершение круга с хорошим временем
        if car['completed_lap'] and car['lap_times']:
            best_lap = min(car['lap_times'])
            fitness += 2000 / best_lap

        # Награда за общий прогресс
        fitness += car['progress'] * 0.02

        return max(0, fitness)

    def _update_display(self):
        """Обновляет отображение"""
        self.ax_main.clear()
        self.ax_main.set_xlim(-200, 200)
        self.ax_main.set_ylim(-200, 200)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3, color=self.COLORS['grid'])
        self.ax_main.set_facecolor(self.COLORS['bg'])

        # Отрисовка трека
        if self.track_data:
            inner_x = np.array(self.track_data['inner_boundary'][0])
            inner_y = np.array(self.track_data['inner_boundary'][1])
            outer_x = np.array(self.track_data['outer_boundary'][0])
            outer_y = np.array(self.track_data['outer_boundary'][1])

            road_x = np.concatenate([inner_x, outer_x[::-1]])
            road_y = np.concatenate([inner_y, outer_y[::-1]])
            self.ax_main.fill(road_x, road_y, color='#4a6572', alpha=0.8, zorder=1)

            self.ax_main.plot(inner_x, inner_y, color=self.COLORS['accent'],
                              linewidth=2, zorder=2)
            self.ax_main.plot(outer_x, outer_y, color=self.COLORS['accent'],
                              linewidth=2, zorder=2)

            # Отрисовка стартовой линии
            if self.start_line_position:
                start_x, start_y = self.start_line_position
                inner_idx = np.argmin([math.hypot(start_x - inner_x[i], start_y - inner_y[i])
                                       for i in range(len(inner_x))])
                outer_idx = np.argmin([math.hypot(start_x - outer_x[i], start_y - outer_y[i])
                                       for i in range(len(outer_x))])

                self.ax_main.plot([inner_x[inner_idx], outer_x[outer_idx]],
                                  [inner_y[inner_idx], outer_y[outer_idx]],
                                  color='white', linewidth=3, linestyle='-', zorder=3)

            # Отрисовка машинок
            self._draw_cars()

        # Статусная информация
        current_time = time.time() - self.generation_start_time if self.is_running else 0
        scaled_time = current_time * SIMULATION_CONFIG['TIME_SCALE']

        status = f"Поколение: {self.ga.generation}\n"
        status += f"Трек: {self.selected_track or 'не выбран'}\n"
        status += f"Время: {scaled_time:.1f} сек (x{SIMULATION_CONFIG['TIME_SCALE']})\n"
        status += f"Лучшее время: {self.ga.best_time:.1f} сек\n"
        status += f"Круги: {sum(1 for car in self.car_objects if car['completed_lap'])}/{SIMULATION_CONFIG['CAR_COUNT']}\n"
        status += f"FPS: {self.fps:.1f}\n"
        status += f"Машинок: {SIMULATION_CONFIG['CAR_COUNT']}"

        self.ax_main.text(0.02, 0.98, status, transform=self.ax_main.transAxes,
                          fontsize=9, color=self.COLORS['text'], va='top',
                          bbox=dict(boxstyle='round', facecolor=self.COLORS['panel'], alpha=0.8))

        self.fig.canvas.draw_idle()

    def _draw_cars(self):
        """Отрисовывает машинки на треке"""
        for artist in self.car_artists:
            try:
                artist.remove()
            except:
                pass
        self.car_artists = []

        track_width = self.track_data.get('width', 20) if self.track_data else 20
        base_zoom = 0.1
        width_scale = 20 / track_width
        zoom_level = base_zoom * width_scale

        for i, car in enumerate(self.car_objects):
            try:
                base_sprite = self.car_sprites.get(car['sprite_state'],
                                                   self.car_sprites.get('straight'))
                rotated_sprite = self._rotate_image(base_sprite, car['body_angle'])

                imagebox = OffsetImage(rotated_sprite, zoom=zoom_level * car['size_scale'])
                ab = AnnotationBbox(imagebox, car['position'], frameon=False)
                self.ax_main.add_artist(ab)
                self.car_artists.append(ab)

                # Отрисовка направления (стрелка показывает направление движения)
                length = 10
                rad_angle = math.radians(car['body_angle'])

                # Если движется назад, стрелка в противоположную сторону
                direction_multiplier = -1 if car['is_reversing'] else 1
                end_x = car['position'][0] + math.cos(rad_angle) * length * direction_multiplier
                end_y = car['position'][1] + math.sin(rad_angle) * length * direction_multiplier

                direction_color = 'red' if car['is_reversing'] else car['color']
                direction_line = self.ax_main.plot([car['position'][0], end_x],
                                                   [car['position'][1], end_y],
                                                   color=direction_color, linewidth=2, zorder=5, alpha=0.8)
                self.car_artists.extend(direction_line)

                # Индикатор заноса
                if car['is_drifting']:
                    drift_length = 8
                    drift_rad = math.radians(car['body_angle'] + 90)
                    drift_x = car['position'][0] + math.cos(drift_rad) * drift_length
                    drift_y = car['position'][1] + math.sin(drift_rad) * drift_length

                    drift_line = self.ax_main.plot([car['position'][0], drift_x],
                                                   [car['position'][1], drift_y],
                                                   color='yellow', linewidth=2, zorder=5, alpha=0.9)
                    self.car_artists.extend(drift_line)

                # Подпись с номером машинки, скоростью и направлением
                direction_symbol = "R" if car['is_reversing'] else "F"
                info_text = f"{i}({abs(car['speed']):.0f}{direction_symbol})"
                if car['off_track']:
                    info_text += "!"
                text = self.ax_main.text(car['position'][0] + 8, car['position'][1] + 8,
                                         info_text, fontsize=7, color=car['color'],
                                         weight='bold', zorder=6)
                self.car_artists.append(text)

            except Exception as e:
                print(f"Ошибка отрисовки машинки {i}: {e}")
                scatter = self.ax_main.scatter(car['position'][0], car['position'][1],
                                               c=car['color'], s=20, zorder=5)
                self.car_artists.append(scatter)

    def _start_simulation(self, event=None):
        """Запускает симуляцию"""
        print("Запуск симуляции...")
        if not self.track_data:
            self._show_message("Сначала выберите трек!")
            return

        self.is_running = True
        self.generation_start_time = time.time()
        self.last_update_time = time.time()
        self.frame_count = 0
        self._run_simulation_loop()

    def _stop_simulation(self, event=None):
        """Останавливает симуляцию"""
        print("Остановка симуляции...")
        self.is_running = False

    def _reset_simulation(self, event=None):
        """Сбрасывает симуляцию"""
        print("Сброс симуляции...")
        self.is_running = False
        self.simulation_step = 0
        self.ga.best_time = float('inf')
        self.ga.generation = 1
        self.ga = GeneticAlgorithm(population_size=SIMULATION_CONFIG['CAR_COUNT'])
        if self.track_data:
            self._initialize_cars()
        self._update_display()

    def _check_generation_completion(self):
        """Проверяет завершение поколения и запускает эволюцию"""
        current_time = time.time() - self.generation_start_time
        scaled_time = current_time * SIMULATION_CONFIG['TIME_SCALE']

        # Условия завершения поколения:
        all_completed = all(car['completed_lap'] for car in self.car_objects)
        all_stuck = all(abs(car['speed']) < 3.0 for car in self.car_objects) and scaled_time > 15

        max_generation_time = SIMULATION_CONFIG['MAX_GENERATION_TIME'] / SIMULATION_CONFIG['TIME_SCALE']

        if all_completed or current_time > max_generation_time or all_stuck:
            print(f"Завершение поколения {self.ga.generation}. Время: {scaled_time:.1f} сек")

            # Вычисляем фитнес для всех машинок
            for i, car in enumerate(self.car_objects):
                self.ga.fitness_scores[i] = self._calculate_fitness(car)
                direction = "R" if car['is_reversing'] else "F"
                print(f"Машинка {i}: фитнес = {self.ga.fitness_scores[i]:.1f}, "
                      f"круги = {car['lap_count']}, скорость = {abs(car['speed']):.1f}{direction}")

            # Запускаем эволюцию
            self.ga.evolve()
            print(f"Переход к поколению {self.ga.generation}")

            # Перезапускаем симуляцию с новым поколением
            self._initialize_cars()
            return True
        return False

    def _run_simulation_loop(self):
        """Основной цикл симуляции в реальном времени"""
        if not self.is_running or not self.track_data:
            return

        center_x = np.array(self.track_data['center_line'][0])
        center_y = np.array(self.track_data['center_line'][1])

        last_fps_time = time.time()

        while self.is_running:
            try:
                current_time = time.time()
                dt = current_time - self.last_update_time

                # Ограничиваем dt для стабильности
                dt = min(dt, 0.033)  # Максимум 30 FPS

                # Обновляем FPS
                self.frame_count += 1
                if current_time - last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - last_fps_time)
                    self.frame_count = 0
                    last_fps_time = current_time

                # Проверяем завершение поколения
                if self._check_generation_completion():
                    self.last_update_time = time.time()
                    continue

                # Обновление физики для каждой машинки
                for car in self.car_objects:
                    if not car['completed_lap']:
                        self._update_car_physics(car, center_x, center_y, dt)

                self.simulation_step += 1
                self.last_update_time = current_time

                # Обновляем отображение
                self._update_display()

                # Небольшая пауза для стабильности
                plt.pause(0.001)

            except Exception as e:
                print(f"Ошибка в симуляции: {e}")
                import traceback
                traceback.print_exc()
                break

    def _show_message(self, text):
        """Показывает сообщение"""
        self.ax_main.set_title(text, fontsize=12, color=self.COLORS['accent'], pad=20)
        self.fig.canvas.draw_idle()
        plt.pause(1.5)
        self.ax_main.set_title(f"Симуляция гонки с ИИ - {SIMULATION_CONFIG['CAR_COUNT']} машинок",
                               fontsize=12, color=self.COLORS['text'], pad=20)


# Запуск приложения
if __name__ == "__main__":
    print("Запуск Гоночного Трекера...")
    print(f"Количество машинок в симуляции: {SIMULATION_CONFIG['CAR_COUNT']}")
    print(f"Масштаб времени: x{SIMULATION_CONFIG['TIME_SCALE']}")
    app = RacingGame()
    app.show_main_menu()

