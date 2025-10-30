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
                                 f'Сохранено треков: {track_count}\nДоступно машинок: {car_count}',
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
        self.track_width = 20
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

        # Основная область рисования
        self.ax_main = plt.axes([0.25, 0.1, 0.7, 0.8])
        self.ax_main.set_xlim(-200, 200)
        self.ax_main.set_ylim(-200, 200)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3, color=self.COLORS['grid'])
        self.ax_main.set_facecolor(self.COLORS['bg'])
        self.ax_main.set_title("Редактор треков - добавляйте точки щелчками",
                               fontsize=12, color=self.COLORS['text'], pad=20)

        # Панель списка треков
        self.ax_list = plt.axes([0.02, 0.1, 0.2, 0.8])
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
        # Основные кнопки
        btn_clear_ax = plt.axes([0.25, 0.02, 0.1, 0.05])
        self.btn_clear = Button(btn_clear_ax, 'ОЧИСТИТЬ',
                                color=self.COLORS['accent'], hovercolor='#c73550')
        self.btn_clear.on_clicked(self._clear_track)

        btn_save_ax = plt.axes([0.36, 0.02, 0.1, 0.05])
        self.btn_save = Button(btn_save_ax, 'СОХРАНИТЬ',
                               color=self.COLORS['accent'], hovercolor='#c73550')
        self.btn_save.on_clicked(self._save_track)

        btn_load_ax = plt.axes([0.47, 0.02, 0.1, 0.05])
        self.btn_load = Button(btn_load_ax, 'ЗАГРУЗИТЬ',
                               color=self.COLORS['accent'], hovercolor='#c73550')
        self.btn_load.on_clicked(self._load_track)

        btn_delete_ax = plt.axes([0.58, 0.02, 0.1, 0.05])
        self.btn_delete = Button(btn_delete_ax, 'УДАЛИТЬ',
                                 color=self.COLORS['accent'], hovercolor='#c73550')
        self.btn_delete.on_clicked(self._delete_track)

        # Слайдер ширины трека
        slider_ax = plt.axes([0.69, 0.02, 0.15, 0.05])
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

        # Заголовок
        self.ax_list.text(0.5, 0.95, 'СОХРАНЕННЫЕ ТРЕКИ',
                          fontsize=12, fontweight='bold', ha='center',
                          color=self.COLORS['text'])

        if not tracks:
            self.ax_list.text(0.5, 0.5, 'Нет сохраненных треков',
                              ha='center', color=self.COLORS['grid'], fontsize=10)
            return

        # Список треков
        for i, track in enumerate(tracks[::-1]):
            y_pos = 0.85 - i * 0.07
            if y_pos < 0.05:
                break

            track_name = track.replace('.json', '')
            color = self.COLORS['accent'] if track == self.selected_track else self.COLORS['text']

            text = self.ax_list.text(0.1, y_pos, f"• {track_name}",
                                     fontsize=9, color=color)
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


class RaceSimulation:
    """Симулятор гонки с улучшенной физикой движения"""

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
        self.car_artists = []  # Для хранения художников машинок

        self.is_running = False
        self.simulation_step = 0

        # Сохраняем ссылки на кнопки
        self.btn_start = None
        self.btn_stop = None
        self.btn_reset = None

    def run(self):
        """Запускает симуляцию"""
        self._setup_ui()
        self._load_tracks_list()
        self._load_car_sprites()
        self._update_display()
        plt.show()

    def _setup_ui(self):
        """Настраивает интерфейс симуляции"""
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor(self.COLORS['bg'])

        # Основная область
        self.ax_main = plt.axes([0.25, 0.1, 0.7, 0.8])
        self.ax_main.set_xlim(-200, 200)
        self.ax_main.set_ylim(-200, 200)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3, color=self.COLORS['grid'])
        self.ax_main.set_facecolor(self.COLORS['bg'])
        self.ax_main.set_title("Симуляция гонки - выберите трек и запустите",
                               fontsize=12, color=self.COLORS['text'], pad=20)

        # Панель списка
        self.ax_list = plt.axes([0.02, 0.1, 0.2, 0.8])
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
        btn_start_ax = plt.axes([0.25, 0.02, 0.12, 0.06])
        self.btn_start = Button(btn_start_ax, 'СТАРТ',
                                color='#27ae60', hovercolor='#229954')
        self.btn_start.on_clicked(self._start_simulation)

        btn_stop_ax = plt.axes([0.38, 0.02, 0.12, 0.06])
        self.btn_stop = Button(btn_stop_ax, 'СТОП',
                               color='#e74c3c', hovercolor='#c0392b')
        self.btn_stop.on_clicked(self._stop_simulation)

        btn_reset_ax = plt.axes([0.51, 0.02, 0.12, 0.06])
        self.btn_reset = Button(btn_reset_ax, 'СБРОС',
                                color='#f39c12', hovercolor='#d68910')
        self.btn_reset.on_clicked(self._reset_simulation)

    def _load_car_sprites(self):
        """Загружает спрайты машинок"""
        try:
            # Ищем файлы с определенными паттернами названий
            sprite_files = {
                'straight': self._find_sprite('*straight*', '*forward*', '*center*'),
                'left': self._find_sprite('*left*', '*turn_left*'),
                'right': self._find_sprite('*right*', '*turn_right*')
            }

            print(f"Найдены спрайты: {sprite_files}")

            # Загружаем спрайты
            for state, file_path in sprite_files.items():
                if file_path:
                    try:
                        img = mpimg.imread(file_path)
                        # Нормализуем изображение чтобы избежать проблем с диапазоном
                        if img.dtype == np.float32 or img.dtype == np.float64:
                            img = np.clip(img, 0, 1)
                        elif img.dtype == np.uint8:
                            img = img.astype(np.float32) / 255.0

                        # Конвертация в RGBA если нужно
                        if img.shape[2] == 3:
                            rgba = np.ones((img.shape[0], img.shape[1], 4))
                            rgba[:, :, :3] = img
                            rgba[:, :, 3] = 1.0  # Полная непрозрачность
                            img = rgba

                        self.car_sprites[state] = img
                        print(f"Загружен спрайт: {state} - {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Ошибка загрузки {file_path}: {e}")
                        self.car_sprites[state] = self._create_dummy_car_image(state)
                else:
                    # Создаем заглушку если спрайт не найден
                    self.car_sprites[state] = self._create_dummy_car_image(state)
                    print(f"Создана заглушка для: {state}")

            # Если не найдено ни одного спрайта, создаем все заглушки
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
                return files[0]  # Возвращаем первый найденный файл
        return None

    def _create_all_dummy_sprites(self):
        """Создает все спрайты-заглушки"""
        for state in ['straight', 'left', 'right']:
            self.car_sprites[state] = self._create_dummy_car_image(state)
        print("Созданы все спрайты-заглушки")

    def _create_dummy_car_image(self, state='straight', size=64):
        """Создает изображение машинки-заглушки с учетом состояния"""
        img = np.ones((size, size, 4))  # Белый фон с прозрачностью

        # Цвета для разных состояний
        colors = {
            'straight': '#e74c3c',  # Красный
            'left': '#3498db',  # Синий
            'right': '#27ae60'  # Зеленый
        }

        color = colors.get(state, '#e74c3c')
        r, g, b = self._hex_to_rgb(color)

        # Рисуем машинку с учетом состояния поворота
        center = size // 2

        for i in range(size):
            for j in range(size):
                # Кузов машинки (прямоугольник с закруглениями)
                dist_from_center = math.hypot(i - center, j - center)

                # Основной кузов
                if (center - 12 <= i <= center + 12 and
                        center - 18 <= j <= center + 18):
                    img[i, j, :3] = [r, g, b]
                    img[i, j, 3] = 1.0

                # Закругленный кузов
                elif dist_from_center < 20:
                    img[i, j, :3] = [r, g, b]
                    img[i, j, 3] = 1.0

        return img

    def _hex_to_rgb(self, hex_color):
        """Конвертирует hex в RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def _get_car_sprite(self, steering_angle):
        """Выбирает спрайт в зависимости от угла поворота колес с учетом вращения на 90 градусов"""
        if steering_angle < -10:
            return 'right'  # Теперь это правый поворот
        elif steering_angle > 10:
            return 'left'  # Теперь это левый поворот
        else:
            return 'straight'

    def _rotate_image(self, image, angle):
        """Вращает изображение на заданный угол с корректировкой начальной ориентации"""
        try:
            # Корректируем угол на 90 градусов для правильной начальной ориентации
            corrected_angle = angle - 90

            # Конвертируем numpy array в PIL Image
            if image.dtype == np.float32 or image.dtype == np.float64:
                img_array = (image * 255).astype(np.uint8)
            else:
                img_array = image

            # Создаем PIL Image
            if img_array.shape[2] == 4:  # RGBA
                pil_img = Image.fromarray(img_array, 'RGBA')
            else:  # RGB
                pil_img = Image.fromarray(img_array, 'RGB')

            # Вращаем изображение
            rotated_img = pil_img.rotate(corrected_angle, expand=True, resample=Image.BICUBIC)

            # Конвертируем обратно в numpy array
            rotated_array = np.array(rotated_img)

            # Нормализуем обратно в [0, 1] если нужно
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
                          fontsize=12, fontweight='bold', ha='center',
                          color=self.COLORS['text'])

        if not tracks:
            self.ax_list.text(0.5, 0.5, 'Нет треков',
                              ha='center', color=self.COLORS['grid'], fontsize=10)
            return

        for i, track in enumerate(tracks[::-1]):
            y_pos = 0.85 - i * 0.07
            if y_pos < 0.05:
                break

            track_name = track.replace('.json', '')
            color = self.COLORS['accent'] if track == self.selected_track else self.COLORS['text']

            text = self.ax_list.text(0.1, y_pos, f"• {track_name}",
                                     fontsize=9, color=color)
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

            self._initialize_cars()
            self._update_display()
            print(f"Трек '{self.selected_track}' загружен!")

        except Exception as e:
            print(f"Ошибка загрузки трека: {e}")

    def _initialize_cars(self):
        """Инициализирует машинки на треке с улучшенной физикой"""
        if not self.track_data:
            return

        center_x = np.array(self.track_data['center_line'][0])
        center_y = np.array(self.track_data['center_line'][1])

        self.car_objects = []
        self.car_artists = []

        # Создаем 3 машинки с разными характеристиками
        for i in range(3):
            # Равномерное распределение по треку
            idx = int(len(center_x) * i / 3)
            pos = (center_x[idx], center_y[idx])

            # Угол направления вдоль трека
            next_idx = (idx + 5) % len(center_x)
            dx = center_x[next_idx] - center_x[idx]
            dy = center_y[next_idx] - center_y[idx]
            track_angle = math.degrees(math.atan2(dy, dx))

            self.car_objects.append({
                'position': pos,
                'body_angle': track_angle,  # Угол кузова машины
                'wheel_angle': 0.0,  # Угол поворота колес
                'target_wheel_angle': 0.0,  # Целевой угол колес
                'speed': 0.8 + i * 0.3,  # Разные максимальные скорости
                'progress': float(idx),
                'max_steering_angle': 25.0,  # Максимальный угол поворота колес
                'steering_speed': 3.0,  # Скорость поворота колес
                'sprite_state': 'straight',
                'color': ['#e74c3c', '#3498db', '#27ae60'][i]  # Разные цвета
            })

    def _update_car_physics(self, car, center_x, center_y, dt=0.05):
        """Обновляет физику движения машинки"""
        # Получаем текущую позицию на треке
        current_idx = int(car['progress']) % len(center_x)
        look_ahead = min(10, len(center_x) // 10)
        target_idx = (current_idx + look_ahead) % len(center_x)

        # Вектор направления к целевой точке
        target_dx = center_x[target_idx] - center_x[current_idx]
        target_dy = center_y[target_idx] - center_y[current_idx]
        target_angle = math.degrees(math.atan2(target_dy, target_dx))

        # Вычисляем разницу углов для определения необходимого поворота
        angle_diff = (target_angle - car['body_angle'] + 180) % 360 - 180

        # Устанавливаем целевой угол поворота колес
        car['target_wheel_angle'] = np.clip(angle_diff * 0.8,
                                            -car['max_steering_angle'],
                                            car['max_steering_angle'])

        # Плавно поворачиваем колеса к целевому углу
        angle_diff_wheel = car['target_wheel_angle'] - car['wheel_angle']
        car['wheel_angle'] += angle_diff_wheel * car['steering_speed'] * dt

        # Выбираем спрайт в зависимости от угла поворота колес
        car['sprite_state'] = self._get_car_sprite(car['wheel_angle'])

        # Обновляем угол кузова машины с учетом поворота колес и скорости
        turn_rate = car['wheel_angle'] * car['speed'] * 0.02
        car['body_angle'] = (car['body_angle'] + turn_rate) % 360

        # Обновляем позицию на основе угла кузова
        rad_angle = math.radians(car['body_angle'])
        dx = math.cos(rad_angle) * car['speed']
        dy = math.sin(rad_angle) * car['speed']

        car['position'] = (car['position'][0] + dx, car['position'][1] + dy)

        # Обновляем прогресс по треку
        car['progress'] = (car['progress'] + car['speed']) % len(center_x)

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

            # Дорога
            road_x = np.concatenate([inner_x, outer_x[::-1]])
            road_y = np.concatenate([inner_y, outer_y[::-1]])
            self.ax_main.fill(road_x, road_y, color='#4a6572', alpha=0.8, zorder=1)

            # Границы
            self.ax_main.plot(inner_x, inner_y, color=self.COLORS['accent'],
                              linewidth=2, zorder=2)
            self.ax_main.plot(outer_x, outer_y, color=self.COLORS['accent'],
                              linewidth=2, zorder=2)

            # Отрисовка машинок
            self._draw_cars()

        # Статус
        status = f"Трек: {self.selected_track or 'не выбран'}\n"
        status += f"Машинок: {len(self.car_objects)}\n"
        status += f"Симуляция: {'ЗАПУЩЕНА' if self.is_running else 'ОСТАНОВЛЕНА'}"

        self.ax_main.text(0.02, 0.98, status, transform=self.ax_main.transAxes,
                          fontsize=11, color=self.COLORS['text'], va='top',
                          bbox=dict(boxstyle='round', facecolor=self.COLORS['panel'], alpha=0.8))

        self.fig.canvas.draw_idle()

    def _draw_cars(self):
        """Отрисовывает машинки на треке с правильным поворотом спрайтов"""
        # Очищаем предыдущих художников
        for artist in self.car_artists:
            try:
                artist.remove()
            except:
                pass
        self.car_artists = []

        for i, car in enumerate(self.car_objects):
            try:
                # Получаем спрайт для текущего состояния
                base_sprite = self.car_sprites.get(car['sprite_state'],
                                                   self.car_sprites.get('straight'))

                # Вращаем спрайт в соответствии с углом кузова машины
                rotated_sprite = self._rotate_image(base_sprite, car['body_angle'])

                # Создание AnnotationBbox для машинки с вращенным спрайтом
                imagebox = OffsetImage(rotated_sprite, zoom=0.2)

                ab = AnnotationBbox(imagebox, car['position'], frameon=False)
                self.ax_main.add_artist(ab)
                self.car_artists.append(ab)

                # Дополнительно рисуем направление машинки (для отладки)
                length = 15
                rad_angle = math.radians(car['body_angle'])
                end_x = car['position'][0] + math.cos(rad_angle) * length
                end_y = car['position'][1] + math.sin(rad_angle) * length

                direction_line = self.ax_main.plot([car['position'][0], end_x],
                                                   [car['position'][1], end_y],
                                                   color=car['color'], linewidth=2, zorder=5, alpha=0.7)
                self.car_artists.extend(direction_line)

            except Exception as e:
                print(f"Ошибка отрисовки машинки: {e}")
                # Резервная отрисовка кружком со стрелкой направления
                scatter = self.ax_main.scatter(car['position'][0], car['position'][1],
                                               c=car['color'], s=100, zorder=5)
                self.car_artists.append(scatter)

                length = 15
                rad_angle = math.radians(car['body_angle'])
                end_x = car['position'][0] + math.cos(rad_angle) * length
                end_y = car['position'][1] + math.sin(rad_angle) * length

                arrow = self.ax_main.arrow(car['position'][0], car['position'][1],
                                           end_x - car['position'][0], end_y - car['position'][1],
                                           head_width=5, head_length=5, fc=car['color'], ec=car['color'])
                self.car_artists.append(arrow)

    def _start_simulation(self, event=None):
        """Запускает симуляцию"""
        print("Запуск симуляции...")
        if not self.track_data:
            self._show_message("Сначала выберите трек!")
            return

        self.is_running = True
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
        if self.track_data:
            self._initialize_cars()
        self._update_display()

    def _run_simulation_loop(self):
        """Основной цикл симуляции с улучшенной физикой"""
        if not self.is_running or not self.track_data:
            return

        center_x = np.array(self.track_data['center_line'][0])
        center_y = np.array(self.track_data['center_line'][1])

        max_steps = 500  # Ограничиваем количество шагов для безопасности

        while self.is_running and self.simulation_step < max_steps:
            try:
                # Обновление физики для каждой машинки
                for car in self.car_objects:
                    self._update_car_physics(car, center_x, center_y)

                self.simulation_step += 1
                self._update_display()
                plt.pause(0.05)  # Контроль скорости анимации

            except Exception as e:
                print(f"Ошибка в симуляции: {e}")
                break

    def _show_message(self, text):
        """Показывает сообщение"""
        self.ax_main.set_title(text, fontsize=12, color=self.COLORS['accent'], pad=20)
        self.fig.canvas.draw_idle()
        plt.pause(2)
        self.ax_main.set_title("Симуляция гонки - выберите трек и запустите",
                               fontsize=12, color=self.COLORS['text'], pad=20)


# Запуск приложения
if __name__ == "__main__":
    print("Запуск Гоночного Трекера...")
    app = RacingGame()
    app.show_main_menu()