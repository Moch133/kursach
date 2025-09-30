import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy import interpolate
import os
import json
import tkinter as tk
from tkinter import simpledialog, messagebox
import random
import math

class RacingGameMenu:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.tracks_folder = "saved_tracks"
        self._create_tracks_folder()
    
    def _create_tracks_folder(self):
        """Создает папку для сохранения треков"""
        if not os.path.exists(self.tracks_folder):
            os.makedirs(self.tracks_folder)
    
    def show_menu(self):
        """Показывает главное меню"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor('#2C3E50')
        
        # Заголовок
        self.ax.text(0.5, 0.8, 'ГОНОЧНЫЙ ТРЕКЕР', 
                    fontsize=24, fontweight='bold', ha='center', color='white')
        self.ax.text(0.5, 0.7, 'Конструктор треков для AI-гонок', 
                    fontsize=14, ha='center', color='#ECF0F1', style='italic')
        
        # Кнопка редактора треков
        button_ax = plt.axes([0.3, 0.4, 0.4, 0.1])
        btn_editor = Button(button_ax, 'РЕДАКТОР ТРЕКОВ', 
                          color='#E74C3C', hovercolor='#C0392B')
        btn_editor.on_clicked(self._open_track_editor)
        
        # Информация
        track_count = self._get_track_count()
        self.ax.text(0.5, 0.2, f'Сохранено треков: {track_count}', 
                    fontsize=12, ha='center', color='#BDC3C7')
        self.ax.text(0.5, 0.1, 'Создавайте собственные трассы для обучения AI!', 
                    fontsize=10, ha='center', color='#7F8C8D')
        
        plt.show()
    
    def _get_track_count(self):
        """Возвращает количество сохраненных треков"""
        try:
            return len([f for f in os.listdir(self.tracks_folder) 
                       if f.endswith('.json')])
        except:
            return 0
    
    def _open_track_editor(self, event):
        """Открывает редактор треков"""
        plt.close(self.fig)
        editor = TrackEditor(self.tracks_folder)
        editor.show_editor()

class TrackGenerator:
    """Класс для генерации случайных треков"""
    
    @staticmethod
    def generate_oval_track(complexity=1.0):
        """Генерирует овальный трек"""
        num_points = max(8, int(8 * complexity))
        points = []
        
        # Основные параметры
        width = 120 + 40 * complexity
        height = 80 + 40 * complexity
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            
            # Добавляем случайные вариации
            var_w = random.uniform(-10, 10) * complexity
            var_h = random.uniform(-10, 10) * complexity
            
            x = (width + var_w) * math.cos(angle)
            y = (height + var_h) * math.sin(angle)
            points.append((x, y))
        
        return points
    
    @staticmethod
    def generate_technical_track(complexity=1.0):
        """Генерирует технический трек с множеством поворотов"""
        num_segments = max(6, int(8 * complexity))
        points = []
        
        current_angle = 0
        x, y = 0, 0
        
        for i in range(num_segments):
            # Случайная длина сегмента
            length = random.uniform(30, 80)
            
            # Случайный угол поворота (более резкие повороты для технических треков)
            angle_change = random.uniform(-math.pi/1.5, math.pi/1.5) * complexity
            
            current_angle += angle_change
            
            x += length * math.cos(current_angle)
            y += length * math.sin(current_angle)
            
            points.append((x, y))
        
        return points
    
    @staticmethod
    def generate_speed_track(complexity=1.0):
        """Генерирует скоростной трек с длинными прямыми"""
        num_segments = max(5, int(6 * complexity))
        points = []
        
        current_angle = 0
        x, y = 0, 0
        
        for i in range(num_segments):
            # Длинные прямые
            length = random.uniform(60, 120)
            
            # Плавные повороты
            angle_change = random.uniform(-math.pi/3, math.pi/3) * complexity
            
            current_angle += angle_change
            
            x += length * math.cos(current_angle)
            y += length * math.sin(current_angle)
            
            points.append((x, y))
        
        return points
    
    @staticmethod
    def generate_random_track(complexity=1.0):
        """Генерирует полностью случайный трек"""
        track_type = random.choice(['oval', 'technical', 'speed', 'mixed'])
        
        if track_type == 'oval':
            return TrackGenerator.generate_oval_track(complexity)
        elif track_type == 'technical':
            return TrackGenerator.generate_technical_track(complexity)
        elif track_type == 'speed':
            return TrackGenerator.generate_speed_track(complexity)
        else:  # mixed
            # Комбинируем разные стили
            base_points = TrackGenerator.generate_oval_track(complexity * 0.7)
            tech_points = TrackGenerator.generate_technical_track(complexity * 0.3)
            
            # Объединяем и перемешиваем точки
            all_points = base_points + tech_points
            random.shuffle(all_points)
            return all_points[:max(10, int(12 * complexity))]

class TrackEditor:
    # Статические константы
    COLORS = {
        'background': '#2C3E50',
        'panel': '#34495E',
        'button': '#E74C3C', 
        'button_hover': '#C0392B',
        'button_delete': '#E74C3C',
        'button_delete_hover': '#C0392B',
        'button_generate': '#27AE60',
        'button_generate_hover': '#229954',
        'text': '#ECF0F1',
        'grid': '#566573'
    }
    
    def __init__(self, tracks_folder):
        self.tracks_folder = tracks_folder
        self.fig = None
        self.ax_editor = None
        self.ax_list = None
        
        # Основные данные
        self.points = []
        self.track_width = 25
        self.current_track = None
        self.selected_track = None
        self.track_closed = False
        
        # Границы трека
        self.inner_x = None
        self.inner_y = None
        self.outer_x = None
        self.outer_y = None
        
        # Для редактирования точек
        self.dragging_point = None
        self.dragging_start_pos = None
        
        # Ссылки на кнопки
        self.btn_clear = None
        self.btn_save = None
        self.btn_load = None
        self.btn_delete = None
        self.btn_generate = None

    def show_editor(self):
        """Показывает редактор треков"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(self.COLORS['background'])
        
        # Создаем панели
        self._create_panels()
        self._create_buttons()
        self._load_track_list()
        
        # Подключаем обработчики
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('pick_event', self._on_track_select)
        
        self._update_display()
        plt.show()
    
    def _create_panels(self):
        """Создает панели интерфейса"""
        # Левая панель - список треков
        self.ax_list = plt.axes([0.02, 0.1, 0.2, 0.8])
        self.ax_list.set_facecolor(self.COLORS['panel'])
        self.ax_list.set_xlim(0, 1)
        self.ax_list.set_ylim(0, 1)
        self.ax_list.set_xticks([])
        self.ax_list.set_yticks([])
        
        # Центральная панель - редактор
        self.ax_editor = plt.axes([0.25, 0.1, 0.7, 0.8])
        self.ax_editor.set_xlim(-200, 200)
        self.ax_editor.set_ylim(-200, 200)
        self.ax_editor.set_aspect('equal')
        self.ax_editor.grid(True, alpha=0.3, color=self.COLORS['grid'])
        self.ax_editor.set_facecolor(self.COLORS['background'])
        self.ax_editor.set_title('РЕДАКТОР ТРЕКОВ - Щелкайте чтобы добавить точки', 
                               fontsize=14, color=self.COLORS['text'], pad=20)
    
    def _create_buttons(self):
        """Создает кнопки управления"""
        button_height = 0.05
        button_width = 0.1
        
        # Кнопки управления
        self.btn_clear = self._create_button(0.25, 0.02, button_width, button_height, 
                          'ОЧИСТИТЬ', self._clear_points)
        self.btn_save = self._create_button(0.36, 0.02, button_width, button_height, 
                          'СОХРАНИТЬ', self._save_track)
        self.btn_load = self._create_button(0.47, 0.02, button_width, button_height, 
                          'ЗАГРУЗИТЬ', self._load_selected_track)
        self.btn_delete = self._create_button(0.58, 0.02, button_width, button_height, 
                          'УДАЛИТЬ', self._delete_track, 
                          color='#E74C3C', hovercolor='#C0392B')
        
        # Кнопка генерации
        self.btn_generate = self._create_button(0.69, 0.02, button_width, button_height, 
                          'СГЕНЕРИРОВАТЬ', self._generate_random,
                          color='#27AE60', hovercolor='#229954')
        
        # Слайдер ширины
        slider_ax = plt.axes([0.80, 0.02, 0.15, button_height])
        slider_ax.set_facecolor(self.COLORS['panel'])
        self.slider_width = Slider(slider_ax, 'Ширина:', 15, 40, 
                                 valinit=self.track_width, valstep=5,
                                 color='#3498DB')
        self.slider_width.on_changed(self._update_track_width)
    
    def _create_button(self, x, y, width, height, text, callback, color=None, hovercolor=None):
        """Создает одну кнопку"""
        ax = plt.axes([x, y, width, height])
        btn_color = color if color else self.COLORS['button']
        btn_hovercolor = hovercolor if hovercolor else self.COLORS['button_hover']
        btn = Button(ax, text, color=btn_color, hovercolor=btn_hovercolor)
        btn.on_clicked(callback)
        return btn
    
    def _generate_random(self, event=None):
        """Генерирует случайный трек"""
        self.points = TrackGenerator.generate_random_track(1.0)
        self.track_closed = True
        if self._generate_track():
            self._update_display()
            print("Случайный трек сгенерирован!")
    
    def _load_track_list(self):
        """Загружает список треков"""
        try:
            track_files = [f for f in os.listdir(self.tracks_folder) 
                         if f.endswith('.json')]
        except:
            track_files = []
        
        self.ax_list.clear()
        self.ax_list.set_facecolor(self.COLORS['panel'])
        self.ax_list.set_xlim(0, 1)
        self.ax_list.set_ylim(0, 1)
        self.ax_list.set_xticks([])
        self.ax_list.set_yticks([])
        
        # Заголовок
        self.ax_list.text(0.5, 0.95, 'СОХРАНЕННЫЕ ТРЕКИ', 
                         fontsize=12, fontweight='bold', 
                         ha='center', color=self.COLORS['text'])
        
        if not track_files:
            self.ax_list.text(0.5, 0.5, 'Нет сохраненных треков', 
                             ha='center', color=self.COLORS['grid'], fontsize=10)
            return
        
        # Список треков
        for i, track_file in enumerate(track_files[::-1]):
            y_pos = 0.85 - i * 0.08
            if y_pos < 0.1:
                break
                
            track_name = track_file.replace('.json', '')
            color = '#E74C3C' if track_file == self.selected_track else self.COLORS['text']
            
            text = self.ax_list.text(0.1, y_pos, f"• {track_name}", 
                            fontsize=9, color=color)
            text.set_picker(True)
    
    def _on_track_select(self, event):
        """Обрабатывает выбор трека"""
        if hasattr(event, 'artist'):
            text = event.artist
            track_files = [f for f in os.listdir(self.tracks_folder) 
                         if f.endswith('.json')]
            
            # Находим индекс выбранного текста
            texts = [child for child in self.ax_list.get_children() 
                    if hasattr(child, 'get_text') and child.get_text().startswith('•')]
            
            if text in texts:
                track_index = texts.index(text)
                self.selected_track = track_files[::-1][track_index]
                self._load_track_list()  # Обновляем выделение
                print(f"Выбран трек: {self.selected_track}")
    
    def _on_click(self, event):
        """Обработчик клика мыши"""
        if event.inaxes != self.ax_editor:
            return
        
        # Если трек замкнут, проверяем клик по точкам для редактирования
        if self.track_closed and self.points:
            # Проверяем клик по точкам (кроме первых двух)
            for i in range(2, len(self.points)):
                point = self.points[i]
                dist = ((event.xdata - point[0])**2 + (event.ydata - point[1])**2)**0.5
                if dist < 8:  # Радиус захвата точки
                    self.dragging_point = i
                    self.dragging_start_pos = (event.xdata, event.ydata)
                    return
        
        # Обычное добавление точек (только если трек не замкнут)
        if not self.track_closed:
            new_point = (float(event.xdata), float(event.ydata))
            
            # Проверяем расстояние до существующих точек
            if self.points:
                min_dist = min(self._distance(new_point, p) for p in self.points)
                if min_dist < 10:
                    return
            
            self.points.append(new_point)
            
            # Проверяем замыкание
            if len(self.points) >= 3:
                first_point = self.points[0]
                if self._distance(new_point, first_point) < 25:
                    self.points[-1] = first_point
                    self.track_closed = True
                    print("Трек замкнут!")
                
                if self._generate_track():
                    self._update_display()
    
    def _on_release(self, event):
        """Обработчик отпускания кнопки мыши"""
        if self.dragging_point is not None:
            self.dragging_point = None
            self.dragging_start_pos = None
            # Перестраиваем трек после перемещения точки
            if self._generate_track():
                self._update_display()
    
    def _on_motion(self, event):
        """Обработчик движения мыши"""
        if (self.dragging_point is not None and event.inaxes == self.ax_editor and 
            event.xdata is not None and event.ydata is not None):
            # Обновляем позицию точки
            self.points[self.dragging_point] = (float(event.xdata), float(event.ydata))
            
            # Если перемещаем последнюю точку (которая равна первой), обновляем и первую
            if self.track_closed and self.dragging_point == len(self.points) - 1:
                self.points[0] = (float(event.xdata), float(event.ydata))
            
            # Перестраиваем трек в реальном времени
            if self._generate_track():
                self._update_display()
    
    def _distance(self, p1, p2):
        """Быстрое вычисление расстояния между точками"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def _generate_track(self):
        """Генерирует трек"""
        if len(self.points) < 3:
            return False
        
        points = self.points.copy()
        if self.track_closed and points[-1] != points[0]:
            points[-1] = points[0]
        
        # Создаем массив точек нужного размера
        points_array = np.array(points)
        
        # Адаптивное количество точек для сглаживания
        smoothness = min(150 + len(points) * 10, 400)
        
        # Интерполяция
        t = np.arange(len(points))
        t_new = np.linspace(0, len(points)-1, smoothness)
        
        try:
            cs_x = interpolate.CubicSpline(t, points_array[:, 0])
            cs_y = interpolate.CubicSpline(t, points_array[:, 1])
            center_x = cs_x(t_new)
            center_y = cs_y(t_new)
        except:
            center_x = np.interp(t_new, t, points_array[:, 0])
            center_y = np.interp(t_new, t, points_array[:, 1])
        
        # Гарантируем замыкание
        if self.track_closed:
            center_x[-1] = center_x[0]
            center_y[-1] = center_y[0]
        
        # Создаем границы
        success = self._create_track_boundaries(center_x, center_y)
        
        if success:
            self.current_track = {
                'center_line': (center_x.tolist(), center_y.tolist()),
                'inner_boundary': (self.inner_x.tolist(), self.inner_y.tolist()),
                'outer_boundary': (self.outer_x.tolist(), self.outer_y.tolist()),
                'points': points_array.tolist(),
                'width': self.track_width,
                'closed': self.track_closed
            }
            return True
        return False
    
    def _create_track_boundaries(self, center_x, center_y):
        """Создает границы трека"""
        # Вычисляем производные
        dx = np.gradient(center_x)
        dy = np.gradient(center_y)
        
        norm = (dx**2 + dy**2)**0.5 + 1e-8
        nx = -dy / norm
        ny = dx / norm
        
        # Находим оптимальную ширину
        current_width = self.track_width
        for _ in range(3):
            half_width = current_width / 2
            
            inner_x = center_x - half_width * nx
            inner_y = center_y - half_width * ny
            outer_x = center_x + half_width * nx
            outer_y = center_y + half_width * ny
            
            if not self._check_self_intersection(inner_x, inner_y):
                self.inner_x, self.inner_y = inner_x, inner_y
                self.outer_x, self.outer_y = outer_x, outer_y
                return True
            
            current_width *= 0.8
        
        # Минимальная ширина
        half_width = 8
        self.inner_x = center_x - half_width * nx
        self.inner_y = center_y - half_width * ny
        self.outer_x = center_x + half_width * nx
        self.outer_y = center_y + half_width * ny
        return True
    
    def _check_self_intersection(self, inner_x, inner_y):
        """Упрощенная проверка самопересечений"""
        points = np.column_stack([inner_x, inner_y])
        n = len(points)
        
        for i in range(0, n, 5):
            for j in range(i + 20, n, 5):
                if ((points[i][0] - points[j][0])**2 + 
                    (points[i][1] - points[j][1])**2 < (self.track_width * 0.4)**2):
                    return True
        return False
    
    def _update_display(self):
        """Обновляет отображение"""
        self.ax_editor.clear()
        self.ax_editor.set_xlim(-200, 200)
        self.ax_editor.set_ylim(-200, 200)
        self.ax_editor.set_aspect('equal')
        self.ax_editor.grid(True, alpha=0.3, color=self.COLORS['grid'])
        self.ax_editor.set_facecolor(self.COLORS['background'])
        
        # Заголовок с инструкциями
        if self.track_closed:
            title = 'Трек замкнут! Перетаскивайте точки для редактирования (первые две точки скрыты)'
        else:
            title = 'Добавляйте точки (первые две точки будут скрыты после замыкания)'
        self.ax_editor.set_title(title, fontsize=12, color=self.COLORS['text'], pad=20)
        
        # Рисуем точки (скрываем первые две если трек замкнут)
        if self.points:
            points_to_display = self.points
            if self.track_closed and len(self.points) > 2:
                points_to_display = self.points[2:]  # Скрываем первые две точки
            
            points_array = np.array(points_to_display)
            
            # Рисуем только видимые точки
            if len(points_array) > 0:
                self.ax_editor.scatter(points_array[:, 0], points_array[:, 1], 
                                     c='#3498DB', s=50, zorder=5, edgecolors='white', 
                                     picker=True)  # Включаем возможность выбора для редактирования
            
            # Линии между точками (все точки)
            all_points_array = np.array(self.points)
            if len(self.points) > 1:
                self.ax_editor.plot(all_points_array[:, 0], all_points_array[:, 1], 
                                  '#3498DB', alpha=0.3, linewidth=1, linestyle='--')
        
        # Рисуем трек
        if self.current_track and self.inner_x is not None:
            # Дорога
            road_x = np.concatenate([self.inner_x, self.outer_x[::-1]])
            road_y = np.concatenate([self.inner_y, self.outer_y[::-1]])
            self.ax_editor.fill(road_x, road_y, color='#7F8C8D', alpha=0.8, zorder=1)
            
            # Границы
            self.ax_editor.plot(np.append(self.inner_x, self.inner_x[0]), 
                              np.append(self.inner_y, self.inner_y[0]), 
                              '#E74C3C', linewidth=2, zorder=2)
            self.ax_editor.plot(np.append(self.outer_x, self.outer_x[0]), 
                              np.append(self.outer_y, self.outer_y[0]), 
                              '#E74C3C', linewidth=2, zorder=2)
        
        # Статус
        status = f"Точек: {len(self.points)}"
        if self.track_closed:
            status += " (замкнут)"
            if len(self.points) > 2:
                status += f"\nВидимых точек: {len(self.points) - 2}"
        
        self.ax_editor.text(0.02, 0.98, status, transform=self.ax_editor.transAxes,
                          fontsize=11, color=self.COLORS['text'], verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor=self.COLORS['panel'], alpha=0.8))
        
        self.fig.canvas.draw_idle()
    
    def _clear_points(self, event=None):
        """Очищает все точки"""
        print("Очистка трека...")
        self.points.clear()
        self.current_track = None
        self.track_closed = False
        self.inner_x = None
        self.inner_y = None
        self.outer_x = None
        self.outer_y = None
        self.dragging_point = None
        self.dragging_start_pos = None
        self._update_display()
    
    def _save_track(self, event=None):
        """Сохраняет трек с выбором названия"""
        print("Попытка сохранения...")
        if not self.current_track or not self.track_closed:
            print("Создайте и замкните трек перед сохранением!")
            return
        
        try:
            # Создаем диалог для ввода названия
            root = tk.Tk()
            root.withdraw()  # Скрываем основное окно
            root.attributes('-topmost', True)  # Поверх всех окон
            
            track_name = simpledialog.askstring(
                "Сохранение трека", 
                "Введите название трека:",
                parent=root
            )
            
            root.destroy()
            
            if not track_name:
                print("Сохранение отменено")
                return
            
            # Заменяем запрещенные символы в имени файла
            track_name = "".join(c for c in track_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            if not track_name:
                track_name = "track"
            
            filename = os.path.join(self.tracks_folder, f"{track_name}.json")
            
            # Проверяем, не существует ли уже файл с таким именем
            if os.path.exists(filename):
                result = messagebox.askyesno(
                    "Файл существует", 
                    f"Трек с названием '{track_name}' уже существует. Перезаписать?",
                    parent=root
                )
                if not result:
                    print("Сохранение отменено")
                    return
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_track, f, separators=(',', ':'), ensure_ascii=False)
            
            print(f"Трек '{track_name}' сохранен!")
            self._load_track_list()
            
        except Exception as e:
            print(f"Ошибка сохранения: {e}")
    
    def _load_selected_track(self, event=None):
        """Загружает выбранный трек"""
        print("Попытка загрузки...")
        if not self.selected_track:
            print("Выберите трек из списка!")
            return
        
        try:
            filename = os.path.join(self.tracks_folder, self.selected_track)
            print(f"Загрузка файла: {filename}")
            
            with open(filename, 'r', encoding='utf-8') as f:
                track_data = json.load(f)
            
            # Восстанавливаем данные
            self.points = [tuple(point) for point in track_data['points']]
            self.track_width = track_data['width']
            self.track_closed = track_data.get('closed', True)
            
            # Восстанавливаем границы
            self.inner_x = np.array(track_data['inner_boundary'][0], dtype=np.float32)
            self.inner_y = np.array(track_data['inner_boundary'][1], dtype=np.float32)
            self.outer_x = np.array(track_data['outer_boundary'][0], dtype=np.float32)
            self.outer_y = np.array(track_data['outer_boundary'][1], dtype=np.float32)
            
            self.current_track = track_data
            self.slider_width.set_val(self.track_width)
            
            self._update_display()
            print("Трек загружен!")
            
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
    
    def _delete_track(self, event=None):
        """Удаляет выбранный трек"""
        print("Попытка удаления...")
        if not self.selected_track:
            print("Выберите трек для удаления!")
            return
        
        try:
            # Создаем диалог подтверждения
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            track_name = self.selected_track.replace('.json', '')
            result = messagebox.askyesno(
                "Удаление трека",
                f"Вы уверены, что хотите удалить трек '{track_name}'?",
                parent=root
            )
            
            root.destroy()
            
            if not result:
                print("Удаление отменено")
                return
            
            filename = os.path.join(self.tracks_folder, self.selected_track)
            
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Трек '{track_name}' удален!")
                
                # Сбрасываем выбор
                self.selected_track = None
                
                # Если удалили загруженный трек, очищаем редактор
                if self.current_track and os.path.exists(filename):
                    try:
                        with open(filename, 'r') as f:
                            deleted_track = json.load(f)
                        if (self.current_track.get('points') == 
                            [tuple(point) for point in deleted_track['points']]):
                            self._clear_points()
                    except:
                        pass
                
                self._load_track_list()
            else:
                print("Файл не найден!")
                
        except Exception as e:
            print(f"Ошибка удаления: {e}")
    
    def _update_track_width(self, val):
        """Обновляет ширину трека"""
        self.track_width = val
        if self.current_track:
            self._generate_track()
            self._update_display()

# Запуск программы
if __name__ == "__main__":
    print("Запуск Гоночного Трекера...")
    menu = RacingGameMenu()
    menu.show_menu()