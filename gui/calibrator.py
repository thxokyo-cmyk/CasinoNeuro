"""
Калибратор и отладочная панель для Roulette AI
Обновленная версия: Выбор областей числа и детекции движения
"""
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import threading
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw

from vision.region_selector import RegionSelector
from utils.config import Config
from vision.number_detector import NumberDetector


class CalibratorWindow:
    def __init__(self, parent, capture_module, number_detector, trainer=None):
        self.parent = parent
        self.capture = capture_module
        self.detector = number_detector
        self.trainer = trainer
        
        self.window = tk.Toplevel(parent)
        self.window.title("🛠 Калибратор и Отладка")
        self.window.geometry("500x650")
        self.window.resizable(False, False)
        
        # Данные для отладки
        self.last_frame = None
        self.last_result = None
        self.calibration_data = {
            'number_region': None,
            'motion_region': None
        }
        self._load_calibration()
        
        # Флаг защиты от двойного нажатия
        self._force_in_progress = False
        
        self._setup_ui()
        
    def _load_calibration(self):
        """Загрузка сохраненных областей"""
        calib_file = Config.get('calibration_file', 'calibration.json')
        if os.path.exists(calib_file):
            try:
                with open(calib_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.calibration_data.update(data)
            except Exception as e:
                print(f"[Calibrator] Error loading calibration: {e}")

    def _save_calibration(self):
        """Сохранение областей"""
        calib_file = Config.get('calibration_file', 'calibration.json')
        try:
            with open(calib_file, 'w', encoding='utf-8') as f:
                json.dump(self.calibration_data, f, indent=2)
            print(f"[Calibrator] Saved to {calib_file}")
        except Exception as e:
            print(f"[Calibrator] Error saving calibration: {e}")

    def _setup_ui(self):
        """Создание интерфейса"""
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Блок 1: Область определения числа ---
        num_frame = ttk.LabelFrame(main_frame, text="1. Определение числа (Template Match)", padding="5")
        num_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_num_region = ttk.Label(num_frame, text="Не выбрано", foreground="orange")
        self.lbl_num_region.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        btn_select_num = ttk.Button(num_frame, text="🎯 Выбрать область числа", command=self._select_number_region)
        btn_select_num.pack(side=tk.RIGHT)

        # --- Блок 2: Область детекции движения (остановки) ---
        mot_frame = ttk.LabelFrame(main_frame, text="2. Детекция остановки (Motion Check)", padding="5")
        mot_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_mot_region = ttk.Label(mot_frame, text="Не выбрано", foreground="orange")
        self.lbl_mot_region.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        btn_select_mot = ttk.Button(mot_frame, text="👁 Выбрать область проверки", command=self._select_motion_region)
        btn_select_mot.pack(side=tk.RIGHT)

        # --- Блок 3: Управление и Тест ---
        ctrl_frame = ttk.LabelFrame(main_frame, text="3. Управление и Тест", padding="5")
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Кнопки управления
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.btn_force = ttk.Button(btn_frame, text="⚡ Force Detection (F3)", command=self._force_detection)
        self.btn_force.pack(side=tk.LEFT, padx=2)
        
        self.btn_train = ttk.Button(btn_frame, text="🧠 Train Model", command=self._train_model)
        self.btn_train.pack(side=tk.LEFT, padx=2)
        
        self.btn_reset = ttk.Button(btn_frame, text="🔄 Reset Session", command=self._reset_session)
        self.btn_reset.pack(side=tk.LEFT, padx=2)

        # Статус
        self.lbl_status = ttk.Label(ctrl_frame, text="Ожидание...", foreground="gray")
        self.lbl_status.pack(fill=tk.X, pady=5)

        # --- Блок 4: Визуализация (Превью) ---
        view_frame = ttk.LabelFrame(main_frame, text="Последний кадр / Результат", padding="5")
        view_frame.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_preview = ttk.Label(view_frame, text="Нет изображения", background="#222", anchor="center")
        self.lbl_preview.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_debug = ttk.Label(view_frame, text="", font=("Consolas", 9), foreground="#0f0", background="#111", justify=tk.LEFT)
        self.lbl_debug.pack(fill=tk.X, side=tk.BOTTOM)

        # Привязка клавиши F3
        self.window.bind('<F3>', lambda e: self._force_detection())
        
        # Обновляем UI с загруженными данными
        self._update_region_labels()

    def _update_region_labels(self):
        """Обновление меток регионов"""
        num_region = self.calibration_data.get('number_region')
        if num_region:
            self.lbl_num_region.config(
                text=f"X:{num_region['left']} Y:{num_region['top']} W:{num_region['width']} H:{num_region['height']}",
                foreground="green"
            )
        
        mot_region = self.calibration_data.get('motion_region')
        if mot_region:
            self.lbl_mot_region.config(
                text=f"X:{mot_region['left']} Y:{mot_region['top']} W:{mot_region['width']} H:{mot_region['height']}",
                foreground="green"
            )

    def _select_number_region(self):
        """Выбор области для поиска числа"""
        self.window.withdraw()
        try:
            selector = RegionSelector()
            region = selector.show()
            if region:
                self.calibration_data['number_region'] = region
                self._save_calibration()
                self.lbl_num_region.config(
                    text=f"X:{region['left']} Y:{region['top']} W:{region['width']} H:{region['height']}",
                    foreground="green"
                )
                # Обновляем регион в детекторе если нужно
                if hasattr(self.detector, 'set_roi'):
                    self.detector.set_roi(region)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось выбрать область: {e}")
        finally:
            self.window.deiconify()

    def _select_motion_region(self):
        """Выбор области для проверки статичности (остановки)"""
        self.window.withdraw()
        try:
            selector = RegionSelector()
            region = selector.show()
            if region:
                self.calibration_data['motion_region'] = region
                self._save_calibration()
                self.lbl_mot_region.config(
                    text=f"X:{region['left']} Y:{region['top']} W:{region['width']} H:{region['height']}",
                    foreground="green"
                )
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось выбрать область: {e}")
        finally:
            self.window.deiconify()

    def _force_detection(self):
        """Принудительное определение числа (ОДИН раз за нажатие)"""
        # Защита от повторного нажатия
        if self._force_in_progress:
            print("[Calibrator] Force detection already in progress, ignoring...")
            return
        
        self._force_in_progress = True
        
        def run():
            try:
                self.lbl_status.config(text="Снимок экрана...", foreground="blue")
                self.window.update()
                
                # 1. Делаем снимок
                frame = self.capture.grab_frame()
                if frame is None:
                    self.lbl_status.config(text="Ошибка захвата!", foreground="red")
                    self._force_in_progress = False
                    return

                self.last_frame = frame
                
                # 2. Определяем область числа
                region = self.calibration_data.get('number_region')
                if not region:
                    # Если область не выбрана, пробуем использовать дефолтную из конфига захвата
                    region = self.capture.current_region
                
                # Кроп области числа
                x, y, w, h = region['left'], region['top'], region['width'], region['height']
                # Проверка границ
                h_full, w_full = frame.shape[:2]
                x = max(0, min(x, w_full - 1))
                y = max(0, min(y, h_full - 1))
                w = min(w, w_full - x)
                h = min(h, h_full - y)
                
                if w <= 0 or h <= 0:
                    self.lbl_status.config(text="Ошибка координат области!", foreground="red")
                    self._force_in_progress = False
                    return

                number_crop = frame[y:y+h, x:x+w]
                
                # 3. Детектируем число (ТОЛЬКО ОДИН РАЗ)
                result = self.detector.detect(number_crop)
                
                self.last_result = result
                num_str = result.get('number', '?') if result else '?'
                color_str = result.get('color', 'unknown') if result else 'unknown'
                conf_str = f"{result.get('confidence', 0):.2f}" if result else '0.00'
                
                # Обновление UI
                self.lbl_status.config(text=f"Результат: {num_str} ({color_str})", foreground="green")
                self.lbl_debug.config(text=f"Num: {num_str}\nColor: {color_str}\nConf: {conf_str}\nRegion: {w}x{h}")
                
                # Показываем кроп
                self._show_preview(number_crop, result)
                
            except Exception as e:
                self.lbl_status.config(text=f"Ошибка: {e}", foreground="red")
                print(f"[Calibrator] Force Detect Error: {e}")
            finally:
                self._force_in_progress = False
        
        # Запуск в отдельном потоке чтобы не фризить UI
        threading.Thread(target=run, daemon=True).start()

    def _train_model(self):
        """Обучение модели / Пересчет шаблонов"""
        def run():
            try:
                self.lbl_status.config(text="Обучение...", foreground="orange")
                self.window.update()
                
                if self.trainer:
                    # Если есть трейнер, запускаем обучение
                    success = self.trainer.retrain()
                    if success:
                        self.lbl_status.config(text="Модель обновлена!", foreground="green")
                    else:
                        self.lbl_status.config(text="Нет данных для обучения", foreground="yellow")
                else:
                    # Иначе просто перегенерируем шаблоны в детекторе
                    if hasattr(self.detector, 'template_matcher'):
                        self.detector.template_matcher.generate_templates()
                        self.lbl_status.config(text="Шаблоны пересозданы", foreground="green")
                    else:
                        self.lbl_status.config(text="Нет модуля для обучения", foreground="red")
                        
            except Exception as e:
                self.lbl_status.config(text=f"Ошибка обучения: {e}", foreground="red")
        
        threading.Thread(target=run, daemon=True).start()

    def _reset_session(self):
        """Сброс сессии"""
        self.last_frame = None
        self.last_result = None
        self.lbl_status.config(text="Сессия сброшена", foreground="gray")
        self.lbl_debug.config(text="")
        self.lbl_preview.config(text="Нет изображения", image="")
        # Здесь можно добавить сброс счетчиков в главном приложении, если передать callback
        print("[Calibrator] Session reset requested")
        
        # Вызываем сброс в главном приложении
        if hasattr(self.parent, 'reset_session'):
            self.parent.reset_session()

    def _show_preview(self, crop, result):
        """Отображение кропа и результата"""
        try:
            # Конвертация BGR -> RGB для PIL
            rgb_frame = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            # Рисуем результат на изображении если есть
            if result and 'bbox' in result:
                draw = ImageDraw.Draw(img)
                bbox = result['bbox']
                draw.rectangle(bbox, outline="green", width=2)
            
            # Ресайз для превью
            max_w, max_h = 300, 150
            ratio = min(max_w / img.width, max_h / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.lbl_preview.config(image=photo, text="")
            self.lbl_preview.image = photo  # Сохраняем ссылку
            
        except Exception as e:
            print(f"Preview error: {e}")
            self.lbl_preview.config(text="Ошибка отображения", image="")
