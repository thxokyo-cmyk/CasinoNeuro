"""
calibrate.py — Инструмент калибровки для GTA5 Roulette AI
Позволяет выделить мышкой область экрана, где отображается результат рулетки.

ИНСТРУКЦИЯ:
1. Запусти GTA5 и открой рулетку в казино
2. Дождись, пока выпадет число (чтобы оно было видно на экране)
3. Запусти этот скрипт
4. Нажми 'C' чтобы сделать скриншот текущего экрана
5. Мышкой выдели ПРЯМОУГОЛЬНИК вокруг числа результата
6. Нажми 'S' чтобы сохранить координаты в config.json
7. Нажми 'Q' чтобы выйти
"""

import mss
import cv2
import numpy as np
import json
import os
import sys

# === Глобальные переменные для мыши ===
drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1
current_frame = None
display_frame = None
regions = {}  # Словарь сохранённых регионов
current_region_name = "result_region"  # Какую область сейчас выделяем
scale_factor = 1.0  # Масштаб отображения


def mouse_callback(event, x, y, flags, param):
    """Обработчик событий мыши"""
    global drawing, start_x, start_y, end_x, end_y, display_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        end_x, end_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x, end_y = x, y
            # Рисуем прямоугольник в реальном времени
            display_frame = current_frame.copy()
            cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            # Показываем координаты
            coord_text = f"({start_x},{start_y}) -> ({end_x},{end_y})"
            cv2.putText(display_frame, coord_text, (start_x, start_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        display_frame = current_frame.copy()
        cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # Рассчитываем реальные координаты (с учётом масштаба)
        real_x1 = int(start_x / scale_factor)
        real_y1 = int(start_y / scale_factor)
        real_x2 = int(end_x / scale_factor)
        real_y2 = int(end_y / scale_factor)

        # Нормализуем (чтобы x1 < x2, y1 < y2)
        left = min(real_x1, real_x2)
        top = min(real_y1, real_y2)
        width = abs(real_x2 - real_x1)
        height = abs(real_y2 - real_y1)

        regions[current_region_name] = {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }

        info_text = f"Region '{current_region_name}': left={left}, top={top}, w={width}, h={height}"
        print(f"  ✅ {info_text}")
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def save_to_config(regions_dict):
    """Сохраняет координаты в config.json"""
    config_path = "config.json"

    # Загружаем существующий конфиг или создаём новый
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "capture": {
                "monitor_index": 1,
                "result_region": {"left": 920, "top": 200, "width": 80, "height": 60},
                "history_region": {"left": 800, "top": 100, "width": 320, "height": 40},
                "capture_interval_ms": 2000
            },
            "ml": {
                "sequence_length": 50,
                "hidden_size": 128,
                "num_layers": 2,
                "learning_rate": 0.001,
                "min_spins_to_predict": 10,
                "retrain_every_n_spins": 20
            },
            "gui": {
                "opacity": 0.9,
                "width": 420,
                "height": 700,
                "position_x": 10,
                "position_y": 100
            }
        }

    # Обновляем регионы
    for name, coords in regions_dict.items():
        config['capture'][name] = coords

    # Сохраняем
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n  💾 Config saved to: {os.path.abspath(config_path)}")
    print(f"  📋 Regions saved: {list(regions_dict.keys())}")


def capture_screen():
    """Делает скриншот всего экрана"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Основной монитор
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def main():
    global current_frame, display_frame, current_region_name, scale_factor

    print("=" * 60)
    print("  🎰 GTA5 Roulette AI — КАЛИБРОВКА")
    print("=" * 60)
    print()
    print("  ИНСТРУКЦИЯ:")
    print("  ─────────────────────────────────────────────")
    print("  1. Открой GTA5 и перейди к столу рулетки")
    print("  2. Дождись, пока выпадет число")
    print("  3. Нажми [C] — сделать скриншот экрана")
    print("  4. Мышкой ВЫДЕЛИ область с числом результата")
    print("  5. Нажми [1] — выделить result_region (число)")
    print("     Нажми [2] — выделить history_region (строка истории)")
    print("  6. Нажми [S] — сохранить в config.json")
    print("  7. Нажми [T] — проверить захват (тест)")
    print("  8. Нажми [Q] — выход")
    print("  ─────────────────────────────────────────────")
    print()

    # Создаём окно
    window_name = "GTA5 Roulette AI - Calibration (Press C to capture)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Начальный экран — инструкция
    placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Press [C] to capture screen", (400, 350),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(placeholder, "Make sure GTA5 roulette is visible!", (350, 400),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    current_frame = placeholder.copy()
    display_frame = placeholder.copy()

    print("  [*] Waiting for input... Press [C] to capture screen.")
    print()

    while True:
        # Показываем текущий кадр
        show_frame = display_frame if display_frame is not None else current_frame
        cv2.imshow(window_name, show_frame)

        key = cv2.waitKey(50) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            print("\n  [*] Exiting calibration.")
            break

        elif key == ord('c'):
            # Захват экрана
            print("  [*] Capturing screen in 3 seconds...")
            print("      (Switch to GTA5 NOW!)")
            cv2.waitKey(3000)  # 3 секунды задержка, чтобы переключиться

            full_screen = capture_screen()
            h, w = full_screen.shape[:2]

            # Масштабируем для отображения (если экран больше окна)
            max_display_w = 1600
            max_display_h = 900
            scale_w = max_display_w / w
            scale_h = max_display_h / h
            scale_factor = min(scale_w, scale_h, 1.0)

            if scale_factor < 1.0:
                display_w = int(w * scale_factor)
                display_h = int(h * scale_factor)
                current_frame = cv2.resize(full_screen, (display_w, display_h))
            else:
                current_frame = full_screen.copy()
                scale_factor = 1.0

            display_frame = current_frame.copy()

            print(f"  ✅ Screen captured! Resolution: {w}x{h}")
            print(f"     Display scale: {scale_factor:.2f}")
            print(f"     Now draw a rectangle around the RESULT NUMBER.")
            print(f"     Current mode: [{current_region_name}]")

        elif key == ord('1'):
            current_region_name = "result_region"
            print(f"  [*] Mode: Selecting RESULT REGION (the winning number)")
            print(f"      Draw a rectangle around where the number appears.")

        elif key == ord('2'):
            current_region_name = "history_region"
            print(f"  [*] Mode: Selecting HISTORY REGION (row of past results)")
            print(f"      Draw a rectangle around the history bar.")

        elif key == ord('s'):
            if regions:
                save_to_config(regions)
                print("  ✅ Configuration saved successfully!")
            else:
                print("  ⚠️  No regions selected yet! Draw a rectangle first.")

        elif key == ord('t'):
            # Тест захвата — показывает что именно захватывается
            if 'result_region' in regions:
                print("  [*] Testing capture of result_region...")
                with mss.mss() as sct:
                    region = regions['result_region']
                    monitor = {
                        "left": region['left'],
                        "top": region['top'],
                        "width": region['width'],
                        "height": region['height']
                    }
                    screenshot = sct.grab(monitor)
                    test_img = np.array(screenshot)
                    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGRA2BGR)

                    # Увеличиваем для просмотра
                    test_img_large = cv2.resize(test_img, None, fx=4, fy=4,
                                               interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Test Capture (result_region)", test_img_large)
                    print(f"  ✅ Captured {region['width']}x{region['height']} pixels")
                    print(f"     Check the 'Test Capture' window!")
            else:
                print("  ⚠️  result_region not defined! Press [1] and draw it.")

        elif key == ord('r'):
            # Перерисовать (сбросить выделение)
            if current_frame is not None:
                display_frame = current_frame.copy()
                print("  [*] Selection cleared. Draw again.")

    cv2.destroyAllWindows()

    # Финальная проверка
    if regions:
        print("\n" + "=" * 60)
        print("  📋 SUMMARY OF CALIBRATED REGIONS:")
        print("  ─────────────────────────────────────────────")
        for name, coords in regions.items():
            print(f"    {name}: left={coords['left']}, top={coords['top']}, "
                  f"width={coords['width']}, height={coords['height']}")
        print("=" * 60)
        print("\n  ✅ You can now run: python main.py")
    else:
        print("\n  ⚠️  No regions were saved. Run calibrate.py again.")


if __name__ == "__main__":
    main()