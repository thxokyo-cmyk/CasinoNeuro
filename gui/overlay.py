"""
Main overlay window with all controls.
"""

import time
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QFrame, QLineEdit,
    QInputDialog, QMessageBox, QCheckBox, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor

from logger import log


class SpinWorker(QThread):
    number_detected = pyqtSignal(str, str)
    status_update = pyqtSignal(str)

    def __init__(self, screen_capture, number_detector, interval_ms=800):
        super().__init__()
        self.screen_capture = screen_capture
        self.number_detector = number_detector
        self.interval = interval_ms / 1000.0
        self.running = True
        self.paused = False

    def run(self):
        while self.running:
            if not self.paused:
                try:
                    img = self.screen_capture.capture_result_region()
                    if img is not None and img.size > 0:
                        result = self.number_detector.process_frame(img)
                        if result:
                            number, color = result
                            self.number_detected.emit(number, color)
                            self.status_update.emit("NEW: " + number + " (" + color + ")")
                except Exception as e:
                    log.error("[Worker] " + str(e))
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused


class HotkeyWorker(QThread):
    hotkey_pressed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        import keyboard
        keyboard.add_hotkey("F2", lambda: self.hotkey_pressed.emit("manual_input"))
        keyboard.add_hotkey("F3", lambda: self.hotkey_pressed.emit("force_capture"))
        keyboard.add_hotkey("F4", lambda: self.hotkey_pressed.emit("toggle_detection"))
        keyboard.add_hotkey("F5", lambda: self.hotkey_pressed.emit("toggle_autobet"))
        keyboard.add_hotkey("F6", lambda: self.hotkey_pressed.emit("open_calibrator"))
        keyboard.add_hotkey("F7", lambda: self.hotkey_pressed.emit("new_session"))
        keyboard.add_hotkey("F8", lambda: self.hotkey_pressed.emit("select_region"))
        keyboard.add_hotkey("ctrl+z", lambda: self.hotkey_pressed.emit("undo_last"))

        while self.running:
            time.sleep(0.1)

    def stop(self):
        self.running = False


class OverlayWindow(QMainWindow):
    manual_input_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("🎰 GTA5 Roulette AI v2.2")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setMinimumWidth(340)

        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("""
            QWidget {
                background-color: #1a1a2e;
                color: #eee;
                font-family: Segoe UI;
            }
            QPushButton {
                background-color: #16213e;
                border: 1px solid #0f3460;
                padding: 6px 10px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0f3460;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QSpinBox {
                background: #16213e;
                border: 1px solid #0f3460;
                padding: 3px;
                border-radius: 3px;
            }
        """)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # === Title bar ===
        title_bar = QFrame()
        title_bar.setStyleSheet("background: #16213e; border-radius: 5px;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 5, 10, 5)

        title = QLabel("🎰 GTA5 Roulette AI v2.2")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title_layout.addWidget(title)
        title_layout.addStretch()

        btn_minimize = QPushButton("−")
        btn_minimize.setFixedSize(25, 25)
        btn_minimize.clicked.connect(self.showMinimized)
        title_layout.addWidget(btn_minimize)

        btn_close = QPushButton("×")
        btn_close.setFixedSize(25, 25)
        btn_close.setStyleSheet("background: #e74c3c;")
        btn_close.clicked.connect(self.close)
        title_layout.addWidget(btn_close)

        layout.addWidget(title_bar)

        # === Status ===
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background: #0f3460; padding: 8px; border-radius: 5px;")
        layout.addWidget(self.status_label)

        # === Session frame ===
        session_frame = QFrame()
        session_frame.setStyleSheet("background: #1e3a5f; border-radius: 5px; padding: 5px;")
        session_layout = QVBoxLayout(session_frame)
        session_layout.setSpacing(4)

        session_row1 = QHBoxLayout()
        self.session_label = QLabel("📍 Session: Active")
        self.session_label.setStyleSheet("font-weight: bold;")
        session_row1.addWidget(self.session_label)
        session_row1.addStretch()
        
        self.btn_new_session = QPushButton("[F7] New Session")
        self.btn_new_session.setStyleSheet("background: #8e44ad;")
        session_row1.addWidget(self.btn_new_session)
        session_layout.addLayout(session_row1)

        session_row2 = QHBoxLayout()
        self.warmup_checkbox = QCheckBox("Warmup (skip first spins)")
        self.warmup_checkbox.setChecked(True)
        session_row2.addWidget(self.warmup_checkbox)
        
        self.warmup_spinbox = QSpinBox()
        self.warmup_spinbox.setRange(0, 50)
        self.warmup_spinbox.setValue(10)
        self.warmup_spinbox.setFixedWidth(50)
        session_row2.addWidget(self.warmup_spinbox)
        session_row2.addWidget(QLabel("spins"))
        session_row2.addStretch()
        session_layout.addLayout(session_row2)

        self.warmup_progress = QLabel("")
        self.warmup_progress.setStyleSheet("color: #f39c12;")
        session_layout.addWidget(self.warmup_progress)

        layout.addWidget(session_frame)

        # === Recommendations ===
        rec_frame = QFrame()
        rec_frame.setStyleSheet("background: #16213e; border-radius: 5px; padding: 5px;")
        rec_layout = QVBoxLayout(rec_frame)

        rec_title = QLabel("📊 Recommendations:")
        rec_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        rec_layout.addWidget(rec_title)

        self.rec_labels = []
        for i in range(4):
            lbl = QLabel("---")
            lbl.setStyleSheet("font-size: 13px; padding: 4px;")
            rec_layout.addWidget(lbl)
            self.rec_labels.append(lbl)

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setStyleSheet("color: #888;")
        rec_layout.addWidget(self.confidence_label)

        layout.addWidget(rec_frame)

        # === History ===
        history_frame = QFrame()
        history_frame.setStyleSheet("background: #16213e; border-radius: 5px; padding: 5px;")
        history_layout = QVBoxLayout(history_frame)

        history_header = QHBoxLayout()
        history_title = QLabel("📜 Last 10 Spins:")
        history_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        history_header.addWidget(history_title)
        history_header.addStretch()

        self.btn_undo = QPushButton("↩ Undo")
        self.btn_undo.setFixedWidth(60)
        self.btn_undo.setStyleSheet("background: #e67e22;")
        history_header.addWidget(self.btn_undo)

        self.btn_clear_history = QPushButton("🗑️")
        self.btn_clear_history.setFixedWidth(30)
        self.btn_clear_history.setStyleSheet("background: #c0392b;")
        self.btn_clear_history.setToolTip("Clear all history")
        history_header.addWidget(self.btn_clear_history)
        history_layout.addLayout(history_header)

        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(130)
        self.history_list.setStyleSheet("""
            QListWidget { background: #1a1a2e; border: none; }
            QListWidget::item { padding: 2px; }
        """)
        history_layout.addWidget(self.history_list)

        layout.addWidget(history_frame)

        # === Auto-bet ===
        self.autobet_frame = QFrame()
        self.autobet_frame.setStyleSheet("background: #2d2d44; border-radius: 5px; padding: 5px;")
        autobet_layout = QVBoxLayout(self.autobet_frame)

        autobet_row1 = QHBoxLayout()
        self.autobet_status = QLabel("🤖 Auto-Bet: OFF")
        self.autobet_status.setFont(QFont("Segoe UI", 10, QFont.Bold))
        autobet_row1.addWidget(self.autobet_status)
        autobet_row1.addStretch()
        autobet_layout.addLayout(autobet_row1)

        autobet_row2 = QHBoxLayout()
        autobet_row2.addWidget(QLabel("Delay:"))
        self.delay_spinbox = QSpinBox()
        self.delay_spinbox.setRange(0, 500)
        self.delay_spinbox.setValue(15)
        self.delay_spinbox.setSuffix(" ms")
        self.delay_spinbox.setFixedWidth(80)
        autobet_row2.addWidget(self.delay_spinbox)
        autobet_row2.addStretch()
        autobet_layout.addLayout(autobet_row2)

        self.calibration_status = QLabel("Calibration: 0/38")
        self.calibration_status.setStyleSheet("color: #888;")
        autobet_layout.addWidget(self.calibration_status)

        layout.addWidget(self.autobet_frame)

        # === Control buttons ===
        btn_frame = QFrame()
        btn_layout = QVBoxLayout(btn_frame)
        btn_layout.setSpacing(4)

        row1 = QHBoxLayout()
        self.btn_manual = QPushButton("[F2] Manual")
        self.btn_manual.clicked.connect(self._manual_input)
        row1.addWidget(self.btn_manual)

        self.btn_force = QPushButton("[F3] Force")
        row1.addWidget(self.btn_force)

        self.btn_pause = QPushButton("[F4] Pause")
        row1.addWidget(self.btn_pause)
        btn_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_autobet = QPushButton("[F5] Auto-Bet")
        self.btn_autobet.setStyleSheet("background: #9b59b6;")
        row2.addWidget(self.btn_autobet)

        self.btn_calibrate = QPushButton("[F6] Calibrate")
        self.btn_calibrate.setStyleSheet("background: #e67e22;")
        row2.addWidget(self.btn_calibrate)
        btn_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_train = QPushButton("🧠 Train Model")
        self.btn_train.setStyleSheet("background: #27ae60;")
        row3.addWidget(self.btn_train)

        self.btn_clear_model = QPushButton("🗑️ Reset Model")
        self.btn_clear_model.setStyleSheet("background: #c0392b;")
        row3.addWidget(self.btn_clear_model)
        btn_layout.addLayout(row3)

        row4 = QHBoxLayout()
        self.btn_select_region = QPushButton("[F8] Select Region")
        self.btn_select_region.setStyleSheet("background: #3498db;")
        row4.addWidget(self.btn_select_region)
        btn_layout.addLayout(row4)

        layout.addWidget(btn_frame)

        # === Stats ===
        self.stats_label = QLabel("Spins: 0 | Session: 0")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.stats_label)

        # Draggable title bar
        self._drag_pos = None
        title_bar.mousePressEvent = self._title_mouse_press
        title_bar.mouseMoveEvent = self._title_mouse_move

        self.resize(340, 720)

    def _title_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()

    def _title_mouse_move(self, event):
        if self._drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)

    def _manual_input(self):
        text, ok = QInputDialog.getText(
            self, "Manual Input",
            "Enter number (0-36 or 00):"
        )
        if ok and text:
            text = text.strip()
            if text == "00" or (text.isdigit() and 0 <= int(text) <= 36):
                self.manual_input_signal.emit(text)
            else:
                QMessageBox.warning(self, "Invalid", "Enter 0-36 or 00")

    def update_status(self, text: str):
        self.status_label.setText(text)

    def update_recommendations(self, recommendations: list, confidence: float):
        colors = {"red": "#e74c3c", "black": "#2c3e50", "green": "#27ae60", "gold": "#f1c40f"}

        for i, lbl in enumerate(self.rec_labels):
            if i < len(recommendations):
                rec = recommendations[i]
                color = colors.get(rec.get("color", ""), "#888")
                prob = rec.get("probability", 0) * 100

                if rec["type"] == "number":
                    text = "#{}: {} ({:.1f}%)".format(rec["rank"], rec["value"], prob)
                else:
                    text = "📍 {} ({:.1f}%)".format(rec["value"], prob)

                lbl.setText(text)
                lbl.setStyleSheet("font-size: 13px; padding: 4px; background: {}; border-radius: 3px;".format(color))
            else:
                lbl.setText("---")
                lbl.setStyleSheet("font-size: 13px; padding: 4px;")

        self.confidence_label.setText("Confidence: {:.0f}%".format(confidence * 100))

    def add_history(self, number: str, color: str):
        colors = {"red": "#e74c3c", "black": "#2c3e50", "green": "#27ae60"}
        item = QListWidgetItem("{} ({})".format(number, color))
        item.setForeground(QColor(colors.get(color, "#888")))
        self.history_list.insertItem(0, item)

        while self.history_list.count() > 10:
            self.history_list.takeItem(self.history_list.count() - 1)

    def remove_last_history(self):
        if self.history_list.count() > 0:
            self.history_list.takeItem(0)

    def clear_history_display(self):
        self.history_list.clear()

    def update_stats(self, total: int, session: int):
        self.stats_label.setText("Spins: {} | Session: {}".format(total, session))

    def update_autobet_status(self, enabled: bool, calibration_progress: dict):
        if enabled:
            self.autobet_status.setText("🤖 Auto-Bet: ON")
            self.autobet_status.setStyleSheet("color: #2ecc71;")
            self.autobet_frame.setStyleSheet("background: #1e5631; border-radius: 5px; padding: 5px;")
        else:
            self.autobet_status.setText("🤖 Auto-Bet: OFF")
            self.autobet_status.setStyleSheet("color: #e74c3c;")
            self.autobet_frame.setStyleSheet("background: #2d2d44; border-radius: 5px; padding: 5px;")

        self.calibration_status.setText("Calibration: {}/{} numbers, {}/{} sectors".format(
            calibration_progress["numbers_done"], calibration_progress["numbers_total"],
            calibration_progress["sectors_done"], calibration_progress["sectors_total"]
        ))

    def update_session_status(self, is_warmup: bool, warmup_current: int, warmup_total: int):
        if is_warmup:
            self.session_label.setText("📍 Session: Warmup...")
            self.session_label.setStyleSheet("font-weight: bold; color: #f39c12;")
            self.warmup_progress.setText("Observing: {}/{} spins".format(warmup_current, warmup_total))
            self.warmup_progress.setVisible(True)
        else:
            self.session_label.setText("📍 Session: Active")
            self.session_label.setStyleSheet("font-weight: bold; color: #2ecc71;")
            self.warmup_progress.setVisible(False)

    def get_warmup_settings(self) -> tuple:
        return (self.warmup_checkbox.isChecked(), self.warmup_spinbox.value())

    def get_bet_delay(self) -> int:
        return self.delay_spinbox.value()