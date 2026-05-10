"""
Visual Calibrator for auto-betting positions.
User clicks on screenshot to set coordinates for each number/sector.
"""

import sys
import os
import mss
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QMessageBox,
    QApplication, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont

from logger import log


class ClickableImageLabel(QLabel):
    """Label that captures mouse clicks and emits coordinates"""
    clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Convert widget coordinates to image coordinates
            x = int((event.x() - self.offset_x) / self.scale_factor)
            y = int((event.y() - self.offset_y) / self.scale_factor)
            self.clicked.emit(x, y)

    def set_scale(self, factor, offset_x=0, offset_y=0):
        self.scale_factor = factor
        self.offset_x = offset_x
        self.offset_y = offset_y


class CalibratorWindow(QMainWindow):
    """Main calibration window"""

    def __init__(self, auto_clicker):
        super().__init__()
        self.auto_clicker = auto_clicker
        self.screenshot = None
        self.current_item = None
        self.original_pixmap = None

        self._setup_ui()
        self._populate_list()
        self._take_screenshot()

    def _setup_ui(self):
        self.setWindowTitle("🎯 Roulette Position Calibrator")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #eee;
                font-family: Segoe UI;
            }
            QPushButton {
                background-color: #16213e;
                border: 1px solid #0f3460;
                padding: 8px 15px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #0f3460;
            }
            QListWidget {
                background-color: #16213e;
                border: 1px solid #0f3460;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #0f3460;
            }
            QListWidget::item:selected {
                background-color: #e94560;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # === Left panel - item list ===
        left_panel = QFrame()
        left_panel.setFixedWidth(280)
        left_panel.setStyleSheet("background: #16213e; border-radius: 10px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("📋 Items to Calibrate")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        self.item_list = QListWidget()
        self.item_list.itemClicked.connect(self._on_item_selected)
        left_layout.addWidget(self.item_list)

        # Progress label
        self.progress_label = QLabel("Progress: 0/0")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 14px; color: #4CAF50; padding: 10px;")
        left_layout.addWidget(self.progress_label)

        # Buttons
        btn_screenshot = QPushButton("📸 New Screenshot")
        btn_screenshot.clicked.connect(self._take_screenshot)
        left_layout.addWidget(btn_screenshot)

        btn_clear = QPushButton("🗑️ Clear Selected")
        btn_clear.clicked.connect(self._clear_current)
        left_layout.addWidget(btn_clear)

        btn_clear_all = QPushButton("💥 Clear ALL")
        btn_clear_all.setStyleSheet("background-color: #c0392b;")
        btn_clear_all.clicked.connect(self._clear_all)
        left_layout.addWidget(btn_clear_all)

        btn_save = QPushButton("💾 Save Calibration")
        btn_save.clicked.connect(self._save)
        btn_save.setStyleSheet("background-color: #27ae60; font-weight: bold; font-size: 14px;")
        left_layout.addWidget(btn_save)

        btn_close = QPushButton("✖️ Close")
        btn_close.clicked.connect(self.close)
        left_layout.addWidget(btn_close)

        layout.addWidget(left_panel)

        # === Right panel - screenshot ===
        right_panel = QFrame()
        right_panel.setStyleSheet("background: #0f0f1a; border-radius: 10px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Instruction label
        self.instruction_label = QLabel("👆 Select an item from the list, then click on the screenshot")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setStyleSheet("""
            font-size: 16px; 
            padding: 15px; 
            background: #16213e; 
            border-radius: 8px;
            color: #fff;
        """)
        right_layout.addWidget(self.instruction_label)

        # Scroll area for image
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")

        self.image_label = ClickableImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.clicked.connect(self._on_image_clicked)
        scroll.setWidget(self.image_label)

        right_layout.addWidget(scroll)

        # Coordinates label
        self.coords_label = QLabel("Last click: --")
        self.coords_label.setAlignment(Qt.AlignCenter)
        self.coords_label.setStyleSheet("color: #888; padding: 5px;")
        right_layout.addWidget(self.coords_label)

        layout.addWidget(right_panel, stretch=1)

    def _populate_list(self):
        """Populate list with all items to calibrate"""
        self.item_list.clear()

        # Numbers section
        header_numbers = QListWidgetItem("═══ NUMBERS ═══")
        header_numbers.setFlags(Qt.NoItemFlags)
        header_numbers.setForeground(QColor("#888"))
        self.item_list.addItem(header_numbers)

        numbers = ["0", "00"] + [str(i) for i in range(1, 37)]
        for num in numbers:
            item = QListWidgetItem("  🔢 " + num)
            item.setData(Qt.UserRole, ("number", num))
            if self.auto_clicker.is_calibrated(num):
                item.setForeground(QColor("#4CAF50"))
                item.setText("  ✅ " + num)
            self.item_list.addItem(item)

        # Sectors section
        header_sectors = QListWidgetItem("═══ SECTORS ═══")
        header_sectors.setFlags(Qt.NoItemFlags)
        header_sectors.setForeground(QColor("#888"))
        self.item_list.addItem(header_sectors)

        sectors = [
            ("1st12 (1-12)", "1st Dozen"),
            ("2nd12 (13-24)", "2nd Dozen"),
            ("3rd12 (25-36)", "3rd Dozen"),
            ("1-18", "Low (1-18)"),
            ("19-36", "High (19-36)"),
            ("EVEN", "Even"),
            ("ODD", "Odd"),
            ("RED", "Red"),
            ("BLACK", "Black"),
        ]

        for sector_key, sector_name in sectors:
            item = QListWidgetItem("  📍 " + sector_name)
            item.setData(Qt.UserRole, ("sector", sector_key))
            if self.auto_clicker.is_calibrated(sector_key):
                item.setForeground(QColor("#4CAF50"))
                item.setText("  ✅ " + sector_name)
            self.item_list.addItem(item)

        # Special buttons
        header_buttons = QListWidgetItem("═══ BUTTONS ═══")
        header_buttons.setFlags(Qt.NoItemFlags)
        header_buttons.setForeground(QColor("#888"))
        self.item_list.addItem(header_buttons)

        special = [
            ("SPIN", "🎡 SPIN Button"),
            ("CLEAR", "🗑️ CLEAR Bets"),
            ("REBET", "🔄 REBET"),
        ]

        for key, name in special:
            item = QListWidgetItem("  " + name)
            item.setData(Qt.UserRole, ("sector", key))
            if self.auto_clicker.is_calibrated(key):
                item.setForeground(QColor("#4CAF50"))
                item.setText("  ✅ " + name.split(" ", 1)[1])
            self.item_list.addItem(item)

        self._update_progress()

    def _update_progress(self):
        """Update progress display"""
        progress = self.auto_clicker.get_calibration_progress()
        total_items = progress["numbers_total"] + progress["sectors_total"]
        done_items = progress["numbers_done"] + progress["sectors_done"]

        percent = (done_items / total_items * 100) if total_items > 0 else 0

        self.progress_label.setText(
            "Progress: {}/{} ({:.0f}%)\nNumbers: {}/{} | Sectors: {}/{}".format(
                done_items, total_items, percent,
                progress["numbers_done"], progress["numbers_total"],
                progress["sectors_done"], progress["sectors_total"]
            )
        )

        if progress["numbers_done"] >= 10:
            self.progress_label.setStyleSheet("font-size: 14px; color: #4CAF50; padding: 10px;")
        else:
            self.progress_label.setStyleSheet("font-size: 14px; color: #f39c12; padding: 10px;")

    def _take_screenshot(self):
        """Capture current screen"""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                # Convert BGRA to RGB
                img = img[:, :, :3]
                img = img[:, :, ::-1].copy()  # BGR to RGB

            self.screenshot = img
            self._update_image()
            log.info("[Calibrator] Screenshot captured: {}x{}".format(img.shape[1], img.shape[0]))

        except Exception as e:
            log.error("[Calibrator] Screenshot failed: " + str(e))
            QMessageBox.warning(self, "Error", "Failed to capture screenshot:\n" + str(e))

    def _update_image(self):
        """Update displayed image with markers"""
        if self.screenshot is None:
            return

        img = self.screenshot.copy()
        h, w = img.shape[:2]

        # Draw existing calibration points
        all_positions = {}
        all_positions.update(self.auto_clicker.positions)
        for k, v in self.auto_clicker.sectors.items():
            if v is not None:
                all_positions[k] = v

        for key, pos in all_positions.items():
            x, y = pos["x"], pos["y"]
            # Draw crosshair
            color = [0, 255, 0]  # Green
            thickness = 2
            size = 15

            # Horizontal line
            for dx in range(-size, size + 1):
                px = x + dx
                if 0 <= px < w:
                    for t in range(-thickness, thickness + 1):
                        py = y + t
                        if 0 <= py < h:
                            img[py, px] = color

            # Vertical line
            for dy in range(-size, size + 1):
                py = y + dy
                if 0 <= py < h:
                    for t in range(-thickness, thickness + 1):
                        px = x + t
                        if 0 <= px < w:
                            img[py, px] = color

        # Convert to QPixmap
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.original_pixmap = pixmap

        # Scale to fit (max 80% of screen)
        screen = QApplication.primaryScreen().geometry()
        max_w = int(screen.width() * 0.7)
        max_h = int(screen.height() * 0.75)

        scaled = pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Calculate scale factor for click coordinate conversion
        self.image_label.set_scale(
            scaled.width() / w,
            0, 0
        )

        self.image_label.setPixmap(scaled)

    def _on_item_selected(self, item: QListWidgetItem):
        """Handle item selection"""
        data = item.data(Qt.UserRole)
        if data is None:
            return  # Header item

        self.current_item = item
        item_type, key = data

        self.instruction_label.setText("👆 Click on position for: <b>{}</b>".format(key))
        self.instruction_label.setStyleSheet("""
            font-size: 18px; 
            padding: 15px; 
            background: #e94560; 
            border-radius: 8px;
            color: #fff;
        """)

    def _on_image_clicked(self, x: int, y: int):
        """Handle click on screenshot"""
        self.coords_label.setText("Last click: ({}, {})".format(x, y))

        if self.current_item is None:
            QMessageBox.information(self, "Info", "Please select an item from the list first!")
            return

        data = self.current_item.data(Qt.UserRole)
        if data is None:
            return

        item_type, key = data

        # Save position
        self.auto_clicker.set_position(key, x, y)

        # Update list item
        self.current_item.setForeground(QColor("#4CAF50"))
        old_text = self.current_item.text()
        # Replace icon with checkmark
        new_text = old_text.replace("🔢", "✅").replace("📍", "✅")
        if "✅" not in new_text:
            new_text = "  ✅" + old_text.split(" ", 2)[-1]
        self.current_item.setText(new_text)

        log.info("[Calibrator] Set {} '{}' to ({}, {})".format(item_type, key, x, y))

        # Update progress
        self._update_progress()

        # Update image to show new marker
        self._update_image()

        # Auto-select next item
        self._select_next_item()

    def _select_next_item(self):
        """Select next uncalibrated item"""
        current_row = self.item_list.row(self.current_item)

        for i in range(current_row + 1, self.item_list.count()):
            item = self.item_list.item(i)
            data = item.data(Qt.UserRole)
            if data is None:
                continue  # Skip headers

            item_type, key = data
            if not self.auto_clicker.is_calibrated(key):
                self.item_list.setCurrentItem(item)
                self._on_item_selected(item)
                return

        # All done!
        self.instruction_label.setText("✅ All items calibrated! Click Save.")
        self.instruction_label.setStyleSheet("""
            font-size: 18px; 
            padding: 15px; 
            background: #27ae60; 
            border-radius: 8px;
            color: #fff;
        """)
        self.current_item = None

    def _clear_current(self):
        """Clear current item's calibration"""
        if self.current_item is None:
            QMessageBox.information(self, "Info", "Select an item first!")
            return

        data = self.current_item.data(Qt.UserRole)
        if data is None:
            return

        item_type, key = data

        if item_type == "number":
            if key in self.auto_clicker.positions:
                del self.auto_clicker.positions[key]
        else:
            self.auto_clicker.sectors[key] = None

        # Update list item appearance
        self.current_item.setForeground(QColor("#eee"))
        old_text = self.current_item.text()
        new_text = old_text.replace("✅", "🔢" if item_type == "number" else "📍")
        self.current_item.setText(new_text)

        self._update_progress()
        self._update_image()
        log.info("[Calibrator] Cleared position for: " + key)

    def _clear_all(self):
        """Clear all calibrations"""
        reply = QMessageBox.question(
            self, "Confirm",
            "Clear ALL calibrated positions?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.auto_clicker.positions.clear()
            for key in self.auto_clicker.sectors:
                self.auto_clicker.sectors[key] = None

            self._populate_list()
            self._update_image()
            log.info("[Calibrator] Cleared all positions")

    def _save(self):
        """Save calibration"""
        if self.auto_clicker.save_calibration():
            progress = self.auto_clicker.get_calibration_progress()
            QMessageBox.information(
                self, "Saved!",
                "Calibration saved!\n\nNumbers: {}/{}\nSectors: {}/{}".format(
                    progress["numbers_done"], progress["numbers_total"],
                    progress["sectors_done"], progress["sectors_total"]
                )
            )
        else:
            QMessageBox.warning(self, "Error", "Failed to save calibration!")

    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        if self.screenshot is not None:
            self._update_image()


def run_calibrator(auto_clicker=None):
    """Run calibrator as standalone or with existing AutoClicker"""
    from autobet.auto_clicker import AutoClicker

    app = QApplication.instance()
    standalone = app is None

    if standalone:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")

    if auto_clicker is None:
        auto_clicker = AutoClicker()

    window = CalibratorWindow(auto_clicker)
    window.show()

    if standalone:
        sys.exit(app.exec_())
    else:
        return window


if __name__ == "__main__":
    # Allow running directly for testing
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from logger import log
    from autobet.auto_clicker import AutoClicker

    run_calibrator()