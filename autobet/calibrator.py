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
    QApplication, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

from logger import log
from autobet.auto_clicker import AutoClicker


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

    def __init__(self, auto_clicker: AutoClicker):
        super().__init__()
        self.auto_clicker = auto_clicker
        self.screenshot = None
        self.current_item = None
        self.marked_positions = {}  # For visual feedback

        self._setup_ui()
        self._populate_list()
        self._take_screenshot()

    def _setup_ui(self):
        self.setWindowTitle("🎯 Roulette Position Calibrator")
        self.setMinimumSize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel - item list
        left_panel = QFrame()
        left_panel.setFixedWidth(250)
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("<b>Items to Calibrate:</b>"))

        self.item_list = QListWidget()
        self.item_list.itemClicked.connect(self._on_item_selected)
        left_layout.addWidget(self.item_list)

        # Progress label
        self.progress_label = QLabel("Progress: 0/0")
        left_layout.addWidget(self.progress_label)

        # Buttons
        btn_screenshot = QPushButton("📸 New Screenshot")
        btn_screenshot.clicked.connect(self._take_screenshot)
        left_layout.addWidget(btn_screenshot)

        btn_clear = QPushButton("🗑️ Clear Current")
        btn_clear.clicked.connect(self._clear_current)
        left_layout.addWidget(btn_clear)

        btn_save = QPushButton("💾 Save Calibration")
        btn_save.clicked.connect(self._save)
        btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        left_layout.addWidget(btn_save)

        btn_close = QPushButton("✖️ Close")
        btn_close.clicked.connect(self.close)
        left_layout.addWidget(btn_close)

        layout.addWidget(left_panel)

        # Right panel - screenshot
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        self.instruction_label = QLabel("Select an item from the list, then click on the screenshot")
        self.instruction_label.setStyleSheet("font-size: 14px; padding: 10px; background: #333; color: white;")
        right_layout.addWidget(self.instruction_label)

        self.image_label = ClickableImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a;")
        self.image_label.clicked.connect(self._on_image_clicked)
        right_layout.addWidget(self.image_label)

        layout.addWidget(right_panel, stretch=1)

    def _populate_list(self):
        """Populate list with all items to calibrate"""
        self.item_list.clear()

        # Numbers
        numbers = ["0", "00"] + [str(i) for i in range(1, 37)]
        for num in numbers:
            item = QListWidgetItem("🔢 " + num)
            item.setData(Qt.UserRole, ("number", num))
            if self.auto_clicker.is_calibrated(num):
                item.setForeground(QColor("#4CAF50"))
                item.setText("✅ " + num)
            self.item_list.addItem(item)

        # Sectors
        sectors = [
            "1st12 (1-12)", "2nd12 (13-24)", "3rd12 (25-36)",
            "1-18", "19-36", "EVEN", "ODD", "RED", "BLACK", "SPIN"
        ]
        for sector in sectors:
            item = QListWidgetItem("📍 " + sector)
            item.setData(Qt.UserRole, ("sector", sector))
            if self.auto_clicker.is_calibrated(sector):
                item.setForeground(QColor("#4CAF50"))
                item.setText("✅ " + sector)
            self.item_list.addItem(item)

        self._update_progress()

    def _update_progress(self):
        progress = self.auto_clicker.get_calibration_progress()
        self.progress_label.setText(
            "Numbers: {}/{} | Sectors: {}/{}".format(
                progress["numbers_done"], progress["numbers_total"],
                progress["sectors_done"], progress["sectors_total"]
            )
        )

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
            log.info("[Calibrator] Screenshot captured")

        except Exception as e:
            log.error("[Calibrator] Screenshot failed: " + str(e))
            QMessageBox.warning(self, "Error", "Failed to capture screenshot: " + str(e))

    def _update_image(self):
        """Update displayed image with markers"""
        if self.screenshot is None:
            return

        img = self.screenshot.copy()
        h, w = img.shape[:2]

        # Draw existing calibration points
        for key, pos in {**self.auto_clicker.positions, **{k: v for k, v in self.auto_clicker.sectors.items() if v}}.items():
            x, y = pos["x"], pos["y"]
            # Draw marker
            cv2_img = img
            # Draw circle
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx*dx + dy*dy <= 25:
                        px, py = x + dx, y + dy
                        if 0 <= px < w and 0 <= py < h:
                            cv2_img[py, px] = [0, 255, 0]  # Green

        # Convert to QPixmap
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit label while maintaining aspect ratio
        label_size = self.image_label.size()
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Calculate scale factor for click coordinate conversion
        scale = scaled.width() / w
        offset_x = (label_size.width() - scaled.width()) // 2
        offset_y = (label_size.height() - scaled.height()) // 2

        self.image_label.set_scale(scale, offset_x, offset_y)
        self.image_label.setPixmap(scaled)

    def _on_item_selected(self, item: QListWidgetItem):
        """Handle item selection"""
        self.current_item = item
        item_type, key = item.data(Qt.UserRole)
        self.instruction_label.setText(
            "👆 Click on the position for: <b>{}</b>".format(key)
        )
        self.instruction_label.setStyleSheet(
            "font-size: 16px; padding: 10px; background: #2196F3; color: white;"
        )

    def _on_image_clicked(self, x: int, y: int):
        """Handle click on screenshot"""
        if self.current_item is None:
            QMessageBox.information(self, "Info", "Please select an item from the list first!")
            return

        item_type, key = self.current_item.data(Qt.UserRole)

        # Save position
        self.auto_clicker.set_position(key, x, y)

        # Update list item
        self.current_item.setForeground(QColor("#4CAF50"))
        self.current_item.setText("✅ " + key)

        log.info("[Calibrator] Set {} '{}' to ({}, {})".format(item_type, key, x, y))

        # Update progress
        self._update_progress()

        # Update image to show new marker
        self._update_image()

        # Auto-select next item
        current_row = self.item_list.row(self.current_item)
        if current_row + 1 < self.item_list.count():
            next_item = self.item_list.item(current_row + 1)
            self.item_list.setCurrentItem(next_item)
            self._on_item_selected(next_item)
        else:
            self.instruction_label.setText("✅ All items calibrated!")
            self.instruction_label.setStyleSheet(
                "font-size: 16px; padding: 10px; background: #4CAF50; color: white;"
            )
            self.current_item = None

    def _clear_current(self):
        """Clear current item's calibration"""
        if self.current_item is None:
            return

        item_type, key = self.current_item.data(Qt.UserRole)

        if item_type == "number":
            if key in self.auto_clicker.positions:
                del self.auto_clicker.positions[key]
        else:
            self.auto_clicker.sectors[key] = None

        self.current_item.setForeground(QColor("white"))
        self.current_item.setText(("🔢 " if item_type == "number" else "📍 ") + key)

        self._update_progress()
        self._update_image()

    def _save(self):
        """Save calibration"""
        if self.auto_clicker.save_calibration():
            QMessageBox.information(self, "Success", "Calibration saved!")
        else:
            QMessageBox.warning(self, "Error", "Failed to save calibration!")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_image()


def run_calibrator(auto_clicker: AutoClicker = None):
    """Run calibrator as standalone or with existing AutoClicker"""
    app = QApplication.instance()
    standalone = app is None

    if standalone:
        app = QApplication(sys.argv)

    if auto_clicker is None:
        auto_clicker = AutoClicker()

    window = CalibratorWindow(auto_clicker)
    window.show()

    if standalone:
        sys.exit(app.exec_())
    else:
        return window


if __name__ == "__main__":
    run_calibrator()