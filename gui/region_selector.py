"""
PyQt5-based region selector with better UX.
"""

import sys
import mss
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QRubberBand
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

from logger import log


class RegionSelectorWindow(QWidget):
    """Fullscreen overlay for selecting capture region"""
    
    region_selected = pyqtSignal(dict)
    cancelled = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        self.start_pos = None
        self.end_pos = None
        self.rubber_band = None
        self.screenshot = None
        
        self._setup_ui()
        self._capture_screen()
        
    def _setup_ui(self):
        # Fullscreen frameless window
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.CrossCursor)
        
        # Rubber band for selection
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        
    def _capture_screen(self):
        """Capture current screen as background"""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            
        # Convert to QPixmap
        h, w = img.shape[:2]
        # BGRA to RGBA
        img_rgba = img.copy()
        img_rgba[:, :, [0, 2]] = img[:, :, [2, 0]]  # Swap R and B
        
        qimg = QImage(img_rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
        self.screenshot = QPixmap.fromImage(qimg)
        self.setFixedSize(w, h)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Draw screenshot
        if self.screenshot:
            painter.drawPixmap(0, 0, self.screenshot)
            
        # Draw semi-transparent overlay
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
        # Draw instructions
        painter.setPen(QPen(QColor(255, 255, 0)))
        painter.setFont(QFont("Arial", 16, QFont.Bold))
        
        instructions = [
            "🎯 SELECT THE RESULT NUMBER REGION",
            "",
            "1. Find where the winning number appears AFTER a spin",
            "2. Click and drag to draw a rectangle around it",
            "3. Release to confirm",
            "",
            "⚠️ Select the BIG number display, NOT the side history panel!",
            "",
            "Press ESC to cancel"
        ]
        
        y = 50
        for line in instructions:
            painter.drawText(20, y, line)
            y += 30
            
        # If we have a selection, highlight it
        if self.start_pos and self.end_pos:
            rect = QRect(self.start_pos, self.end_pos).normalized()
            
            # Clear the selected area (show original screenshot)
            painter.setClipRect(rect)
            painter.drawPixmap(0, 0, self.screenshot)
            painter.setClipRect(self.rect())
            
            # Draw border
            painter.setPen(QPen(QColor(0, 255, 0), 3))
            painter.drawRect(rect)
            
            # Draw size label
            size_text = "{}x{} pixels".format(rect.width(), rect.height())
            painter.setPen(QPen(QColor(0, 255, 0)))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(rect.left(), rect.top() - 10, size_text)
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.rubber_band.setGeometry(QRect(self.start_pos, QSize()))
            self.rubber_band.show()
            
    def mouseMoveEvent(self, event):
        if self.start_pos:
            self.end_pos = event.pos()
            self.rubber_band.setGeometry(QRect(self.start_pos, self.end_pos).normalized())
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_pos:
            self.end_pos = event.pos()
            self.rubber_band.hide()
            
            rect = QRect(self.start_pos, self.end_pos).normalized()
            
            if rect.width() > 20 and rect.height() > 20:
                region = {
                    "left": rect.left(),
                    "top": rect.top(),
                    "width": rect.width(),
                    "height": rect.height()
                }
                log.info("[RegionSelector] Selected: {}".format(region))
                self.region_selected.emit(region)
                self.close()
            else:
                # Too small, reset
                self.start_pos = None
                self.end_pos = None
                self.update()
                
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            log.info("[RegionSelector] Cancelled")
            self.cancelled.emit()
            self.close()


def select_region_dialog() -> dict:
    """
    Show region selector and return selected region.
    Returns None if cancelled.
    """
    result = {"region": None}
    
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    def on_selected(region):
        result["region"] = region
        
    selector = RegionSelectorWindow()
    selector.region_selected.connect(on_selected)
    selector.show()
    
    # Wait for window to close
    while selector.isVisible():
        app.processEvents()
        
    return result["region"]