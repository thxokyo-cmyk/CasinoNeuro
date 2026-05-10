"""
Screen capture module with visual region selector.
"""

import mss
import numpy as np
import cv2
import json
import os
from typing import Optional, Tuple

from logger import log


class ScreenCapture:
    def __init__(self, config_file: str = "capture_region.json"):
        self.sct = mss.mss()
        self.config_file = config_file
        
        # Default region (will be overwritten by config)
        self.result_region = None
        
        # Load saved region
        self._load_region()
        
        if self.result_region:
            log.info("[Capture] Loaded region: {}".format(self.result_region))
        else:
            log.warning("[Capture] No region configured! Use F8 to select.")

    def _load_region(self):
        """Load capture region from config file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    self.result_region = data.get("result_region")
                    log.info("[Capture] Region loaded from " + self.config_file)
            except Exception as e:
                log.error("[Capture] Failed to load region: " + str(e))

    def save_region(self, region: dict):
        """Save capture region to config file"""
        self.result_region = region
        try:
            with open(self.config_file, "w") as f:
                json.dump({"result_region": region}, f, indent=2)
            log.info("[Capture] Region saved: {}".format(region))
        except Exception as e:
            log.error("[Capture] Failed to save region: " + str(e))

    def capture_result_region(self) -> Optional[np.ndarray]:
        """Capture the result display region"""
        if not self.result_region:
            return None
            
        try:
            screenshot = self.sct.grab(self.result_region)
            img = np.array(screenshot)
            # Convert BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except Exception as e:
            log.error("[Capture] Error: " + str(e))
            return None

    def capture_full_screen(self) -> Optional[np.ndarray]:
        """Capture full primary monitor"""
        try:
            monitor = self.sct.monitors[1]
            screenshot = self.sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except Exception as e:
            log.error("[Capture] Full screen error: " + str(e))
            return None

    def get_monitor_info(self) -> dict:
        """Get primary monitor info"""
        return self.sct.monitors[1]

    def is_configured(self) -> bool:
        """Check if capture region is configured"""
        return self.result_region is not None


class RegionSelector:
    """
    Visual region selector - allows user to draw a rectangle on screen.
    """
    
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.selected_region = None
        
    def select_region(self) -> Optional[dict]:
        """
        Open fullscreen window and let user select region.
        Returns dict with {left, top, width, height} or None if cancelled.
        """
        import mss
        
        with mss.mss() as sct:
            # Capture full screen
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        self.original_img = img.copy()
        self.display_img = img.copy()
        self.selecting = False
        self.selected_region = None
        
        # Create window
        window_name = "SELECT RESULT REGION - Draw rectangle, then press ENTER (ESC to cancel)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # Add instructions overlay
        self._draw_instructions()
        
        while True:
            cv2.imshow(window_name, self.display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            elif key == 13 or key == 10:  # Enter
                if self.selected_region:
                    cv2.destroyAllWindows()
                    return self.selected_region
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.selecting = True
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.display_img = self.original_img.copy()
            cv2.rectangle(self.display_img, self.start_point, (x, y), (0, 255, 0), 2)
            self._draw_instructions()
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (x, y)
            
            # Calculate region
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            
            width = x2 - x1
            height = y2 - y1
            
            if width > 10 and height > 10:
                self.selected_region = {
                    "left": x1,
                    "top": y1,
                    "width": width,
                    "height": height
                }
                
                # Draw final rectangle
                self.display_img = self.original_img.copy()
                cv2.rectangle(self.display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Show size
                text = "{}x{} - Press ENTER to confirm".format(width, height)
                cv2.putText(self.display_img, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self._draw_instructions()
    
    def _draw_instructions(self):
        """Draw instruction text on image"""
        instructions = [
            "INSTRUCTIONS:",
            "1. Find where the RESULT NUMBER appears after a spin",
            "2. Draw a rectangle around ONLY that number",
            "3. Press ENTER to confirm, ESC to cancel",
            "",
            "TIP: Select the BIG number that shows after the ball lands,",
            "NOT the history panel on the side!"
        ]
        
        y = 30
        for line in instructions:
            cv2.putText(self.display_img, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y += 25