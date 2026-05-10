"""
Spin State Detector - Detects when the roulette has stopped.

Three methods:
1. Text detection on screen (wait for specific text like "Result", "Win", etc.)
2. Frame comparison (compare last 2 frames with previous 2, detect change)
3. Static region detection (check if roulette wheel area is static for 2+ seconds)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import time
from collections import deque

from logger import log


class SpinStateDetector:
    """Detects when roulette spin has ended and number can be read"""
    
    def __init__(self):
        log.info("[SpinState] Initializing...")
        
        # Frame history for comparison
        self.frame_history = deque(maxlen=10)
        self.result_region_history = deque(maxlen=20)
        
        # Static detection
        self.static_start_time = None
        self.static_threshold = 2.0  # seconds
        
        # Method selection
        self.method = "combined"  # "text", "change", "static", "combined"
        
        # Text detection (OCR-based)
        self.trigger_texts = ["RESULT", "WIN", "NO MORE BETS", "NUMBER"]
        
        # Change detection thresholds
        self.change_threshold = 0.15  # 15% difference = spinning
        self.stable_count_needed = 3
        
        # Result region (where the number appears)
        self.result_region = None
        
        log.info("[SpinState] Ready!")
    
    def set_result_region(self, region: dict):
        """Set the region where the result number appears"""
        self.result_region = region
        log.info(f"[SpinState] Result region set: {region}")
    
    def process_frame(self, frame: np.ndarray, full_screen: bool = False) -> str:
        """
        Process frame and return state:
        - "spinning": wheel is still moving
        - "stopped": wheel has stopped, ready to read number
        - "uncertain": not enough data
        """
        if frame is None or frame.size == 0:
            return "uncertain"
        
        # Add to history
        self.frame_history.append(frame.copy())
        
        # Need at least 3 frames for comparison
        if len(self.frame_history) < 3:
            return "uncertain"
        
        results = {}
        
        # Method 1: Change detection
        results["change"] = self._detect_change()
        
        # Method 2: Static detection (if result region defined)
        if self.result_region:
            results["static"] = self._detect_static(frame)
        
        # Method 3: Text detection (optional, requires OCR)
        # results["text"] = self._detect_trigger_text(frame)
        
        # Combine results based on method
        if self.method == "change":
            return results.get("change", "uncertain")
        elif self.method == "static":
            return results.get("static", "uncertain")
        elif self.method == "combined":
            return self._combine_results(results)
        else:
            return "uncertain"
    
    def _detect_change(self) -> str:
        """
        Method 2: Compare last 2 frames with previous 2 frames.
        If significant change detected = spinning.
        If stable = stopped.
        """
        if len(self.frame_history) < 4:
            return "uncertain"
        
        frames = list(self.frame_history)
        
        # Compare recent vs older frames
        recent_1 = frames[-1]
        recent_2 = frames[-2]
        older_1 = frames[-3]
        older_2 = frames[-4]
        
        # Calculate differences
        diff_recent = self._frame_difference(recent_1, recent_2)
        diff_older = self._frame_difference(older_1, older_2)
        diff_cross = self._frame_difference(recent_1, older_1)
        
        avg_diff = (diff_recent + diff_older + diff_cross) / 3
        
        log.debug(f"[SpinState] Change detection: avg_diff={avg_diff:.3f}")
        
        if avg_diff > self.change_threshold:
            # Still changing = spinning
            self.static_start_time = None
            return "spinning"
        else:
            # Stable = potentially stopped
            if self.static_start_time is None:
                self.static_start_time = time.time()
            
            elapsed = time.time() - self.static_start_time
            
            if elapsed >= 1.0:  # Stable for 1 second
                return "stopped"
            else:
                return "uncertain"
    
    def _detect_static(self, frame: np.ndarray) -> str:
        """
        Method 3: Check if result region is static for 2+ seconds.
        """
        if self.result_region is None:
            return "uncertain"
        
        # Extract result region
        x = self.result_region.get("x", 0)
        y = self.result_region.get("y", 0)
        w = self.result_region.get("width", 100)
        h = self.result_region.get("height", 100)
        
        # Ensure region is within frame bounds
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w <= 0 or h <= 0:
            return "uncertain"
        
        roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Resize to standard size for comparison
        gray_resized = cv2.resize(gray, (50, 50))
        
        # Add to history
        self.result_region_history.append(gray_resized)
        
        if len(self.result_region_history) < 5:
            return "uncertain"
        
        # Check stability
        recent = list(self.result_region_history)[-5:]
        
        diffs = []
        for i in range(len(recent) - 1):
            diff = self._frame_difference(recent[i], recent[i+1])
            diffs.append(diff)
        
        avg_diff = sum(diffs) / len(diffs) if diffs else 1.0
        
        log.debug(f"[SpinState] Static detection: avg_diff={avg_diff:.3f}")
        
        if avg_diff > 0.1:  # Still changing
            self.static_start_time = None
            return "spinning"
        else:
            if self.static_start_time is None:
                self.static_start_time = time.time()
            
            elapsed = time.time() - self.static_start_time
            
            if elapsed >= self.static_threshold:
                return "stopped"
            elif elapsed >= 1.0:
                return "uncertain"
            else:
                return "spinning"
    
    def _detect_trigger_text(self, frame: np.ndarray) -> str:
        """
        Method 1: Wait for specific text on screen.
        Requires OCR (pytesseract).
        """
        try:
            import pytesseract
        except ImportError:
            return "uncertain"
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # OCR
        try:
            text = pytesseract.image_to_string(thresh, config="--psm 6")
            text = text.upper().strip()
            
            for trigger in self.trigger_texts:
                if trigger in text:
                    log.debug(f"[SpinState] Trigger text detected: {trigger}")
                    return "stopped"
        except Exception as e:
            log.error(f"[SpinState] OCR error: {e}")
        
        return "uncertain"
    
    def _frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate normalized difference between two frames"""
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1.copy()
        
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2.copy()
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1.astype(float), gray2.astype(float))
        
        # Normalize
        diff_normalized = diff / 255.0
        
        # Return mean difference
        return float(np.mean(diff_normalized))
    
    def _combine_results(self, results: dict) -> str:
        """Combine multiple detection methods"""
        # If any method says "spinning", we're spinning
        if any(v == "spinning" for v in results.values()):
            return "spinning"
        
        # If all available methods say "stopped", we're stopped
        stopped_count = sum(1 for v in results.values() if v == "stopped")
        total_methods = len(results)
        
        if stopped_count >= total_methods and stopped_count > 0:
            return "stopped"
        
        return "uncertain"
    
    def reset(self):
        """Reset detector state"""
        self.frame_history.clear()
        self.result_region_history.clear()
        self.static_start_time = None
        log.info("[SpinState] Reset")


class NumberCaptureTrigger:
    """
    Triggers number capture based on configurable strategy.
    
    Strategies:
    1. "immediate": Capture immediately when color changes
    2. "stable": Wait for frame stability (default)
    3. "manual": Only capture on manual trigger
    4. "smart": Combine multiple signals
    """
    
    def __init__(self, strategy: str = "stable"):
        self.strategy = strategy
        self.state_detector = SpinStateDetector()
        self.last_state = "uncertain"
        self.transition_time = None
        
        log.info(f"[CaptureTrigger] Initialized with strategy: {strategy}")
    
    def should_capture(self, frame: np.ndarray, detected_color: str) -> bool:
        """
        Determine if we should capture the number now.
        
        Args:
            frame: Current frame
            detected_color: Color detected in this frame
        
        Returns:
            True if should capture, False otherwise
        """
        if frame is None:
            return False
        
        if self.strategy == "immediate":
            # Capture as soon as we have a color
            return detected_color is not None
        
        elif self.strategy == "stable":
            # Wait for stable state
            state = self.state_detector.process_frame(frame)
            
            if state != self.last_state:
                log.debug(f"[CaptureTrigger] State changed: {self.last_state} → {state}")
                if state == "stopped":
                    self.transition_time = time.time()
            
            self.last_state = state
            
            if state == "stopped":
                # Additional delay after stopping to ensure number is visible
                if self.transition_time and (time.time() - self.transition_time) > 0.5:
                    return True
            
            return False
        
        elif self.strategy == "manual":
            # Never auto-capture
            return False
        
        elif self.strategy == "smart":
            # Smart combination
            state = self.state_detector.process_frame(frame)
            
            # Fast path: if color changed and we were spinning
            if self.last_state == "spinning" and state == "stopped":
                self.transition_time = time.time()
            
            self.last_state = state
            
            if state == "stopped":
                # Wait a bit for the number display to settle
                if self.transition_time:
                    elapsed = time.time() - self.transition_time
                    
                    # Capture between 0.5 and 3 seconds after stopping
                    if 0.5 <= elapsed <= 3.0:
                        return True
            
            return False
        
        return False
    
    def force_capture(self) -> bool:
        """Force a capture regardless of state"""
        log.info("[CaptureTrigger] Force capture triggered")
        return True
    
    def reset(self):
        """Reset trigger state"""
        self.state_detector.reset()
        self.last_state = "uncertain"
        self.transition_time = None
