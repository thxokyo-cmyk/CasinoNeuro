"""
Number Detector v4.0 - COLOR-FIRST APPROACH

Algorithm:
1. Detect background COLOR first (most reliable)
2. Run OCR to get digit candidates
3. VALIDATE: only accept numbers that match the detected color
4. If mismatch: find similar-looking number with correct color

This ensures we NEVER report "4 red" (4 is black) or "1 black" (1 is red).
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import os
import time

from logger import log
from utils.roulette_logic import (
    NUMBER_TO_COLOR, 
    COLOR_TO_NUMBERS, 
    get_color,
    is_valid_combination,
    find_similar_numbers,
    OCR_CONFUSION_MAP
)


class NumberDetector:
    def __init__(self, debug=True):
        log.info("[Vision] Initializing Number Detector v4.0 (Color-First)...")
        self.debug = debug
        self.debug_dir = "debug_captures"

        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        # === Tesseract Setup ===
        self.use_tesseract = False
        self.pytesseract = None
        self._init_tesseract()

        # === Detection State ===
        self.last_recorded_number = None
        self.current_reading = None
        self.reading_count = 0
        self.readings_needed = 3  # Need 3 consistent reads
        self.last_record_time = 0
        self.cooldown_seconds = 8.0
        self.no_reading_count = 0

        # === Color Detection (HSV ranges) ===
        # Green (0, 00)
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])
        
        # Red (two ranges because red wraps around in HSV)
        self.red_lower1 = np.array([0, 70, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 70, 50])
        self.red_upper2 = np.array([180, 255, 255])
        
        # Black is detected by absence of green/red
        
        log.info("[Vision] Detector ready! Color-first validation enabled.")

    def _init_tesseract(self):
        """Initialize Tesseract OCR"""
        try:
            import pytesseract
            
            paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
            
            for path in paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    tessdata = os.path.join(os.path.dirname(path), "tessdata")
                    if os.path.exists(tessdata):
                        os.environ["TESSDATA_PREFIX"] = tessdata
                    self.use_tesseract = True
                    self.pytesseract = pytesseract
                    log.info("[Vision] Tesseract found: " + path)
                    return
                    
            log.warning("[Vision] Tesseract not found!")
            
        except ImportError:
            log.error("[Vision] pytesseract not installed!")

    def process_frame(self, image: np.ndarray) -> Optional[Tuple[str, str]]:
        """
        Process a frame and detect the roulette number.
        Returns (number, color) or None.
        """
        if image is None or image.size == 0:
            return None

        # === Step 1: Detect Color (Most Reliable) ===
        detected_color = self._detect_color_robust(image)
        
        # === Step 2: OCR Detection ===
        ocr_candidates = self._run_ocr_multi(image)
        
        # === Step 3: Validate and Correct ===
        result = self._validate_and_correct(ocr_candidates, detected_color, image)
        
        if result is None:
            self.no_reading_count += 1
            if self.no_reading_count > 10:
                self.current_reading = None
                self.reading_count = 0
            return None

        number, color = result
        self.no_reading_count = 0

        # === Stability Check ===
        if number == self.last_recorded_number:
            return None  # Same as last recorded

        if number == self.current_reading:
            self.reading_count += 1
        else:
            self.current_reading = number
            self.reading_count = 1
            log.debug("[Vision] New candidate: {} ({}) [1/{}]".format(
                number, color, self.readings_needed))

        # === Confirmation ===
        if self.reading_count >= self.readings_needed:
            now = time.time()
            if now - self.last_record_time < self.cooldown_seconds:
                return None  # Still in cooldown

            self.last_recorded_number = number
            self.last_record_time = now
            self.reading_count = 0

            log.info("[Vision] ✓ CONFIRMED: {} ({})".format(number, color))
            
            if self.debug:
                self._save_debug(image, "confirmed_{}_{}".format(number, color))

            return (number, color)

        return None

    def _detect_color_robust(self, image: np.ndarray) -> str:
        """
        Robust color detection from background.
        Samples multiple regions and uses voting.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        
        # Sample regions: corners and edges (avoid center where digit is)
        regions = []
        
        # Corners
        margin = max(5, min(h, w) // 6)
        regions.append(hsv[0:margin, 0:margin])  # Top-left
        regions.append(hsv[0:margin, w-margin:w])  # Top-right
        regions.append(hsv[h-margin:h, 0:margin])  # Bottom-left
        regions.append(hsv[h-margin:h, w-margin:w])  # Bottom-right
        
        # Edges (middle of each side)
        mid_h, mid_w = h // 2, w // 2
        edge_size = margin
        regions.append(hsv[0:edge_size, mid_w-edge_size:mid_w+edge_size])  # Top
        regions.append(hsv[h-edge_size:h, mid_w-edge_size:mid_w+edge_size])  # Bottom
        regions.append(hsv[mid_h-edge_size:mid_h+edge_size, 0:edge_size])  # Left
        regions.append(hsv[mid_h-edge_size:mid_h+edge_size, w-edge_size:w])  # Right
        
        # Count color votes
        green_votes = 0
        red_votes = 0
        black_votes = 0
        
        for region in regions:
            if region.size == 0:
                continue
                
            color = self._classify_region_color(region)
            if color == "green":
                green_votes += 1
            elif color == "red":
                red_votes += 1
            else:
                black_votes += 1
        
        # Decision with thresholds
        total_votes = green_votes + red_votes + black_votes
        
        if green_votes >= 2:  # Green is distinctive
            return "green"
        elif red_votes > black_votes and red_votes >= 2:
            return "red"
        elif black_votes >= 2:
            return "black"
        else:
            # Fallback: analyze whole background
            return self._detect_color_fallback(hsv)

    def _classify_region_color(self, hsv_region: np.ndarray) -> str:
        """Classify a single region's color"""
        if hsv_region.size == 0:
            return "black"
        
        # Green check
        green_mask = cv2.inRange(hsv_region, self.green_lower, self.green_upper)
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        if green_ratio > 0.15:
            return "green"
        
        # Red check
        red_mask1 = cv2.inRange(hsv_region, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_region, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / red_mask.size
        
        if red_ratio > 0.15:
            return "red"
        
        return "black"

    def _detect_color_fallback(self, hsv: np.ndarray) -> str:
        """Fallback color detection using whole image"""
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / red_mask.size
        
        if green_ratio > 0.08:
            return "green"
        elif red_ratio > 0.08:
            return "red"
        else:
            return "black"

    def _run_ocr_multi(self, image: np.ndarray) -> List[str]:
        """
        Run OCR with multiple preprocessing methods.
        Returns list of unique candidates.
        """
        if not self.use_tesseract:
            return []
        
        candidates = []
        
        # Preprocessing methods
        preprocessors = [
            ("standard", self._preprocess_standard),
            ("highcontrast", self._preprocess_highcontrast),
            ("adaptive", self._preprocess_adaptive),
            ("morphology", self._preprocess_morphology),
        ]
        
        for name, preprocess_func in preprocessors:
            try:
                processed = preprocess_func(image)
                if processed is not None:
                    text = self._ocr_single(processed)
                    if text and text not in candidates:
                        candidates.append(text)
                        log.debug("[Vision] OCR ({}): '{}'".format(name, text))
            except Exception as e:
                log.error("[Vision] Preprocess {} error: {}".format(name, str(e)))
        
        return candidates

    def _preprocess_standard(self, image: np.ndarray) -> np.ndarray:
        """Standard preprocessing"""
        scale = 4
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    def _preprocess_highcontrast(self, image: np.ndarray) -> np.ndarray:
        """High contrast with CLAHE"""
        scale = 4
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    def _preprocess_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Adaptive threshold"""
        scale = 4
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 5)
        return cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    def _preprocess_morphology(self, image: np.ndarray) -> np.ndarray:
        """Morphological preprocessing to clean up digits"""
        scale = 4
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphology
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cv2.copyMakeBorder(cleaned, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    def _ocr_single(self, processed: np.ndarray) -> Optional[str]:
        """Run single OCR pass"""
        if not self.use_tesseract or processed is None:
            return None
        
        try:
            # Multiple PSM modes for better detection
            configs = [
                "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789",  # Single char
                "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789",   # Single line
                "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789",   # Single word
                "--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789",  # Raw line
            ]
            
            for config in configs:
                text = self.pytesseract.image_to_string(processed, config=config)
                text = self._clean_ocr_text(text)
                
                if text and self._is_valid_ocr(text):
                    return text
                    
        except Exception as e:
            log.error("[Vision] OCR error: " + str(e))
        
        return None

    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR output"""
        text = text.strip().replace(" ", "").replace("\n", "")
        
        # Common OCR mistakes
        replacements = {
            "O": "0", "o": "0",
            "l": "1", "I": "1", "|": "1",
            "Z": "2", "z": "2",
            "S": "5", "s": "5",
            "B": "8",
            "g": "9", "q": "9",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Keep only digits
        text = "".join(c for c in text if c.isdigit())
        
        return text

    def _is_valid_ocr(self, text: str) -> bool:
        """Check if OCR result is potentially valid"""
        if not text:
            return False
        if text == "00":
            return True
        try:
            num = int(text)
            return 0 <= num <= 36
        except ValueError:
            return False

    def _validate_and_correct(self, ocr_candidates: List[str], 
                              detected_color: str,
                              image: np.ndarray) -> Optional[Tuple[str, str]]:
        """
        Validate OCR against detected color.
        If mismatch, find a valid number that matches the color.
        
        THIS IS THE KEY FUNCTION - ensures number-color consistency!
        """
        
        valid_numbers = COLOR_TO_NUMBERS.get(detected_color, [])
        
        if not ocr_candidates:
            log.debug("[Vision] No OCR candidates, color={}".format(detected_color))
            return None
        
        log.debug("[Vision] OCR candidates: {}, color={}".format(ocr_candidates, detected_color))
        
        # === Priority 1: Exact match with correct color ===
        for candidate in ocr_candidates:
            if candidate in valid_numbers:
                log.debug("[Vision] ✓ Exact match: {} is {}".format(candidate, detected_color))
                return (candidate, detected_color)
        
        # === Priority 2: Handle special cases ===
        
        # Special: 00 detection
        if detected_color == "green":
            # Check if image looks like double zero
            if self._looks_like_double_zero(image):
                log.debug("[Vision] ✓ Detected 00 (green + double zero shape)")
                return ("00", "green")
            else:
                # Single 0 - but check OCR for hints
                # If OCR strongly suggests "00", use that
                for candidate in ocr_candidates:
                    if candidate == "00" or candidate == "0":
                        pass  # Both are valid green
                
                # Default to single 0 unless it clearly looks like 00
                return ("0", "green")
        
        # === Priority 3: Find similar number with correct color ===
        for candidate in ocr_candidates:
            similar = find_similar_numbers(candidate, detected_color)
            if similar:
                best_match = similar[0]
                log.debug("[Vision] ✓ Corrected: {} → {} (color match)".format(
                    candidate, best_match))
                return (best_match, detected_color)
        
        # === Priority 4: Use OCR confusion map ===
        for candidate in ocr_candidates:
            if candidate in OCR_CONFUSION_MAP:
                for possible in OCR_CONFUSION_MAP[candidate]:
                    if possible in valid_numbers:
                        log.debug("[Vision] ✓ Confusion fix: {} → {}".format(
                            candidate, possible))
                        return (possible, detected_color)
        
        # === Priority 5: Structural analysis for 1/7/11 problem ===
        structure = self._analyze_structure(image)
        
        if detected_color == "red":
            # Red numbers with similar shapes: 1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36
            if structure["is_single_stroke"]:
                if structure["has_top_bar"]:
                    return ("7", "red")  # 7 has horizontal top
                else:
                    return ("1", "red")  # 1 is just vertical
        
        elif detected_color == "black":
            # Black numbers: 2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35
            if structure["is_double_stroke"]:
                return ("11", "black")  # Two vertical strokes
            elif structure["is_single_stroke"]:
                if structure["has_top_bar"]:
                    return ("17", "black")  # Check if it looks like 17
                else:
                    return ("4", "black")  # Could be 4 misread as 1
        
        # === Fallback: Trust color, use first valid similar ===
        for candidate in ocr_candidates:
            # Just find ANY number with the right color
            similar = find_similar_numbers(candidate, detected_color)
            if similar:
                log.warning("[Vision] Fallback match: {} → {} (forced color match)".format(
                    candidate, similar[0]))
                return (similar[0], detected_color)
        
        log.warning("[Vision] No valid match found for {} with color {}".format(
            ocr_candidates, detected_color))
        return None

    def _looks_like_double_zero(self, image: np.ndarray) -> bool:
        """
        Check if image contains "00" (two zeros).
        Uses contour analysis.
        """
        scale = 4
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h_img, w_img = thresh.shape[:2]
        min_area = (h_img * w_img) * 0.01
        
        # Find significant contours
        significant = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / h if h > 0 else 0
                significant.append({
                    "x": x, "w": w, "aspect": aspect
                })
        
        # Sort by x position
        significant.sort(key=lambda c: c["x"])
        
        # Check for two round shapes side by side
        if len(significant) >= 2:
            # Both should be roughly circular (aspect close to 1)
            c1, c2 = significant[0], significant[1]
            
            both_round = (0.5 < c1["aspect"] < 1.5) and (0.5 < c2["aspect"] < 1.5)
            gap = c2["x"] - (c1["x"] + c1["w"])
            has_gap = gap > 0
            
            if both_round and has_gap:
                log.debug("[Vision] Looks like 00: two round shapes with gap")
                return True
        
        return False

    def _analyze_structure(self, image: np.ndarray) -> dict:
        """
        Analyze digit structure to help distinguish 1/7/11.
        """
        scale = 4
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        h_img, w_img = thresh.shape[:2]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = (h_img * w_img) * 0.02
        components = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / h if h > 0 else 0
                
                # Check top region for horizontal bar
                top_region = thresh[y:y + max(1, h // 4), x:x + w]
                top_white = np.sum(top_region > 0) / max(1, top_region.size)
                
                components.append({
                    "x": x, "y": y, "w": w, "h": h,
                    "aspect": aspect,
                    "top_white": top_white,
                    "is_narrow": aspect < 0.4
                })
        
        components.sort(key=lambda c: c["x"])
        
        # Analyze
        is_single_stroke = len(components) == 1 and components[0]["is_narrow"]
        is_double_stroke = (
            len(components) == 2 and 
            components[0]["is_narrow"] and 
            components[1]["is_narrow"]
        )
        has_top_bar = any(c["top_white"] > 0.5 for c in components)
        
        return {
            "is_single_stroke": is_single_stroke,
            "is_double_stroke": is_double_stroke,
            "has_top_bar": has_top_bar,
            "num_components": len(components),
        }

    def force_detect(self, image: np.ndarray) -> Optional[Tuple[str, str]]:
        """Force detection without cooldown"""
        if image is None or image.size == 0:
            return None
        
        detected_color = self._detect_color_robust(image)
        ocr_candidates = self._run_ocr_multi(image)
        result = self._validate_and_correct(ocr_candidates, detected_color, image)
        
        if result:
            self.last_recorded_number = result[0]
            self.last_record_time = time.time()
            log.info("[Vision] FORCE: {} ({})".format(result[0], result[1]))
        
        return result

    def reset_state(self):
        """Reset detection state"""
        self.last_recorded_number = None
        self.current_reading = None
        self.reading_count = 0
        self.no_reading_count = 0
        log.info("[Vision] State reset")

    def _save_debug(self, image: np.ndarray, label: str):
        """Save debug image"""
        if self.debug:
            try:
                path = os.path.join(self.debug_dir, "{}_latest.png".format(label))
                cv2.imwrite(path, image)
            except Exception:
                pass