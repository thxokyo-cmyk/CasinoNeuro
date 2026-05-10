"""
Number Detector v5.0 - TEMPLATE MATCHING APPROACH

Вместо OCR используем шаблонное сопоставление для цифр.
Это намного надежнее для фиксированного набора символов (0-36, 00).

Алгоритм:
1. Detect background COLOR first (most reliable)
2. Extract digit region
3. Match against pre-rendered templates for 0-9, 00
4. VALIDATE: only accept numbers that match the detected color
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import os
import time
from pathlib import Path

from logger import log
from utils.roulette_logic import (
    NUMBER_TO_COLOR, 
    COLOR_TO_NUMBERS, 
    get_color,
    is_valid_combination,
    find_similar_numbers,
    OCR_CONFUSION_MAP
)


class TemplateMatcher:
    """Template matching for roulette digits"""
    
    def __init__(self):
        self.templates = {}
        self.template_dir = "digit_templates"
        self._generate_templates()
        
    def _generate_templates(self):
        """Generate synthetic templates for digits 0-9 and 00"""
        os.makedirs(self.template_dir, exist_ok=True)
        
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_PLAIN,
        ]
        
        scales = [1.5, 2.0, 2.5, 3.0]
        thicknesses = [2, 3]
        
        # Generate templates for each digit
        for digit in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "00"]:
            self.templates[digit] = []
            
            for font in fonts:
                for scale in scales:
                    for thickness in thicknesses:
                        template = self._render_digit(digit, font, scale, thickness)
                        if template is not None:
                            self.templates[digit].append(template)
                            
                            # Save for debugging
                            if len(self.templates[digit]) == 1:
                                path = os.path.join(self.template_dir, f"{digit}.png")
                                cv2.imwrite(path, template)
        
        log.info(f"[TemplateMatcher] Generated {sum(len(t) for t in self.templates.values())} templates")
    
    def _render_digit(self, text: str, font: int, scale: float, thickness: int) -> np.ndarray:
        """Render a digit as a binary template"""
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        # Create image with padding
        img_width = text_width + 20
        img_height = text_height + 20
        img = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Draw text centered
        x = (img_width - text_width) // 2
        y = (img_height + text_height) // 2
        
        cv2.putText(img, text, (x, y), font, scale, 255, thickness)
        
        # Crop to content
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Add small padding
            pad = 2
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img_width - x, w + 2 * pad)
            h = min(img_height - y, h + 2 * pad)
            return img[y:y+h, x:x+w]
        
        return None
    
    def match(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Match image against all digit templates.
        Returns list of (digit, confidence) sorted by confidence.
        """
        if image is None or image.size == 0:
            return []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to get binary image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours to locate digit region
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get largest contour (the digit)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 50:  # Too small
            return []
        
        x, y, w, h = cv2.boundingRect(largest)
        
        # Extract digit ROI - use the full bounding box
        roi = gray[y:y+h, x:x+w]
        roi_thresh = thresh[y:y+h, x:x+w]
        
        results = []
        
        # Match against all templates
        for digit, templates in self.templates.items():
            best_score = 0
            
            for template in templates:
                # Skip if template aspect ratio is very different from ROI
                template_aspect = template.shape[1] / template.shape[0] if template.shape[0] > 0 else 0
                roi_aspect = roi.shape[1] / roi.shape[0] if roi.shape[0] > 0 else 0
                
                # Filter out single-digit templates for double-digit ROI and vice versa
                if abs(template_aspect - roi_aspect) > 0.5:
                    continue
                
                # Resize template to match ROI scale
                template_resized = cv2.resize(template, (roi.shape[1], roi.shape[0]))
                _, template_bin = cv2.threshold(template_resized, 128, 255, cv2.THRESH_BINARY)
                
                # Calculate match score using correlation
                score = self._calculate_match(roi_thresh, template_bin)
                best_score = max(best_score, score)
            
            if best_score > 0.5:  # Minimum threshold
                results.append((digit, best_score))
        
        # Sort by confidence
        results.sort(key=lambda x: -x[1])
        
        return results
    
    def _calculate_match(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate match score between two binary images"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Normalize
        img1_norm = img1.astype(float) / 255.0
        img2_norm = img2.astype(float) / 255.0
        
        # Correlation
        correlation = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
        
        # IoU (Intersection over Union)
        intersection = np.sum((img1 > 0) & (img2 > 0))
        union = np.sum((img1 > 0) | (img2 > 0))
        iou = intersection / union if union > 0 else 0
        
        # Combined score
        score = 0.5 * max(0, correlation) + 0.5 * iou
        
        return score


class NumberDetector:
    def __init__(self, debug=True):
        log.info("[Vision] Initializing Number Detector v5.0 (Template Matching)...")
        self.debug = debug
        self.debug_dir = "debug_captures"

        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Template matcher
        self.matcher = TemplateMatcher()

        # === Detection State ===
        self.last_recorded_number = None
        self.current_reading = None
        self.reading_count = 0
        self.readings_needed = 3  # Need 3 consistent reads
        self.last_record_time = 0
        self.cooldown_seconds = 8.0
        self.no_reading_count = 0
        
        # === Frame history for change detection ===
        self.frame_history = []
        self.max_frame_history = 5
        
        # === Static detection ===
        self.static_start_time = None
        self.static_threshold = 2.0  # seconds

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
        
        log.info("[Vision] Detector ready! Template matching enabled.")

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

    def _extract_digit_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract the digit region from the image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 50:
            return None
        
        x, y, w, h = cv2.boundingRect(largest)
        
        # Extract and return
        return gray[y:y+h, x:x+w]

    def process_frame(self, image: np.ndarray) -> Optional[Tuple[str, str]]:
        """
        Process a frame and detect the roulette number.
        Returns (number, color) or None.
        """
        if image is None or image.size == 0:
            return None

        # === Step 1: Detect Color (Most Reliable) ===
        detected_color = self._detect_color_robust(image)
        
        # === Step 2: Template Matching ===
        matches = self.matcher.match(image)
        
        # Get top candidates
        ocr_candidates = [m[0] for m in matches[:5] if m[1] > 0.5]
        
        log.debug(f"[Vision] Template matches: {ocr_candidates}")
        
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

    def _validate_and_correct(self, ocr_candidates: List[str], 
                              detected_color: str,
                              image: np.ndarray) -> Optional[Tuple[str, str]]:
        """
        Validate template matching against detected color.
        If mismatch, find a valid number that matches the color.
        """
        
        valid_numbers = COLOR_TO_NUMBERS.get(detected_color, [])
        
        # === SPECIAL CASE: Green (0 or 00) - Color is king! ===
        if detected_color == "green":
            # For green, we ONLY have two options: 0 or 00
            # Check template matches - prioritize 00 if both present
            if "00" in ocr_candidates:
                log.debug("[Vision] ✓ Detected 00 (green + template match)")
                return ("00", "green")
            
            # Check for "0" but also look at aspect ratio of the digit region
            if "0" in ocr_candidates:
                # Additional check: if image is wider than tall, likely 00
                h, w = image.shape[:2]
                if w > h * 1.5:  # Wide image suggests two digits
                    log.debug("[Vision] ✓ Detected 00 (green + wide aspect)")
                    return ("00", "green")
                
                # Also check contour analysis
                if self._looks_like_double_zero(image):
                    log.debug("[Vision] ✓ Detected 00 (green + double zero shape)")
                    return ("00", "green")
                
                log.debug("[Vision] ✓ Detected 0 (green + template match)")
                return ("0", "green")
            
            # Fallback: use shape analysis
            if self._looks_like_double_zero(image):
                log.debug("[Vision] ✓ Detected 00 (green + double zero shape)")
                return ("00", "green")
            
            # Default to 0
            return ("0", "green")
        
        # === For red/black: need template confirmation ===
        if not ocr_candidates:
            log.debug("[Vision] No template matches, color={}".format(detected_color))
            return None
        
        log.debug("[Vision] Template candidates: {}, color={}".format(ocr_candidates, detected_color))
        
        # === Priority 1: Exact match with correct color ===
        for candidate in ocr_candidates:
            if candidate in valid_numbers:
                log.debug("[Vision] ✓ Exact match: {} is {}".format(candidate, detected_color))
                return (candidate, detected_color)
        
        # === Priority 2: Find similar number with correct color ===
        for candidate in ocr_candidates:
            similar = find_similar_numbers(candidate, detected_color)
            if similar:
                best_match = similar[0]
                log.debug("[Vision] ✓ Corrected: {} → {} (color match)".format(
                    candidate, best_match))
                return (best_match, detected_color)
        
        # === Priority 3: Use OCR confusion map ===
        for candidate in ocr_candidates:
            if candidate in OCR_CONFUSION_MAP:
                for possible in OCR_CONFUSION_MAP[candidate]:
                    if possible in valid_numbers:
                        log.debug("[Vision] ✓ Confusion fix: {} → {}".format(
                            candidate, possible))
                        return (possible, detected_color)
        
        # === Fallback: First valid number of this color ===
        if valid_numbers:
            log.warning("[Vision] FALLBACK: using first valid {} number: {}".format(
                detected_color, valid_numbers[0]))
            return (valid_numbers[0], detected_color)
        
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
            c1, c2 = significant[0], significant[1]
            
            both_round = (0.5 < c1["aspect"] < 1.5) and (0.5 < c2["aspect"] < 1.5)
            gap = c2["x"] - (c1["x"] + c1["w"])
            has_gap = gap > 0
            
            if both_round and has_gap:
                log.debug("[Vision] Looks like 00: two round shapes with gap")
                return True
        
        return False

    def force_detect(self, image: np.ndarray) -> Optional[Tuple[str, str]]:
        """Force detection without cooldown"""
        if image is None or image.size == 0:
            return None
        
        detected_color = self._detect_color_robust(image)
        matches = self.matcher.match(image)
        ocr_candidates = [m[0] for m in matches[:5] if m[1] > 0.5]
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
        self.static_start_time = None
        log.info("[Vision] State reset")

    def _save_debug(self, image: np.ndarray, label: str):
        """Save debug image"""
        if self.debug:
            try:
                path = os.path.join(self.debug_dir, "{}_latest.png".format(label))
                cv2.imwrite(path, image)
            except Exception:
                pass
