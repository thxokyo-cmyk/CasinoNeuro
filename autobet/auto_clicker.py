"""
Auto-betting module with configurable delays.
"""

import json
import os
import time
import pyautogui
from typing import Dict, List, Optional

from logger import log


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01  # Minimal internal pause


class AutoClicker:
    def __init__(self, calibration_file: str = "calibration.json"):
        self.calibration_file = calibration_file
        self.positions: Dict[str, dict] = {}
        self.enabled = False
        
        # === CONFIGURABLE DELAYS ===
        self.bet_delay_ms = 15  # Delay between bets in milliseconds
        self.click_duration = 0.05  # Click hold duration
        
        self.sectors = {
            "1st12 (1-12)": None,
            "2nd12 (13-24)": None,
            "3rd12 (25-36)": None,
            "1-18": None,
            "19-36": None,
            "EVEN": None,
            "ODD": None,
            "RED": None,
            "BLACK": None,
            "SPIN": None,
            "CLEAR": None,
            "REBET": None,
        }

        self.load_calibration()

    def set_bet_delay(self, delay_ms: int):
        """Set delay between bets in milliseconds"""
        self.bet_delay_ms = max(0, delay_ms)
        log.info("[AutoBet] Bet delay set to {}ms".format(self.bet_delay_ms))

    def load_calibration(self) -> bool:
        if not os.path.exists(self.calibration_file):
            log.info("[AutoBet] No calibration file: " + self.calibration_file)
            return False

        try:
            with open(self.calibration_file, "r") as f:
                data = json.load(f)

            self.positions = data.get("numbers", {})
            loaded_sectors = data.get("sectors", {})
            for key in self.sectors:
                if key in loaded_sectors:
                    self.sectors[key] = loaded_sectors[key]
            
            # Load settings
            settings = data.get("settings", {})
            self.bet_delay_ms = settings.get("bet_delay_ms", 15)

            log.info("[AutoBet] Loaded {} numbers, {} sectors, delay={}ms".format(
                len(self.positions),
                len([s for s in self.sectors.values() if s]),
                self.bet_delay_ms
            ))
            return True

        except Exception as e:
            log.error("[AutoBet] Load failed: " + str(e))
            return False

    def save_calibration(self) -> bool:
        try:
            data = {
                "numbers": self.positions,
                "sectors": self.sectors,
                "settings": {
                    "bet_delay_ms": self.bet_delay_ms
                }
            }
            with open(self.calibration_file, "w") as f:
                json.dump(data, f, indent=2)
            log.info("[AutoBet] Saved to " + self.calibration_file)
            return True
        except Exception as e:
            log.error("[AutoBet] Save failed: " + str(e))
            return False

    def set_position(self, key: str, x: int, y: int):
        if key in self.sectors:
            self.sectors[key] = {"x": x, "y": y}
        else:
            self.positions[key] = {"x": x, "y": y}

    def get_position(self, key: str) -> Optional[dict]:
        if key in self.sectors and self.sectors[key]:
            return self.sectors[key]
        return self.positions.get(key)

    def is_calibrated(self, key: str) -> bool:
        return self.get_position(key) is not None

    def get_calibration_progress(self) -> dict:
        all_numbers = ["0", "00"] + [str(i) for i in range(1, 37)]
        calibrated_numbers = sum(1 for n in all_numbers if n in self.positions)
        calibrated_sectors = sum(1 for s in self.sectors.values() if s is not None)

        return {
            "numbers_total": len(all_numbers),
            "numbers_done": calibrated_numbers,
            "sectors_total": len(self.sectors),
            "sectors_done": calibrated_sectors,
            "complete": calibrated_numbers == len(all_numbers)
        }

    def click_number(self, number: str) -> bool:
        pos = self.positions.get(number)
        if not pos:
            log.warning("[AutoBet] Number {} not calibrated".format(number))
            return False
        return self._click(pos["x"], pos["y"], "number " + number)

    def click_sector(self, sector: str) -> bool:
        pos = self.sectors.get(sector)
        if not pos:
            log.warning("[AutoBet] Sector {} not calibrated".format(sector))
            return False
        return self._click(pos["x"], pos["y"], "sector " + sector)

    def click_spin(self) -> bool:
        pos = self.sectors.get("SPIN")
        if not pos:
            log.warning("[AutoBet] SPIN not calibrated")
            return False
        return self._click(pos["x"], pos["y"], "SPIN")

    def click_clear(self) -> bool:
        pos = self.sectors.get("CLEAR")
        if not pos:
            return False
        return self._click(pos["x"], pos["y"], "CLEAR")

    def _click(self, x: int, y: int, label: str = "") -> bool:
        try:
            pyautogui.click(x, y, _pause=False)
            log.debug("[AutoBet] Click {} at ({}, {})".format(label, x, y))
            
            # Apply configured delay
            if self.bet_delay_ms > 0:
                time.sleep(self.bet_delay_ms / 1000.0)
            
            return True
        except Exception as e:
            log.error("[AutoBet] Click failed: " + str(e))
            return False

    def place_bets(self, recommendations: List[dict]) -> int:
        if not self.enabled:
            return 0

        successful = 0

        for rec in recommendations:
            if rec["type"] == "number":
                if self.click_number(rec["value"]):
                    successful += 1
            elif rec["type"] == "sector":
                if self.click_sector(rec["value"]):
                    successful += 1

        log.info("[AutoBet] Placed {}/{} bets".format(successful, len(recommendations)))
        return successful

    def enable(self):
        progress = self.get_calibration_progress()
        if progress["numbers_done"] < 10:
            log.warning("[AutoBet] Need more calibration ({}/38)".format(progress["numbers_done"]))
            return False
        self.enabled = True
        log.info("[AutoBet] ENABLED (delay={}ms)".format(self.bet_delay_ms))
        return True

    def disable(self):
        self.enabled = False
        log.info("[AutoBet] DISABLED")

    def toggle(self) -> bool:
        if self.enabled:
            self.disable()
        else:
            self.enable()
        return self.enabled