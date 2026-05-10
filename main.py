"""
GTA5 Roulette AI v2.2
- Fixed region selection
- Added F8 hotkey for visual region selector
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer

from logger import log
from capture.screen_capture import ScreenCapture
from vision.number_detector import NumberDetector
from vision.spin_state_detector import NumberCaptureTrigger, SpinStateDetector
from data.database import SpinDatabase
from ml.trainer import RouletteTrainer
from ml.predictor import RoulettePredictor, get_color
from gui.overlay import OverlayWindow, SpinWorker, HotkeyWorker
from autobet.auto_clicker import AutoClicker


class RouletteApp:
    def __init__(self):
        log.info("=" * 50)
        log.info("GTA5 Roulette AI v2.2 Starting...")
        log.info("=" * 50)

        self.app = QApplication(sys.argv)
        self.app.setStyle("Fusion")

        # Components
        self.capture = ScreenCapture()
        self.detector = NumberDetector(debug=True)
        self.state_detector = SpinStateDetector()
        self.capture_trigger = NumberCaptureTrigger(strategy="smart")
        self.database = SpinDatabase()
        self.trainer = RouletteTrainer({"sequence_length": 20})
        self.predictor = RoulettePredictor(self.trainer)
        self.auto_clicker = AutoClicker()

        # GUI
        self.window = OverlayWindow()
        self.window.show()
        self.window.move(50, 50)

        # Session state
        self.session_spins = 0
        self.session_id = self._generate_session_id()
        self.warmup_mode = False
        self.warmup_count = 0

        # Check if region is configured
        if not self.capture.is_configured():
            self.window.update_status("⚠️ Press F8 to select capture region!")
            log.warning("[Main] No capture region configured!")

        # Wire up buttons
        self._connect_signals()

        # Detection worker - starts PAUSED initially
        self.worker = SpinWorker(self.capture, self.detector, self.capture_trigger, interval_ms=500)
        self.worker.number_detected.connect(self._on_number_detected)
        self.worker.status_update.connect(self.window.update_status)
        self.worker.paused = True  # Start paused, user must press START
        self.worker.start()

        # Hotkey worker
        self.hotkey_worker = HotkeyWorker()
        self.hotkey_worker.hotkey_pressed.connect(self._on_hotkey)
        self.hotkey_worker.start()

        # Initial update
        self._update_predictions()
        self._update_autobet_display()
        self._load_recent_history()

        status = "Ready! Press START to begin detection. F8=SelectRegion F2=Manual F3=Force"
        if not self.capture.is_configured():
            status = "⚠️ FIRST: Press F8 to select where the result number appears!"
        self.window.update_status(status)
        self.window.update_detection_status(False)  # Show OFF state
        log.info("Application ready! Waiting for user to press START...")

    def _connect_signals(self):
        self.window.btn_start.clicked.connect(self._toggle_detection_start)
        self.window.btn_force.clicked.connect(self._force_capture)
        self.window.btn_pause.clicked.connect(self._toggle_detection)
        self.window.btn_train.clicked.connect(self._train_model)
        self.window.btn_autobet.clicked.connect(self._toggle_autobet)
        self.window.btn_calibrate.clicked.connect(self._open_calibrator)
        self.window.btn_new_session.clicked.connect(self._new_session)
        self.window.btn_undo.clicked.connect(self._undo_last)
        self.window.btn_clear_history.clicked.connect(self._clear_history)
        self.window.btn_clear_model.clicked.connect(self._clear_model)
        self.window.manual_input_signal.connect(self._on_manual_input)
        self.window.delay_spinbox.valueChanged.connect(self._on_delay_changed)

    def _generate_session_id(self) -> str:
        import time
        return "session_" + str(int(time.time()))

    def _load_recent_history(self):
        recent = self.database.get_recent_spins(10)
        for number, color, timestamp, source in recent:
            self.window.add_history(number, color)

    def _select_capture_region(self):
        """Open region selector"""
        log.info("[Main] Opening region selector...")
        self.window.update_status("Select the result number area...")
        
        # Pause detection during selection
        was_paused = self.worker.paused
        self.worker.paused = True
        
        # Hide overlay temporarily
        self.window.hide()
        
        try:
            from gui.region_selector import select_region_dialog
            region = select_region_dialog()
            
            if region:
                self.capture.save_region(region)
                self.window.update_status("✅ Region saved! Size: {}x{}".format(
                    region["width"], region["height"]))
                log.info("[Main] Region configured: {}".format(region))
            else:
                self.window.update_status("Region selection cancelled")
                
        except Exception as e:
            log.error("[Main] Region selection error: " + str(e))
            self.window.update_status("Error: " + str(e))
            
            # Fallback to OpenCV selector
            try:
                from capture.screen_capture import RegionSelector
                selector = RegionSelector()
                region = selector.select_region()
                
                if region:
                    self.capture.save_region(region)
                    self.window.update_status("✅ Region saved!")
            except Exception as e2:
                log.error("[Main] Fallback selector also failed: " + str(e2))
        
        # Restore overlay
        self.window.show()
        self.worker.paused = was_paused

    def _on_number_detected(self, number: str, color: str):
        self._record_spin(number, color, source="auto")

    def _record_spin(self, number: str, color: str, source: str = "auto"):
        warmup_enabled, warmup_total = self.window.get_warmup_settings()
        
        if self.warmup_mode and warmup_enabled:
            self.warmup_count += 1
            self.window.update_session_status(True, self.warmup_count, warmup_total)
            
            if self.warmup_count >= warmup_total:
                self.warmup_mode = False
                self.window.update_session_status(False, 0, 0)
                log.info("[Session] Warmup complete!")
                self.window.update_status("Warmup complete! Now active.")
            else:
                log.info("[Session] Warmup {}/{}: {} ({})".format(
                    self.warmup_count, warmup_total, number, color))
                self.window.update_status("Warmup {}/{}: {}".format(
                    self.warmup_count, warmup_total, number))
            
            self.database.add_spin(number, color, source, self.session_id)
            self.window.add_history(number, color)
            self.session_spins += 1
            total = self.database.get_total_spins()
            self.window.update_stats(total, self.session_spins)
            return

        self.database.add_spin(number, color, source, self.session_id)
        self.window.add_history(number, color)
        self.session_spins += 1

        total = self.database.get_total_spins()
        self.window.update_stats(total, self.session_spins)

        self._update_predictions()

        if self.auto_clicker.enabled:
            self._execute_autobet()

        log.info("[Main] Recorded: {} ({}) | Total: {}".format(number, color, total))

    def _update_predictions(self):
        recent = self.database.get_recent_numbers(50)
        if recent:
            recs, confidence = self.predictor.get_recommendations(recent)
            self.window.update_recommendations(recs, confidence)

    def _force_capture(self):
        if not self.capture.is_configured():
            self.window.update_status("⚠️ Press F8 first to select region!")
            return
            
        self.window.update_status("FORCE CAPTURE...")
        try:
            img = self.capture.capture_result_region()
            if img is not None:
                # Save debug image
                import cv2
                cv2.imwrite("debug_captures/force_capture_latest.png", img)
                
                result = self.detector.force_detect(img)
                if result:
                    number, color = result
                    self._record_spin(number, color, source="force")
                    self.window.update_status("Captured: {} ({})".format(number, color))
                else:
                    self.window.update_status("No number detected - check region (F8)")
            else:
                self.window.update_status("Capture failed - configure region (F8)")
        except Exception as e:
            log.error("Force capture: " + str(e))
            self.window.update_status("Error: " + str(e))

    def _toggle_detection_start(self):
        """Toggle detection on/off with START/STOP button"""
        if self.worker.paused:
            # Start detection
            self.worker.paused = False
            self.window.update_detection_status(True)
            self.window.update_status("🔍 Scanning for results... Press STOP to pause")
            log.info("[Main] Detection STARTED")
            
            # Force immediate capture after starting
            QTimer.singleShot(500, self._force_capture)
        else:
            # Stop detection
            self.worker.paused = True
            self.window.update_detection_status(False)
            self.window.update_status("⏸️ Detection stopped. Press START to resume.")
            log.info("[Main] Detection STOPPED")

    def _toggle_detection(self):
        """Pause/Resume detection (only available when running)"""
        if self.worker.paused:
            # Currently stopped, can't pause
            return
            
        paused = self.worker.toggle_pause()
        if paused:
            self.window.btn_pause.setText("[F4] Resume")
            self.window.update_status("PAUSED")
        else:
            self.window.btn_pause.setText("[F4] Pause")
            self.window.update_status("SCANNING")

    def _train_model(self):
        self.window.update_status("Training...")
        numbers = self.database.get_recent_numbers(1000)
        if len(numbers) < 50:
            self.window.update_status("Need 50+ spins to train")
            return

        loss = self.trainer.train(numbers, epochs=30)
        if loss >= 0:
            self.window.update_status("Training done! Loss: {:.4f}".format(loss))
        else:
            self.window.update_status("Training skipped")

        self._update_predictions()

    def _toggle_autobet(self):
        self.auto_clicker.set_bet_delay(self.window.get_bet_delay())
        
        if self.auto_clicker.enabled:
            self.auto_clicker.disable()
        else:
            progress = self.auto_clicker.get_calibration_progress()
            if progress["numbers_done"] < 10:
                self.window.update_status("Calibrate first! (F6)")
                return
            self.auto_clicker.enable()

        self._update_autobet_display()

    def _update_autobet_display(self):
        progress = self.auto_clicker.get_calibration_progress()
        self.window.update_autobet_status(self.auto_clicker.enabled, progress)

        if self.auto_clicker.enabled:
            self.window.btn_autobet.setText("[F5] Stop Auto")
            self.window.btn_autobet.setStyleSheet("background: #27ae60;")
        else:
            self.window.btn_autobet.setText("[F5] Auto-Bet")
            self.window.btn_autobet.setStyleSheet("background: #9b59b6;")

    def _execute_autobet(self):
        recent = self.database.get_recent_numbers(50)
        recs, _ = self.predictor.get_recommendations(recent)

        if recs:
            self.auto_clicker.place_bets(recs[:3])

    def _open_calibrator(self):
        from gui.calibrator import CalibratorWindow
        self.calibrator_window = CalibratorWindow(self.auto_clicker)
        self.calibrator_window.show()
        self.calibrator_window.destroyed.connect(self._update_autobet_display)

    def _new_session(self):
        warmup_enabled, warmup_total = self.window.get_warmup_settings()
        
        self.session_id = self._generate_session_id()
        self.session_spins = 0
        self.detector.reset_state()
        
        if warmup_enabled and warmup_total > 0:
            self.warmup_mode = True
            self.warmup_count = 0
            self.window.update_session_status(True, 0, warmup_total)
            self.window.update_status("New session! Warmup: 0/{}".format(warmup_total))
            log.info("[Session] New session with warmup ({} spins)".format(warmup_total))
        else:
            self.warmup_mode = False
            self.window.update_session_status(False, 0, 0)
            self.window.update_status("New session started!")
            log.info("[Session] New session (no warmup)")

        self.window.update_stats(self.database.get_total_spins(), 0)

    def _undo_last(self):
        if self.database.remove_last_spin():
            self.window.remove_last_history()
            self.session_spins = max(0, self.session_spins - 1)
            self.window.update_stats(self.database.get_total_spins(), self.session_spins)
            self._update_predictions()
            self.window.update_status("Undone last spin")
            log.info("[Main] Undone last spin")
        else:
            self.window.update_status("Nothing to undo")

    def _clear_history(self):
        reply = QMessageBox.question(
            self.window, "Confirm",
            "Clear ALL spin history?\nThis cannot be undone!",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.database.clear_all()
            self.window.clear_history_display()
            self.session_spins = 0
            self.window.update_stats(0, 0)
            self._update_predictions()
            self.window.update_status("History cleared")
            log.info("[Main] History cleared")

    def _clear_model(self):
        reply = QMessageBox.question(
            self.window, "Confirm",
            "Reset the trained model?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            model_path = "roulette_model.pth"
            if os.path.exists(model_path):
                os.remove(model_path)
                log.info("[Main] Model deleted")
            
            self.trainer = RouletteTrainer({"sequence_length": 20})
            self.predictor = RoulettePredictor(self.trainer)
            
            self._update_predictions()
            self.window.update_status("Model reset")

    def _on_manual_input(self, number: str):
        color = get_color(int(number) if number != "00" else 0)
        self._record_spin(number, color, source="manual")
        self.window.update_status("Manual: {} ({})".format(number, color))

    def _on_delay_changed(self, value: int):
        self.auto_clicker.set_bet_delay(value)

    def _on_hotkey(self, action: str):
        actions = {
            "manual_input": lambda: QTimer.singleShot(0, self.window._manual_input),
            "force_capture": lambda: QTimer.singleShot(0, self._force_capture),
            "toggle_detection": lambda: QTimer.singleShot(0, self._toggle_detection),
            "toggle_autobet": lambda: QTimer.singleShot(0, self._toggle_autobet),
            "open_calibrator": lambda: QTimer.singleShot(0, self._open_calibrator),
            "new_session": lambda: QTimer.singleShot(0, self._new_session),
            "undo_last": lambda: QTimer.singleShot(0, self._undo_last),
            "select_region": lambda: QTimer.singleShot(0, self._select_capture_region),
        }
        if action in actions:
            actions[action]()
    
    def _start_detection_hotkey(self):
        """Handle F4 hotkey for starting/stopping detection when not running"""
        if self.worker.paused and not hasattr(self, '_detection_started'):
            # If completely stopped, use start/stop toggle
            self._toggle_detection_start()
        else:
            # If running, use pause/resume
            self._toggle_detection()

    def run(self):
        return self.app.exec_()

    def cleanup(self):
        self.worker.stop()
        self.hotkey_worker.stop()
        self.worker.wait()
        self.hotkey_worker.wait()


if __name__ == "__main__":
    app = RouletteApp()
    try:
        sys.exit(app.run())
    finally:
        app.cleanup()