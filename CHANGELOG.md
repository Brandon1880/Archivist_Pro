# Changelog

All notable changes to the **Archivist Pro** project are documented in this file.

## [v4.7.2] - 2026-02-17 (Modular UI Fix)
### Fixed
- **Settings Tab:** Restored missing "Misc" settings section (OCR toggle, Debug Log, Video Clips toggle) that was accidentally removed during the modular refactoring.
- **UI Layout:** Adjusted grid spacing in the settings menu to prevent buttons from overlapping.

## [v4.7.1] - 2026-02-17 (Modular Architecture)
### Changed
- **Refactoring:** Split the monolithic `scanner.py` into a modular 5-file structure for better maintainability:
  - `main.py`: UI and Process Management.
  - `engine.py`: AI Workers and DirectML Logic.
  - `database.py`: SQLite "Neural Vault" management.
  - `helpers.py`: Utility functions (Paths, JSON).
  - `filters.py`: Detection category definitions.

## [v4.7] - 2026-02-17 (Corruption Resistance)
### Added
- **Pothole Protection:** The engine now detects corrupted frames (read errors) and skips them instead of crashing the worker.
- **Tolerance Counter:** Workers will now retry up to **50 bad frames** before abandoning a video file.

## [v4.6.1] - 2026-02-17 (Anti-Freeze)
### Fixed
- **Ghost Workers:** Fixed a critical bug where crashed workers failed to send a "Done" signal, causing the queue to hang.
- **Dead Man's Switch:** Implemented a `try...finally` block in the worker engine to guarantee a termination signal is always sent.

## [v4.6] - 2026-02-17 (Frame Counters)
### Added
- **Real-Time Counters:** Replaced "Estimated Time" with exact `Current Frame / Total Frame` tracking.
- **Chunk Summation:** In Focus Mode, progress from multiple workers is now summed dynamically to show total speed.

## [v4.5.2] - 2026-02-17 (Status Logging)
### Added
- **Focus Logging:** The log window now explicitly prints `>>> STARTING` and `✔ COMPLETED` messages.
### Fixed
- **Log Visibility:** Status messages now bypass the "Debug Mode" filter so they are always visible.

## [v4.5.1] - 2026-02-17 (Queue Logic Fix)
### Fixed
- **Focus Mode:** Fixed a race condition where adding videos immediately dispatched them to workers.
- **Queue Flushing:** Added a safety check to wipe stale tasks from the queue before starting a new batch.

## [v4.5] - 2026-02-17 (Optimization)
### Changed
- **UI Throttling:** Reduced worker-to-UI communication frequency from every 200 frames to every 1% of the video.
### Added
- **ETC Restoration:** Re-implemented the "Estimated Time to Completion" calculation on file ingestion.

## [v4.4] - 2026-02-16 (Restored Features)
### Added
- **Numeric Sliders:** Threshold sliders now display their exact float value (e.g., 0.40).
- **First Pass Storage:** Restored the `FIRST_PASS` folder creation to save all raw detections.
### Fixed
- **UI Elements:** Restored missing checkboxes for specific Detection Filters and the Debug Log toggle.

## [v4.3] - 2026-02-16 (Feature Complete)
### Added
- **Neural Vault:** Integrated SQLite database (`scan_history.db`) to track scanned files.
- **Settings Persistence:** Added `user_settings.json` to save slider positions and toggles.
- **Smart Ingestion:** "Add Folder" now automatically sets the Subject Name to the folder name.

## [v4.2.2] - 2026-02-16 (UI & Filters)
### Changed
- **Filter Update:** Moved `BUTTOCKS_EXPOSED` from "Critical" to "Warning" category.
### Added
- **Force Stop:** Added a dedicated Red "Stop" button to kill all background processes immediately.

## [v4.2.1] - 2026-02-16 (Hardware Acceleration)
### Added
- **DirectML Injection:** Forced ONNX Runtime to use the `DmlExecutionProvider` for AMD/NPU support.
- **Model Auto-Detect:** Script now scans `.NudeNet` folder for any valid `.onnx` model.

## [v4.0.0] - 2026-01-25 (The "Pro" UI Update)
### Changed
- **UI Overhaul:** Abandoned standard `tkinter` for `CustomTkinter` (Modern Dark Mode).
- **Dashboard:** Added the "Worker Grid" to visualize thread status.

## [v3.0.0] - 2025-12-20 (The "Swarm" Update)
### Added
- **Multiprocessing:** Rewrote core engine to use all CPU cores (Swarm Architecture).
- **Queue System:** Implemented `multiprocessing.Queue` for task distribution.

## [v2.0.0] - 2025-10-01 (The AI Shift)
### Changed
- **Detection Engine:** Replaced OpenCV Haar Cascades with **NudeNet** (ONNX) for higher accuracy.
- **Verification:** Added "Double Check" crop logic to reduce false positives.

## [v1.0.0] - 2025-09-15 (Initial Release)
### Added
- **Core Functionality:** Basic frame-grabber using OpenCV.