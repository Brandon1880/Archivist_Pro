# Archivist_Pro
Archivist Pro is a modular, multi-threaded AI application designed to autonomously scan large volumes of video content for specific visual targets. Built for speed and stability, it leverages DirectML Hardware Acceleration to run advanced computer vision models directly on GPUs, bypassing the traditional CPU bottlenecks of standard Python scripts.

Technical Overview
Unlike basic frame-grabbers, Archivist Pro utilizes a "Filter-Verify-Archive" pipeline:

Ingestion: Accepts localized video files or recursively scans directories, automatically skipping files previously processed via its internal SQLite Neural Vault.

Fast-Path Detection: Uses a lightweight ONNX model (NudeNet) running on the GPU to scan frames at high speed (up to 30 workers in parallel).

Verification Logic: When a target is found, the system performs a "Double-Check" by cropping the Region of Interest (ROI) and re-evaluating it to eliminate false positives.

Archival: Positive hits are saved as high-res images and 5-second context video clips, sorted into a granular folder structure based on the specific content detected.

Key Features
🧠 AI & Hardware Acceleration
DirectML Injection: Custom-written logic forces the ONNX Runtime to bypass the CPU and utilize AMD Ryzen AI / Radeon / NVIDIA hardware for inference.

Multi-Threaded Swarm: Spawns independent worker processes for every CPU core, allowing simultaneous scanning of 30+ video streams.

Focus Mode: Can dedicate all system resources to a single large video file by splitting it into dynamic chunks and processing them in parallel.

🛡️ Stability & Intelligence
Corruption Resistance: Includes a "Pothole Protection" engine that detects corrupted video frames and skips them without crashing the workflow.

Motion Gating: Smart algorithms analyze pixel differences to skip static scenes (like empty rooms), ensuring the AI only processes active footage.

Dead Man’s Switch: Advanced process management ensures that if a worker thread hangs or crashes, it is automatically killed and reported to prevent queue freezes.

📊 Data Management
The Neural Vault: An integrated SQLite database tracks every file scanned (by hash/name), preventing redundant processing across sessions.

OCR Integration: Tesseract engine runs in the background to scrape on-screen text (usernames, watermarks) from valid hits.

First-Pass Logging: Saves raw detection data alongside verified hits for model fine-tuning and debugging.
