import cv2
import os
import time
import gc
import re
import glob
import numpy as np
import pytesseract
import threading
from nudenet import NudeDetector
from filters import FILTERS
from helpers import resolve_path, generate_clip

# Global for Worker stop/start
OUTPUT_ROOT = "Archived_Scans"

def verify_crop(frame, box, detector, thresh):
    if not detector: return True
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    pad_x, pad_y = int(bw * 0.2), int(bh * 0.2)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return False
    detections = detector.detect(crop)
    for d in detections:
        if d.get('score', 0) >= thresh: return True
    return False

def video_worker(task, msg_queue, worker_idx):
    import onnxruntime as ort
    try:
        # Task Parsing
        safe_path = resolve_path(task['path'])
        display_name = os.path.basename(task['path'])
        subject, settings = task['subject'], task['settings']
        active_filters = settings['active_filters'] 
        start_frame, end_frame = task.get('start_frame', 0), task.get('end_frame', None)
        
        msg_queue.put({'type': 'WORKER_START', 'idx': worker_idx, 'file': display_name})

        # AI Hardware Injection
        detector = None
        try:
            opts = ort.SessionOptions()
            opts.enable_mem_pattern, opts.execution_mode = False, ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [('DmlExecutionProvider', {'device_id': 0})]
            home = os.path.expanduser("~")
            possible = glob.glob(os.path.join(home, ".NudeNet", "*.onnx"))
            model_path = possible[0] if possible else NudeDetector().onnx_session._model_path
            for p in possible:
                if "640m" in p: model_path = p; break
                if "320n" in p: model_path = p; break
            detector = NudeDetector()
            detector.detector = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        except: detector = NudeDetector()

        label_map = {lbl: cat for cat, labels in FILTERS.items() for lbl in labels}
        scan_height = 480 # Balanced default
        
        cap = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
        if not cap.isOpened(): cap = cv2.VideoCapture(safe_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        base_root = os.path.join(OUTPUT_ROOT, subject, re.sub(r'[^\w\-_\. ]', '_', os.path.splitext(display_name)[0]))
        skip_interval = 1 if settings.get('deep_scan') else int(fps / 2)
        ui_update_interval = max(50, int((end_frame - start_frame if end_frame else total) / 100))

        frame_count, frames_since_update = start_frame, 0
        prev_gray, last_preview, bad_streak = None, time.time(), 0

        while True:
            if end_frame and frame_count >= end_frame: break
            ret, frame = cap.read()
            if not ret or frame is None:
                bad_streak += 1
                if bad_streak >= 50: break
                frame_count += 1; continue
            bad_streak = 0

            if frame_count % skip_interval == 0:
                # Preview Grid
                if time.time() - last_preview > 2.0:
                    try:
                        _, buf = cv2.imencode(".jpg", cv2.resize(frame, (120, 80), interpolation=cv2.INTER_NEAREST))
                        msg_queue.put({'type': 'GRID_PREVIEW', 'idx': worker_idx, 'data': buf.tobytes()})
                        last_preview = time.time()
                    except: pass

                # AI Scan
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                is_static = False
                if settings.get('motion_gate') and prev_gray is not None:
                    if np.sum(cv2.threshold(cv2.absdiff(prev_gray, gray), 25, 255, cv2.THRESH_BINARY)[1]) < 25000: is_static = True
                prev_gray = gray

                if detector and not is_static:
                    orig_h, orig_w = frame.shape[:2]
                    detections = detector.detect(cv2.resize(frame, (int(scan_height * (orig_w/orig_h)), scan_height)))
                    hit_recorded = False
                    for det in detections:
                        label, score = det.get('class', 'UNK'), det.get('score', 0)
                        if label.upper() in active_filters and score >= settings['threshold1']:
                            ts = frame_count / fps
                            t_str = f"{int(ts)//3600:02d}-{int(ts)%3600//60:02d}-{int(ts)%60:02d}"
                            cat = label_map.get(label.upper(), "UNK")
                            fp_dir = os.path.join(base_root, "FIRST_PASS", cat)
                            os.makedirs(fp_dir, exist_ok=True)
                            f_name = f"Frame-{frame_count}_{t_str}_{label.title()}.jpg"
                            cv2.imwrite(os.path.join(fp_dir, f_name), frame)

                            if not settings['double_check'] or verify_crop(frame, det['box'], detector, settings['threshold2']):
                                v_dir = os.path.join(base_root, "VERIFIED", cat)
                                os.makedirs(v_dir, exist_ok=True)
                                cv2.imwrite(os.path.join(v_dir, f_name), frame)
                                if settings.get('generate_clips') and not hit_recorded:
                                    threading.Thread(target=generate_clip, args=(safe_path, ts, 5, os.path.join(v_dir, f_name.replace(".jpg", ".mp4"))), daemon=True).start()
                                    hit_recorded = True
                                msg_queue.put({'type': 'HIT', 'path': os.path.join(v_dir, f_name)})

            frames_since_update += 1
            if frames_since_update >= ui_update_interval:
                msg_queue.put({'type': 'TICK', 'video': display_name, 'count': frames_since_update})
                frames_since_update = 0
            if frame_count % 50 == 0: gc.collect()
            frame_count += 1

        if not end_frame: msg_queue.put({'type': 'DB_UPDATE', 'file': display_name})
        cap.release()

    except Exception as e: msg_queue.put({'type': 'LOG', 'msg': f"❌ W{worker_idx} CRASH: {e}"})
    finally:
        msg_queue.put({'type': 'STATUS', 'video': display_name if 'display_name' in locals() else "Unknown", 'status': 'Done'})
        msg_queue.put({'type': 'WORKER_STOP', 'idx': worker_idx})

def worker_wrapper(task_q, msg_q, worker_idx):
    time.sleep(worker_idx * 0.5)
    while True:
        try:
            task = task_q.get(timeout=1)
            video_worker(task, msg_q, worker_idx)
        except: continue