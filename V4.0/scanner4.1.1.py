import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import os
# CRITICAL: Forces Tesseract to be single-threaded to prevent CPU thrashing
os.environ['OMP_THREAD_LIMIT'] = '1'
import multiprocessing
import re
import pytesseract
import threading
import time
import queue 
import numpy as np 
import ctypes
import gc
from datetime import datetime
from moviepy import VideoFileClip
from PIL import Image

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
OUTPUT_ROOT = "Archived_Scans" 

# --- ESTIMATION CONSTANTS ---
ASSUMED_FPS_BASELINE = 300 

# --- DETECTION CATEGORIES ---
FILTERS = {
    "CRITICAL": [
        "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", 
        "ANUS_EXPOSED", "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED"
    ],
    "WARNING": [
        "FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED",
        "MALE_GENITALIA_COVERED", "BUTTOCKS_COVERED", "ANUS_COVERED"
    ],
    "MINOR": [
        "BELLY_EXPOSED", "BELLY_COVERED", "MALE_BREAST_EXPOSED",
        "FEET_EXPOSED", "FEET_COVERED", "ARMPITS_EXPOSED", "ARMPITS_COVERED"
    ]
}

# --- AI LOADER ---
try:
    from nudenet import NudeDetector
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("!!! WARNING: NudeNet not installed. !!!")

# --- HELPER: ROBUST PATH NORMALIZER ---
def resolve_path(path):
    """
    Handles Windows Long Paths (>260 chars).
    Priority 1: Try 8.3 Short Path
    Priority 2: Try Extended UNC Path (\\?\)
    Priority 3: Return Original (Fail-safe)
    """
    abs_path = os.path.abspath(path)
    
    # Strategy 1: Windows 8.3 Short Path
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(abs_path, buffer, 1024) > 0:
            return buffer.value
    except Exception:
        pass 

    # Strategy 2: Extended Length Prefix (\\?\)
    if len(abs_path) > 240 and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path

    # Strategy 3: Return absolute path as-is
    return abs_path

# --- HELPER: Verification ---
def verify_crop(frame, box, detector, thresh):
    if not detector: return True
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    pad_x = int(bw * 0.2)
    pad_y = int(bh * 0.2)
    x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x); y2 = min(h, y + bh + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return False
    
    detections = detector.detect(crop)
    for d in detections:
        score = d.get('score', 0)
        if score >= thresh: return True
    return False

# --- HELPER: CLIP GENERATION ---
def generate_clip(video_path, start_time, duration, output_path):
    """
    Generates a 5-second clip around the start_time.
    Uses MoviePy v2.0+ syntax (.subclipped) and 'ultrafast' preset.
    """
    try:
        with VideoFileClip(video_path) as clip:
            max_duration = clip.duration if clip.duration else (start_time + duration + 100)
            end_time = min(max_duration, start_time + (duration / 2))
            start_t = max(0, start_time - (duration / 2))
            
            new_clip = clip.subclipped(start_t, end_time)
            
            new_clip.write_videofile(
                output_path, 
                codec='libx264', 
                audio=False, 
                preset='ultrafast', 
                threads=4, 
                logger=None
            )
    except Exception as e:
        print(f"!!! CLIP ERROR: {e}")

# --- WORKER PROCESS ---
def video_worker(task, msg_queue, worker_idx):
    # 1. SETUP PATHS
    safe_path = resolve_path(task['path']) 
    display_name = os.path.basename(task['path'])
    
    subject = task['subject']
    settings = task['settings']
    active_filters = settings['active_filters'] 
    
    start_frame = task.get('start_frame', 0)
    end_frame = task.get('end_frame', None) 
    
    worker_id = f"{display_name} [{start_frame}-{end_frame if end_frame else 'END'}]"
    print(f"   -> Worker #{worker_idx} started: {worker_id}")

    # --- RESOLUTION LOGIC ---
    scan_preset = settings.get('scan_preset', 'Balanced')
    if scan_preset == 'Max Speed':
        scan_height = 240
    elif scan_preset == 'Balanced':
        scan_height = 480
    else: # High Precision
        scan_height = 720

    # Initial status update
    msg_queue.put({'type': 'WORKER_START', 'idx': worker_idx, 'file': display_name})

    label_map = {}
    for cat, labels in FILTERS.items():
        for lbl in labels:
            label_map[lbl] = cat

    # Initialize AI with DirectML Acceleration
    try:
        if AI_AVAILABLE:
            # DmlExecutionProvider = DirectML (Works on AMD Radeon iGPUs)
            # CPUExecutionProvider = Fallback if DML fails
            detector = NudeDetector(providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        else:
            detector = None
    except Exception as e:
        print(f"!!! AI CRASHED: {e}")
        return

    # Folder Setup
    video_name_no_ext = os.path.splitext(display_name)[0]
    clean_video_name = re.sub(r'[^\w\-_\. ]', '_', video_name_no_ext)
    base_root = os.path.join(OUTPUT_ROOT, subject, clean_video_name)
    os.makedirs(base_root, exist_ok=True)

    # --- VIDEO LOADING ---
    try:
        cap = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(safe_path)
            
        if not cap.isOpened():
            msg_queue.put({'type': 'LOG', 'msg': f"❌ ERROR: Could not open {display_name}"})
            msg_queue.put({'type': 'STATUS', 'video': display_name, 'status': 'Failed'})
            msg_queue.put({'type': 'WORKER_STOP', 'idx': worker_idx})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
    except Exception as e:
        msg_queue.put({'type': 'LOG', 'msg': f"❌ CRASH: {display_name} - {e}"})
        msg_queue.put({'type': 'WORKER_STOP', 'idx': worker_idx})
        return
    
    if start_frame == 0:
        msg_queue.put({
            'type': 'INIT_VIDEO', 
            'video': display_name, 
            'total_frames': total_video_frames
        })

    deep_scan = settings.get('deep_scan', False)
    skip_interval = 1 if deep_scan else int(fps / 2)
    
    frame_count = start_frame
    frames_processed_batch = 0 
    prev_gray = None
    motion_gate = settings.get('motion_gate', False)
    no_frame_counter = 0
    
    # LIVE FEED THROTTLE
    last_preview_time = time.time() + (worker_idx * 0.1) # Stagger starts

    while True:
        if end_frame and frame_count >= end_frame: break
        ret, frame = cap.read()
        
        if not ret or frame is None: 
            no_frame_counter += 1
            if no_frame_counter > 50: break
            frame_count += 1
            continue
        
        no_frame_counter = 0 

        if frame_count % skip_interval == 0:
            
            # --- WORKER PREVIEW (THUMBNAIL) ---
            if time.time() - last_preview_time > 2.0:
                try:
                    # OPTIMIZATION: Use Nearest Neighbor (Fastest) for thumbnails
                    preview = cv2.resize(frame, (120, 80), interpolation=cv2.INTER_NEAREST)
                    _, buf = cv2.imencode(".jpg", preview)
                    msg_queue.put({'type': 'GRID_PREVIEW', 'idx': worker_idx, 'data': buf.tobytes()})
                    last_preview_time = time.time()
                except: pass

            # Motion Gating
            is_static = False
            # Check for existing gray variable or create if needed
            if 'gray' in locals():
                pass 
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if motion_gate:
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    if np.sum(thresh) < 25000: is_static = True 
                prev_gray = gray
            
            # AI Scan
            if detector and not is_static:
                try:
                    aspect = orig_w / orig_h
                    new_w = int(scan_height * aspect)
                    small_frame = cv2.resize(frame, (new_w, scan_height))
                    detections = detector.detect(small_frame)
                    
                    hit_found = False
                    hit_verified = False
                    label_found = "Unknown"

                    scale_x = orig_w / new_w
                    scale_y = orig_h / scan_height

                    for det in detections:
                        label = det.get('class', det.get('label', 'UNKNOWN'))
                        score = det.get('score', 0)
                        
                        if label.upper() not in active_filters: continue
                        if score < settings['threshold1']: continue
                        
                        sb = det['box']
                        real_box = [int(sb[0]*scale_x), int(sb[1]*scale_y), int(sb[2]*scale_x), int(sb[3]*scale_y)]
                        
                        hit_found = True
                        label_found = label
                        
                        if settings['double_check']:
                            if verify_crop(frame, real_box, detector, settings['threshold2']):
                                hit_verified = True
                        else:
                            hit_verified = True
                        
                        break 

                    if hit_found:
                        timestamp = frame_count / fps
                        ts = int(timestamp)
                        t_str = f"{ts//3600:02d}-{ts%3600//60:02d}-{ts%60:02d}"
                        cat_folder = label_map.get(label_found.upper(), "UNCATEGORIZED")
                        clean_label = label_found.replace(" ", "_").title()
                        f_name = f"Frame-{frame_count}_{t_str}_{clean_label}.jpg"
                        
                        fp_dir = os.path.join(base_root, "FIRST_PASS", cat_folder)
                        os.makedirs(fp_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(fp_dir, f_name), frame)
                        
                        if hit_verified:
                            v_dir = os.path.join(base_root, "VERIFIED", cat_folder)
                            os.makedirs(v_dir, exist_ok=True)
                            
                            # 1. Save Verified Image
                            cv2.imwrite(os.path.join(v_dir, f_name), frame)

                            # 2. Trigger Video Clip (Background Thread)
                            if settings.get('generate_clips', True):
                                clip_name = f_name.replace(".jpg", ".mp4")
                                clip_out = os.path.join(v_dir, clip_name)
                                
                                threading.Thread(
                                    target=generate_clip, 
                                    args=(safe_path, timestamp, 5, clip_out), 
                                    daemon=True
                                ).start()

                            msg_queue.put({'type': 'HIT', 'path': os.path.join(v_dir, f_name), 'video': display_name})

                except Exception as e:
                    print(f"!!! DETECTION ERROR: {e}")

            # --- ROBUST OCR SCAN ---
            if settings.get('scan_ocr', False) and (frame_count % int(fps) == 0):
                try:
                    if 'gray' not in locals(): 
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    _, thresh_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    
                    text_data = pytesseract.image_to_string(thresh_img, config='--psm 6')
                    clean_text = re.sub(r'[^A-Za-z0-9@_.]', ' ', text_data).strip()
                    
                    if len(clean_text) > 4:
                         msg_queue.put({'type': 'LOG', 'msg': f"📝 [OCR] {display_name}: {clean_text[:25]}"})
                except Exception:
                    pass 

        frames_processed_batch += 1
        
        # OPTIMIZATION: Report every 200 frames instead of 50 to reduce IPC overhead
        if frames_processed_batch >= 200:
            msg_queue.put({'type': 'TICK', 'video': display_name, 'count': frames_processed_batch})
            frames_processed_batch = 0
        
        # --- MEMORY LEAK PROTECTION ---
        if frame_count % 50 == 0:
            gc.collect()

        frame_count += 1
    
    if frames_processed_batch > 0:
        msg_queue.put({'type': 'TICK', 'video': display_name, 'count': frames_processed_batch})
    
    msg_queue.put({'type': 'STATUS', 'video': display_name, 'status': 'Done'})
    msg_queue.put({'type': 'WORKER_STOP', 'idx': worker_idx})
    cap.release()

# --- TOP LEVEL WRAPPER ---
def worker_wrapper(task_q, msg_q, worker_idx):
    while True:
        try:
            task = task_q.get(timeout=1) 
            video_worker(task, msg_q, worker_idx)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"!!! WORKER ERROR: {e}")

# --- MAIN UI ---
class ArchivistPro(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        self.title("Archivist Pro v4.1 - HIGH PERFORMANCE")
        self.geometry("1400x900")

        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.msg_queue = self.manager.Queue()
        
        self.prog_bars = {} 
        self.prog_data = {} 
        self.pool = [] 
        
        # GRID VARS
        self.grid_slots = {} 
        
        self.stop_event = multiprocessing.Event()
        self.is_running = False
        self.task_stack_local = [] 
        
        # FEATURE: ACTIVE CHUNK TRACKER
        self.active_chunks = 0

        self._setup_layout()
        self.check_messages()
        
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
             messagebox.showerror("Error", f"Tesseract not found at:\n{pytesseract.pytesseract.tesseract_cmd}")

    def _setup_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar, text="ARCHIVIST PRO v4.1", font=("Impact", 28)).pack(pady=(20, 10))
        
        self.subj_input = ctk.CTkEntry(self.sidebar, placeholder_text="Subject Name...")
        self.subj_input.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="+ Add Videos", command=self.add_videos, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="+ Add Folder", command=self.add_folder, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="🔍 TEST FILE", command=self.test_file_compatibility, fg_color="#e67e22").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Clear UI Queue", command=self.clear_ui, fg_color="#c0392b").pack(fill="x", padx=20, pady=5)
        
        # ROADMAP ITEM 3: REFRESH BUTTON
        ctk.CTkButton(self.sidebar, text="Force Refresh UI", command=self.force_refresh, fg_color="#e67e22").pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="⚠ FORCE STOP ⚠", command=self.stop_processing, fg_color="#c0392b", hover_color="#922b21").pack(side="bottom", fill="x", padx=20, pady=10)

        self.btn_start = ctk.CTkButton(self.sidebar, text="START / RESUME QUEUE", command=self.start_processing, fg_color="#27ae60", height=60, font=("Arial", 16, "bold"))
        self.btn_start.pack(side="bottom", fill="x", padx=20, pady=10)

        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.tab_scan = self.tabs.add("  LIVE DASHBOARD  ")
        self.tab_settings = self.tabs.add("  SETTINGS  ")
        self._build_dashboard(self.tab_scan)
        self._build_settings(self.tab_settings)

    def _build_dashboard(self, parent):
        # WORKER GRID AREA (Scrollable to fit 32+ cores)
        ctk.CTkLabel(parent, text="Worker Grid (One screen per CPU Thread)").pack(anchor="w", padx=10)
        
        self.worker_grid_frame = ctk.CTkScrollableFrame(parent, height=300, fg_color="#1a1a1a")
        self.worker_grid_frame.pack(fill="x", padx=10, pady=5)
        
        # Just a placeholder until Start is clicked
        self.lbl_grid_placeholder = ctk.CTkLabel(self.worker_grid_frame, text="Grid will initialize on Start...", text_color="gray")
        self.lbl_grid_placeholder.pack(pady=20)

        # LATEST HIT
        hit_frame = ctk.CTkFrame(parent, height=150)
        hit_frame.pack(fill="x", padx=10, pady=10)
        self.lbl_hit = ctk.CTkLabel(hit_frame, text="[ Latest Critical Hit ]", text_color="gray")
        self.lbl_hit.pack(expand=True, fill="both", padx=5, pady=5)
        
        # QUEUE
        header = ctk.CTkFrame(parent, height=30)
        header.pack(fill="x", padx=10, pady=(5,0))
        ctk.CTkLabel(header, text="Video Queue", font=("Arial", 12, "bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="Progress & ETC", font=("Arial", 12, "bold")).pack(side="right", padx=10)

        self.queue_area = ctk.CTkScrollableFrame(parent, height=200)
        self.queue_area.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(parent, text="System Log").pack(anchor="w", padx=10)
        self.log_box = ctk.CTkTextbox(parent, height=100, font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=5)

    def _init_worker_grid(self, num_cores):
        # 1. Cleanup
        for widget in self.worker_grid_frame.winfo_children(): widget.destroy()
        self.grid_slots = {}
        
        # 2. Dynamic Column Calculation
        cols = max(8, min(num_cores, 14))
        
        # 3. Grid Column Configuration
        for i in range(cols):
            self.worker_grid_frame.grid_columnconfigure(i, weight=1)

        # 4. Generate Worker Cards
        for i in range(num_cores):
            f = ctk.CTkFrame(self.worker_grid_frame)
            f.grid(row=i//cols, column=i%cols, padx=3, pady=3, sticky="ew")
            
            img_lbl = ctk.CTkLabel(f, text=f"W{i}", fg_color="black", width=120, height=80)
            img_lbl.pack(pady=(5,0), expand=True)
            
            txt_lbl = ctk.CTkLabel(f, text="Idle", font=("Arial", 10))
            txt_lbl.pack(pady=(0,5))
            
            self.grid_slots[i] = {'img': img_lbl, 'txt': txt_lbl}

    def _build_settings(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        
        frame_perf = ctk.CTkFrame(parent)
        frame_perf.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(frame_perf, text="PERFORMANCE TUNING", font=("Arial", 14, "bold")).pack(pady=10)

        self.var_preset = ctk.StringVar(value="Balanced")
        self.opt_preset = ctk.CTkOptionMenu(frame_perf, values=["Max Speed", "Balanced", "High Precision"], 
                                            variable=self.var_preset, width=200)
        self.opt_preset.pack(pady=10)
        ctk.CTkLabel(frame_perf, text="Max Speed: 240p\nBalanced: 480p\nHigh Precision: 720p", 
                     font=("Arial", 10), text_color="gray").pack(pady=5)

        self.var_focus = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(frame_perf, text="FOCUS MODE (1 Video at a time)", variable=self.var_focus, progress_color="#e74c3c").pack(anchor="w", padx=20, pady=10)
        
        self.var_motion = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(frame_perf, text="Motion Gating (Skip Static Frames)", variable=self.var_motion).pack(anchor="w", padx=20, pady=10)

        self.entry_res = ctk.CTkEntry(frame_perf, placeholder_text="320") 

        ctk.CTkLabel(frame_perf, text="THRESHOLDS", font=("Arial", 14, "bold")).pack(pady=(20,10))
        self.lbl_t1 = ctk.CTkLabel(frame_perf, text="First Scan: 0.40")
        self.lbl_t1.pack(anchor="w", padx=20)
        self.slider_t1 = ctk.CTkSlider(frame_perf, from_=0.1, to=0.9, command=lambda v: self.lbl_t1.configure(text=f"First Scan: {v:.2f}"))
        self.slider_t1.set(0.4)
        self.slider_t1.pack(fill="x", padx=20, pady=5)
        
        self.lbl_t2 = ctk.CTkLabel(frame_perf, text="Verify: 0.60")
        self.lbl_t2.pack(anchor="w", padx=20)
        self.slider_t2 = ctk.CTkSlider(frame_perf, from_=0.1, to=0.9, command=lambda v: self.lbl_t2.configure(text=f"Verify: {v:.2f}"))
        self.slider_t2.set(0.6)
        self.slider_t2.pack(fill="x", padx=20, pady=5)
        
        self.var_verify = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(frame_perf, text="Double-Check (Zoom & Verify)", variable=self.var_verify).pack(anchor="w", padx=20, pady=5)

        frame_filt = ctk.CTkFrame(parent)
        frame_filt.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(frame_filt, text="TARGET FILTERS", font=("Arial", 14, "bold")).pack(pady=10)

        self.chk_filters = {}
        for category, labels in FILTERS.items():
            ctk.CTkLabel(frame_filt, text=f"--- {category} ---", text_color="gray").pack(anchor="w", padx=10, pady=(5,0))
            for lbl in labels:
                clean_name = lbl.replace("_", " ").title().replace("Exposed", "").replace("Covered", "(C)")
                var = ctk.BooleanVar(value=True)
                ctk.CTkCheckBox(frame_filt, text=clean_name, variable=var).pack(anchor="w", padx=20, pady=2)
                self.chk_filters[lbl] = var
        
        ctk.CTkLabel(frame_filt, text="MISC", font=("Arial", 14, "bold")).pack(pady=(20,10))
        self.var_ocr = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(frame_filt, text="Scan Usernames (OCR)", variable=self.var_ocr).pack(anchor="w", padx=20, pady=5)
        self.var_deep = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(frame_filt, text="Deep Scan (Every Frame)", variable=self.var_deep).pack(anchor="w", padx=20, pady=5)
        self.var_debug = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(frame_filt, text="Debug Log", variable=self.var_debug, progress_color="orange").pack(anchor="w", padx=20, pady=5)
        
        # ROADMAP ITEM 4: CLIP TOGGLE
        self.var_clips = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(frame_filt, text="Generate Video Clips", variable=self.var_clips).pack(anchor="w", padx=20, pady=5)

    def get_active_filters(self):
        active = []
        for lbl, var in self.chk_filters.items():
            if var.get(): active.append(lbl)
        return active

    def get_settings(self):
        return {
            "threshold1": self.slider_t1.get(),
            "threshold2": self.slider_t2.get(),
            "deep_scan": self.var_deep.get(),
            "double_check": self.var_verify.get(),
            "scan_ocr": self.var_ocr.get(),
            "debug": self.var_debug.get(),
            "active_filters": self.get_active_filters(),
            "motion_gate": self.var_motion.get(),
            "scan_preset": self.var_preset.get(),
            "generate_clips": self.var_clips.get(),
        }

    def test_file_compatibility(self):
        f = filedialog.askopenfilename(title="Select Video", filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov")])
        if not f: return
        
        # Use helper instead of raw ctypes
        safe_path = resolve_path(f) 

        report = f"Report for:\n{os.path.basename(f)}\n\n"
        
        cap = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                report += f"✅ FFMPEG: SUCCESS\n   - {w}x{h}, {frames} frames"
                messagebox.showinfo("Success", report)
            else:
                messagebox.showerror("Error", "Opened but frame is empty.")
        else:
            messagebox.showerror("Error", "Failed to open video container.")
        cap.release()

    def add_videos(self):
        files = filedialog.askopenfilenames(filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov")])
        if files: self.stack_files(files)

    def add_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
            self.stack_files(files)

    def clear_ui(self):
        self.prog_bars, self.prog_data, self.task_stack_local = {}, {}, []
        for w in self.queue_area.winfo_children(): w.destroy()
        
    def force_refresh(self):
        # Manually drain queue if stuck
        self.check_messages()
        self.update()

    def stack_files(self, files):
        subj, sett = self.subj_input.get().strip() or "Default_Subject", self.get_settings()
        
        print(f"DEBUG: Processing {len(files)} files...") # Debug Print

        for f in files:
            # --- TASK QUEUE LOGIC ---
            task = {"path": f, "subject": subj, "settings": settings}
            self.task_queue.put(task)
            self.task_stack_local.append(task) # <--- CRITICAL FOR FOCUS MODE
            
            video_name = os.path.basename(f) 
            
            # --- IMMEDIATE METADATA READ ---
            try:
                # Use helper instead of raw ctypes
                safe_path = resolve_path(f)
                
                cap_temp = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
                if not cap_temp.isOpened(): cap_temp = cv2.VideoCapture(safe_path)
                
                if cap_temp.isOpened():
                    t_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Pre-fill data for UI
                    self.prog_data[video_name] = {'total': t_frames, 'done': 0, 'start_time': 0}
                    
                    # Calculate Initial ETC (Heuristic)
                    est_seconds = t_frames / ASSUMED_FPS_BASELINE
                    if est_seconds > 3600:
                        est_str = f"Est: {int(est_seconds//3600)}h {int((est_seconds%3600)//60)}m"
                    else:
                        est_str = f"Est: {int(est_seconds//60)}:{int(est_seconds%60):02d}"
                else:
                    t_frames = 0
                    est_str = "Wait..."
                cap_temp.release()
            except:
                t_frames = 0
                est_str = "Wait..."

            # --- UI ROW ---
            row = ctk.CTkFrame(self.queue_area)
            row.pack(fill="x", padx=5, pady=2)
            
            lbl = ctk.CTkLabel(row, text=f"{video_name}", anchor="w", width=300)
            lbl.pack(side="left", padx=10)
            
            etc_lbl = ctk.CTkLabel(row, text=est_str, width=100)
            etc_lbl.pack(side="right", padx=10)
            
            bar = ctk.CTkProgressBar(row)
            bar.set(0)
            bar.pack(side="right", fill="x", expand=True, padx=10)
            
            self.prog_bars[video_name] = {'bar': bar, 'label': etc_lbl}
            
        self.log(f"Added {len(files)} files to active queue.")

    def log(self, msg):
        self.log_box.insert("end", f"{msg}\n")
        
        # OPTIMIZATION: Auto-trim logs to keep RAM usage low (Max 1000 lines)
        self.log_box.delete("1.0", "end-1000l")
        
        self.log_box.see("end")

    def check_messages(self):
        try:
            # OPTIMIZATION: Process max 100 messages per tick to prevent GUI freeze
            messages_processed = 0
            while not self.msg_queue.empty() and messages_processed < 100:
                m = self.msg_queue.get_nowait()
                messages_processed += 1
                
                # --- GRID PREVIEWS ---
                if m['type'] == 'GRID_PREVIEW':
                    idx = m['idx']
                    if idx in self.grid_slots:
                        try:
                            nparr = np.frombuffer(m['data'], np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            pil = Image.fromarray(rgb)
                            ctk_img = ctk.CTkImage(light_image=pil, size=(120, 80))
                            self.grid_slots[idx]['img'].configure(text="", image=ctk_img)
                        except: pass
                
                # --- ACTIVE CHUNK TRACKING ---
                elif m['type'] == 'WORKER_STOP':
                    if self.active_chunks > 0: self.active_chunks -= 1
                    if m['idx'] in self.grid_slots:
                        self.grid_slots[m['idx']]['txt'].configure(text="Idle", text_color="gray")
                        self.grid_slots[m['idx']]['img'].configure(image=None, text=f"W{m['idx']}")

                elif m['type'] == 'WORKER_START':
                    if m['idx'] in self.grid_slots:
                        fname = m['file']
                        short_name = (fname[:15] + '..') if len(fname) > 15 else fname
                        self.grid_slots[m['idx']]['txt'].configure(text=short_name, text_color="green")

                elif m['type'] == 'TICK' and m['video'] in self.prog_bars:
                    d = self.prog_data.get(m['video'], {'done': 0, 'total': 1})
                    d['done'] += m['count']; self.prog_bars[m['video']]['bar'].set(d['done']/d['total'])
                elif m['type'] == 'INIT_VIDEO': self.prog_data[m['video']] = {'total': m['total_frames'], 'done': 0}
                elif m['type'] == 'HIT':
                    self.lbl_hit.configure(image=ctk.CTkImage(Image.open(m['path']), size=(300, 200)), text="")
                elif m['type'] == 'LOG':
                    self.log(m['msg'])
        except Exception:
            pass # Prevent GUI freeze if queue gets corrupted
        finally:
            self.after(20, self.check_messages)

    def start_processing(self):
        if self.is_running:
            self.log("Batch is already running.")
            return
        self.stop_event.clear()
        self.is_running = True
        self.btn_start.configure(state="disabled", text="RUNNING...", fg_color="#e74c3c")
        self.tabs.set("  LIVE DASHBOARD  ") 
        
        # ROADMAP ITEM 1: 30 WORKERS
        # We assume 32-thread CPU, so we reserve 2 for Windows
        num_cores = max(1, multiprocessing.cpu_count() - 2)
        
        # BUILD GRID NOW THAT WE KNOW CORES
        self._init_worker_grid(num_cores)
        
        if self.var_focus.get():
            self.log(f"--- FOCUS MODE ENABLED (Using {num_cores} Threads) ---")
            while not self.task_queue.empty():
                try: self.task_queue.get_nowait()
                except: pass
            
            if not self.task_stack_local:
                self.log("No videos found in queue.")
                return

            threading.Thread(target=self.generate_focus_tasks, args=(num_cores,), daemon=True).start()
        else:
            self.log(f"--- STANDARD MODE (File Parallelism) ---")
        
        print(f"--- RYZEN AI MAX MODE: STARTING POOL WITH {num_cores} THREADS ---")
        threading.Thread(target=self.run_dynamic_pool, args=(num_cores,), daemon=True).start()

    def generate_focus_tasks(self, cores):
        settings = self.get_settings()
        
        # ROADMAP ITEM 2: SERIAL CHUNKING
        # We iterate through videos one by one
        for task_template in self.task_stack_local:
            raw_path = task_template['path']
            subj = task_template['subject']
            
            # Use helper instead of raw ctypes
            safe_path = resolve_path(raw_path)

            try:
                cap = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(safe_path)

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                if total_frames <= 0:
                    self.log(f"⚠ WARNING: Could not read length of {os.path.basename(raw_path)}. Running in Standard Mode (1 Core).")
                    single_task = {
                        "path": raw_path, "subject": subj, "settings": settings,
                        "start_frame": 0, "end_frame": None, "is_chunk": False
                    }
                    self.task_queue.put(single_task)
                    continue

                frames_per_chunk = total_frames // cores
                self.log(f"Splitting {os.path.basename(raw_path)} into {cores} chunks (~{frames_per_chunk} frames each)...")
                
                # Push chunks to queue
                for i in range(cores):
                    start = i * frames_per_chunk
                    end = (i + 1) * frames_per_chunk if i < cores - 1 else total_frames
                    
                    chunk_task = {
                        "path": raw_path, 
                        "subject": subj,
                        "settings": settings,
                        "start_frame": start,
                        "end_frame": end,
                        "is_chunk": True
                    }
                    self.task_queue.put(chunk_task)
                
                # BARRIER: Wait for queue to effectively empty AND workers to report completion
                self.active_chunks = cores 
                
                # Wait until all active chunks drop to 0
                while self.active_chunks > 0 and self.is_running:
                    time.sleep(1)
                
                if not self.is_running: break
                
                # Short cool-down before next video to let RAM settle
                time.sleep(2)
                self.log(f"Finished processing: {os.path.basename(raw_path)}")

            except Exception as e:
                self.log(f"Error chunking video: {e}")

    def stop_processing(self):
        if self.is_running:
            self.log("!!! TERMINATING ALL PROCESSES !!!")
            self.stop_event.set()
            for p in self.pool:
                p.terminate()
            self.pool = []
            self.is_running = False
            self.btn_start.configure(state="normal", text="SCAN CANCELLED", fg_color="#c0392b")

    def run_dynamic_pool(self, cores):
        self.pool = []
        for i in range(cores):
            # Pass i as worker_idx
            p = multiprocessing.Process(target=worker_wrapper, args=(self.task_queue, self.msg_queue, i))
            p.daemon = True
            p.start()
            self.pool.append(p)
        while not self.stop_event.is_set():
            time.sleep(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = ArchivistPro()
    app.mainloop()