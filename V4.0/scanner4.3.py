import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import os
import multiprocessing
import re
import pytesseract
import threading
import time
import queue 
import numpy as np 
import ctypes
import gc
import json
import sqlite3
import glob
from datetime import datetime
from moviepy import VideoFileClip
from PIL import Image

# ==========================================
#        GLOBAL CONFIGURATION
# ==========================================

# 1. Performance Constraints
# Forces Tesseract to run on a single thread to prevent it from fighting
# with the 30+ video worker threads for CPU time.
os.environ['OMP_THREAD_LIMIT'] = '1'

# 2. External Tools Paths
# Update this if Tesseract is installed in a custom location.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 3. File System Paths
OUTPUT_ROOT = "Archived_Scans" 
DB_FILE = "scan_history.db"
SETTINGS_FILE = "user_settings.json"

# 4. Estimation Constants
# Used to guess how long a scan will take before it starts.
ASSUMED_FPS_BASELINE = 300 

# ==========================================
#        DETECTION FILTER DEFINITIONS
# ==========================================
# These lists map the raw NudeNet labels to our 3 severity categories.
FILTERS = {
    "CRITICAL": [
        "FEMALE_GENITALIA_EXPOSED", 
        "MALE_GENITALIA_EXPOSED", 
        "ANUS_EXPOSED", 
        "FEMALE_BREAST_EXPOSED"
    ],
    "WARNING": [
        "BUTTOCKS_EXPOSED", 
        "FEMALE_BREAST_COVERED", 
        "FEMALE_GENITALIA_COVERED", 
        "MALE_GENITALIA_COVERED", 
        "BUTTOCKS_COVERED", 
        "ANUS_COVERED"
    ],
    "MINOR": [
        "BELLY_EXPOSED", 
        "BELLY_COVERED", 
        "MALE_BREAST_EXPOSED", 
        "FEET_EXPOSED", 
        "FEET_COVERED", 
        "ARMPITS_EXPOSED", 
        "ARMPITS_COVERED"
    ]
}

# ==========================================
#        AI ENGINE LOADER
# ==========================================
try:
    from nudenet import NudeDetector
    import onnxruntime as ort
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("!!! CRITICAL WARNING: NudeNet or ONNX Runtime libraries are missing. !!!")
    print("!!! Please run: pip install nudenet onnxruntime-directml !!!")

# ==========================================
#        DATABASE ENGINE (THE NEURAL VAULT)
# ==========================================
def init_db():
    """
    Initializes the SQLite database to track scanned files.
    Creates the table if it doesn't exist.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS history (
                filename TEXT PRIMARY KEY, 
                date_scanned TEXT, 
                status TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database Init Error: {e}")

def check_history(filename):
    """
    Checks if a file has already been successfully scanned.
    Returns True if found, False otherwise.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT status FROM history WHERE filename=?", (filename,))
        result = c.fetchone()
        conn.close()
        return result is not None
    except:
        return False

def mark_scanned(filename):
    """
    Marks a file as completed in the database.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT OR REPLACE INTO history VALUES (?, ?, ?)", 
                  (filename, timestamp, "COMPLETED"))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database Write Error: {e}")

# ==========================================
#        SYSTEM HELPER FUNCTIONS
# ==========================================
def resolve_path(path):
    """
    Fixes the Windows 260-character path limit issue.
    Prepends \\?\ to absolute paths to allow long file access.
    """
    abs_path = os.path.abspath(path)
    try:
        # Try to get the 8.3 short path first
        buffer = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(abs_path, buffer, 1024) > 0:
            return buffer.value
    except: 
        pass 
    
    # Fallback to extended path prefix
    if len(abs_path) > 240 and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path

def verify_crop(frame, box, detector, thresh):
    """
    The 'Double Check' Logic.
    Crops the detected area and feeds it back into the AI as a standalone image.
    This drastically reduces false positives (e.g., elbows looking like buttocks).
    """
    if not detector: return True
    
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    
    # Add 20% padding around the crop for context
    pad_x = int(bw * 0.2)
    pad_y = int(bh * 0.2)
    
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0: return False
    
    detections = detector.detect(crop)
    for d in detections:
        if d.get('score', 0) >= thresh: 
            return True
            
    return False

def generate_clip(video_path, start_time, duration, output_path):
    """
    Extracts a 5-second video clip around the detected timestamp.
    Runs on a background thread to avoid blocking the scanner.
    """
    try:
        with VideoFileClip(video_path) as clip:
            max_duration = clip.duration if clip.duration else (start_time + duration + 100)
            
            # Center the clip on the hit
            end_t = min(max_duration, start_time + (duration / 2))
            start_t = max(0, start_time - (duration / 2))
            
            new_clip = clip.subclipped(start_t, end_t)
            
            # Use 'ultrafast' preset to minimize CPU usage
            new_clip.write_videofile(
                output_path, 
                codec='libx264', 
                audio=False, 
                preset='ultrafast', 
                threads=4, 
                logger=None
            )
    except Exception as e:
        print(f"Clip Generation Failed: {e}")

# ==========================================
#        CORE WORKER LOGIC (The Engine)
# ==========================================
def video_worker(task, msg_queue, worker_idx):
    """
    The main scanning process. Runs entirely independent of the UI.
    Contains the DirectML injection logic.
    """
    # Import locally to ensure process isolation
    import onnxruntime as ort
    
    # 1. Parse Task Data
    safe_path = resolve_path(task['path'])
    display_name = os.path.basename(task['path'])
    subject = task['subject']
    settings = task['settings']
    active_filters = settings['active_filters'] 
    
    start_frame = task.get('start_frame', 0)
    end_frame = task.get('end_frame', None)
    
    # Notify UI we are alive
    msg_queue.put({'type': 'WORKER_START', 'idx': worker_idx, 'file': display_name})

    # 2. Hardware Acceleration Injection (DirectML)
    detector = None
    if AI_AVAILABLE:
        try:
            # Configure Session Options for AMD Stability
            opts = ort.SessionOptions()
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Force DirectML Provider
            providers = [('DmlExecutionProvider', {'device_id': 0})]
            
            # Auto-Detect Model File
            home = os.path.expanduser("~")
            nudenet_dir = os.path.join(home, ".NudeNet")
            possible_models = glob.glob(os.path.join(nudenet_dir, "*.onnx"))
            
            if not possible_models:
                # Emergency Download if missing
                temp = NudeDetector()
                model_path = temp.onnx_session._model_path 
            else:
                # Prefer 640m (Standard) or 320n (Fast)
                model_path = possible_models[0]
                for p in possible_models:
                    if "640m" in p: model_path = p; break
                    if "320n" in p: model_path = p; break

            # Initialize Wrapper
            detector = NudeDetector()
            
            # INJECT: Overwrite the session with our GPU-forced settings
            detector.detector = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            
            # Verify Hardware
            active = detector.detector.get_providers()
            if 'DmlExecutionProvider' not in active:
                msg_queue.put({'type': 'LOG', 'msg': f"⚠ Worker {worker_idx} GPU rejected. Running on CPU."})
                
        except Exception as e:
            # Fallback to standard CPU mode if GPU fails
            detector = NudeDetector() 
            msg_queue.put({'type': 'LOG', 'msg': f"Worker {worker_idx} Init Error: {e}"})

    # 3. Setup Scanning Parameters
    # Create the filter map for fast lookups
    label_map = {}
    for cat, labels in FILTERS.items():
        for lbl in labels:
            label_map[lbl] = cat

    # Resolution Logic
    scan_preset = settings.get('scan_preset', 'Balanced')
    if scan_preset == 'Max Speed':
        scan_height = 240
    elif scan_preset == 'Balanced':
        scan_height = 480
    else:
        scan_height = 720

    # 4. Open Video Stream
    try:
        cap = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
        if not cap.isOpened(): 
            cap = cv2.VideoCapture(safe_path)
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if start_frame > 0: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
    except Exception as e:
        msg_queue.put({'type': 'STATUS', 'video': display_name, 'status': 'Error'})
        msg_queue.put({'type': 'WORKER_STOP', 'idx': worker_idx})
        return
    
    # Send total frames to UI for progress bar
    if start_frame == 0:
        msg_queue.put({
            'type': 'INIT_VIDEO', 
            'video': display_name, 
            'total_frames': total_video_frames
        })

    # 5. Prepare Output Folders
    clean_video_name = re.sub(r'[^\w\-_\. ]', '_', os.path.splitext(display_name)[0])
    base_root = os.path.join(OUTPUT_ROOT, subject, clean_video_name)
    
    # 6. Runtime Variables
    deep_scan = settings.get('deep_scan', False)
    skip_interval = 1 if deep_scan else int(fps / 2) # Scan every frame OR every 0.5s
    
    frame_count = start_frame
    frames_batch = 0
    prev_gray = None
    last_preview = time.time()

    # ==========================
    #      MAIN SCAN LOOP
    # ==========================
    while True:
        # Check chunk limits
        if end_frame and frame_count >= end_frame: 
            break
            
        ret, frame = cap.read()
        
        # End of stream check
        if not ret or frame is None: 
            break

        # Processing Interval Check
        if frame_count % skip_interval == 0:
            
            # --- A. GRID THUMBNAIL (Every 2s) ---
            if time.time() - last_preview > 2.0:
                try:
                    # Nearest Neighbor resize is fastest
                    preview = cv2.resize(frame, (120, 80), interpolation=cv2.INTER_NEAREST)
                    _, buf = cv2.imencode(".jpg", preview)
                    msg_queue.put({'type': 'GRID_PREVIEW', 'idx': worker_idx, 'data': buf.tobytes()})
                    last_preview = time.time()
                except: pass

            # --- B. MOTION GATING ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_static = False
            
            if settings.get('motion_gate') and prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                # If pixel change is negligible, skip AI
                if np.sum(thresh) < 25000: 
                    is_static = True
            
            prev_gray = gray

            # --- C. AI INFERENCE ---
            if detector and not is_static:
                # Resize for AI
                aspect = orig_w / orig_h
                small = cv2.resize(frame, (int(scan_height * aspect), scan_height))
                
                # Run NudeNet
                detections = detector.detect(small)
                
                # MULTI-TAG SUPPORT: 
                # We iterate through ALL detections to allow multiple hits per frame
                processed_frame_hit = False
                
                for det in detections:
                    label = det.get('class', det.get('label', 'UNK'))
                    score = det.get('score', 0)
                    
                    # Check Filters & Threshold
                    if label.upper() in active_filters and score >= settings['threshold1']:
                        
                        # Calculate Box
                        sb = det['box']
                        scale_x = orig_w / (int(scan_height * aspect))
                        scale_y = orig_h / scan_height
                        real_box = [
                            int(sb[0]*scale_x), int(sb[1]*scale_y), 
                            int(sb[2]*scale_x), int(sb[3]*scale_y)
                        ]
                        
                        # Verify Hit
                        if not settings['double_check'] or verify_crop(frame, real_box, detector, settings['threshold2']):
                            
                            # Calculate Timestamp
                            ts = frame_count / fps
                            t_str = f"{int(ts)//3600:02d}-{int(ts)%3600//60:02d}-{int(ts)%60:02d}"
                            
                            # Save Image
                            cat_folder = label_map.get(label.upper(), "UNCATEGORIZED")
                            v_dir = os.path.join(base_root, "VERIFIED", cat_folder)
                            os.makedirs(v_dir, exist_ok=True)
                            
                            f_path = os.path.join(v_dir, f"Frame-{frame_count}_{t_str}_{label.title()}.jpg")
                            cv2.imwrite(f_path, frame)
                            
                            # Generate Video Clip (Only once per frame to avoid duplicates)
                            if settings.get('generate_clips') and not processed_frame_hit:
                                threading.Thread(
                                    target=generate_clip, 
                                    args=(safe_path, ts, 5, f_path.replace(".jpg", ".mp4")), 
                                    daemon=True
                                ).start()
                                processed_frame_hit = True
                            
                            # Notify UI
                            msg_queue.put({'type': 'HIT', 'path': f_path, 'video': display_name})

            # --- D. OCR SCANNING ---
            if settings.get('scan_ocr') and (frame_count % int(fps) == 0):
                try:
                    _, thresh_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    txt = pytesseract.image_to_string(thresh_img, config='--psm 6')
                    clean_txt = re.sub(r'[^A-Za-z0-9@_.]', ' ', txt).strip()
                    if len(clean_txt) > 4: 
                        msg_queue.put({'type': 'LOG', 'msg': f"📝 [OCR] {display_name}: {clean_txt[:25]}"})
                except: pass

        # Update Progress Counter
        frames_batch += 1
        # Only report to UI every 200 frames to save CPU
        if frames_batch >= 200:
            msg_queue.put({'type': 'TICK', 'video': display_name, 'count': frames_batch})
            frames_batch = 0
            
        # Garbage Collection (Vital for long runs)
        if frame_count % 50 == 0: 
            gc.collect() 
            
        frame_count += 1
    
    # 7. Finalize Task
    if not end_frame: 
        # Only mark as "Complete" if we scanned the whole file, not just a chunk
        msg_queue.put({'type': 'DB_UPDATE', 'file': display_name})

    msg_queue.put({'type': 'STATUS', 'video': display_name, 'status': 'Done'})
    msg_queue.put({'type': 'WORKER_STOP', 'idx': worker_idx})
    cap.release()

# --- WORKER THREAD WRAPPER ---
def worker_wrapper(task_q, msg_q, worker_idx):
    """
    Wraps the worker process to handle the staggered startup and errors.
    """
    # Stagger Start: 0.5s delay * Worker Index
    # Prevents 30 workers from hitting the GPU memory simultaneously
    time.sleep(worker_idx * 0.5)
    
    while True:
        try:
            # Wait for a task
            task = task_q.get(timeout=1) 
            video_worker(task, msg_q, worker_idx)
        except queue.Empty: 
            continue
        except Exception as e: 
            print(f"Worker Wrapper Error: {e}")

# ==========================================
#        MAIN GUI CLASS (CustomTkinter)
# ==========================================
class ArchivistPro(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # System Setup
        init_db() 
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        
        self.title("Archivist Pro v4.3 - FEATURE COMPLETE")
        self.geometry("1400x900")
        
        # Multiprocessing Manager
        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.msg_queue = self.manager.Queue()
        
        # Runtime State
        self.prog_bars = {} 
        self.prog_data = {} 
        self.pool = [] 
        self.grid_slots = {} 
        self.stop_event = multiprocessing.Event()
        self.is_running = False
        self.task_stack_local = [] 
        self.active_chunks = 0

        # Build UI
        self._setup_layout()
        
        # Load User Preferences
        self.load_settings() 
        
        # Start Message Loop
        self.check_messages()
        
        # Handle Exit
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_layout(self):
        """
        Constructs the Grid Layout and all UI widgets.
        """
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar, text="ARCHIVIST PRO v4.3", font=("Impact", 28)).pack(pady=(20, 10))
        
        self.subj_input = ctk.CTkEntry(self.sidebar, placeholder_text="Subject Name...")
        self.subj_input.pack(fill="x", padx=20, pady=5)
        
        # Control Buttons
        ctk.CTkButton(self.sidebar, text="+ Add Videos", command=self.add_videos, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="+ Add Folder", command=self.add_folder, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        
        # Maintenance Buttons
        ctk.CTkButton(self.sidebar, text="Force Refresh UI", command=self.force_refresh, fg_color="#e67e22").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Clear UI", command=self.clear_ui, fg_color="#c0392b").pack(fill="x", padx=20, pady=5)
        
        # Action Buttons
        self.btn_stop = ctk.CTkButton(self.sidebar, text="⚠ FORCE STOP", command=self.stop_processing, fg_color="#c0392b", hover_color="#922b21")
        self.btn_stop.pack(side="bottom", fill="x", padx=20, pady=10)
        
        self.btn_start = ctk.CTkButton(self.sidebar, text="START QUEUE", command=self.start_processing, fg_color="#27ae60", height=60, font=("Arial", 16, "bold"))
        self.btn_start.pack(side="bottom", fill="x", padx=20, pady=10)

        # --- MAIN AREA ---
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.tab_scan = self.tabs.add(" DASHBOARD ")
        self.tab_settings = self.tabs.add(" SETTINGS ")
        
        self._build_dashboard(self.tab_scan)
        self._build_settings(self.tab_settings)

    def _build_dashboard(self, parent):
        # 1. Worker Grid (Visual Thread Monitor)
        ctk.CTkLabel(parent, text="Worker Grid (One cell per CPU Thread)").pack(anchor="w", padx=10)
        self.worker_grid_frame = ctk.CTkScrollableFrame(parent, height=350, fg_color="#1a1a1a")
        self.worker_grid_frame.pack(fill="x", padx=10, pady=5)
        
        # 2. Latest Hit Preview
        self.lbl_hit = ctk.CTkLabel(parent, text="[ Latest Critical Hit ]", height=150, fg_color="#2c3e50")
        self.lbl_hit.pack(fill="x", padx=10, pady=10)
        
        # 3. Video Queue
        ctk.CTkLabel(parent, text="Active Queue").pack(anchor="w", padx=10)
        self.queue_area = ctk.CTkScrollableFrame(parent, height=200)
        self.queue_area.pack(fill="x", padx=10, pady=5)
        
        # 4. System Log
        self.log_box = ctk.CTkTextbox(parent, height=100, font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=5)

    def _init_worker_grid(self, num_cores):
        # Clears and rebuilds the grid based on core count
        for w in self.worker_grid_frame.winfo_children(): 
            w.destroy()
            
        cols = max(8, min(num_cores, 14))
        
        for i in range(cols): 
            self.worker_grid_frame.grid_columnconfigure(i, weight=1)
            
        for i in range(num_cores):
            f = ctk.CTkFrame(self.worker_grid_frame)
            f.grid(row=i//cols, column=i%cols, padx=3, pady=3, sticky="ew")
            
            # Placeholder for Thumbnail
            ctk.CTkLabel(f, text=f"W{i}", width=120, height=80, fg_color="black").pack(pady=5)
            
            # Status Text
            lbl_txt = ctk.CTkLabel(f, text="Idle", font=("Arial", 10))
            lbl_txt.pack(pady=(0,5))
            
            # Store references to update later
            self.grid_slots[i] = {
                'img': f.winfo_children()[0], 
                'txt': lbl_txt
            }

    def _build_settings(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # --- Left Column: Performance ---
        left_frame = ctk.CTkFrame(parent)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(left_frame, text="PERFORMANCE", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Sliders
        ctk.CTkLabel(left_frame, text="Confidence Threshold 1").pack()
        self.slider_t1 = ctk.CTkSlider(left_frame, from_=0.1, to=0.9)
        self.slider_t1.pack(pady=5)
        
        ctk.CTkLabel(left_frame, text="Verify Threshold 2").pack()
        self.slider_t2 = ctk.CTkSlider(left_frame, from_=0.1, to=0.9)
        self.slider_t2.pack(pady=5)
        
        # Preset Dropdown
        self.var_preset = ctk.StringVar(value="Balanced")
        self.opt_preset = ctk.CTkOptionMenu(left_frame, values=["Max Speed", "Balanced", "High Precision"], variable=self.var_preset)
        self.opt_preset.pack(pady=10)
        
        # Toggles
        self.var_focus = ctk.BooleanVar(value=False)
        self.var_motion = ctk.BooleanVar(value=False)
        self.var_deep = ctk.BooleanVar(value=False)
        
        ctk.CTkSwitch(left_frame, text="FOCUS MODE (One Video)", variable=self.var_focus).pack(pady=5, anchor="w")
        ctk.CTkSwitch(left_frame, text="Motion Gating (Skip Static)", variable=self.var_motion).pack(pady=5, anchor="w")
        ctk.CTkSwitch(left_frame, text="Deep Scan (Every Frame - Slow)", variable=self.var_deep).pack(pady=5, anchor="w")

        # --- Right Column: Features ---
        right_frame = ctk.CTkFrame(parent)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(right_frame, text="FEATURES", font=("Arial", 14, "bold")).pack(pady=10)
        
        self.var_verify = ctk.BooleanVar(value=True)
        self.var_ocr = ctk.BooleanVar(value=False)
        self.var_clips = ctk.BooleanVar(value=True)
        self.var_skip = ctk.BooleanVar(value=True) # New "Skip Scanned" toggle
        
        ctk.CTkSwitch(right_frame, text="Double-Check Verification", variable=self.var_verify).pack(pady=5, anchor="w")
        ctk.CTkSwitch(right_frame, text="Scan Usernames (OCR)", variable=self.var_ocr).pack(pady=5, anchor="w")
        ctk.CTkSwitch(right_frame, text="Generate Video Clips", variable=self.var_clips).pack(pady=5, anchor="w")
        ctk.CTkSwitch(right_frame, text="Skip Already Scanned", variable=self.var_skip).pack(pady=5, anchor="w")

    # ==========================================
    #        SETTINGS PERSISTENCE (JSON)
    # ==========================================
    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f: 
                    data = json.load(f)
                    
                self.slider_t1.set(data.get('t1', 0.4))
                self.slider_t2.set(data.get('t2', 0.6))
                self.var_preset.set(data.get('preset', 'Balanced'))
                self.var_focus.set(data.get('focus', False))
                self.var_motion.set(data.get('motion', False))
                self.var_ocr.set(data.get('ocr', False))
                self.var_deep.set(data.get('deep', False))
                self.var_skip.set(data.get('skip', True))
                
                print("Settings Loaded Successfully.")
            except Exception as e:
                print(f"Settings Load Error: {e}")

    def save_settings(self):
        data = {
            't1': self.slider_t1.get(), 
            't2': self.slider_t2.get(),
            'preset': self.var_preset.get(), 
            'focus': self.var_focus.get(),
            'motion': self.var_motion.get(), 
            'ocr': self.var_ocr.get(),
            'deep': self.var_deep.get(), 
            'skip': self.var_skip.get()
        }
        try:
            with open(SETTINGS_FILE, 'w') as f: 
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Settings Save Error: {e}")

    def on_close(self):
        """
        Triggered when closing the window.
        Saves settings and stops workers.
        """
        self.save_settings()
        self.stop_processing()
        self.destroy()

    # ==========================================
    #        FILE INGESTION LOGIC
    # ==========================================
    def add_videos(self):
        files = filedialog.askopenfilenames(
            parent=self, 
            filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov *.wmv")]
        )
        if files: 
            self.stack_files(files)

    def add_folder(self):
        folder = filedialog.askdirectory(parent=self)
        if folder:
            # SMART FOLDER INGESTION: 
            # Automatically update the Subject Input to match the folder name
            folder_name = os.path.basename(folder)
            self.subj_input.delete(0, "end")
            self.subj_input.insert(0, folder_name)
            
            files = [
                os.path.join(folder, f) for f in os.listdir(folder) 
                if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
            ]
            self.stack_files(files)

    def stack_files(self, files):
        subj = self.subj_input.get().strip() or "Default_Subject"
        sett = self.get_settings_dict()
        
        count = 0
        skipped = 0
        
        for f in files:
            if not os.path.exists(f): continue
            
            # --- DATABASE CHECK ---
            # If "Skip Scanned" is ON, we check the DB before adding
            if self.var_skip.get() and check_history(os.path.basename(f)):
                print(f"Skipping {os.path.basename(f)} (Already Scanned)")
                skipped += 1
                continue

            # Add to Queue
            task = {"path": f, "subject": subj, "settings": sett}
            self.task_queue.put(task)
            self.task_stack_local.append(task)
            
            # Create UI Row
            row = ctk.CTkFrame(self.queue_area)
            row.pack(fill="x", pady=2)
            
            ctk.CTkLabel(row, text=os.path.basename(f), width=300, anchor="w").pack(side="left", padx=5)
            
            bar = ctk.CTkProgressBar(row)
            bar.set(0)
            bar.pack(side="right", fill="x", expand=True, padx=5)
            
            self.prog_bars[os.path.basename(f)] = {'bar': bar}
            count += 1
            
        self.log_box.insert("end", f"Queued {count} files. Skipped {skipped}.\n")
        self.log_box.see("end")

    def get_settings_dict(self):
        """
        Compiles the current UI state into a settings dictionary for workers.
        """
        active_filters = list(FILTERS["CRITICAL"] + FILTERS["WARNING"] + FILTERS["MINOR"])
        return {
            'threshold1': self.slider_t1.get(),
            'threshold2': self.slider_t2.get(),
            'scan_preset': self.var_preset.get(),
            'deep_scan': self.var_deep.get(),
            'double_check': self.var_verify.get(),
            'scan_ocr': self.var_ocr.get(),
            'motion_gate': self.var_motion.get(),
            'generate_clips': self.var_clips.get(),
            'active_filters': active_filters
        }

    # ==========================================
    #        MESSAGE LOOP & EVENT HANDLING
    # ==========================================
    def check_messages(self):
        try:
            cnt = 0
            # Process up to 100 messages per tick to keep UI responsive
            while not self.msg_queue.empty() and cnt < 100:
                m = self.msg_queue.get_nowait()
                cnt += 1
                
                # Update Grid Thumbnail
                if m['type'] == 'GRID_PREVIEW' and m['idx'] in self.grid_slots:
                    try:
                        pil = Image.fromarray(cv2.cvtColor(cv2.imdecode(np.frombuffer(m['data'], np.uint8), 1), cv2.COLOR_BGR2RGB))
                        self.grid_slots[m['idx']]['img'].configure(image=ctk.CTkImage(pil, size=(120, 80)), text="")
                    except: pass
                
                # Update Worker Status
                elif m['type'] == 'WORKER_START': 
                    self.grid_slots[m['idx']]['txt'].configure(text=m['file'][:15], text_color="green")
                elif m['type'] == 'WORKER_STOP': 
                    self.grid_slots[m['idx']]['txt'].configure(text="Idle", text_color="gray")
                
                # Update Progress Bar
                elif m['type'] == 'TICK' and m['video'] in self.prog_bars: 
                    self.prog_bars[m['video']]['bar'].step() 
                
                # Show Latest Hit
                elif m['type'] == 'HIT':
                    try: 
                        self.lbl_hit.configure(image=ctk.CTkImage(Image.open(m['path']), size=(300, 200)), text="")
                    except: pass
                
                # Log Message
                elif m['type'] == 'LOG': 
                    self.log_box.insert("end", f"{m['msg']}\n")
                    self.log_box.see("end")
                
                # Database Update
                elif m['type'] == 'DB_UPDATE': 
                    mark_scanned(m['file']) # Update DB on Main Thread to avoid locks
                    
        except: 
            pass
        finally: 
            self.after(20, self.check_messages)

    # ==========================================
    #        PROCESS MANAGEMENT
    # ==========================================
    def start_processing(self):
        if self.is_running: return
        self.stop_event.clear()
        self.is_running = True
        
        self.btn_start.configure(state="disabled", text="RUNNING...", fg_color="#e74c3c")
        
        # Calculate optimal cores (Total - 2)
        num_cores = max(1, multiprocessing.cpu_count() - 2)
        
        self._init_worker_grid(num_cores)
        
        # Focus Mode: Split one file into chunks
        if self.var_focus.get(): 
            threading.Thread(target=self.generate_focus_tasks, args=(num_cores,), daemon=True).start()
        
        # Start Worker Pool
        for i in range(num_cores):
            p = multiprocessing.Process(
                target=worker_wrapper, 
                args=(self.task_queue, self.msg_queue, i)
            )
            p.daemon = True
            p.start()
            self.pool.append(p)
    
    def generate_focus_tasks(self, cores):
        """
        Splits a single large video into 'cores' amount of chunks for parallel scanning.
        """
        sett = self.get_settings_dict()
        for t in self.task_stack_local:
            try:
                cap = cv2.VideoCapture(resolve_path(t['path']))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                chunk = total // cores
                self.active_chunks = cores
                
                for i in range(cores): 
                    start = i * chunk
                    end = (i + 1) * chunk if i < cores - 1 else total
                    
                    self.task_queue.put({
                        "path": t['path'], 
                        "subject": t['subject'], 
                        "settings": sett, 
                        "start_frame": start, 
                        "end_frame": end
                    })
                    
                # Barrier: Wait for all chunks to finish
                while self.active_chunks > 0 and self.is_running: 
                    time.sleep(1)
            except Exception as e:
                print(f"Focus Task Error: {e}")

    def stop_processing(self):
        """
        Hard kill of all child processes.
        """
        self.is_running = False
        self.stop_event.set()
        
        for p in self.pool: 
            if p.is_alive():
                p.terminate()
                
        self.pool = []
        self.btn_start.configure(state="normal", text="SCAN CANCELLED", fg_color="#c0392b")
        print("All processes terminated.")
        
    def force_refresh(self): 
        self.update()
        
    def clear_ui(self):
        for w in self.queue_area.winfo_children(): 
            w.destroy()
        self.prog_bars = {}
        self.task_stack_local = []

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ArchivistPro().mainloop()