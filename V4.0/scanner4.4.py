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
os.environ['OMP_THREAD_LIMIT'] = '1'

# 2. External Tools Paths
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 3. File System Paths
OUTPUT_ROOT = "Archived_Scans" 
DB_FILE = "scan_history.db"
SETTINGS_FILE = "user_settings.json"

# 4. Estimation Constants
ASSUMED_FPS_BASELINE = 300 

# ==========================================
#        DETECTION FILTER DEFINITIONS
# ==========================================
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

# ==========================================
#        DATABASE ENGINE (THE NEURAL VAULT)
# ==========================================
def init_db():
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
    abs_path = os.path.abspath(path)
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(abs_path, buffer, 1024) > 0:
            return buffer.value
    except: 
        pass 
    if len(abs_path) > 240 and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path

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
        if d.get('score', 0) >= thresh: 
            return True
    return False

def generate_clip(video_path, start_time, duration, output_path):
    try:
        with VideoFileClip(video_path) as clip:
            max_duration = clip.duration if clip.duration else (start_time + duration + 100)
            end_t = min(max_duration, start_time + (duration / 2))
            start_t = max(0, start_time - (duration / 2))
            new_clip = clip.subclipped(start_t, end_t)
            new_clip.write_videofile(output_path, codec='libx264', audio=False, preset='ultrafast', threads=4, logger=None)
    except Exception as e:
        print(f"Clip Generation Failed: {e}")

# ==========================================
#        CORE WORKER LOGIC (The Engine)
# ==========================================
def video_worker(task, msg_queue, worker_idx):
    import onnxruntime as ort
    
    # 1. Parse Task Data
    safe_path = resolve_path(task['path'])
    display_name = os.path.basename(task['path'])
    subject = task['subject']
    settings = task['settings']
    active_filters = settings['active_filters'] 
    
    start_frame = task.get('start_frame', 0)
    end_frame = task.get('end_frame', None)
    
    msg_queue.put({'type': 'WORKER_START', 'idx': worker_idx, 'file': display_name})

    # 2. Hardware Acceleration Injection (DirectML)
    detector = None
    if AI_AVAILABLE:
        try:
            opts = ort.SessionOptions()
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [('DmlExecutionProvider', {'device_id': 0})]
            
            home = os.path.expanduser("~")
            nudenet_dir = os.path.join(home, ".NudeNet")
            possible_models = glob.glob(os.path.join(nudenet_dir, "*.onnx"))
            
            if not possible_models:
                temp = NudeDetector()
                model_path = temp.onnx_session._model_path 
            else:
                model_path = possible_models[0]
                for p in possible_models:
                    if "640m" in p: model_path = p; break
                    if "320n" in p: model_path = p; break

            detector = NudeDetector()
            detector.detector = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            
            active = detector.detector.get_providers()
            if 'DmlExecutionProvider' not in active:
                msg_queue.put({'type': 'LOG', 'msg': f"⚠ Worker {worker_idx} GPU rejected. Running on CPU."})
                
        except Exception as e:
            detector = NudeDetector() 
            msg_queue.put({'type': 'LOG', 'msg': f"Worker {worker_idx} Init Error: {e}"})

    # 3. Setup Scanning Parameters
    label_map = {}
    for cat, labels in FILTERS.items():
        for lbl in labels:
            label_map[lbl] = cat

    scan_height = 240 if settings.get('scan_preset') == 'Max Speed' else 480 if settings.get('scan_preset') == 'Balanced' else 720

    # 4. Open Video Stream
    try:
        cap = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
        if not cap.isOpened(): cap = cv2.VideoCapture(safe_path)
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
    except Exception as e:
        msg_queue.put({'type': 'STATUS', 'video': display_name, 'status': 'Error'})
        msg_queue.put({'type': 'WORKER_STOP', 'idx': worker_idx})
        return
    
    if start_frame == 0:
        msg_queue.put({'type': 'INIT_VIDEO', 'video': display_name, 'total_frames': total_video_frames})

    # 5. Prepare Output Folders
    clean_video_name = re.sub(r'[^\w\-_\. ]', '_', os.path.splitext(display_name)[0])
    base_root = os.path.join(OUTPUT_ROOT, subject, clean_video_name)
    
    # 6. Runtime Variables
    deep_scan = settings.get('deep_scan', False)
    skip_interval = 1 if deep_scan else int(fps / 2) 
    
    frame_count = start_frame
    frames_batch = 0
    prev_gray = None
    last_preview = time.time()

    while True:
        if end_frame and frame_count >= end_frame: break
        ret, frame = cap.read()
        if not ret or frame is None: break

        if frame_count % skip_interval == 0:
            
            # A. GRID THUMBNAIL
            if time.time() - last_preview > 2.0:
                try:
                    preview = cv2.resize(frame, (120, 80), interpolation=cv2.INTER_NEAREST)
                    _, buf = cv2.imencode(".jpg", preview)
                    msg_queue.put({'type': 'GRID_PREVIEW', 'idx': worker_idx, 'data': buf.tobytes()})
                    last_preview = time.time()
                except: pass

            # B. MOTION GATING
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_static = False
            
            if settings.get('motion_gate') and prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                if np.sum(thresh) < 25000: is_static = True
            
            prev_gray = gray

            # C. AI INFERENCE
            if detector and not is_static:
                aspect = orig_w / orig_h
                small = cv2.resize(frame, (int(scan_height * aspect), scan_height))
                detections = detector.detect(small)
                
                processed_frame_hit = False
                
                for det in detections:
                    label = det.get('class', det.get('label', 'UNK'))
                    score = det.get('score', 0)
                    
                    if label.upper() in active_filters and score >= settings['threshold1']:
                        
                        ts = frame_count / fps
                        t_str = f"{int(ts)//3600:02d}-{int(ts)%3600//60:02d}-{int(ts)%60:02d}"
                        cat_folder = label_map.get(label.upper(), "UNCATEGORIZED")

                        # --- RESTORED: FIRST PASS SAVING (The "Missing" Images) ---
                        # Save the raw detection BEFORE double-checking
                        fp_dir = os.path.join(base_root, "FIRST_PASS", cat_folder)
                        os.makedirs(fp_dir, exist_ok=True)
                        f_name = f"Frame-{frame_count}_{t_str}_{label.title()}.jpg"
                        cv2.imwrite(os.path.join(fp_dir, f_name), frame)
                        
                        # --- VERIFICATION LOGIC ---
                        sb = det['box']
                        scale_x = orig_w / (int(scan_height * aspect))
                        scale_y = orig_h / scan_height
                        real_box = [int(sb[0]*scale_x), int(sb[1]*scale_y), int(sb[2]*scale_x), int(sb[3]*scale_y)]
                        
                        if not settings['double_check'] or verify_crop(frame, real_box, detector, settings['threshold2']):
                            
                            # Save to Verified
                            v_dir = os.path.join(base_root, "VERIFIED", cat_folder)
                            os.makedirs(v_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(v_dir, f_name), frame)
                            
                            # Clip Generation
                            if settings.get('generate_clips') and not processed_frame_hit:
                                threading.Thread(
                                    target=generate_clip, 
                                    args=(safe_path, ts, 5, os.path.join(v_dir, f_name.replace(".jpg", ".mp4"))), 
                                    daemon=True
                                ).start()
                                processed_frame_hit = True
                            
                            msg_queue.put({'type': 'HIT', 'path': os.path.join(v_dir, f_name), 'video': display_name})

            # D. OCR SCANNING
            if settings.get('scan_ocr') and (frame_count % int(fps) == 0):
                try:
                    _, thresh_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    txt = pytesseract.image_to_string(thresh_img, config='--psm 6')
                    clean_txt = re.sub(r'[^A-Za-z0-9@_.]', ' ', txt).strip()
                    if len(clean_txt) > 4: 
                        msg_queue.put({'type': 'LOG', 'msg': f"📝 [OCR] {display_name}: {clean_txt[:25]}"})
                except: pass

        frames_batch += 1
        if frames_batch >= 200:
            msg_queue.put({'type': 'TICK', 'video': display_name, 'count': frames_batch})
            frames_batch = 0
        if frame_count % 50 == 0: gc.collect() 
        frame_count += 1
    
    if not end_frame: 
        msg_queue.put({'type': 'DB_UPDATE', 'file': display_name})

    msg_queue.put({'type': 'STATUS', 'video': display_name, 'status': 'Done'})
    msg_queue.put({'type': 'WORKER_STOP', 'idx': worker_idx})
    cap.release()

def worker_wrapper(task_q, msg_q, worker_idx):
    time.sleep(worker_idx * 0.5)
    while True:
        try:
            task = task_q.get(timeout=1) 
            video_worker(task, msg_q, worker_idx)
        except queue.Empty: continue
        except Exception as e: print(f"Worker Wrapper Error: {e}")

# ==========================================
#        MAIN GUI CLASS
# ==========================================
class ArchivistPro(ctk.CTk):
    def __init__(self):
        super().__init__()
        init_db() 
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        
        self.title("Archivist Pro v4.4 - RESTORED SUITE")
        self.geometry("1400x900")
        
        self.manager = multiprocessing.Manager()
        self.task_queue, self.msg_queue = self.manager.Queue(), self.manager.Queue()
        
        self.prog_bars, self.prog_data, self.pool, self.grid_slots = {}, {}, [], {}
        self.stop_event, self.is_running, self.task_stack_local = multiprocessing.Event(), False, []
        self.active_chunks = 0
        self.chk_filters = {} # Storage for checkboxes

        self._setup_layout()
        self.load_settings() 
        self.check_messages()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_layout(self):
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar, text="ARCHIVIST PRO v4.4", font=("Impact", 28)).pack(pady=(20, 10))
        self.subj_input = ctk.CTkEntry(self.sidebar, placeholder_text="Subject Name..."); self.subj_input.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="+ Add Videos", command=self.add_videos, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="+ Add Folder", command=self.add_folder, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Force Refresh UI", command=self.force_refresh, fg_color="#e67e22").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Clear UI", command=self.clear_ui, fg_color="#c0392b").pack(fill="x", padx=20, pady=5)
        
        self.btn_stop = ctk.CTkButton(self.sidebar, text="⚠ FORCE STOP", command=self.stop_processing, fg_color="#c0392b", hover_color="#922b21")
        self.btn_stop.pack(side="bottom", fill="x", padx=20, pady=10)
        self.btn_start = ctk.CTkButton(self.sidebar, text="START QUEUE", command=self.start_processing, fg_color="#27ae60", height=60, font=("Arial", 16, "bold"))
        self.btn_start.pack(side="bottom", fill="x", padx=20, pady=10)

        # --- MAIN AREA ---
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.tab_scan = self.tabs.add(" DASHBOARD "); self.tab_settings = self.tabs.add(" SETTINGS ")
        
        self._build_dashboard(self.tab_scan)
        self._build_settings(self.tab_settings)

    def _build_dashboard(self, parent):
        ctk.CTkLabel(parent, text="Worker Grid").pack(anchor="w", padx=10)
        self.worker_grid_frame = ctk.CTkScrollableFrame(parent, height=350, fg_color="#1a1a1a")
        self.worker_grid_frame.pack(fill="x", padx=10, pady=5)
        self.lbl_hit = ctk.CTkLabel(parent, text="[ Latest Hit ]", height=150, fg_color="#2c3e50")
        self.lbl_hit.pack(fill="x", padx=10, pady=10)
        self.queue_area = ctk.CTkScrollableFrame(parent, height=200); self.queue_area.pack(fill="x", padx=10, pady=5)
        self.log_box = ctk.CTkTextbox(parent, height=100, font=("Consolas", 12)); self.log_box.pack(fill="both", expand=True, padx=10, pady=5)

    def _init_worker_grid(self, num_cores):
        for w in self.worker_grid_frame.winfo_children(): w.destroy()
        cols = max(8, min(num_cores, 14))
        for i in range(cols): self.worker_grid_frame.grid_columnconfigure(i, weight=1)
        for i in range(num_cores):
            f = ctk.CTkFrame(self.worker_grid_frame); f.grid(row=i//cols, column=i%cols, padx=3, pady=3, sticky="ew")
            ctk.CTkLabel(f, text=f"W{i}", width=120, height=80, fg_color="black").pack(pady=5)
            lbl = ctk.CTkLabel(f, text="Idle", font=("Arial", 10)); lbl.pack(pady=(0,5))
            self.grid_slots[i] = {'img': f.winfo_children()[0], 'txt': lbl}

    def _build_settings(self, parent):
        parent.grid_columnconfigure(0, weight=1); parent.grid_columnconfigure(1, weight=1)

        # --- LEFT: PERFORMANCE & SLIDERS ---
        left = ctk.CTkFrame(parent); left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(left, text="TUNING", font=("Arial", 14, "bold")).pack(pady=10)
        
        # RESTORED: Numeric Labels for Sliders
        self.lbl_t1 = ctk.CTkLabel(left, text="First Scan: 0.40")
        self.lbl_t1.pack()
        self.slider_t1 = ctk.CTkSlider(left, from_=0.1, to=0.9, command=lambda v: self.lbl_t1.configure(text=f"First Scan: {v:.2f}"))
        self.slider_t1.pack(pady=5)
        
        self.lbl_t2 = ctk.CTkLabel(left, text="Verify: 0.60")
        self.lbl_t2.pack()
        self.slider_t2 = ctk.CTkSlider(left, from_=0.1, to=0.9, command=lambda v: self.lbl_t2.configure(text=f"Verify: {v:.2f}"))
        self.slider_t2.pack(pady=5)
        
        self.var_preset = ctk.StringVar(value="Balanced")
        ctk.CTkOptionMenu(left, values=["Max Speed", "Balanced", "High Precision"], variable=self.var_preset).pack(pady=10)
        
        self.var_focus, self.var_motion, self.var_deep = ctk.BooleanVar(), ctk.BooleanVar(), ctk.BooleanVar()
        ctk.CTkSwitch(left, text="Focus Mode", variable=self.var_focus).pack(anchor="w", pady=5, padx=20)
        ctk.CTkSwitch(left, text="Motion Gating", variable=self.var_motion).pack(anchor="w", pady=5, padx=20)
        ctk.CTkSwitch(left, text="Deep Scan (Slow)", variable=self.var_deep).pack(anchor="w", pady=5, padx=20)

        # --- RIGHT: FILTERS & MISC ---
        right = ctk.CTkFrame(parent); right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # RESTORED: Filter Selection UI
        ctk.CTkLabel(right, text="FILTERS", font=("Arial", 14, "bold")).pack(pady=10)
        scroll_filt = ctk.CTkScrollableFrame(right, height=300)
        scroll_filt.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.chk_filters = {}
        for cat, labels in FILTERS.items():
            ctk.CTkLabel(scroll_filt, text=f"--- {cat} ---", text_color="gray").pack(anchor="w")
            for lbl in labels:
                clean = lbl.replace("_", " ").title().replace("Exposed", "").replace("Covered", "(C)")
                var = ctk.BooleanVar(value=True)
                ctk.CTkCheckBox(scroll_filt, text=clean, variable=var).pack(anchor="w", padx=10, pady=2)
                self.chk_filters[lbl] = var
        
        ctk.CTkLabel(right, text="MISC", font=("Arial", 14, "bold")).pack(pady=10)
        self.var_verify, self.var_ocr, self.var_clips, self.var_skip = ctk.BooleanVar(value=True), ctk.BooleanVar(), ctk.BooleanVar(value=True), ctk.BooleanVar(value=True)
        self.var_debug = ctk.BooleanVar(value=False) # RESTORED: Debug
        
        ctk.CTkSwitch(right, text="Double-Check", variable=self.var_verify).pack(anchor="w", padx=20)
        ctk.CTkSwitch(right, text="Scan Usernames (OCR)", variable=self.var_ocr).pack(anchor="w", padx=20)
        ctk.CTkSwitch(right, text="Generate Video Clips", variable=self.var_clips).pack(anchor="w", padx=20)
        ctk.CTkSwitch(right, text="Skip Scanned", variable=self.var_skip).pack(anchor="w", padx=20)
        ctk.CTkSwitch(right, text="Debug Log", variable=self.var_debug).pack(anchor="w", padx=20)

    def get_active_filters(self):
        return [lbl for lbl, var in self.chk_filters.items() if var.get()]

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f: data = json.load(f)
                self.slider_t1.set(data.get('t1', 0.4))
                self.lbl_t1.configure(text=f"First Scan: {data.get('t1', 0.4):.2f}") # Update Label
                self.slider_t2.set(data.get('t2', 0.6))
                self.lbl_t2.configure(text=f"Verify: {data.get('t2', 0.6):.2f}") # Update Label
                self.var_preset.set(data.get('preset', 'Balanced'))
                self.var_focus.set(data.get('focus', False))
                self.var_motion.set(data.get('motion', False))
                self.var_ocr.set(data.get('ocr', False))
                self.var_deep.set(data.get('deep', False))
                self.var_skip.set(data.get('skip', True))
                self.var_debug.set(data.get('debug', False))
                
                # Load Filters
                saved_filters = data.get('filters', [])
                if saved_filters:
                    for lbl, var in self.chk_filters.items():
                        var.set(lbl in saved_filters)
            except: pass

    def save_settings(self):
        data = {
            't1': self.slider_t1.get(), 't2': self.slider_t2.get(),
            'preset': self.var_preset.get(), 'focus': self.var_focus.get(),
            'motion': self.var_motion.get(), 'ocr': self.var_ocr.get(),
            'deep': self.var_deep.get(), 'skip': self.var_skip.get(),
            'debug': self.var_debug.get(), 'filters': self.get_active_filters()
        }
        with open(SETTINGS_FILE, 'w') as f: json.dump(data, f)

    def on_close(self):
        self.save_settings(); self.stop_processing(); self.destroy()

    def add_videos(self):
        files = filedialog.askopenfilenames(parent=self, filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov")])
        if files: self.stack_files(files)

    def add_folder(self):
        folder = filedialog.askdirectory(parent=self)
        if folder:
            self.subj_input.delete(0, "end"); self.subj_input.insert(0, os.path.basename(folder))
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
            self.stack_files(files)

    def stack_files(self, files):
        subj, sett = self.subj_input.get().strip() or "Default", self.get_settings_dict()
        cnt = 0
        for f in files:
            if not os.path.exists(f): continue
            if self.var_skip.get() and check_history(os.path.basename(f)): continue
            
            task = {"path": f, "subject": subj, "settings": sett}
            self.task_queue.put(task); self.task_stack_local.append(task)
            
            row = ctk.CTkFrame(self.queue_area); row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=os.path.basename(f), width=300, anchor="w").pack(side="left")
            bar = ctk.CTkProgressBar(row); bar.set(0); bar.pack(side="right", fill="x", expand=True)
            self.prog_bars[os.path.basename(f)] = {'bar': bar}
            cnt += 1
        self.log_box.insert("end", f"Queued {cnt} files.\n")

    def get_settings_dict(self):
        return {
            'threshold1': self.slider_t1.get(), 'threshold2': self.slider_t2.get(),
            'scan_preset': self.var_preset.get(), 'deep_scan': self.var_deep.get(),
            'double_check': self.var_verify.get(), 'scan_ocr': self.var_ocr.get(),
            'motion_gate': self.var_motion.get(), 'generate_clips': self.var_clips.get(),
            'active_filters': self.get_active_filters()
        }

    def check_messages(self):
        try:
            cnt = 0
            while not self.msg_queue.empty() and cnt < 100:
                m = self.msg_queue.get_nowait(); cnt += 1
                if m['type'] == 'GRID_PREVIEW' and m['idx'] in self.grid_slots:
                    pil = Image.fromarray(cv2.cvtColor(cv2.imdecode(np.frombuffer(m['data'], np.uint8), 1), cv2.COLOR_BGR2RGB))
                    self.grid_slots[m['idx']]['img'].configure(image=ctk.CTkImage(pil, size=(120, 80)), text="")
                elif m['type'] == 'WORKER_START': self.grid_slots[m['idx']]['txt'].configure(text=m['file'][:10], text_color="green")
                elif m['type'] == 'WORKER_STOP': self.grid_slots[m['idx']]['txt'].configure(text="Idle", text_color="gray")
                elif m['type'] == 'TICK' and m['video'] in self.prog_bars: self.prog_bars[m['video']]['bar'].step() 
                elif m['type'] == 'HIT':
                    try: self.lbl_hit.configure(image=ctk.CTkImage(Image.open(m['path']), size=(300, 200)), text="")
                    except: pass
                elif m['type'] == 'LOG': 
                    if self.var_debug.get() or "Error" in m['msg']: # Debug Filter
                        self.log_box.insert("end", f"{m['msg']}\n"); self.log_box.see("end")
                elif m['type'] == 'DB_UPDATE': mark_scanned(m['file']) 
        except: pass
        finally: self.after(20, self.check_messages)

    def start_processing(self):
        if self.is_running: return
        self.stop_event.clear(); self.is_running = True
        self.btn_start.configure(state="disabled", text="RUNNING...", fg_color="#e74c3c")
        num_cores = max(1, multiprocessing.cpu_count() - 2)
        self._init_worker_grid(num_cores)
        if self.var_focus.get(): threading.Thread(target=self.generate_focus_tasks, args=(num_cores,), daemon=True).start()
        for i in range(num_cores):
            p = multiprocessing.Process(target=worker_wrapper, args=(self.task_queue, self.msg_queue, i)); p.daemon = True; p.start(); self.pool.append(p)
    
    def generate_focus_tasks(self, cores):
        sett = self.get_settings_dict()
        for t in self.task_stack_local:
            cap = cv2.VideoCapture(resolve_path(t['path'])); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            chunk = total // cores; self.active_chunks = cores
            for i in range(cores): self.task_queue.put({"path": t['path'], "subject": t['subject'], "settings": sett, "start_frame": i*chunk, "end_frame": (i+1)*chunk if i<cores-1 else total})
            while self.active_chunks > 0 and self.is_running: time.sleep(1)

    def stop_processing(self):
        self.is_running = False; self.stop_event.set()
        for p in self.pool: p.terminate()
        self.pool = []; self.btn_start.configure(state="normal", text="SCAN CANCELLED", fg_color="#c0392b")
        
    def force_refresh(self): self.update()
    def clear_ui(self):
        for w in self.queue_area.winfo_children(): w.destroy()
        self.prog_bars = {}; self.task_stack_local = []

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ArchivistPro().mainloop()