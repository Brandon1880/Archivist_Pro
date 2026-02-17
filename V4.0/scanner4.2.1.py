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
from datetime import datetime
from moviepy import VideoFileClip
from PIL import Image

# --- CRITICAL CONFIGURATION ---
# Forces Tesseract to be single-threaded to prevent CPU thrashing
os.environ['OMP_THREAD_LIMIT'] = '1'

# Path to Tesseract OCR Engine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Root folder for saving results
OUTPUT_ROOT = "Archived_Scans" 

# Frames Per Second baseline for ETC calculation
ASSUMED_FPS_BASELINE = 300 

# --- DETECTION FILTERS ---
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

# --- AI LIBRARY LOADER ---
try:
    from nudenet import NudeDetector
    import onnxruntime as ort
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("!!! WARNING: NudeNet or ONNX Runtime not installed. !!!")

# --- HELPER: ROBUST PATH NORMALIZER (Fixes Windows Long Path Errors) ---
def resolve_path(path):
    abs_path = os.path.abspath(path)
    # Strategy 1: Try Windows 8.3 Short Path
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(abs_path, buffer, 1024) > 0:
            return buffer.value
    except Exception:
        pass 
    # Strategy 2: Add Extended Length Prefix
    if len(abs_path) > 240 and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path

# --- HELPER: VERIFY DETECTION (Double Check) ---
def verify_crop(frame, box, detector, thresh):
    if not detector: return True
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    
    # Add padding to context
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
        if d.get('score', 0) >= thresh: return True
    return False

# --- HELPER: GENERATE VIDEO CLIP ---
def generate_clip(video_path, start_time, duration, output_path):
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

# ==========================================
#        CORE WORKER LOGIC
# ==========================================
def video_worker(task, msg_queue, worker_idx):
    import onnxruntime as ort # Force local import for sub-process
    import glob # Needed for file searching
    
    safe_path = resolve_path(task['path']) 
    display_name = os.path.basename(task['path'])
    subject, settings = task['subject'], task['settings']
    active_filters = settings['active_filters'] 
    start_frame, end_frame = task.get('start_frame', 0), task.get('end_frame', None) 
    
    msg_queue.put({'type': 'WORKER_START', 'idx': worker_idx, 'file': display_name})

    # --- FORCED DIRECTML INJECTION (AUTO-DETECT) ---
    detector = None
    if AI_AVAILABLE:
        try:
            # 1. Define Session Options
            opts = ort.SessionOptions()
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # 2. Define Provider (DirectML)
            providers = [('DmlExecutionProvider', {'device_id': 0})]
            
            # 3. AUTO-LOCATE MODEL
            home = os.path.expanduser("~")
            nudenet_dir = os.path.join(home, ".NudeNet")
            
            # Find ANY .onnx file in that folder
            possible_models = glob.glob(os.path.join(nudenet_dir, "*.onnx"))
            
            if not possible_models:
                # If directory is empty, try to trigger a download by initializing standard detector
                # This might be slow but fixes the missing file issue
                print(f"   -> Worker {worker_idx}: No models found. Initializing default...")
                temp = NudeDetector() 
                model_path = temp.onnx_session._model_path # Grab the path it downloaded to
            else:
                # Prefer '640m' or 'base' if available, otherwise take the first one
                model_path = possible_models[0]
                for p in possible_models:
                    if "640m" in p: model_path = p; break
                    if "base" in p: model_path = p; break

            # 4. Initialize NudeDetector wrapper
            detector = NudeDetector()
            
            # 5. INJECTION: Override with the found model
            detector.detector = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            
            active = detector.detector.get_providers()
            print(f"   -> Worker #{worker_idx} Model: {os.path.basename(model_path)} | Backend: {active}")
            
            if 'DmlExecutionProvider' not in active:
                msg_queue.put({'type': 'LOG', 'msg': f"⚠ Worker {worker_idx} GPU rejected. Running on CPU."})
                
        except Exception as e:
            print(f"!!! GPU INJECTION FAIL (Worker {worker_idx}): {e}")
            detector = NudeDetector() # Fallback to standard CPU loading

    # [Rest of the worker code remains the same...]
    # Copy the rest of the function from the previous full code block starting from:
    # label_map = {lbl: cat for cat, labels in FILTERS.items() for lbl in labels}
    
    # ... (Paste the rest of the function here or use the full block below)

    label_map = {lbl: cat for cat, labels in FILTERS.items() for lbl in labels}
    scan_height = 240 if settings.get('scan_preset') == 'Max Speed' else 480 if settings.get('scan_preset') == 'Balanced' else 720

    try:
        cap = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
        if not cap.isOpened(): cap = cv2.VideoCapture(safe_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    except: return
    
    if start_frame == 0:
        msg_queue.put({'type': 'INIT_VIDEO', 'video': display_name, 'total_frames': total_video_frames})

    clean_video_name = re.sub(r'[^\w\-_\. ]', '_', os.path.splitext(display_name)[0])
    base_root = os.path.join(OUTPUT_ROOT, subject, clean_video_name)
    os.makedirs(base_root, exist_ok=True)

    skip_interval = 1 if settings.get('deep_scan') else int(fps / 2)
    frame_count, frames_batch = start_frame, 0
    prev_gray, last_preview = None, time.time()

    while True:
        if end_frame and frame_count >= end_frame: break
        ret, frame = cap.read()
        if not ret or frame is None: break

        if frame_count % skip_interval == 0:
            if time.time() - last_preview > 2.0:
                try:
                    preview = cv2.resize(frame, (120, 80), interpolation=cv2.INTER_NEAREST)
                    _, buf = cv2.imencode(".jpg", preview)
                    msg_queue.put({'type': 'GRID_PREVIEW', 'idx': worker_idx, 'data': buf.tobytes()})
                    last_preview = time.time()
                except: pass

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_static = False
            if settings.get('motion_gate') and prev_gray is not None:
                if np.sum(cv2.threshold(cv2.absdiff(prev_gray, gray), 25, 255, cv2.THRESH_BINARY)[1]) < 25000: is_static = True
            prev_gray = gray

            if detector and not is_static:
                aspect = orig_w / orig_h
                small = cv2.resize(frame, (int(scan_height * aspect), scan_height))
                detections = detector.detect(small)
                for det in detections:
                    label, score = det.get('class', 'UNK'), det.get('score', 0)
                    if label.upper() in active_filters and score >= settings['threshold1']:
                        sb = det['box']
                        scale_x, scale_y = orig_w / (int(scan_height * aspect)), orig_h / scan_height
                        real_box = [int(sb[0]*scale_x), int(sb[1]*scale_y), int(sb[2]*scale_x), int(sb[3]*scale_y)]
                        if not settings['double_check'] or verify_crop(frame, real_box, detector, settings['threshold2']):
                            ts = frame_count / fps
                            t_str = f"{int(ts)//3600:02d}-{int(ts)%3600//60:02d}-{int(ts)%60:02d}"
                            v_dir = os.path.join(base_root, "VERIFIED", label_map.get(label.upper(), "UNK"))
                            os.makedirs(v_dir, exist_ok=True)
                            f_path = os.path.join(v_dir, f"Frame-{frame_count}_{t_str}_{label.title()}.jpg")
                            cv2.imwrite(f_path, frame)
                            if settings.get('generate_clips'):
                                threading.Thread(target=generate_clip, args=(safe_path, ts, 5, f_path.replace(".jpg", ".mp4")), daemon=True).start()
                            msg_queue.put({'type': 'HIT', 'path': f_path, 'video': display_name})
                            break

            if settings.get('scan_ocr') and (frame_count % int(fps) == 0):
                try:
                    _, thresh_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    txt = pytesseract.image_to_string(thresh_img, config='--psm 6')
                    clean_txt = re.sub(r'[^A-Za-z0-9@_.]', ' ', txt).strip()
                    if len(clean_txt) > 4: msg_queue.put({'type': 'LOG', 'msg': f"📝 [OCR] {display_name}: {clean_txt[:25]}"})
                except: pass

        frames_batch += 1
        if frames_batch >= 200:
            msg_queue.put({'type': 'TICK', 'video': display_name, 'count': frames_batch})
            frames_batch = 0
        if frame_count % 50 == 0: gc.collect() 
        frame_count += 1
    
    msg_queue.put({'type': 'STATUS', 'video': display_name, 'status': 'Done'})
    msg_queue.put({'type': 'WORKER_STOP', 'idx': worker_idx})
    cap.release()

# --- TOP LEVEL WRAPPER (STAGGERED STARTUP) ---
def worker_wrapper(task_q, msg_q, worker_idx):
    # STAGGERED DELAY: 0.5s per worker
    # This prevents all 30 workers from hitting GPU memory simultaneously
    time.sleep(worker_idx * 0.5)
    
    while True:
        try:
            task = task_q.get(timeout=1) 
            video_worker(task, msg_q, worker_idx)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"!!! WORKER ERROR: {e}")

# ==========================================
#        USER INTERFACE (UI)
# ==========================================
class ArchivistPro(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        self.title("Archivist Pro v4.2.1 - MANUAL GPU INJECTION")
        self.geometry("1400x900")

        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.msg_queue = self.manager.Queue()
        
        self.prog_bars = {} 
        self.prog_data = {} 
        self.pool = [] 
        
        # Grid UI Storage
        self.grid_slots = {} 
        
        self.stop_event = multiprocessing.Event()
        self.is_running = False
        self.task_stack_local = [] 
        self.active_chunks = 0

        self._setup_layout()
        self.check_messages()
        
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
             messagebox.showerror("Error", f"Tesseract not found at:\n{pytesseract.pytesseract.tesseract_cmd}")

    def _setup_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar, text="ARCHIVIST PRO v4.2.1", font=("Impact", 28)).pack(pady=(20, 10))
        
        self.subj_input = ctk.CTkEntry(self.sidebar, placeholder_text="Subject Name...")
        self.subj_input.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="+ Add Videos", command=self.add_videos, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="+ Add Folder", command=self.add_folder, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Force Refresh UI", command=self.force_refresh, fg_color="#e67e22").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Clear UI Queue", command=self.clear_ui, fg_color="#c0392b").pack(fill="x", padx=20, pady=5)
        
        self.btn_start = ctk.CTkButton(self.sidebar, text="START QUEUE", command=self.start_processing, fg_color="#27ae60", height=60, font=("Arial", 16, "bold"))
        self.btn_start.pack(side="bottom", fill="x", padx=20, pady=10)

        # Main Area Tabs
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.tab_scan = self.tabs.add("  LIVE DASHBOARD  ")
        self.tab_settings = self.tabs.add("  SETTINGS  ")
        
        self._build_dashboard(self.tab_scan)
        self._build_settings(self.tab_settings)

    def _build_dashboard(self, parent):
        # Worker Grid
        ctk.CTkLabel(parent, text="Worker Grid (One screen per CPU Thread)").pack(anchor="w", padx=10)
        self.worker_grid_frame = ctk.CTkScrollableFrame(parent, height=350, fg_color="#1a1a1a")
        self.worker_grid_frame.pack(fill="x", padx=10, pady=5)
        
        # Latest Hit
        self.lbl_hit = ctk.CTkLabel(parent, text="[ Latest Critical Hit ]", height=150, fg_color="#2c3e50")
        self.lbl_hit.pack(fill="x", padx=10, pady=10)
        
        # Queue List
        self.queue_area = ctk.CTkScrollableFrame(parent, height=200)
        self.queue_area.pack(fill="x", padx=10, pady=5)
        
        # Log Box
        self.log_box = ctk.CTkTextbox(parent, height=100, font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=5)

    def _init_worker_grid(self, num_cores):
        # Clear existing
        for widget in self.worker_grid_frame.winfo_children(): 
            widget.destroy()
        
        self.grid_slots = {}
        cols = max(8, min(num_cores, 14))
        
        for i in range(cols):
            self.worker_grid_frame.grid_columnconfigure(i, weight=1)

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
        
        # Col 1: Performance
        frame_perf = ctk.CTkFrame(parent)
        frame_perf.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(frame_perf, text="PERFORMANCE TUNING", font=("Arial", 14, "bold")).pack(pady=10)

        self.var_preset = ctk.StringVar(value="Balanced")
        self.opt_preset = ctk.CTkOptionMenu(frame_perf, values=["Max Speed", "Balanced", "High Precision"], 
                                            variable=self.var_preset, width=200)
        self.opt_preset.pack(pady=10)

        self.var_focus = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(frame_perf, text="FOCUS MODE (1 Video at a time)", variable=self.var_focus, progress_color="#e74c3c").pack(anchor="w", padx=20, pady=10)
        
        self.var_motion = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(frame_perf, text="Motion Gating (Skip Static Frames)", variable=self.var_motion).pack(anchor="w", padx=20, pady=10)

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

        # Col 2: Filters & Misc
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

    # --- BUTTON ACTIONS ---
    def add_videos(self):
        try:
            files = filedialog.askopenfilenames(
                parent=self, 
                title="Select Video Files",
                filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov *.wmv"), ("All Files", "*.*")]
            )
            if files: self.stack_files(files)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file dialog:\n{e}")

    def add_folder(self):
        folder = filedialog.askdirectory(parent=self)
        if folder:
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
            self.stack_files(files)

    def clear_ui(self):
        self.prog_bars, self.prog_data, self.task_stack_local = {}, {}, []
        for w in self.queue_area.winfo_children(): w.destroy()
        
    def force_refresh(self):
        self.check_messages()
        self.update()

    def stack_files(self, files):
        subj, sett = self.subj_input.get().strip() or "Default_Subject", self.get_settings()
        
        for f in files:
            try:
                if not os.path.exists(f): continue

                task = {"path": f, "subject": subj, "settings": sett}
                self.task_queue.put(task)
                self.task_stack_local.append(task)
                
                video_name = os.path.basename(f) 
                
                # Setup UI Row
                row = ctk.CTkFrame(self.queue_area)
                row.pack(fill="x", padx=5, pady=2)
                ctk.CTkLabel(row, text=f"{video_name}", anchor="w", width=300).pack(side="left", padx=10)
                
                est_lbl = ctk.CTkLabel(row, text="Wait...", width=100)
                est_lbl.pack(side="right", padx=10)
                
                bar = ctk.CTkProgressBar(row)
                bar.set(0)
                bar.pack(side="right", fill="x", expand=True, padx=10)
                
                self.prog_bars[video_name] = {'bar': bar, 'label': est_lbl}

                # Quick metadata read (Background)
                try:
                    cap = cv2.VideoCapture(resolve_path(f))
                    if cap.isOpened():
                        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        self.prog_data[video_name] = {'total': tf, 'done': 0}
                        est_str = f"Est: {tf/ASSUMED_FPS_BASELINE/60:.1f}m"
                        est_lbl.configure(text=est_str)
                    cap.release()
                except: pass

            except Exception as e:
                print(f"Error adding {f}: {e}")
        self.log(f"Added {len(files)} files to queue.")

    def log(self, msg):
        self.log_box.insert("end", f"{msg}\n")
        self.log_box.delete("1.0", "end-1000l")
        self.log_box.see("end")

    def check_messages(self):
        try:
            processed = 0
            while not self.msg_queue.empty() and processed < 100:
                m = self.msg_queue.get_nowait()
                processed += 1
                
                if m['type'] == 'GRID_PREVIEW' and m['idx'] in self.grid_slots:
                    try:
                        nparr = np.frombuffer(m['data'], np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        self.grid_slots[m['idx']]['img'].configure(image=ctk.CTkImage(pil, size=(120, 80)), text="")
                    except: pass
                
                elif m['type'] == 'WORKER_STOP':
                    if self.active_chunks > 0: self.active_chunks -= 1
                    if m['idx'] in self.grid_slots:
                        self.grid_slots[m['idx']]['txt'].configure(text="Idle", text_color="gray")
                        self.grid_slots[m['idx']]['img'].configure(image=None, text=f"W{m['idx']}")

                elif m['type'] == 'WORKER_START':
                    if m['idx'] in self.grid_slots:
                        self.grid_slots[m['idx']]['txt'].configure(text=m['file'][:15], text_color="green")

                elif m['type'] == 'TICK' and m['video'] in self.prog_bars:
                    d = self.prog_data.get(m['video'], {'done': 0, 'total': 1})
                    d['done'] += m['count']
                    self.prog_bars[m['video']]['bar'].set(d['done']/d['total'])

                elif m['type'] == 'HIT':
                    try:
                        self.lbl_hit.configure(image=ctk.CTkImage(Image.open(m['path']), size=(300, 200)), text="")
                    except: pass
                
                elif m['type'] == 'LOG':
                    self.log(m['msg'])
        except:
            pass 
        finally:
            self.after(20, self.check_messages)

    def start_processing(self):
        if self.is_running: return
        self.stop_event.clear()
        self.is_running = True
        self.btn_start.configure(state="disabled", text="RUNNING...", fg_color="#e74c3c")
        
        # Calculate Cores
        # Recommendation: Use CPU_COUNT - 2 for safety
        num_cores = max(1, multiprocessing.cpu_count() - 2)
        
        self.log(f"Starting Scan with {num_cores} Workers...")
        self._init_worker_grid(num_cores)
        
        if self.var_focus.get():
            threading.Thread(target=self.generate_focus_tasks, args=(num_cores,), daemon=True).start()
        
        for i in range(num_cores):
            p = multiprocessing.Process(target=worker_wrapper, args=(self.task_queue, self.msg_queue, i))
            p.daemon = True
            p.start()
            self.pool.append(p)
            
        threading.Thread(target=self.run_dynamic_pool, args=(num_cores,), daemon=True).start()

    def generate_focus_tasks(self, cores):
        sett = self.get_settings()
        for task_template in self.task_stack_local:
            raw_path = task_template['path']
            subj = task_template['subject']
            
            try:
                cap = cv2.VideoCapture(resolve_path(raw_path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                if total <= 0: continue
                
                chunk_size = total // cores
                self.log(f"Processing: {os.path.basename(raw_path)} ({cores} Chunks)")
                self.active_chunks = cores 
                
                for i in range(cores):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size if i < cores - 1 else total
                    self.task_queue.put({
                        "path": raw_path, "subject": subj, "settings": sett, 
                        "start_frame": start, "end_frame": end
                    })
                
                # Barrier Wait
                while self.active_chunks > 0 and self.is_running:
                    time.sleep(1)
                
                if not self.is_running: break
                time.sleep(2) # Cooldown
                self.log(f"Finished: {os.path.basename(raw_path)}")

            except Exception as e:
                self.log(f"Error chunking video: {e}")

    def run_dynamic_pool(self, cores):
        while not self.stop_event.is_set():
            time.sleep(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = ArchivistPro()
    app.mainloop()