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
os.environ['OMP_THREAD_LIMIT'] = '1'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
OUTPUT_ROOT = "Archived_Scans" 
DB_FILE = "scan_history.db"
SETTINGS_FILE = "user_settings.json"
ASSUMED_FPS_BASELINE = 300 

FILTERS = {
    "CRITICAL": ["FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED", "FEMALE_BREAST_EXPOSED"],
    "WARNING": ["BUTTOCKS_EXPOSED", "FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED", "MALE_GENITALIA_COVERED", "BUTTOCKS_COVERED", "ANUS_COVERED"],
    "MINOR": ["BELLY_EXPOSED", "BELLY_COVERED", "MALE_BREAST_EXPOSED", "FEET_EXPOSED", "FEET_COVERED", "ARMPITS_EXPOSED", "ARMPITS_COVERED"]
}

try:
    from nudenet import NudeDetector
    import onnxruntime as ort
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("!!! WARNING: Libraries Missing !!!")

# ==========================================
#        DATABASE & HELPERS
# ==========================================
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS history (filename TEXT PRIMARY KEY, date_scanned TEXT, status TEXT)''')
        conn.commit(); conn.close()
    except: pass

def check_history(filename):
    try:
        conn = sqlite3.connect(DB_FILE); c = conn.cursor()
        c.execute("SELECT status FROM history WHERE filename=?", (filename,))
        result = c.fetchone(); conn.close()
        return result is not None
    except: return False

def mark_scanned(filename):
    try:
        conn = sqlite3.connect(DB_FILE); c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO history VALUES (?, ?, ?)", (filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "COMPLETED"))
        conn.commit(); conn.close()
    except: pass

def resolve_path(path):
    abs_path = os.path.abspath(path)
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(abs_path, buffer, 1024) > 0: return buffer.value
    except: pass 
    if len(abs_path) > 240 and not abs_path.startswith("\\\\?\\"): return "\\\\?\\" + abs_path
    return abs_path

def verify_crop(frame, box, detector, thresh):
    if not detector: return True
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    pad_x = int(bw * 0.2); pad_y = int(bh * 0.2)
    x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x); y2 = min(h, y + bh + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return False
    detections = detector.detect(crop)
    for d in detections:
        if d.get('score', 0) >= thresh: return True
    return False

def generate_clip(video_path, start_time, duration, output_path):
    try:
        with VideoFileClip(video_path) as clip:
            max_dur = clip.duration if clip.duration else (start_time + duration + 100)
            end_t = min(max_dur, start_time + (duration / 2))
            start_t = max(0, start_time - (duration / 2))
            new_clip = clip.subclipped(start_t, end_t)
            new_clip.write_videofile(output_path, codec='libx264', audio=False, preset='ultrafast', threads=4, logger=None)
    except Exception as e: print(f"Clip Error: {e}")

# ==========================================
#        WORKER ENGINE
# ==========================================
def video_worker(task, msg_queue, worker_idx):
    import onnxruntime as ort
    safe_path = resolve_path(task['path'])
    display_name = os.path.basename(task['path'])
    subject, settings = task['subject'], task['settings']
    active_filters = settings['active_filters'] 
    start_frame, end_frame = task.get('start_frame', 0), task.get('end_frame', None)
    
    msg_queue.put({'type': 'WORKER_START', 'idx': worker_idx, 'file': display_name})

    # DirectML Injection
    detector = None
    if AI_AVAILABLE:
        try:
            opts = ort.SessionOptions()
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [('DmlExecutionProvider', {'device_id': 0})]
            home = os.path.expanduser("~")
            possible = glob.glob(os.path.join(home, ".NudeNet", "*.onnx"))
            if not possible: temp = NudeDetector(); model_path = temp.onnx_session._model_path 
            else:
                model_path = possible[0]
                for p in possible:
                    if "640m" in p: model_path = p; break
                    if "320n" in p: model_path = p; break
            detector = NudeDetector()
            detector.detector = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            if 'DmlExecutionProvider' not in detector.detector.get_providers():
                msg_queue.put({'type': 'LOG', 'msg': f"⚠ W{worker_idx}: GPU Failed. Using CPU."})
        except: detector = NudeDetector() 

    label_map = {lbl: cat for cat, labels in FILTERS.items() for lbl in labels}
    scan_height = 240 if settings.get('scan_preset') == 'Max Speed' else 480 if settings.get('scan_preset') == 'Balanced' else 720

    try:
        cap = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
        if not cap.isOpened(): cap = cv2.VideoCapture(safe_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    except: return
    
    clean_name = re.sub(r'[^\w\-_\. ]', '_', os.path.splitext(display_name)[0])
    base_root = os.path.join(OUTPUT_ROOT, subject, clean_name)
    
    deep_scan = settings.get('deep_scan', False)
    skip_interval = 1 if deep_scan else int(fps / 2)
    
    # UI THROTTLE CALCULATION
    # Only update UI every 1% of frames to prevent lag
    chunk_size = (end_frame - start_frame) if end_frame else total
    ui_update_interval = max(50, int(chunk_size / 100)) 

    frame_count = start_frame
    frames_since_update = 0
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
                
                processed_hit = False
                for det in detections:
                    label, score = det.get('class', 'UNK'), det.get('score', 0)
                    if label.upper() in active_filters and score >= settings['threshold1']:
                        
                        sb = det['box']
                        scale_x, scale_y = orig_w / (int(scan_height * aspect)), orig_h / scan_height
                        real_box = [int(sb[0]*scale_x), int(sb[1]*scale_y), int(sb[2]*scale_x), int(sb[3]*scale_y)]
                        
                        ts = frame_count / fps
                        t_str = f"{int(ts)//3600:02d}-{int(ts)%3600//60:02d}-{int(ts)%60:02d}"
                        cat = label_map.get(label.upper(), "UNK")

                        # FIRST PASS SAVE
                        fp_dir = os.path.join(base_root, "FIRST_PASS", cat)
                        os.makedirs(fp_dir, exist_ok=True)
                        f_name = f"Frame-{frame_count}_{t_str}_{label.title()}.jpg"
                        cv2.imwrite(os.path.join(fp_dir, f_name), frame)

                        if not settings['double_check'] or verify_crop(frame, real_box, detector, settings['threshold2']):
                            v_dir = os.path.join(base_root, "VERIFIED", cat)
                            os.makedirs(v_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(v_dir, f_name), frame)
                            
                            if settings.get('generate_clips') and not processed_hit:
                                threading.Thread(target=generate_clip, args=(safe_path, ts, 5, os.path.join(v_dir, f_name.replace(".jpg", ".mp4"))), daemon=True).start()
                                processed_hit = True
                            
                            msg_queue.put({'type': 'HIT', 'path': os.path.join(v_dir, f_name), 'video': display_name})

            if settings.get('scan_ocr') and (frame_count % int(fps) == 0):
                try:
                    _, thresh_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    txt = pytesseract.image_to_string(thresh_img, config='--psm 6')
                    clean = re.sub(r'[^A-Za-z0-9@_.]', ' ', txt).strip()
                    if len(clean) > 4: msg_queue.put({'type': 'LOG', 'msg': f"📝 OCR: {clean[:20]}"})
                except: pass

        # OPTIMIZED UI UPDATES (Reduces Lag)
        frames_since_update += 1
        if frames_since_update >= ui_update_interval:
            msg_queue.put({'type': 'TICK', 'video': display_name, 'count': frames_since_update})
            frames_since_update = 0
            
        if frame_count % 50 == 0: gc.collect() 
        frame_count += 1
    
    if frames_since_update > 0:
        msg_queue.put({'type': 'TICK', 'video': display_name, 'count': frames_since_update})

    if not end_frame: msg_queue.put({'type': 'DB_UPDATE', 'file': display_name})
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
        except: pass

# ==========================================
#        MAIN GUI
# ==========================================
class ArchivistPro(ctk.CTk):
    def __init__(self):
        super().__init__()
        init_db() 
        ctk.set_appearance_mode("dark")
        self.title("Archivist Pro v4.5 - STABLE")
        self.geometry("1400x900")
        
        self.manager = multiprocessing.Manager()
        self.task_queue, self.msg_queue = self.manager.Queue(), self.manager.Queue()
        self.prog_bars, self.prog_data, self.pool, self.grid_slots = {}, {}, [], {}
        self.stop_event, self.is_running, self.task_stack_local = multiprocessing.Event(), False, []
        self.active_chunks = 0
        self.chk_filters = {}

        self._setup_layout()
        self.load_settings() 
        self.check_messages()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_layout(self):
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(self.sidebar, text="ARCHIVIST PRO v4.5", font=("Impact", 28)).pack(pady=20)
        self.subj_input = ctk.CTkEntry(self.sidebar, placeholder_text="Subject Name..."); self.subj_input.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="+ Add Videos", command=self.add_videos, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="+ Add Folder", command=self.add_folder, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Force Refresh UI", command=self.force_refresh, fg_color="#e67e22").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Clear UI", command=self.clear_ui, fg_color="#c0392b").pack(fill="x", padx=20, pady=5)
        
        self.btn_stop = ctk.CTkButton(self.sidebar, text="⚠ FORCE STOP", command=self.stop_processing, fg_color="#c0392b", hover_color="#922b21")
        self.btn_stop.pack(side="bottom", fill="x", padx=20, pady=10)
        self.btn_start = ctk.CTkButton(self.sidebar, text="START QUEUE", command=self.start_processing, fg_color="#27ae60", height=60, font=("Arial", 16, "bold"))
        self.btn_start.pack(side="bottom", fill="x", padx=20, pady=10)

        self.tabs = ctk.CTkTabview(self); self.tabs.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.tab_scan = self.tabs.add(" DASHBOARD "); self.tab_settings = self.tabs.add(" SETTINGS ")
        
        self._build_dashboard(self.tab_scan); self._build_settings(self.tab_settings)

    def _build_dashboard(self, parent):
        ctk.CTkLabel(parent, text="Worker Grid").pack(anchor="w", padx=10)
        self.worker_grid_frame = ctk.CTkScrollableFrame(parent, height=350, fg_color="#1a1a1a"); self.worker_grid_frame.pack(fill="x", padx=10, pady=5)
        self.lbl_hit = ctk.CTkLabel(parent, text="[ Latest Hit ]", height=150, fg_color="#2c3e50"); self.lbl_hit.pack(fill="x", padx=10, pady=10)
        self.queue_area = ctk.CTkScrollableFrame(parent, height=200); self.queue_area.pack(fill="x", padx=10, pady=5)
        self.log_box = ctk.CTkTextbox(parent, height=100, font=("Consolas", 12)); self.log_box.pack(fill="both", expand=True, padx=10, pady=5)

    def _init_worker_grid(self, num_cores):
        for w in self.worker_grid_frame.winfo_children(): w.destroy()
        cols = max(8, min(num_cores, 14))
        for i in range(cols): self.worker_grid_frame.grid_columnconfigure(i, weight=1)
        for i in range(num_cores):
            f = ctk.CTkFrame(self.worker_grid_frame); f.grid(row=i//cols, column=i%cols, padx=3, pady=3, sticky="ew")
            ctk.CTkLabel(f, text=f"W{i}", width=120, height=80, fg_color="black").pack(pady=5)
            self.grid_slots[i] = {'img': f.winfo_children()[0], 'txt': ctk.CTkLabel(f, text="Idle", font=("Arial", 10))}
            self.grid_slots[i]['txt'].pack(pady=(0,5))

    def _build_settings(self, parent):
        parent.grid_columnconfigure(0, weight=1); parent.grid_columnconfigure(1, weight=1)
        left = ctk.CTkFrame(parent); left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.lbl_t1 = ctk.CTkLabel(left, text="First Scan: 0.40"); self.lbl_t1.pack()
        self.slider_t1 = ctk.CTkSlider(left, from_=0.1, to=0.9, command=lambda v: self.lbl_t1.configure(text=f"First Scan: {v:.2f}")); self.slider_t1.pack(pady=5)
        self.lbl_t2 = ctk.CTkLabel(left, text="Verify: 0.60"); self.lbl_t2.pack()
        self.slider_t2 = ctk.CTkSlider(left, from_=0.1, to=0.9, command=lambda v: self.lbl_t2.configure(text=f"Verify: {v:.2f}")); self.slider_t2.pack(pady=5)
        self.var_preset = ctk.StringVar(value="Balanced")
        ctk.CTkOptionMenu(left, values=["Max Speed", "Balanced", "High Precision"], variable=self.var_preset).pack(pady=10)
        self.var_focus, self.var_motion, self.var_deep = ctk.BooleanVar(), ctk.BooleanVar(), ctk.BooleanVar()
        ctk.CTkSwitch(left, text="Focus Mode", variable=self.var_focus).pack(anchor="w", pady=5, padx=20)
        ctk.CTkSwitch(left, text="Motion Gating", variable=self.var_motion).pack(anchor="w", pady=5, padx=20)
        ctk.CTkSwitch(left, text="Deep Scan (Slow)", variable=self.var_deep).pack(anchor="w", pady=5, padx=20)

        right = ctk.CTkFrame(parent); right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        scroll_filt = ctk.CTkScrollableFrame(right, height=300); scroll_filt.pack(fill="both", expand=True, padx=5, pady=5)
        self.chk_filters = {}
        for cat, labels in FILTERS.items():
            ctk.CTkLabel(scroll_filt, text=f"--- {cat} ---", text_color="gray").pack(anchor="w")
            for lbl in labels:
                var = ctk.BooleanVar(value=True)
                ctk.CTkCheckBox(scroll_filt, text=lbl.replace("_", " ").title(), variable=var).pack(anchor="w", padx=10, pady=2)
                self.chk_filters[lbl] = var
        self.var_verify, self.var_ocr, self.var_clips, self.var_skip, self.var_debug = ctk.BooleanVar(value=True), ctk.BooleanVar(), ctk.BooleanVar(value=True), ctk.BooleanVar(value=True), ctk.BooleanVar()
        ctk.CTkSwitch(right, text="Double-Check", variable=self.var_verify).pack(anchor="w", padx=20)
        ctk.CTkSwitch(right, text="Scan Usernames (OCR)", variable=self.var_ocr).pack(anchor="w", padx=20)
        ctk.CTkSwitch(right, text="Generate Clips", variable=self.var_clips).pack(anchor="w", padx=20)
        ctk.CTkSwitch(right, text="Skip Scanned", variable=self.var_skip).pack(anchor="w", padx=20)
        ctk.CTkSwitch(right, text="Debug Log", variable=self.var_debug).pack(anchor="w", padx=20)

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f: data = json.load(f)
                self.slider_t1.set(data.get('t1', 0.4)); self.lbl_t1.configure(text=f"First Scan: {data.get('t1', 0.4):.2f}")
                self.slider_t2.set(data.get('t2', 0.6)); self.lbl_t2.configure(text=f"Verify: {data.get('t2', 0.6):.2f}")
                self.var_preset.set(data.get('preset', 'Balanced')); self.var_focus.set(data.get('focus', False))
                self.var_motion.set(data.get('motion', False)); self.var_ocr.set(data.get('ocr', False))
                self.var_deep.set(data.get('deep', False)); self.var_skip.set(data.get('skip', True))
                self.var_debug.set(data.get('debug', False))
                if 'filters' in data:
                    for lbl, var in self.chk_filters.items(): var.set(lbl in data['filters'])
            except: pass

    def save_settings(self):
        data = {
            't1': self.slider_t1.get(), 't2': self.slider_t2.get(), 'preset': self.var_preset.get(),
            'focus': self.var_focus.get(), 'motion': self.var_motion.get(), 'ocr': self.var_ocr.get(),
            'deep': self.var_deep.get(), 'skip': self.var_skip.get(), 'debug': self.var_debug.get(),
            'filters': [lbl for lbl, var in self.chk_filters.items() if var.get()]
        }
        with open(SETTINGS_FILE, 'w') as f: json.dump(data, f)

    def on_close(self): self.save_settings(); self.stop_processing(); self.destroy()

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
            
            # --- FIX: QUEUE LOGIC REWRITTEN ---
            # Do NOT put into task_queue yet. Store in local stack.
            task = {"path": f, "subject": subj, "settings": sett}
            self.task_stack_local.append(task)
            
            row = ctk.CTkFrame(self.queue_area); row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=os.path.basename(f), width=300, anchor="w").pack(side="left")
            est_lbl = ctk.CTkLabel(row, text="Calc...", width=100); est_lbl.pack(side="right")
            bar = ctk.CTkProgressBar(row); bar.set(0); bar.pack(side="right", fill="x", expand=True)
            self.prog_bars[os.path.basename(f)] = {'bar': bar, 'lbl': est_lbl} # Store label ref
            
            # --- FIX: RESTORED ETC CALCULATION ---
            threading.Thread(target=self.calc_eta, args=(f, est_lbl), daemon=True).start()
            cnt += 1
        self.log_box.insert("end", f"Queued {cnt} files. Click START to begin.\n")

    def calc_eta(self, path, label_widget):
        try:
            cap = cv2.VideoCapture(resolve_path(path))
            if cap.isOpened():
                tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                mins = tf / ASSUMED_FPS_BASELINE / 60
                label_widget.configure(text=f"Est: {mins:.1f}m")
            cap.release()
        except: pass

    def get_settings_dict(self):
        return {
            'threshold1': self.slider_t1.get(), 'threshold2': self.slider_t2.get(),
            'scan_preset': self.var_preset.get(), 'deep_scan': self.var_deep.get(),
            'double_check': self.var_verify.get(), 'scan_ocr': self.var_ocr.get(),
            'motion_gate': self.var_motion.get(), 'generate_clips': self.var_clips.get(),
            'active_filters': [lbl for lbl, var in self.chk_filters.items() if var.get()]
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
                elif m['type'] == 'TICK' and m['video'] in self.prog_bars: 
                    # Progress Update
                    pb = self.prog_bars[m['video']]['bar']
                    curr = pb.get()
                    # Increment by small amount (simulated) or calculate real % if you passed total frames
                    # Simplified: just step
                    pb.step() 
                elif m['type'] == 'HIT':
                    try: self.lbl_hit.configure(image=ctk.CTkImage(Image.open(m['path']), size=(300, 200)), text="")
                    except: pass
                elif m['type'] == 'LOG': 
                    if self.var_debug.get() or "Error" in m['msg']: self.log_box.insert("end", f"{m['msg']}\n"); self.log_box.see("end")
                elif m['type'] == 'DB_UPDATE': mark_scanned(m['file']) 
        except: pass
        finally: self.after(20, self.check_messages)

    def start_processing(self):
        if self.is_running: return
        self.stop_event.clear(); self.is_running = True
        self.btn_start.configure(state="disabled", text="RUNNING...", fg_color="#e74c3c")
        num_cores = max(1, multiprocessing.cpu_count() - 2)
        self._init_worker_grid(num_cores)
        
        # --- FIX: CORE QUEUE LOGIC ---
        if self.var_focus.get():
            threading.Thread(target=self.generate_focus_tasks, args=(num_cores,), daemon=True).start()
        else:
            # Standard Mode: Dump everything to queue
            for t in self.task_stack_local: self.task_queue.put(t)
            
        for i in range(num_cores):
            p = multiprocessing.Process(target=worker_wrapper, args=(self.task_queue, self.msg_queue, i)); p.daemon = True; p.start(); self.pool.append(p)
    
    def generate_focus_tasks(self, cores):
        sett = self.get_settings_dict()
        for t in self.task_stack_local:
            try:
                cap = cv2.VideoCapture(resolve_path(t['path'])); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
                chunk = total // cores; self.active_chunks = cores
                for i in range(cores): self.task_queue.put({"path": t['path'], "subject": t['subject'], "settings": sett, "start_frame": i*chunk, "end_frame": (i+1)*chunk if i<cores-1 else total})
                while self.active_chunks > 0 and self.is_running: time.sleep(1)
            except: pass

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