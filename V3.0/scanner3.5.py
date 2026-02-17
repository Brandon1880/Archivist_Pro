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
from datetime import datetime
from moviepy import VideoFileClip
from PIL import Image

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
OUTPUT_ROOT = "Archived_Scans" 

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

# --- HELPER: WINDOWS SHORT PATH ---
def get_short_path(long_path):
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        GetShortPathName = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathName(long_path, buffer, 1024)
        return buffer.value
    except:
        return long_path

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

# --- WORKER PROCESS ---
def video_worker(task, msg_queue):
    # 1. SETUP PATHS
    raw_path = os.path.abspath(task['path'])
    safe_path = get_short_path(raw_path) 
    
    # FIX: Use the full filename (with extension) as ID to match UI
    display_name = os.path.basename(raw_path) 
    
    subject = task['subject']
    settings = task['settings']
    active_filters = settings['active_filters'] 
    
    start_frame = task.get('start_frame', 0)
    end_frame = task.get('end_frame', None) 
    
    worker_id = f"{display_name} [{start_frame}-{end_frame if end_frame else 'END'}]"
    print(f"   -> Worker started: {worker_id}")

    # Notify UI we started
    msg_queue.put({'type': 'STATUS', 'video': display_name, 'status': 'Starting...'})

    label_map = {}
    for cat, labels in FILTERS.items():
        for lbl in labels:
            label_map[lbl] = cat

    # Initialize AI
    try:
        if AI_AVAILABLE:
            try:
                detector = NudeDetector(providers=['DmlExecutionProvider'])
            except:
                detector = NudeDetector()
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
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
    except Exception as e:
        msg_queue.put({'type': 'LOG', 'msg': f"❌ CRASH: {display_name} - {e}"})
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
    
    scan_height = int(settings.get('scan_res', 320))
    motion_gate = settings.get('motion_gate', False)
    
    no_frame_counter = 0

    while True:
        if end_frame and frame_count >= end_frame:
            break

        ret, frame = cap.read()
        if not ret: break

        if frame is None or frame.size == 0:
            no_frame_counter += 1
            if no_frame_counter > 50:
                break
            frame_count += 1
            continue
        
        no_frame_counter = 0 

        if frame_count % skip_interval == 0:
            
            # Motion Gating
            is_static = False
            if motion_gate:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    if np.sum(thresh) < 25000: is_static = True 
                prev_gray = gray
            
            # OCR
            if settings['scan_ocr'] and not is_static:
                try:
                    if not motion_gate: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray)
                    handles = re.findall(r'@[\w\d.]+|ig:\s*[\w\d.]+', text, re.I)
                    # Debug print only
                except: pass

            # AI
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
                        
                        x,y,w,h = real_box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        break 

                    if hit_found:
                        timestamp = frame_count / fps
                        seconds = int(timestamp)
                        t_str = f"{seconds//3600:02d}-{seconds%3600//60:02d}-{seconds%60:02d}"
                        cat_folder = label_map.get(label_found.upper(), "UNCATEGORIZED")
                        
                        # --- NEW FILENAME FORMAT ---
                        # Frame-Timestamp-Target
                        clean_label = label_found.replace(" ", "_").title()
                        f_name = f"Frame-{frame_count}_{t_str}_{clean_label}.jpg"
                        
                        fp_dir = os.path.join(base_root, "FIRST_PASS", cat_folder)
                        os.makedirs(fp_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(fp_dir, f_name), frame)
                        
                        if hit_verified:
                            v_dir = os.path.join(base_root, "VERIFIED", cat_folder)
                            os.makedirs(v_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(v_dir, f_name), frame)
                            msg_queue.put({'type': 'HIT', 'path': os.path.join(v_dir, f_name), 'video': display_name})

                except Exception as e:
                    print(f"!!! DETECTION ERROR: {e}")

        frames_processed_batch += 1
        if frames_processed_batch >= 50:
            msg_queue.put({'type': 'TICK', 'video': display_name, 'count': frames_processed_batch})
            frames_processed_batch = 0

        frame_count += 1
    
    if frames_processed_batch > 0:
        msg_queue.put({'type': 'TICK', 'video': display_name, 'count': frames_processed_batch})
    
    msg_queue.put({'type': 'STATUS', 'video': display_name, 'status': 'Done'})
    cap.release()

# --- TOP LEVEL WRAPPER ---
def worker_wrapper(task_q, msg_q):
    while True:
        try:
            task = task_q.get(timeout=1) 
            video_worker(task, msg_q)
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
        self.title("Archivist Pro v3.5 - BUGFIX & RENAME")
        self.geometry("1400x900")

        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.msg_queue = self.manager.Queue()
        
        self.prog_bars = {} 
        self.prog_data = {} 
        self.pool = [] 
        self.stop_event = multiprocessing.Event()
        self.is_running = False
        self.task_stack_local = [] 

        self._setup_layout()
        self.check_messages()
        
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
             messagebox.showerror("Error", f"Tesseract not found at:\n{pytesseract.pytesseract.tesseract_cmd}")

    def _setup_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar, text="ARCHIVIST PRO v3.5", font=("Impact", 28)).pack(pady=(20, 10))
        
        self.subj_input = ctk.CTkEntry(self.sidebar, placeholder_text="Subject Name...")
        self.subj_input.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="+ Add Videos", command=self.add_videos, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="+ Add Folder", command=self.add_folder, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="🔍 TEST FILE", command=self.test_file_compatibility, fg_color="#e67e22").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Clear UI Queue", command=self.clear_ui, fg_color="#c0392b").pack(fill="x", padx=20, pady=5)
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
        preview_frame = ctk.CTkFrame(parent, height=350, fg_color="black")
        preview_frame.pack(fill="x", padx=10, pady=10)
        self.lbl_feed = ctk.CTkLabel(preview_frame, text="[ Live Video Feed ]", text_color="gray")
        self.lbl_feed.pack(side="left", expand=True, fill="both")
        self.lbl_hit = ctk.CTkLabel(preview_frame, text="[ Latest Critical Hit ]", text_color="gray")
        self.lbl_hit.pack(side="right", expand=True, fill="both")
        
        header = ctk.CTkFrame(parent, height=30)
        header.pack(fill="x", padx=10, pady=(5,0))
        ctk.CTkLabel(header, text="Video Queue", font=("Arial", 12, "bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="Progress & ETC", font=("Arial", 12, "bold")).pack(side="right", padx=10)

        self.queue_area = ctk.CTkScrollableFrame(parent, height=250)
        self.queue_area.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(parent, text="System Log").pack(anchor="w", padx=10)
        self.log_box = ctk.CTkTextbox(parent, height=150, font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=5)

    def _build_settings(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        
        frame_perf = ctk.CTkFrame(parent)
        frame_perf.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(frame_perf, text="PERFORMANCE TUNING", font=("Arial", 14, "bold")).pack(pady=10)

        self.var_focus = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(frame_perf, text="FOCUS MODE (Single Video/All Cores)", variable=self.var_focus, progress_color="#e74c3c").pack(anchor="w", padx=20, pady=10)
        
        self.var_motion = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(frame_perf, text="Motion Gating (Skip Static Frames)", variable=self.var_motion).pack(anchor="w", padx=20, pady=10)

        ctk.CTkLabel(frame_perf, text="AI Scan Resolution (Default: 320)").pack(anchor="w", padx=20, pady=(10,0))
        self.entry_res = ctk.CTkEntry(frame_perf, placeholder_text="320")
        self.entry_res.insert(0, "320")
        self.entry_res.pack(fill="x", padx=20, pady=5)

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

    def get_active_filters(self):
        active = []
        for lbl, var in self.chk_filters.items():
            if var.get(): active.append(lbl)
        return active

    def get_settings(self):
        try:
            res = int(self.entry_res.get())
        except:
            res = 320
        return {
            "threshold1": self.slider_t1.get(),
            "threshold2": self.slider_t2.get(),
            "deep_scan": self.var_deep.get(),
            "double_check": self.var_verify.get(),
            "scan_ocr": self.var_ocr.get(),
            "debug": self.var_debug.get(),
            "active_filters": self.get_active_filters(),
            "motion_gate": self.var_motion.get(),
            "scan_res": res
        }

    def test_file_compatibility(self):
        f = filedialog.askopenfilename(title="Select Video", filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov")])
        if not f: return
        
        try:
            buffer = ctypes.create_unicode_buffer(1024)
            ctypes.windll.kernel32.GetShortPathNameW(os.path.abspath(f), buffer, 1024)
            safe_path = buffer.value
        except:
            safe_path = f

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
        self.prog_bars = {}
        self.prog_data = {}
        self.task_stack_local = []
        for widget in self.queue_area.winfo_children(): widget.destroy()

    def stack_files(self, files):
        subj = self.subj_input.get().strip() or "Default_Subject"
        settings = self.get_settings()
        for f in files:
            task = {"path": f, "subject": subj, "settings": settings}
            self.task_queue.put(task)
            self.task_stack_local.append(task)
            
            # FIX: Use the full filename as the ID (same as Worker)
            video_name = os.path.basename(f) 
            
            row = ctk.CTkFrame(self.queue_area)
            row.pack(fill="x", padx=5, pady=2)
            
            lbl = ctk.CTkLabel(row, text=f"{video_name}", anchor="w", width=300)
            lbl.pack(side="left", padx=10)
            
            etc_lbl = ctk.CTkLabel(row, text="Waiting...", width=100)
            etc_lbl.pack(side="right", padx=10)
            
            bar = ctk.CTkProgressBar(row)
            bar.set(0)
            bar.pack(side="right", fill="x", expand=True, padx=10)
            
            self.prog_bars[video_name] = {'bar': bar, 'label': etc_lbl}
            
        self.log(f"Added {len(files)} files to active queue.")

    def log(self, msg):
        self.log_box.insert("end", f"{msg}\n")
        self.log_box.see("end")

    def check_messages(self):
        while not self.msg_queue.empty():
            msg = self.msg_queue.get_nowait()
            
            if msg['type'] == 'STATUS':
                v = msg['video']
                if v in self.prog_bars:
                    self.prog_bars[v]['label'].configure(text=msg['status'])

            elif msg['type'] == 'INIT_VIDEO':
                v_name = msg['video']
                t_frames = msg['total_frames']
                if v_name not in self.prog_data:
                    self.prog_data[v_name] = {
                        'total': t_frames, 
                        'done': 0, 
                        'start_time': time.time()
                    }

            elif msg['type'] == 'TICK':
                v_name = msg['video']
                count = msg['count']
                
                if v_name in self.prog_data and v_name in self.prog_bars:
                    data = self.prog_data[v_name]
                    data['done'] += count
                    
                    if data['total'] > 0:
                        pct = data['done'] / data['total']
                    else:
                        pct = 0
                    
                    self.prog_bars[v_name]['bar'].set(pct)
                    
                    elapsed = time.time() - data['start_time']
                    if pct > 0.01:
                        rate = pct / elapsed 
                        remaining_pct = 1.0 - pct
                        rem_seconds = int(remaining_pct / rate)
                        etc_str = f"{rem_seconds//60}:{rem_seconds%60:02d}"
                        self.prog_bars[v_name]['label'].configure(text=f"{int(pct*100)}% | {etc_str}")
                    else:
                        self.prog_bars[v_name]['label'].configure(text="Scanning...")

            elif msg['type'] == 'HIT':
                try:
                    pil = Image.open(msg['path'])
                    ctk_img = ctk.CTkImage(light_image=pil, size=(300, 200))
                    self.lbl_hit.configure(image=ctk_img, text="")
                    self.log(f"⚠ HIT: {os.path.basename(msg['path'])}")
                except: pass
            
            elif msg['type'] == 'LOG':
                self.log(msg['msg'])
                
        self.after(50, self.check_messages)

    def start_processing(self):
        if self.is_running:
            self.log("Batch is already running.")
            return
        self.stop_event.clear()
        self.is_running = True
        self.btn_start.configure(state="disabled", text="RUNNING...", fg_color="#e74c3c")
        self.tabs.set("  LIVE DASHBOARD  ") 
        num_cores = multiprocessing.cpu_count()
        
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
        
        for task_template in self.task_stack_local:
            raw_path = task_template['path']
            subj = task_template['subject']
            
            try:
                buffer = ctypes.create_unicode_buffer(1024)
                ctypes.windll.kernel32.GetShortPathNameW(raw_path, buffer, 1024)
                safe_path = buffer.value
            except:
                safe_path = raw_path

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
            p = multiprocessing.Process(target=worker_wrapper, args=(self.task_queue, self.msg_queue))
            p.daemon = True
            p.start()
            self.pool.append(p)
        while not self.stop_event.is_set():
            time.sleep(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = ArchivistPro()
    app.mainloop()