import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import os
import hashlib
import multiprocessing
import re
import pytesseract
import threading
import time
import queue # Standard python queue for thread safety
from datetime import datetime
from moviepy import VideoFileClip
from fpdf import FPDF
from PIL import Image, ImageTk

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
    path = task['path']
    subject = task['subject']
    settings = task['settings']
    active_filters = settings['active_filters'] 
    
    label_map = {}
    for cat, labels in FILTERS.items():
        for lbl in labels:
            label_map[lbl] = cat

    detector = NudeDetector() if AI_AVAILABLE else None
    video_name = os.path.splitext(os.path.basename(path))[0]
    
    base_save_dir = os.path.join(OUTPUT_ROOT, subject, video_name)
    os.makedirs(base_save_dir, exist_ok=True)

    try:
        with open(path, 'rb') as f:
            v_hash = hashlib.md5(f.read(8192)).hexdigest()
    except Exception as e:
        msg_queue.put({'type': 'LOG', 'msg': f"[ERROR] Could not read {video_name}: {e}"})
        return {}

    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    skip_interval = 1 if settings['deep_scan'] else int(fps / 2)
    frame_count = 0
    hit_count = 0
    found_handles = set()

    if settings['debug']:
        msg_queue.put({'type': 'LOG', 'msg': f"[DEBUG] Starting: {video_name}"})

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_count % skip_interval == 0:
            timestamp = frame_count / fps
            
            if frame_count % (skip_interval * 5) == 0:
                msg_queue.put({
                    'type': 'PROGRESS', 
                    'video': video_name, 
                    'val': (frame_count / total_frames)
                })

            if frame_count % (skip_interval * 10) == 0:
                small = cv2.resize(frame, (320, 180))
                _, buf = cv2.imencode('.jpg', small)
                msg_queue.put({'type': 'PREVIEW', 'data': buf, 'video': video_name})

            if settings['scan_ocr']:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray)
                handles = re.findall(r'@[\w\d.]+|ig:\s*[\w\d.]+', text, re.I)
                for h in handles: 
                    if h not in found_handles:
                        found_handles.add(h)
                        if settings['debug']:
                            msg_queue.put({'type': 'LOG', 'msg': f"[OCR] Found handle: {h}"})

            hit_found = False
            label_found = "Unknown"
            
            if detector:
                detections = detector.detect(frame)
                for det in detections:
                    label = det.get('class', det.get('label', 'UNKNOWN'))
                    score = det.get('score', 0)
                    
                    if label.upper() not in active_filters: continue
                    if score < settings['threshold1']: continue
                    
                    if settings['double_check']:
                        if not verify_crop(frame, det['box'], detector, settings['threshold2']):
                            continue
                    
                    hit_found = True
                    label_found = label
                    x, y, w, h = det['box']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    break 

            if hit_found:
                hit_count += 1
                category_folder = label_map.get(label_found.upper(), "UNCATEGORIZED")
                final_save_dir = os.path.join(base_save_dir, category_folder)
                os.makedirs(final_save_dir, exist_ok=True)

                seconds = int(timestamp)
                time_str = f"{seconds//3600:02d}-{(seconds%3600)//60:02d}-{seconds%60:02d}"
                clean_label = label_found.replace(" ", "_").title()
                filename = f"{clean_label}_{time_str}_Frame-{frame_count}.jpg"
                full_path = os.path.join(final_save_dir, filename)
                
                cv2.imwrite(full_path, frame)
                msg_queue.put({'type': 'HIT', 'path': full_path, 'video': video_name})
                
                if settings['debug']:
                    msg_queue.put({'type': 'LOG', 'msg': f"[HIT] Saved to {category_folder}"})

                try:
                    with VideoFileClip(path) as video:
                        clip = video.subclip(max(0, timestamp - 2), min(video.duration, timestamp + 3))
                        clip_name = f"{clean_label}_{time_str}_CLIP.mp4"
                        clip.write_videofile(os.path.join(final_save_dir, clip_name), 
                                             codec="libx264", audio=False, logger=None)
                except: pass

        frame_count += 1
    
    msg_queue.put({'type': 'PROGRESS', 'video': video_name, 'val': 1.0})
    cap.release()
    return {"subject": subject, "file": video_name, "hits": hit_count, "handles": list(found_handles), "hash": v_hash}

# --- MAIN UI ---
class ArchivistPro(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        
        self.title("Archivist Pro v1.3 - Dynamic Queue")
        self.geometry("1400x900")

        # THREAD-SAFE QUEUE for tasks
        # We use a real Queue() for tasks now, not just a list
        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.msg_queue = self.manager.Queue()
        
        self.prog_bars = {} 
        self.pool = None
        self.stop_event = multiprocessing.Event()
        self.is_running = False

        self._setup_layout()
        self.check_messages()

    def _setup_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar, text="ARCHIVIST PRO v1.3", font=("Impact", 28)).pack(pady=(20, 10))
        
        self.subj_input = ctk.CTkEntry(self.sidebar, placeholder_text="Subject Name...")
        self.subj_input.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="+ Add Videos", command=self.add_videos, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="+ Add Folder", command=self.add_folder, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Clear UI Queue", command=self.clear_ui, fg_color="#e67e22").pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="⚠ FORCE STOP ⚠", command=self.stop_processing, 
                      fg_color="#c0392b", hover_color="#922b21").pack(side="bottom", fill="x", padx=20, pady=10)

        self.btn_start = ctk.CTkButton(self.sidebar, text="START / RESUME QUEUE", command=self.start_processing, 
                                       fg_color="#27ae60", height=60, font=("Arial", 16, "bold"))
        self.btn_start.pack(side="bottom", fill="x", padx=20, pady=10)

        # --- RIGHT MAIN AREA ---
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.tab_scan = self.tabs.add("  LIVE DASHBOARD  ")
        self.tab_settings = self.tabs.add("  CONFIGURATION  ")

        self._build_dashboard(self.tab_scan)
        self._build_settings(self.tab_settings)

    def _build_dashboard(self, parent):
        preview_frame = ctk.CTkFrame(parent, height=350, fg_color="black")
        preview_frame.pack(fill="x", padx=10, pady=10)
        
        self.lbl_feed = ctk.CTkLabel(preview_frame, text="[ Live Video Feed ]", text_color="gray")
        self.lbl_feed.pack(side="left", expand=True, fill="both")
        
        self.lbl_hit = ctk.CTkLabel(preview_frame, text="[ Latest Critical Hit ]", text_color="gray")
        self.lbl_hit.pack(side="right", expand=True, fill="both")

        ctk.CTkLabel(parent, text="Batch Progress (Individual Videos)").pack(anchor="w", padx=10)
        self.queue_area = ctk.CTkScrollableFrame(parent, height=250)
        self.queue_area.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(parent, text="System Log").pack(anchor="w", padx=10)
        self.log_box = ctk.CTkTextbox(parent, height=150, font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=5)

    def _build_settings(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # SENSITIVITY
        frame_sens = ctk.CTkFrame(parent)
        frame_sens.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(frame_sens, text="SCAN SENSITIVITY", font=("Arial", 14, "bold")).pack(pady=10)

        self.lbl_t1 = ctk.CTkLabel(frame_sens, text="First Scan Threshold: 0.40")
        self.lbl_t1.pack(anchor="w", padx=20)
        self.slider_t1 = ctk.CTkSlider(frame_sens, from_=0.1, to=0.9, command=lambda v: self.lbl_t1.configure(text=f"First Scan Threshold: {v:.2f}"))
        self.slider_t1.set(0.4)
        self.slider_t1.pack(fill="x", padx=20, pady=5)

        self.lbl_t2 = ctk.CTkLabel(frame_sens, text="Verification Threshold: 0.60")
        self.lbl_t2.pack(anchor="w", padx=20, pady=(20,0))
        self.slider_t2 = ctk.CTkSlider(frame_sens, from_=0.1, to=0.9, command=lambda v: self.lbl_t2.configure(text=f"Verification Threshold: {v:.2f}"))
        self.slider_t2.set(0.6)
        self.slider_t2.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(frame_sens, text="OPTIONS", font=("Arial", 14, "bold")).pack(pady=(30,10))
        self.var_deep = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(frame_sens, text="Deep Scan (Check Every Frame)", variable=self.var_deep).pack(anchor="w", padx=20, pady=5)
        self.var_verify = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(frame_sens, text="Double-Check (Zoom & Verify)", variable=self.var_verify).pack(anchor="w", padx=20, pady=5)
        self.var_ocr = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(frame_sens, text="Scan Usernames (OCR)", variable=self.var_ocr).pack(anchor="w", padx=20, pady=5)
        self.var_debug = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(frame_sens, text="Show Debug Info in Log", variable=self.var_debug, progress_color="orange").pack(anchor="w", padx=20, pady=5)

        # FILTERS
        frame_filt = ctk.CTkFrame(parent)
        frame_filt.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(frame_filt, text="TARGET FILTERS", font=("Arial", 14, "bold")).pack(pady=10)

        self.chk_filters = {}
        for category, labels in FILTERS.items():
            ctk.CTkLabel(frame_filt, text=f"--- {category} ---", text_color="gray").pack(anchor="w", padx=10, pady=(10,0))
            for lbl in labels:
                clean_name = lbl.replace("_", " ").title().replace("Exposed", "").replace("Covered", "(C)")
                var = ctk.BooleanVar(value=True)
                ctk.CTkCheckBox(frame_filt, text=clean_name, variable=var).pack(anchor="w", padx=20, pady=2)
                self.chk_filters[lbl] = var

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
            "active_filters": self.get_active_filters()
        }

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
        for widget in self.queue_area.winfo_children(): widget.destroy()

    def stack_files(self, files):
        subj = self.subj_input.get().strip() or "Default_Subject"
        settings = self.get_settings()
        
        for f in files:
            task = {"path": f, "subject": subj, "settings": settings}
            
            # --- PUSH DIRECTLY TO QUEUE ---
            self.task_queue.put(task)
            
            video_name = os.path.splitext(os.path.basename(f))[0]
            row = ctk.CTkFrame(self.queue_area)
            row.pack(fill="x", padx=5, pady=2)
            
            lbl = ctk.CTkLabel(row, text=f"{video_name}", anchor="w", width=300)
            lbl.pack(side="left", padx=10)
            
            bar = ctk.CTkProgressBar(row)
            bar.set(0)
            bar.pack(side="right", fill="x", expand=True, padx=10)
            
            self.prog_bars[video_name] = bar
            
        self.log(f"Added {len(files)} files to active queue.")

    def log(self, msg):
        self.log_box.insert("end", f"{msg}\n")
        self.log_box.see("end")

    def check_messages(self):
        import numpy as np
        while not self.msg_queue.empty():
            msg = self.msg_queue.get_nowait()
            
            if msg['type'] == 'HIT':
                pil = Image.open(msg['path'])
                ctk_img = ctk.CTkImage(light_image=pil, size=(300, 200))
                self.lbl_hit.configure(image=ctk_img, text="")
                self.log(f"⚠ HIT: {os.path.basename(msg['path'])}")
            
            elif msg['type'] == 'PREVIEW':
                try:
                    nparr = np.frombuffer(msg['data'], np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ctk_img = ctk.CTkImage(light_image=Image.fromarray(rgb), size=(300, 200))
                    self.lbl_feed.configure(image=ctk_img, text="")
                except Exception: pass

            elif msg['type'] == 'PROGRESS':
                vid_name = msg.get('video')
                val = msg.get('val', 0)
                if vid_name and vid_name in self.prog_bars:
                    self.prog_bars[vid_name].set(val)

            elif msg['type'] == 'LOG':
                self.log(msg['msg'])

        self.after(50, self.check_messages)

    def start_processing(self):
        if self.is_running:
            self.log("Batch is already running. New files added to queue will be picked up automatically.")
            return

        self.stop_event.clear()
        self.is_running = True
        self.btn_start.configure(state="disabled", text="RUNNING (ADD MORE ANYTIME)", fg_color="#e74c3c")
        self.tabs.set("  LIVE DASHBOARD  ") 
        
        num_cores = 15 # Set this to your desired core count
        threading.Thread(target=self.run_dynamic_pool, args=(num_cores,), daemon=True).start()

    def stop_processing(self):
        if self.pool:
            self.log("!!! TERMINATING ALL PROCESSES !!!")
            self.stop_event.set()
            self.pool.terminate()
            self.pool.join()
            self.pool = None
            self.is_running = False
            self.btn_start.configure(state="normal", text="SCAN CANCELLED", fg_color="#c0392b")
            self.log("Batch stopped by user.")

    # --- DYNAMIC POOL CONSUMER ---
    def run_dynamic_pool(self, cores):
        def worker_wrapper(task_q, msg_q):
            # Each process grabs tasks forever until told to stop
            while True:
                try:
                    # Non-blocking check for stop signal would go here ideally
                    task = task_q.get(timeout=1) # Wait 1s for a task
                    video_worker(task, msg_q)
                except queue.Empty:
                    # If queue empty, wait a bit and try again (keep worker alive)
                    # For a cleaner exit in this specific wrapper we can just loop
                    continue
                except Exception as e:
                    # msg_q.put({'type': 'LOG', 'msg': f"Worker Error: {e}"})
                    pass

        # We can't use pool.map() for infinite dynamic queues easily.
        # Instead, we launch N processes that consume the shared task_queue
        self.pool = []
        for i in range(cores):
            p = multiprocessing.Process(target=worker_wrapper, args=(self.task_queue, self.msg_queue))
            p.daemon = True
            p.start()
            self.pool.append(p)
        
        # Monitor Loop (Main Thread)
        # This keeps the "Run" state active until you hit Stop
        while not self.stop_event.is_set():
            time.sleep(1)
        
        # Cleanup is handled by stop_processing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = ArchivistPro()
    app.mainloop()