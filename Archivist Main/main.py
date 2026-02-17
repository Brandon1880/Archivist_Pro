import customtkinter as ctk
import multiprocessing
import threading
import time
import os
import cv2
import numpy as np
from tkinter import filedialog
from PIL import Image

# MODULAR IMPORTS
import database as db
import helpers as hp
import engine
from filters import FILTERS

class ArchivistPro(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # System Init
        db.init_db()
        ctk.set_appearance_mode("dark")
        self.title("Archivist Pro v4.7.2 - MODULAR FIXED")
        self.geometry("1400x900")
        
        # Multiprocessing Setup
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
        self.chk_filters = {} 

        # Build GUI
        self._setup_layout()
        self._apply_saved_settings()
        
        # Start Listeners
        self.check_messages()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar, text="ARCHIVIST PRO", font=("Impact", 32)).pack(pady=20)
        
        self.subj_input = ctk.CTkEntry(self.sidebar, placeholder_text="Subject Name...")
        self.subj_input.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="+ Add Videos", command=self.add_videos, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="+ Add Folder", command=self.add_folder, fg_color="#34495e").pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(self.sidebar, text="Force Refresh UI", command=self.force_refresh, fg_color="#e67e22").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.sidebar, text="Clear Queue", command=self.clear_ui, fg_color="#c0392b").pack(fill="x", padx=20, pady=5)
        
        self.btn_stop = ctk.CTkButton(self.sidebar, text="⚠ FORCE STOP", command=self.stop_processing, fg_color="#c0392b", hover_color="#922b21")
        self.btn_stop.pack(side="bottom", fill="x", padx=20, pady=10)
        
        self.btn_start = ctk.CTkButton(self.sidebar, text="START QUEUE", command=self.start_processing, fg_color="#27ae60", height=60, font=("Arial", 16, "bold"))
        self.btn_start.pack(side="bottom", fill="x", padx=20, pady=10)

        # --- MAIN TABS ---
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.tab_scan = self.tabs.add(" DASHBOARD ")
        self.tab_settings = self.tabs.add(" SETTINGS ")
        
        self._build_dashboard(self.tab_scan)
        self._build_settings(self.tab_settings)

    def _build_dashboard(self, parent):
        # Worker Grid
        ctk.CTkLabel(parent, text="Live Worker Status").pack(anchor="w", padx=10)
        self.worker_grid_frame = ctk.CTkScrollableFrame(parent, height=350, fg_color="#1a1a1a")
        self.worker_grid_frame.pack(fill="x", padx=10, pady=5)
        
        # Latest Hit
        self.lbl_hit = ctk.CTkLabel(parent, text="[ Latest Hit Preview ]", height=150, fg_color="#2c3e50")
        self.lbl_hit.pack(fill="x", padx=10, pady=10)
        
        # Queue List
        ctk.CTkLabel(parent, text="Processing Queue").pack(anchor="w", padx=10)
        self.queue_area = ctk.CTkScrollableFrame(parent, height=200)
        self.queue_area.pack(fill="x", padx=10, pady=5)
        
        # Logs
        self.log_box = ctk.CTkTextbox(parent, height=100, font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=5)

    def _build_settings(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # --- LEFT COLUMN: PERFORMANCE ---
        left = ctk.CTkFrame(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(left, text="PERFORMANCE TUNING", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Sliders
        self.lbl_t1 = ctk.CTkLabel(left, text="First Pass Confidence: 0.40")
        self.lbl_t1.pack(anchor="w", padx=20)
        self.slider_t1 = ctk.CTkSlider(left, from_=0.1, to=0.9, command=lambda v: self.lbl_t1.configure(text=f"First Pass Confidence: {v:.2f}"))
        self.slider_t1.pack(fill="x", padx=20, pady=5)
        
        self.lbl_t2 = ctk.CTkLabel(left, text="Verification Confidence: 0.60")
        self.lbl_t2.pack(anchor="w", padx=20)
        self.slider_t2 = ctk.CTkSlider(left, from_=0.1, to=0.9, command=lambda v: self.lbl_t2.configure(text=f"Verification Confidence: {v:.2f}"))
        self.slider_t2.pack(fill="x", padx=20, pady=5)
        
        # Scan Logic
        ctk.CTkLabel(left, text="SCAN LOGIC", text_color="gray").pack(pady=(20,5))
        self.var_preset = ctk.StringVar(value="Balanced")
        ctk.CTkOptionMenu(left, values=["Max Speed", "Balanced", "High Precision"], variable=self.var_preset).pack(pady=5)
        
        self.var_focus = ctk.BooleanVar(value=False)
        self.var_motion = ctk.BooleanVar(value=False)
        self.var_deep = ctk.BooleanVar(value=False)
        
        ctk.CTkSwitch(left, text="Focus Mode (1 Video at a time)", variable=self.var_focus).pack(anchor="w", padx=20, pady=10)
        ctk.CTkSwitch(left, text="Motion Gating (Skip Static)", variable=self.var_motion).pack(anchor="w", padx=20, pady=10)
        ctk.CTkSwitch(left, text="Deep Scan (Every Frame - Slow)", variable=self.var_deep).pack(anchor="w", padx=20, pady=10)

        # --- RIGHT COLUMN: FILTERS & OUTPUT ---
        right = ctk.CTkFrame(parent)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Filter Scroll List
        ctk.CTkLabel(right, text="DETECTION TARGETS", font=("Arial", 14, "bold")).pack(pady=10)
        scroll_filt = ctk.CTkScrollableFrame(right, height=300)
        scroll_filt.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.chk_filters = {}
        for cat, labels in FILTERS.items():
            ctk.CTkLabel(scroll_filt, text=f"--- {cat} ---", text_color="gray").pack(anchor="w")
            for lbl in labels:
                clean_name = lbl.replace("_", " ").title().replace("Exposed", "").replace("Covered", "(C)")
                var = ctk.BooleanVar(value=True)
                ctk.CTkCheckBox(scroll_filt, text=clean_name, variable=var).pack(anchor="w", padx=10, pady=2)
                self.chk_filters[lbl] = var
        
        # Output Options (RESTORED SECTION)
        ctk.CTkLabel(right, text="OUTPUT & FEATURES", font=("Arial", 14, "bold")).pack(pady=10)
        
        self.var_verify = ctk.BooleanVar(value=True)
        self.var_ocr = ctk.BooleanVar(value=False)
        self.var_clips = ctk.BooleanVar(value=True)
        self.var_skip = ctk.BooleanVar(value=True)
        self.var_debug = ctk.BooleanVar(value=False)
        
        ctk.CTkSwitch(right, text="Double-Check Verification", variable=self.var_verify).pack(anchor="w", padx=20, pady=5)
        ctk.CTkSwitch(right, text="Scan Usernames (OCR)", variable=self.var_ocr).pack(anchor="w", padx=20, pady=5)
        ctk.CTkSwitch(right, text="Generate Video Clips", variable=self.var_clips).pack(anchor="w", padx=20, pady=5)
        ctk.CTkSwitch(right, text="Skip Already Scanned", variable=self.var_skip).pack(anchor="w", padx=20, pady=5)
        ctk.CTkSwitch(right, text="Show Debug Logs", variable=self.var_debug, progress_color="orange").pack(anchor="w", padx=20, pady=5)

    def _apply_saved_settings(self):
        s = hp.load_user_settings()
        if s:
            self.slider_t1.set(s.get('t1', 0.4)); self.lbl_t1.configure(text=f"First Pass Confidence: {s.get('t1', 0.4):.2f}")
            self.slider_t2.set(s.get('t2', 0.6)); self.lbl_t2.configure(text=f"Verification Confidence: {s.get('t2', 0.6):.2f}")
            self.var_preset.set(s.get('preset', 'Balanced'))
            self.var_focus.set(s.get('focus', False))
            self.var_motion.set(s.get('motion', False))
            self.var_deep.set(s.get('deep', False))
            self.var_skip.set(s.get('skip', True))
            self.var_verify.set(s.get('verify', True))
            self.var_ocr.set(s.get('ocr', False))
            self.var_clips.set(s.get('clips', True))
            self.var_debug.set(s.get('debug', False))
            
            saved_filters = s.get('filters', [])
            if saved_filters:
                for lbl, var in self.chk_filters.items():
                    var.set(lbl in saved_filters)

    def on_close(self):
        hp.save_user_settings({
            't1': self.slider_t1.get(), 't2': self.slider_t2.get(),
            'preset': self.var_preset.get(), 'focus': self.var_focus.get(),
            'motion': self.var_motion.get(), 'deep': self.var_deep.get(),
            'skip': self.var_skip.get(), 'verify': self.var_verify.get(),
            'ocr': self.var_ocr.get(), 'clips': self.var_clips.get(),
            'debug': self.var_debug.get(),
            'filters': [l for l, v in self.chk_filters.items() if v.get()]
        })
        self.stop_processing()
        self.destroy()

    # --- ACTION HANDLERS ---
    def add_videos(self):
        files = filedialog.askopenfilenames(parent=self, filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov")])
        if files: self.stack_files(files)

    def add_folder(self):
        folder = filedialog.askdirectory(parent=self)
        if folder:
            self.subj_input.delete(0, "end"); self.subj_input.insert(0, os.path.basename(folder))
            self.stack_files([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))])

    def stack_files(self, files):
        subj, sett = self.subj_input.get() or "Default", self.get_settings_dict()
        cnt = 0
        for f in files:
            if self.var_skip.get() and db.check_history(os.path.basename(f)): continue
            self.task_stack_local.append({"path": f, "subject": subj, "settings": sett})
            
            row = ctk.CTkFrame(self.queue_area); row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=os.path.basename(f), width=300, anchor="w").pack(side="left", padx=5)
            lbl = ctk.CTkLabel(row, text="Waiting...", width=150); lbl.pack(side="right")
            bar = ctk.CTkProgressBar(row); bar.set(0); bar.pack(side="right", fill="x", expand=True)
            self.prog_bars[os.path.basename(f)] = {'bar': bar, 'lbl': lbl}
            self.prog_data[os.path.basename(f)] = {'total': 1, 'done': 0}
            threading.Thread(target=self.init_file_info, args=(f,), daemon=True).start()
            cnt += 1
        self.log_box.insert("end", f"Queued {cnt} files. Click START to begin.\n")

    def init_file_info(self, path):
        try:
            cap = cv2.VideoCapture(hp.resolve_path(path))
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.prog_data[os.path.basename(path)]['total'] = total
                self.prog_bars[os.path.basename(path)]['lbl'].configure(text=f"0 / {total:,}")
            cap.release()
        except: pass

    def get_settings_dict(self):
        return {
            'threshold1': self.slider_t1.get(), 'threshold2': self.slider_t2.get(),
            'scan_preset': self.var_preset.get(), 'deep_scan': self.var_deep.get(),
            'double_check': self.var_verify.get(), 'scan_ocr': self.var_ocr.get(),
            'motion_gate': self.var_motion.get(), 'generate_clips': self.var_clips.get(),
            'active_filters': [l for l, v in self.chk_filters.items() if v.get()]
        }

    # --- LOOP ---
    def check_messages(self):
        try:
            while not self.msg_queue.empty():
                m = self.msg_queue.get_nowait()
                
                if m['type'] == 'GRID_PREVIEW' and m['idx'] in self.grid_slots:
                    try:
                        img = Image.fromarray(cv2.cvtColor(cv2.imdecode(np.frombuffer(m['data'], np.uint8), 1), cv2.COLOR_BGR2RGB))
                        self.grid_slots[m['idx']]['img'].configure(image=ctk.CTkImage(img, size=(120, 80)), text="")
                    except: pass
                    
                elif m['type'] == 'WORKER_START': 
                    self.grid_slots[m['idx']]['txt'].configure(text=m['file'][:12], text_color="green")
                    
                elif m['type'] == 'WORKER_STOP': 
                    self.grid_slots[m['idx']]['txt'].configure(text="Idle", text_color="gray")
                    if self.active_chunks > 0: self.active_chunks -= 1
                    
                elif m['type'] == 'TICK':
                    d = self.prog_data[m['video']]
                    d['done'] += m['count']
                    # Visual Update
                    pct = d['done'] / max(1, d['total'])
                    self.prog_bars[m['video']]['bar'].set(pct)
                    self.prog_bars[m['video']]['lbl'].configure(text=f"{d['done']:,} / {d['total']:,}")
                    
                elif m['type'] == 'HIT': 
                    try: self.lbl_hit.configure(image=ctk.CTkImage(Image.open(m['path']), size=(300, 200)), text="")
                    except: pass
                    
                elif m['type'] == 'LOG': 
                    if self.var_debug.get() or any(x in m['msg'] for x in ["Error", ">>>", "✔"]):
                        self.log_box.insert("end", f"{m['msg']}\n"); self.log_box.see("end")
                        
                elif m['type'] == 'DB_UPDATE': 
                    db.mark_scanned(m['file'])
        except: pass
        finally: self.after(20, self.check_messages)

    # --- PROCESS CONTROL ---
    def start_processing(self):
        if self.is_running: return
        self.is_running = True
        self.btn_start.configure(state="disabled", text="RUNNING...", fg_color="#e74c3c")
        
        num_cores = multiprocessing.cpu_count() - 2
        
        # Init Grid
        for widget in self.worker_grid_frame.winfo_children(): widget.destroy()
        cols = 8
        for i in range(cols): self.worker_grid_frame.grid_columnconfigure(i, weight=1)
        for i in range(num_cores):
            f = ctk.CTkFrame(self.worker_grid_frame)
            f.grid(row=i//cols, column=i%cols, padx=3, pady=3, sticky="ew")
            img = ctk.CTkLabel(f, text=f"W{i}", width=120, height=80, fg_color="black"); img.pack(pady=2)
            txt = ctk.CTkLabel(f, text="Idle", font=("Arial", 10)); txt.pack(pady=2)
            self.grid_slots[i] = {'img': img, 'txt': txt}

        if self.var_focus.get(): threading.Thread(target=self.run_focus, args=(num_cores,), daemon=True).start()
        else:
            for t in self.task_stack_local: self.task_queue.put(t)
            
        for i in range(num_cores):
            p = multiprocessing.Process(target=engine.worker_wrapper, args=(self.task_queue, self.msg_queue, i))
            p.daemon = True; p.start(); self.pool.append(p)

    def run_focus(self, cores):
        for t in self.task_stack_local:
            try:
                cap = cv2.VideoCapture(hp.resolve_path(t['path']))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
                self.active_chunks = cores
                chunk = total // cores
                
                self.msg_queue.put({'type': 'LOG', 'msg': f">>> FOCUS START: {os.path.basename(t['path'])}"})
                
                for i in range(cores): 
                    start = i * chunk
                    end = (i + 1) * chunk if i < cores - 1 else total
                    self.task_queue.put({**t, "start_frame": start, "end_frame": end})
                
                # Wait for all chunks to finish
                while self.active_chunks > 0 and self.is_running: 
                    time.sleep(1)
                
                if self.is_running:
                    self.msg_queue.put({'type': 'LOG', 'msg': f"✔ COMPLETED: {os.path.basename(t['path'])}"})
            except: pass

    def stop_processing(self):
        self.is_running = False
        self.stop_event.set()
        for p in self.pool: p.terminate()
        self.pool = []
        self.btn_start.configure(state="normal", text="SCAN CANCELLED", fg_color="#c0392b")
    
    def force_refresh(self): self.update()
    def clear_ui(self):
        for w in self.queue_area.winfo_children(): w.destroy()
        self.prog_bars = {}; self.task_stack_local = []

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ArchivistPro().mainloop()