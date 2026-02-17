import tkinter as tk

from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2

import os

import threading

from datetime import datetime



# --- CONFIGURATION ---

DEFAULT_THRESHOLD = 0.40  

VERIFY_THRESHOLD = 0.55   

OUTPUT_ROOT = "Scans"



ALL_LABELS = {

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



# --- SAFE IMPORT BLOCK ---

try:

    from nudenet import NudeDetector

    AI_AVAILABLE = True

except ImportError:

    try:

        from nudenet.nudenet import NudeDetector

        AI_AVAILABLE = True

    except ImportError:

        AI_AVAILABLE = False



class NudeScannerApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Archivist Video Scanner")

        self.root.geometry("1100x900")



        self.video_queue = []

        self.stop_event = False

        self.detector = None

        self.chk_vars = {} 

        self.deep_scan_var = tk.BooleanVar(value=False)

        self.double_check_var = tk.BooleanVar(value=True)

        self.debug_mode_var = tk.BooleanVar(value=False)



        # --- HEADER ---

        tk.Label(root, text="Archivist Video Scanner", font=("Segoe UI", 16, "bold")).pack(pady=5)

        

        # --- MAIN CONTAINER ---

        main_frame = tk.Frame(root)

        main_frame.pack(fill="both", expand=True, padx=20)



        left_col = tk.Frame(main_frame)

        left_col.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 10))



        right_col = tk.LabelFrame(main_frame, text="Target Filters", padx=10, pady=10)

        right_col.pack(side=tk.RIGHT, fill="both", padx=(10, 0))



        # --- 1. SUBJECT INFO (NEW) ---

        subj_frame = tk.LabelFrame(left_col, text="1. Subject Information", padx=10, pady=10)

        subj_frame.pack(fill="x", pady=5)

        

        tk.Label(subj_frame, text="Subject Name:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)

        self.ent_subject = tk.Entry(subj_frame, font=("Arial", 10), width=30)

        self.ent_subject.pack(side=tk.LEFT, padx=10)

        self.ent_subject.insert(0, "Unknown_Subject") # Default

        tk.Label(subj_frame, text="(Creates a main folder for this person)", font=("Arial", 8), fg="gray").pack(side=tk.LEFT)



        # --- 2. QUEUE ---

        q_frame = tk.LabelFrame(left_col, text="2. Video Queue", padx=10, pady=10)

        q_frame.pack(fill="both", expand=True, pady=5)

        

        btn_box = tk.Frame(q_frame)

        btn_box.pack(fill="x", pady=5)

        tk.Button(btn_box, text="Add Videos...", command=self.add_files, bg="#e1bee7", width=15).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_box, text="Clear Queue", command=self.clear_queue, bg="#ddd", width=15).pack(side=tk.LEFT, padx=5)

        

        self.list_queue = tk.Listbox(q_frame, height=5, bg="#f3e5f5")

        self.list_queue.pack(fill="both", expand=True, padx=5, pady=5)



        # --- 3. SETTINGS ---

        sett_frame = tk.LabelFrame(left_col, text="3. Settings", padx=10, pady=5)

        sett_frame.pack(fill="x", pady=5)



        self.lbl_thresh = tk.Label(sett_frame, text=f"Threshold: {int(DEFAULT_THRESHOLD*100)}%")

        self.lbl_thresh.pack(anchor="w")

        self.scale_sens = ttk.Scale(sett_frame, from_=0.10, to=0.90, value=DEFAULT_THRESHOLD, command=self.update_thresh)

        self.scale_sens.pack(fill="x")

        

        tk.Checkbutton(sett_frame, text="Secondary Verification (Zoom & Check)", variable=self.double_check_var, fg="green").pack(anchor="w")

        tk.Checkbutton(sett_frame, text="Deep Scan (Check Every Frame)", variable=self.deep_scan_var, fg="blue").pack(anchor="w")

        tk.Checkbutton(sett_frame, text="Debug Log", variable=self.debug_mode_var, fg="red").pack(anchor="w")



        # --- LOG ---

        self.lbl_prog = tk.Label(left_col, text="Ready", font=("Arial", 10, "bold"))

        self.lbl_prog.pack(pady=(10, 0))

        self.prog_bar = ttk.Progressbar(left_col, orient="horizontal", mode="determinate")

        self.prog_bar.pack(fill="x", pady=5)

        

        self.log_text = scrolledtext.ScrolledText(left_col, height=10)

        self.log_text.pack(fill="both", expand=True)



        self.create_checkboxes(right_col)



        self.btn_start = tk.Button(root, text="START ARCHIVING", command=self.start_batch, bg="#4CAF50", fg="white", font=("Arial", 14, "bold"), height=2)

        self.btn_start.pack(fill="x", padx=20, pady=10)



        if AI_AVAILABLE:

            threading.Thread(target=self.load_model, daemon=True).start()

        else:

            self.log("CRITICAL: NudeNet library missing.")



    def create_checkboxes(self, parent):

        for cat, labels in ALL_LABELS.items():

            tk.Label(parent, text=cat, font=("Arial", 9, "bold")).pack(anchor="w", pady=(5,0))

            for lbl in labels:

                var = tk.BooleanVar(value=True)

                display_name = lbl.replace("_", " ").title().replace("Exposed", "").replace("Covered", "(Cov)")

                self.chk_vars[lbl] = var

                tk.Checkbutton(parent, text=display_name, variable=var).pack(anchor="w")



    def update_thresh(self, val):

        self.lbl_thresh.config(text=f"Threshold: {int(float(val)*100)}%")



    def load_model(self):

        try:

            self.log("Loading AI Model...")

            self.detector = NudeDetector()

            self.log("Model Loaded.")

        except Exception as e:

            self.log(f"Error loading model: {e}")



    def log(self, msg):

        def _l():

            self.log_text.insert(tk.END, msg + "\n")

            self.log_text.see(tk.END)

        self.root.after(0, _l)



    def add_files(self):

        files = filedialog.askopenfilenames(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])

        for f in files:

            if f not in self.video_queue:

                self.video_queue.append(f)

                self.list_queue.insert(tk.END, f"[WAITING] {os.path.basename(f)}")



    def clear_queue(self):

        self.video_queue = []

        self.list_queue.delete(0, tk.END)



    def start_batch(self):

        if not self.video_queue:

            messagebox.showwarning("Empty Queue", "Add videos first.")

            return

        if not self.detector: return



        self.stop_event = False

        self.btn_start.config(state=tk.DISABLED, text="PROCESSING...", bg="#f44336")

        self.log_text.delete(1.0, tk.END)

        threading.Thread(target=self.process_queue, daemon=True).start()



    def process_queue(self):

        total_videos = len(self.video_queue)

        

        # Get Subject Name for folder creation

        subject_name = self.ent_subject.get().strip()

        if not subject_name: subject_name = "Unknown_Subject"

        

        # Sanitize folder name

        subject_name = "".join([c for c in subject_name if c.isalpha() or c.isdigit() or c==' ' or c=='_']).rstrip()



        for idx, video_path in enumerate(self.video_queue):

            if self.stop_event: break

            

            self.root.after(0, lambda i=idx: self.list_queue.itemconfig(i, {'bg':'#fff9c4'}))

            self.log(f"\n--- Processing {idx+1}/{total_videos}: {os.path.basename(video_path)} ---")

            

            self.scan_single_video(video_path, subject_name)

            

            self.root.after(0, lambda i=idx: self.list_queue.itemconfig(i, {'bg':'#c8e6c9', 'fg':'#1b5e20'}))



        self.log("\nBatch Complete.")

        self.root.after(0, lambda: self.btn_start.config(state=tk.NORMAL, text="START ARCHIVING", bg="#4CAF50"))



    def verify_detection(self, frame_rgb, box):

        """Crops and re-checks the area."""

        try:

            h, w = frame_rgb.shape[:2]

            x, y, bw, bh = box

            pad_x = int(bw * 0.2)

            pad_y = int(bh * 0.2)

            x1 = max(0, x - pad_x)

            y1 = max(0, y - pad_y)

            x2 = min(w, x + bw + pad_x)

            y2 = min(h, y + bh + pad_y)

            crop = frame_rgb[y1:y2, x1:x2]

            

            if crop.shape[0] < 10 or crop.shape[1] < 10: return True



            crop_detections = self.detector.detect(crop)

            for d in crop_detections:

                if d['score'] > VERIFY_THRESHOLD:

                    return True

            return False 

        except:

            return True



    def scan_single_video(self, video_path, subject_name):

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        

        # --- NEW FOLDER STRUCTURE ---

        # Scans / [SubjectName] / [VideoName] / [Severity]

        base_output_dir = os.path.join(OUTPUT_ROOT, subject_name, video_name)



        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)

        

        skip = 1 if self.deep_scan_var.get() else max(1, int(fps / 2))

        

        frame_count = 0

        found_count = 0

        thresh = self.scale_sens.get()

        do_double_check = self.double_check_var.get()



        while True:

            if self.stop_event: break

            success, frame = cap.read()

            if not success: break



            if frame_count % skip == 0:

                pct = (frame_count / total_frames) * 100

                self.root.after(0, lambda p=pct: self.prog_bar.config(value=p))

                self.root.after(0, lambda p=pct: self.lbl_prog.config(text=f"Scanning... {int(p)}%"))



                try:

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    detections = self.detector.detect(rgb)

                    

                    annotated = frame.copy()

                    hit = False

                    max_sev = "MINOR"

                    

                    # Store the labels found in this frame to name the file

                    frame_labels = []



                    for det in detections:

                        lbl = det.get('class', det.get('label', 'UNKNOWN'))

                        score = det['score']

                        box = det['box']



                        if score < thresh: continue

                        if not self.chk_vars.get(lbl, tk.BooleanVar(value=False)).get(): continue



                        if do_double_check:

                            if not self.verify_detection(rgb, box):

                                if self.debug_mode_var.get(): print(f"Rejected: {lbl}")

                                continue



                        hit = True

                        frame_labels.append(lbl)

                        

                        x, y, w, h = box

                        if lbl in ALL_LABELS["CRITICAL"]: 

                            color = (0, 0, 255)

                            max_sev = "CRITICAL"

                        elif lbl in ALL_LABELS["WARNING"]: 

                            color = (0, 255, 255)

                            if max_sev == "MINOR": max_sev = "WARNING"

                        else: 

                            color = (0, 255, 0)



                        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

                        cv2.putText(annotated, f"{lbl} {int(score*100)}%", (x, y-10), 0, 0.6, color, 2)



                    if hit:

                        found_count += 1

                        

                        # --- TIMESTAMP CALCULATION ---

                        sec = frame_count / fps

                        h_ts = int(sec // 3600)

                        m_ts = int((sec % 3600) // 60)

                        s_ts = int(sec % 60)

                        

                        # Format: HH-MM-SS

                        time_str = f"{h_ts:02d}-{m_ts:02d}-{s_ts:02d}"

                        

                        # --- FILENAME GENERATION ---

                        # Pick the "worst" thing found to name the file

                        primary_alert = frame_labels[0]

                        for l in frame_labels:

                            if l in ALL_LABELS["CRITICAL"]:

                                primary_alert = l

                                break

                        

                        # Clean label for filename (e.g. "FEMALE_BREAST_EXPOSED" -> "Female_Breast_Exposed")

                        clean_alert = primary_alert.replace("_", " ").title().replace(" ", "_")

                        

                        # Final Name: AlertType_HH-MM-SS_Frame-XXXX.jpg

                        filename = f"{clean_alert}_{time_str}_Frame-{frame_count}.jpg"

                        

                        # Create Folder Just-In-Time

                        final_path = os.path.join(base_output_dir, max_sev)

                        if not os.path.exists(final_path): os.makedirs(final_path)

                        

                        cv2.imwrite(os.path.join(final_path, filename), annotated)

                        

                        self.log(f"*** SAVED: {filename}")



                except Exception as e:

                    print(e)



            frame_count += 1



        cap.release()

        self.log(f"Finished. Found {found_count} items.")



if __name__ == "__main__":

    root = tk.Tk()

    app = NudeScannerApp(root)

    root.mainloop()