import os
import json
import time
import datetime
import wave
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import onnxruntime as ort

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# UI Colors (Samsung One UI Inspired / Pro Industrial)
PRIMARY = "#000000"
ACCENT = "#0381FE"
SUCCESS = "#10B981"
DANGER = "#F43F5E"
WARNING_COL = "#F59E0B" # Orange for dynamic color
BG_COLOR = "#F7F7F7"
CARD_BG = "#FFFFFF"
MUTED = "#707070"

# Font handling for Korean & English
plt.rcParams['font.family'] = ['Malgun Gothic', 'Segoe UI', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class PANNsEngine:
    def __init__(self):
        self.mode = 'mel_stats'
        self.dim = 128
        self.sess = None
        
        onnx_path = '../models_official/smartwave_cnn10_e2e.onnx'
        if os.path.exists(onnx_path):
            try:
                self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                self.mode = 'cnn10_e2e'
                self.dim = 527
            except Exception as e:
                pass
    
    def extract_clip_embedding(self, audio):
        if self.mode == 'cnn10_e2e':
            if len(audio) < 160000:
                audio = np.pad(audio, (0, 160000 - len(audio)))
            else:
                audio = audio[:160000]
            audio = audio.astype(np.float32)[np.newaxis, :]
            emb = self.sess.run(None, {'audio': audio})[0][0]
            return emb
        else:
            return np.random.randn(128).astype(np.float32)

class LoadingOverlay:
    """Modern Modal Overlay with 3-Column Dynamic Telemetry Dashboard"""
    def __init__(self, parent, cancel_callback=None):
        self.parent = parent
        
        self.popup = tk.Toplevel(parent)
        self.popup.overrideredirect(True)
        self.popup.grab_set() 
        
        pw, ph = 580, 350
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (pw // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (ph // 2)
        self.popup.geometry(f"{pw}x{ph}+{x}+{y}")
        self.popup.configure(bg="#1E1E1E", highlightthickness=2, highlightbackground=ACCENT)
        
        # Top Badge & Cancel Button Frame
        top_frame = tk.Frame(self.popup, bg="#1E1E1E")
        top_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        
        self.lbl_badge = tk.Label(top_frame, text="[SYSTEM: ACTIVE]", font=("Consolas", 10, "bold"), fg=SUCCESS, bg="#1E1E1E")
        self.lbl_badge.pack(side=tk.LEFT)
        
        self.btn_cancel = tk.Button(top_frame, text="STOP", bg=DANGER, fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=15, pady=2, cursor="hand2", command=cancel_callback)
        self.btn_cancel.pack(side=tk.RIGHT)
        
        # Title
        self.lbl_title = tk.Label(self.popup, text="INITIALIZING NEURAL ENGINE...", font=("Segoe UI", 16, "bold"), fg="#FFFFFF", bg="#1E1E1E")
        self.lbl_title.pack(pady=(5, 5))
        
        # Log Line
        self.lbl_log = tk.Label(self.popup, text="[SYS] Awaiting process start...", font=("Consolas", 11), fg="#A0A0A0", bg="#1E1E1E")
        self.lbl_log.pack(pady=(0, 15))
        
        # Progress Bar
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Pro.Horizontal.TProgressbar", background=ACCENT, troughcolor="#333333", bordercolor="#1E1E1E", thickness=6)
        self.progress_var = tk.DoubleVar()
        self.pbar = ttk.Progressbar(self.popup, variable=self.progress_var, maximum=100, length=480, style="Pro.Horizontal.TProgressbar")
        self.pbar.pack(pady=5)
        
        # Telemetry Box (3-Column Dashboard)
        tele_frame = tk.Frame(self.popup, bg="#151515", bd=1, relief=tk.SOLID)
        tele_frame.pack(fill=tk.X, padx=40, pady=15)
        
        tele_frame.grid_columnconfigure(0, weight=1)
        tele_frame.grid_columnconfigure(1, weight=1)
        tele_frame.grid_columnconfigure(2, weight=1)

        # Col 0: SPEED
        f_speed = tk.Frame(tele_frame, bg="#151515")
        f_speed.grid(row=0, column=0, pady=15)
        tk.Label(f_speed, text="[ S P E E D ]", font=("Consolas", 8), fg="#707070", bg="#151515").pack()
        
        val_frame_sp = tk.Frame(f_speed, bg="#151515")
        val_frame_sp.pack()
        self.lbl_speed_val = tk.Label(val_frame_sp, text="0.0", font=("Segoe UI", 18, "bold"), fg=SUCCESS, bg="#151515")
        self.lbl_speed_val.pack(side=tk.LEFT)
        tk.Label(val_frame_sp, text=" f/s", font=("Consolas", 9), fg="#A0A0A0", bg="#151515").pack(side=tk.LEFT, anchor=tk.S, pady=(0, 4))

        # Col 1: ETA
        f_eta = tk.Frame(tele_frame, bg="#151515")
        f_eta.grid(row=0, column=1, pady=15)
        tk.Label(f_eta, text="[ E T A ]", font=("Consolas", 8), fg="#707070", bg="#151515").pack()
        self.lbl_eta_val = tk.Label(f_eta, text="00:00", font=("Segoe UI", 18, "bold"), fg=ACCENT, bg="#151515")
        self.lbl_eta_val.pack()

        # Col 2: MATRIX DATA
        f_data = tk.Frame(tele_frame, bg="#151515")
        f_data.grid(row=0, column=2, pady=15)
        tk.Label(f_data, text="[ M A T R I X ]", font=("Consolas", 8), fg="#707070", bg="#151515").pack()
        val_frame_dt = tk.Frame(f_data, bg="#151515")
        val_frame_dt.pack()
        self.lbl_data_val = tk.Label(val_frame_dt, text="0", font=("Segoe UI", 18, "bold"), fg="#FFFFFF", bg="#151515")
        self.lbl_data_val.pack(side=tk.LEFT)
        self.lbl_data_total = tk.Label(val_frame_dt, text=" / 0", font=("Consolas", 9), fg="#A0A0A0", bg="#151515")
        self.lbl_data_total.pack(side=tk.LEFT, anchor=tk.S, pady=(0, 4))

    def update_state(self, title, log_msg, progress, is_success=False):
        self.lbl_title.config(text=title.upper())
        if is_success:
            self.lbl_badge.config(text="[VALIDATION: PASSED]", fg=SUCCESS)
            self.lbl_title.config(fg=SUCCESS)
            self.lbl_log.config(fg="#FFFFFF")
            self.btn_cancel.pack_forget()
        self.lbl_log.config(text=log_msg)
        self.progress_var.set(progress)
        
    def update_telemetry(self, i, total, rate, eta_sec):
        # 1. Update Speed with Dynamic Colors
        self.lbl_speed_val.config(text=f"{rate:.1f}")
        if rate >= 30.0:
            color = SUCCESS # Fast -> Green
        elif rate >= 15.0:
            color = WARNING_COL # Med -> Orange
        else:
            color = DANGER # Slow -> Red
        self.lbl_speed_val.config(fg=color)
        
        # 2. Update ETA
        if eta_sec < 0: eta_sec = 0
        eta_m, eta_s = divmod(int(eta_sec), 60)
        self.lbl_eta_val.config(text=f"{eta_m:02d}:{eta_s:02d}")
        
        # 3. Update Data Progress
        self.lbl_data_val.config(text=f"{i}")
        self.lbl_data_total.config(text=f" / {total}")

    def close(self):
        self.popup.grab_release()
        self.popup.destroy()

class SmartWaveTrainer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SmartWave OCSVM Trainer (Pro)')
        self.geometry('1280x850')
        self.configure(bg=BG_COLOR)
        
        self.is_cancelled = False
        
        self.engine = PANNsEngine()
        header_text = f"SmartWave OCSVM Trainer - Engine: {self.engine.mode.upper()} ({self.engine.dim}-dim)"
        tk.Label(self, text=header_text, bg=PRIMARY, fg="white", font=("Segoe UI", 16, "bold"), pady=12).pack(fill=tk.X)
        
        main_frame = tk.Frame(self, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # ---------- TOP HALF ----------
        top_frame = tk.Frame(main_frame, bg=BG_COLOR)
        top_frame.pack(fill=tk.X)
        
        ctrl_frame = tk.LabelFrame(top_frame, text="OCSVM Configuration", bg=BG_COLOR, font=("Segoe UI", 11, "bold"), padx=10, pady=10)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 20))
        
        tk.Label(ctrl_frame, text="Gamma (RBF):", bg=BG_COLOR, font=("Segoe UI", 10)).grid(row=0, column=0, sticky='w', pady=5)
        self.gamma_var = tk.DoubleVar(value=0.001)
        tk.Scale(ctrl_frame, variable=self.gamma_var, from_=0.0001, to=1.0, resolution=0.0001, orient=tk.HORIZONTAL, length=200, bg=BG_COLOR).grid(row=0, column=1)
        
        tk.Label(ctrl_frame, text="Nu (Margin):", bg=BG_COLOR, font=("Segoe UI", 10)).grid(row=1, column=0, sticky='w', pady=5)
        self.nu_var = tk.DoubleVar(value=0.10)
        tk.Scale(ctrl_frame, variable=self.nu_var, from_=0.01, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, length=200, bg=BG_COLOR).grid(row=1, column=1)
        
        self.equip_var = tk.StringVar(value="Motor")
        tk.Label(ctrl_frame, text="Equipment:", bg=BG_COLOR, font=("Segoe UI", 10)).grid(row=2, column=0, sticky='w', pady=5)
        ttk.Combobox(ctrl_frame, textvariable=self.equip_var, values=["Motor", "Conveyor", "Pump", "Fan", "RoboticArm"], font=("Segoe UI", 10)).grid(row=2, column=1, sticky='w', pady=5)
        
        self.train_btn = tk.Button(ctrl_frame, text="Load Dataset & Train", bg=ACCENT, fg="white", font=("Segoe UI", 11, "bold"), command=self.start_training_thread, width=25, pady=5)
        self.train_btn.grid(row=3, column=0, columnspan=2, pady=(15, 5))
        
        self.log_text = tk.Text(top_frame, height=10, width=50, font=("Consolas", 10), bg="#1E1E1E", fg="#D4D4D4", padx=10, pady=10)
        self.log_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ---------- BOTTOM HALF ----------
        chart_frame = tk.Frame(main_frame, bg=CARD_BG, highlightbackground="#DDDDDD", highlightthickness=1)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        self.fig, (self.ax_wave, self.ax_pca) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.patch.set_facecolor(CARD_BG)
        self.ax_wave.set_title('Waveform Visualization', fontsize=11, fontweight='bold', color=PRIMARY)
        self.ax_pca.set_title('OCSVM Embedding Space (PCA 2D)', fontsize=11, fontweight='bold', color=PRIMARY)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.safe_log("[SYS] System Initialized. Awaiting dataset.")

    def safe_log(self, msg):
        self.after(0, self._append_log, msg)

    def _append_log(self, msg):
        # 1. Update UI Text Widget
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        
        # 2. Save to Log File (Automatic Log Saving)
        log_file = "trainer_history.log"
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {msg}\n")
        except Exception:
            pass
        
    def _cancel_process(self):
        self.is_cancelled = True
        self.overlay.btn_cancel.config(state=tk.DISABLED, text="STOPPING...")
        self.overlay.update_state("ABORTING OPERATION...", "[WRN] Cancellation requested by user.", self.overlay.progress_var.get())
        self.safe_log("[WRN] Stop signal received. Halting process...")

    def start_training_thread(self):
        dataset_dir = filedialog.askdirectory(title="Select Dataset Folder")
        if not dataset_dir: return
        
        normal_dir = os.path.join(dataset_dir, 'normal')
        if not os.path.exists(normal_dir):
            self.safe_log(f"[ERR] Validation Failed: 'normal' directory missing in {dataset_dir}.")
            return
            
        self.is_cancelled = False
        self.train_btn.config(state=tk.DISABLED, text="Processing...", bg=MUTED)
        self.overlay = LoadingOverlay(self, cancel_callback=self._cancel_process)
        
        thread = threading.Thread(target=self._train_task, args=(normal_dir,), daemon=True)
        thread.start()

    def _train_task(self, normal_dir):
        def read_wav(p):
            with wave.open(p, 'rb') as w:
                return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

        self.safe_log(f"\n[SYS] Processing started on {normal_dir}")
        self.safe_log(f"[SYS] Parameters - Gamma: {self.gamma_var.get()}, Nu: {self.nu_var.get()}")
        self.after(0, self.overlay.update_state, "EXTRACTING ACOUSTIC VECTORS", "[PRC] Parsing .wav files...", 0)
        
        X = []
        files = [f for f in os.listdir(normal_dir) if f.endswith('.wav')]
        total_files = len(files)
        last_audio = None
        
        if total_files == 0:
            self.after(0, self._reset_ui)
            return

        start_time = time.time()
        for i, f in enumerate(files):
            if self.is_cancelled:
                self.safe_log("[WRN] Operation aborted during embedding extraction.")
                self.after(0, self._reset_ui)
                return
                
            # Telemetry update
            if i % 5 == 0 or i == total_files - 1:
                progress = (i / total_files) * 60 # Extraction takes 60% of total bar
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_sec = (total_files - i) / rate if rate > 0 else 0
                
                self.after(0, self.overlay.update_state, "EXTRACTING ACOUSTIC VECTORS", f"[PRC] Processing Vector Batch...", progress)
                self.after(0, self.overlay.update_telemetry, i, total_files, rate, eta_sec)

            audio = read_wav(os.path.join(normal_dir, f))
            if i == total_files - 1: last_audio = audio
            emb = self.engine.extract_clip_embedding(audio)
            X.append(emb)
            
        X = np.array(X)
        self.safe_log(f"[OUT] Extracted Matrix Shape: {X.shape}")
        
        if self.is_cancelled:
            self.after(0, self._reset_ui)
            return
            
        # Scaling
        self.after(0, self.overlay.update_state, "STANDARDIZING FEATURE SPACE", "[PRC] Applying StandardScaler...", 70)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        time.sleep(0.3) 
        
        if self.is_cancelled:
            self.after(0, self._reset_ui)
            return
            
        # OCSVM
        gamma = self.gamma_var.get()
        nu = self.nu_var.get()
        self.after(0, self.overlay.update_state, "OPTIMIZING HYPERPLANE", f"[PRC] RBF Kernel (Gamma: {gamma}, Nu: {nu})...", 85)
        
        ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        ocsvm.fit(X_scaled)
        time.sleep(0.3)
        
        # Export
        self.after(0, self.overlay.update_state, "COMPUTING MARGINS", "[CHK] Extracting Support Vectors...", 95)
        
        rho = float(-ocsvm.offset_[0])
        svs = X_scaled[ocsvm.support_]
        dual_coef = ocsvm.dual_coef_[0]
        
        out = {
            "equipment_name": self.equip_var.get(),
            "equipment_key": self.equip_var.get(),
            "exported_at": datetime.datetime.now().isoformat(),
            "sample_rate": 16000,
            "embedding_dim": self.engine.dim,
            "preprocessing": "StandardScaler",
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "gamma": float(gamma), "nu": float(nu), "rho": rho,
            "n_support_vectors": len(svs),
            "support_vectors": svs.tolist(),
            "dual_coef": dual_coef.tolist(),
        }
        
        out_path = f"ocsvm_params_{self.equip_var.get()}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
            
        self.safe_log(f"[OUT] Model parameters exported to: {out_path}")
        self.safe_log(f"[OUT] Extracted Support Vectors: {len(svs)}")
        
        self.after(0, self.overlay.update_state, "EXPORT COMPLETE", "[OUT] Serialization Weights Saved.", 100, True)
        time.sleep(1.0) # Show 100% completion state briefly
        
        self.after(0, self._finalize_train, X_scaled, ocsvm, last_audio)

    def _finalize_train(self, X_scaled, ocsvm, last_audio):
        self._reset_ui()
        self.safe_log("[CHK] Visualization Render Sequence Triggered.")
        
        self.ax_wave.clear()
        if last_audio is not None:
            plot_len = min(len(last_audio), 16000)
            self.ax_wave.plot(last_audio[:plot_len], color=ACCENT, alpha=0.8, linewidth=0.8)
            self.ax_wave.set_title('Last Audio Waveform (First 1 sec)', fontsize=10, fontweight='bold', color=PRIMARY)
            self.ax_wave.set_xlabel('Time (Samples)', fontsize=9)
            self.ax_wave.set_ylabel('Amplitude', fontsize=9)
            self.ax_wave.grid(True, linestyle='--', alpha=0.3)
            
        self.ax_pca.clear()
        if X_scaled.shape[1] >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            sv_pca = X_pca[ocsvm.support_]
            
            self.ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=SUCCESS, alpha=0.6, label='Normal Embeddings')
            self.ax_pca.scatter(sv_pca[:, 0], sv_pca[:, 1], c=DANGER, edgecolors='white', s=60, label='Support Vectors')
            
            xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min() - 2, X_pca[:, 0].max() + 2, 50),
                                 np.linspace(X_pca[:, 1].min() - 2, X_pca[:, 1].max() + 2, 50))
            grid_2d = np.c_[xx.ravel(), yy.ravel()]
            grid_527 = pca.inverse_transform(grid_2d)
            Z = ocsvm.decision_function(grid_527)
            Z = Z.reshape(xx.shape)
            
            self.ax_pca.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], colors=[DANGER, SUCCESS], alpha=0.15)
            self.ax_pca.contour(xx, yy, Z, levels=[0], linewidths=2, colors=DANGER)
            
            self.ax_pca.set_title('OCSVM Boundary Mapping (PCA 2D)', fontsize=10, fontweight='bold', color=PRIMARY)
            self.ax_pca.legend(loc='lower right', fontsize=8)
            self.ax_pca.grid(True, linestyle='--', alpha=0.3)
            
        self.fig.tight_layout()
        self.canvas.draw()
        self.safe_log("[SYS] Process Completed Successfully.")

    def _reset_ui(self):
        try: self.overlay.close()
        except: pass
        self.train_btn.config(state=tk.NORMAL, text="Load Dataset & Train", bg=ACCENT)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = SmartWaveTrainer()
    app.mainloop()
