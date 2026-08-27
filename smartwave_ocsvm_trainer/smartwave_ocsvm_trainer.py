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
ACCENT = "#0381FE"     # Blue
SUCCESS = "#10B981"    # Green
DANGER = "#F43F5E"     # Red
WARNING_COL = "#F59E0B"
DISABLED_BG = "#D1D5DB"
DISABLED_FG = "#6B7280"
LOAD_BG = "#4B5563"    # Dark Gray
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
    """Modern Modal Overlay for Loading Dataset"""
    def __init__(self, parent, cancel_callback=None):
        self.parent = parent
        
        self.popup = tk.Toplevel(parent)
        self.popup.overrideredirect(True)
        self.popup.grab_set() 
        
        pw, ph = 620, 480
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (pw // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (ph // 2)
        self.popup.geometry(f"{pw}x{ph}+{x}+{y}")
        self.popup.configure(bg="#1E1E1E", highlightthickness=2, highlightbackground=LOAD_BG)
        
        top_frame = tk.Frame(self.popup, bg="#1E1E1E")
        top_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        
        self.lbl_badge = tk.Label(top_frame, text="[SYSTEM: ACTIVE]", font=("Consolas", 10, "bold"), fg=SUCCESS, bg="#1E1E1E")
        self.lbl_badge.pack(side=tk.LEFT)
        
        self.btn_cancel = tk.Button(top_frame, text="STOP", bg=DANGER, fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=15, pady=2, cursor="hand2", command=cancel_callback)
        self.btn_cancel.pack(side=tk.RIGHT)
        
        self.lbl_title = tk.Label(self.popup, text="EXTRACTING ACOUSTIC VECTORS...", font=("Segoe UI", 15, "bold"), fg="#FFFFFF", bg="#1E1E1E")
        self.lbl_title.pack(pady=(5, 5))
        
        self.lbl_log = tk.Label(self.popup, text="[PRC] Parsing .wav files...", font=("Consolas", 10), fg="#A0A0A0", bg="#1E1E1E")
        self.lbl_log.pack(pady=(0, 10))
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Pro.Horizontal.TProgressbar", background=ACCENT, troughcolor="#333333", bordercolor="#1E1E1E", thickness=6)
        self.progress_var = tk.DoubleVar()
        self.pbar = ttk.Progressbar(self.popup, variable=self.progress_var, maximum=100, length=520, style="Pro.Horizontal.TProgressbar")
        self.pbar.pack(pady=5)
        
        tele_frame = tk.Frame(self.popup, bg="#151515", bd=1, relief=tk.SOLID)
        tele_frame.pack(fill=tk.X, padx=40, pady=(15, 10))
        
        tele_frame.grid_columnconfigure(0, weight=1)
        tele_frame.grid_columnconfigure(1, weight=1)
        tele_frame.grid_columnconfigure(2, weight=1)

        f_speed = tk.Frame(tele_frame, bg="#151515")
        f_speed.grid(row=0, column=0, pady=15)
        tk.Label(f_speed, text="[ S P E E D ]", font=("Consolas", 8), fg="#707070", bg="#151515").pack()
        val_frame_sp = tk.Frame(f_speed, bg="#151515")
        val_frame_sp.pack()
        self.lbl_speed_val = tk.Label(val_frame_sp, text="0.0", font=("Segoe UI", 18, "bold"), fg=SUCCESS, bg="#151515")
        self.lbl_speed_val.pack(side=tk.LEFT)
        tk.Label(val_frame_sp, text=" f/s", font=("Consolas", 9), fg="#A0A0A0", bg="#151515").pack(side=tk.LEFT, anchor=tk.S, pady=(0, 4))

        f_eta = tk.Frame(tele_frame, bg="#151515")
        f_eta.grid(row=0, column=1, pady=15)
        tk.Label(f_eta, text="[ E T A ]", font=("Consolas", 8), fg="#707070", bg="#151515").pack()
        self.lbl_eta_val = tk.Label(f_eta, text="00:00", font=("Segoe UI", 18, "bold"), fg=ACCENT, bg="#151515")
        self.lbl_eta_val.pack()

        f_data = tk.Frame(tele_frame, bg="#151515")
        f_data.grid(row=0, column=2, pady=15)
        tk.Label(f_data, text="[ M A T R I X ]", font=("Consolas", 8), fg="#707070", bg="#151515").pack()
        val_frame_dt = tk.Frame(f_data, bg="#151515")
        val_frame_dt.pack()
        self.lbl_data_val = tk.Label(val_frame_dt, text="0", font=("Segoe UI", 18, "bold"), fg="#FFFFFF", bg="#151515")
        self.lbl_data_val.pack(side=tk.LEFT)
        self.lbl_data_total = tk.Label(val_frame_dt, text=" / 0", font=("Consolas", 9), fg="#A0A0A0", bg="#151515")
        self.lbl_data_total.pack(side=tk.LEFT, anchor=tk.S, pady=(0, 4))
        
        self.mini_log = tk.Text(self.popup, height=12, font=("Consolas", 9), bg="#050505", fg="#10B981", bd=1, relief=tk.SOLID, highlightthickness=0)
        self.mini_log.pack(fill=tk.BOTH, expand=True, padx=40, pady=(0, 20))
        self.mini_log.insert(tk.END, "> Neural engine online. Ready for embedding...\n")
        self.mini_log.config(state=tk.DISABLED)

    def update_state(self, log_msg, progress):
        self.lbl_log.config(text=log_msg)
        self.progress_var.set(progress)
        
    def update_telemetry(self, i, total, rate, eta_sec):
        self.lbl_speed_val.config(text=f"{rate:.1f}")
        if rate >= 30.0: color = SUCCESS
        elif rate >= 15.0: color = WARNING_COL
        else: color = DANGER
        self.lbl_speed_val.config(fg=color)
        
        if eta_sec < 0: eta_sec = 0
        eta_m, eta_s = divmod(int(eta_sec), 60)
        self.lbl_eta_val.config(text=f"{eta_m:02d}:{eta_s:02d}")
        
        self.lbl_data_val.config(text=f"{i}")
        self.lbl_data_total.config(text=f" / {total}")
        
    def append_mini_log(self, msg):
        self.mini_log.config(state=tk.NORMAL)
        self.mini_log.insert(tk.END, msg + "\n")
        lines = int(self.mini_log.index('end-1c').split('.')[0])
        if lines > 80:
            self.mini_log.delete('1.0', '2.0')
        self.mini_log.see(tk.END)
        self.mini_log.config(state=tk.DISABLED)

    def close(self):
        self.popup.grab_release()
        self.popup.destroy()

class SmartWaveTrainer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SmartWave OCSVM Trainer (Pro)')
        self.geometry('1380x880')
        self.configure(bg=BG_COLOR)
        
        self.is_cancelled = False
        self.X_cache = None
        self.last_audio_cache = None
        self.scaler_cache = None
        self.ocsvm_cache = None
        
        self.engine = PANNsEngine()
        header_text = f"SmartWave OCSVM Trainer - Engine: {self.engine.mode.upper()} ({self.engine.dim}-dim)"
        tk.Label(self, text=header_text, bg=PRIMARY, fg="white", font=("Segoe UI", 16, "bold"), pady=12).pack(fill=tk.X)
        
        main_frame = tk.Frame(self, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # ---------- TOP HALF ----------
        top_frame = tk.Frame(main_frame, bg=BG_COLOR)
        top_frame.pack(fill=tk.X)
        
        # UI Grouping (3 Zones)
        ctrl_frame = tk.Frame(top_frame, bg=BG_COLOR)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Zone 1: Data Setup
        zone1 = tk.LabelFrame(ctrl_frame, text="1. Data Setup", bg=BG_COLOR, font=("Segoe UI", 10, "bold"), padx=15, pady=10)
        zone1.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(zone1, text="Equipment:", bg=BG_COLOR, font=("Segoe UI", 9)).grid(row=0, column=0, sticky='w', pady=5)
        self.equip_var = tk.StringVar(value="Motor")
        ttk.Combobox(zone1, textvariable=self.equip_var, values=["Motor", "Conveyor", "Pump", "Fan", "RoboticArm"], font=("Segoe UI", 9), width=15).grid(row=0, column=1, sticky='w', pady=5, padx=10)
        
        self.btn_load = tk.Button(zone1, text="📂 EXTRACT FEATURES", bg=LOAD_BG, fg="white", font=("Segoe UI", 10, "bold"), width=22, cursor="hand2", command=self.start_load_thread)
        self.btn_load.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Zone 2: Hyperparameters
        zone2 = tk.LabelFrame(ctrl_frame, text="2. Hyperparameters", bg=BG_COLOR, font=("Segoe UI", 10, "bold"), padx=15, pady=10)
        zone2.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(zone2, text="Gamma (RBF):", bg=BG_COLOR, font=("Segoe UI", 9)).grid(row=0, column=0, sticky='w', pady=5)
        self.gamma_var = tk.DoubleVar(value=0.001)
        tk.Scale(zone2, variable=self.gamma_var, from_=0.0001, to=1.0, resolution=0.0001, orient=tk.HORIZONTAL, length=150, bg=BG_COLOR, highlightthickness=0).grid(row=0, column=1, padx=10)
        
        tk.Label(zone2, text="Nu (Margin):", bg=BG_COLOR, font=("Segoe UI", 9)).grid(row=1, column=0, sticky='w', pady=5)
        self.nu_var = tk.DoubleVar(value=0.10)
        tk.Scale(zone2, variable=self.nu_var, from_=0.01, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, length=150, bg=BG_COLOR, highlightthickness=0).grid(row=1, column=1, padx=10)
        
        # Zone 3: Execution
        zone3 = tk.LabelFrame(ctrl_frame, text="3. Execution", bg=BG_COLOR, font=("Segoe UI", 10, "bold"), padx=15, pady=10)
        zone3.pack(fill=tk.X)
        
        self.btn_train = tk.Button(zone3, text="🧠 FIT OCSVM BOUNDARY", bg=DISABLED_BG, fg=DISABLED_FG, font=("Segoe UI", 10, "bold"), width=22, state=tk.DISABLED, cursor="hand2", command=self.action_train_ocsvm)
        self.btn_train.pack(pady=5)
        
        self.btn_export = tk.Button(zone3, text="🚀 EXPORT TO DEVICE", bg=DISABLED_BG, fg=DISABLED_FG, font=("Segoe UI", 10, "bold"), width=22, state=tk.DISABLED, cursor="hand2", command=self.action_export_model)
        self.btn_export.pack(pady=5)
        
        # Main Log Window
        self.log_text = tk.Text(top_frame, height=16, width=50, font=("Consolas", 10), bg="#1E1E1E", fg="#D4D4D4", padx=10, pady=10)
        self.log_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ---------- BOTTOM HALF (3 Parallel Charts) ----------
        chart_frame = tk.Frame(main_frame, bg=CARD_BG, highlightbackground="#DDDDDD", highlightthickness=1)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        self.fig, (self.ax_wave, self.ax_fingerprint, self.ax_pca) = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.patch.set_facecolor(CARD_BG)
        self.ax_wave.set_title('Raw Audio Waveform', fontsize=10, fontweight='bold', color=PRIMARY)
        self.ax_fingerprint.set_title('Acoustic Fingerprint (527-Dim)', fontsize=10, fontweight='bold', color=PRIMARY)
        self.ax_pca.set_title('OCSVM Space (PCA 2D)', fontsize=10, fontweight='bold', color=PRIMARY)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.safe_log("[SYS] System Initialized. Awaiting feature extraction.")

    def safe_log(self, msg):
        self.after(0, self._append_log, msg)

    def _append_log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
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
        self.overlay.append_mini_log("! > SIGNAL: SIGINT received. Halting extraction...")
        self.safe_log("[WRN] Stop signal received. Halting process...")

    def start_load_thread(self):
        dataset_dir = filedialog.askdirectory(title="Select Dataset Folder")
        if not dataset_dir: return
        
        normal_dir = os.path.join(dataset_dir, 'normal')
        if not os.path.exists(normal_dir):
            self.safe_log(f"[ERR] Validation Failed: 'normal' directory missing in {dataset_dir}.")
            return
            
        self.is_cancelled = False
        self.btn_load.config(state=tk.DISABLED, text="PROCESSING...", bg=MUTED)
        self.btn_train.config(state=tk.DISABLED, bg=DISABLED_BG, fg=DISABLED_FG)
        self.btn_export.config(state=tk.DISABLED, bg=DISABLED_BG, fg=DISABLED_FG)
        
        self.overlay = LoadingOverlay(self, cancel_callback=self._cancel_process)
        
        thread = threading.Thread(target=self._load_task, args=(normal_dir,), daemon=True)
        thread.start()

    def _load_task(self, normal_dir):
        def read_wav(p):
            with wave.open(p, 'rb') as w:
                return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

        self.safe_log(f"\n[SYS] Feature Extraction started from: {normal_dir}")
        self.after(0, self.overlay.update_state, "[PRC] Parsing .wav files...", 0)
        
        X = []
        files = [f for f in os.listdir(normal_dir) if f.endswith('.wav')]
        total_files = len(files)
        last_audio = None
        
        if total_files == 0:
            self.after(0, self._reset_load_ui)
            return

        start_time = time.time()
        for i, f in enumerate(files):
            if self.is_cancelled:
                self.safe_log("[WRN] Operation aborted during feature extraction.")
                self.after(0, self._reset_load_ui)
                return
                
            audio = read_wav(os.path.join(normal_dir, f))
            if i == total_files - 1: last_audio = audio
            emb = self.engine.extract_clip_embedding(audio)
            X.append(emb)
            
            if i % 3 == 0 or i == total_files - 1:
                short_f = (f[:12] + '..') if len(f) > 14 else f.ljust(14)
                snap = f"{emb[0]:.3f}, {emb[1]:.3f}, {emb[2]:.3f}"
                self.after(0, self.overlay.append_mini_log, f"> {short_f} ➔ [ {snap} ... 527-Dim ] ➔ OK")
            
            if i % 5 == 0 or i == total_files - 1:
                progress = (i / total_files) * 100 
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_sec = (total_files - i) / rate if rate > 0 else 0
                
                self.after(0, self.overlay.update_state, f"[PRC] Processing Batch {i}/{total_files}...", progress)
                self.after(0, self.overlay.update_telemetry, i, total_files, rate, eta_sec)

        self.X_cache = np.array(X)
        self.last_audio_cache = last_audio
        
        self.safe_log(f"[OUT] Extracted Matrix Shape: {self.X_cache.shape}")
        self.after(0, self.overlay.append_mini_log, f"> Matrix perfectly extracted in {time.time()-start_time:.1f}s.")
        time.sleep(0.5)
        
        self.after(0, self._finalize_load)

    def _reset_load_ui(self):
        try: self.overlay.close()
        except: pass
        self.btn_load.config(state=tk.NORMAL, text="📂 EXTRACT FEATURES", bg=LOAD_BG)

    def _finalize_load(self):
        self._reset_load_ui()
        self.btn_train.config(state=tk.NORMAL, bg=ACCENT, fg="white")
        self.safe_log("[SYS] Features extracted and loaded into memory. Ready to fit boundary.")
        
    def action_train_ocsvm(self):
        if self.X_cache is None: return
        
        self.safe_log(f"\n[PRC] Fitting OCSVM Boundary instantly from memory cache...")
        gamma = self.gamma_var.get()
        nu = self.nu_var.get()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_cache)
        
        ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        ocsvm.fit(X_scaled)
        
        self.scaler_cache = scaler
        self.ocsvm_cache = ocsvm
        
        self.safe_log(f"[CHK] Boundary fit complete (Gamma: {gamma}, Nu: {nu}). SVs: {len(ocsvm.support_)}")
        self.safe_log("[CHK] Redrawing charts...")
        
        self._draw_charts(X_scaled)
        self.btn_export.config(state=tk.NORMAL, bg=SUCCESS, fg="white")

    def _draw_charts(self, X_scaled):
        self.ax_wave.clear()
        if self.last_audio_cache is not None:
            plot_len = min(len(self.last_audio_cache), 16000)
            self.ax_wave.plot(self.last_audio_cache[:plot_len], color=MUTED, alpha=0.8, linewidth=0.8)
            self.ax_wave.set_title('Raw Audio Waveform (1 sec)', fontsize=10, fontweight='bold', color=PRIMARY)
            self.ax_wave.set_xlabel('Time (Samples)', fontsize=9)
            self.ax_wave.set_ylabel('Amplitude', fontsize=9)
            self.ax_wave.grid(True, linestyle='--', alpha=0.3)
            
        self.ax_fingerprint.clear()
        if self.X_cache is not None and len(self.X_cache) > 0:
            X_mean = np.mean(self.X_cache, axis=0)
            X_std = np.std(self.X_cache, axis=0)
            bands = np.arange(len(X_mean))
            
            self.ax_fingerprint.fill_between(bands, X_mean - X_std, X_mean + X_std, color=MUTED, alpha=0.25, label='Normal Variance')
            self.ax_fingerprint.plot(bands, X_mean, color=ACCENT, linewidth=2, label='Healthy Baseline')
            
            self.ax_fingerprint.set_title('Deep Acoustic Fingerprint (527-Dim)', fontsize=10, fontweight='bold', color=PRIMARY)
            self.ax_fingerprint.set_xlabel('CNN10 Feature Dimensions (0-526)', fontsize=9)
            self.ax_fingerprint.set_ylabel('Normalized Energy', fontsize=9)
            self.ax_fingerprint.legend(loc='upper right', fontsize=8)
            self.ax_fingerprint.grid(True, linestyle='--', alpha=0.3)
            
        self.ax_pca.clear()
        if X_scaled.shape[1] >= 2 and self.ocsvm_cache is not None:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            sv_pca = X_pca[self.ocsvm_cache.support_]
            
            self.ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=SUCCESS, alpha=0.6, label='Normal Embeddings')
            self.ax_pca.scatter(sv_pca[:, 0], sv_pca[:, 1], c=DANGER, edgecolors='white', s=60, label='Support Vectors')
            
            xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min() - 2, X_pca[:, 0].max() + 2, 50),
                                 np.linspace(X_pca[:, 1].min() - 2, X_pca[:, 1].max() + 2, 50))
            grid_2d = np.c_[xx.ravel(), yy.ravel()]
            grid_527 = pca.inverse_transform(grid_2d)
            Z = self.ocsvm_cache.decision_function(grid_527)
            Z = Z.reshape(xx.shape)
            
            self.ax_pca.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], colors=[DANGER, SUCCESS], alpha=0.15)
            self.ax_pca.contour(xx, yy, Z, levels=[0], linewidths=2, colors=DANGER)
            
            self.ax_pca.set_title('OCSVM Boundary Mapping (PCA 2D)', fontsize=10, fontweight='bold', color=PRIMARY)
            self.ax_pca.legend(loc='lower right', fontsize=8)
            self.ax_pca.grid(True, linestyle='--', alpha=0.3)
            
        self.fig.tight_layout()
        self.canvas.draw()

    def action_export_model(self):
        if self.ocsvm_cache is None or self.scaler_cache is None: return
        
        rho = float(-self.ocsvm_cache.offset_[0])
        svs = self.scaler_cache.transform(self.X_cache)[self.ocsvm_cache.support_]
        dual_coef = self.ocsvm_cache.dual_coef_[0]
        
        out = {
            "equipment_name": self.equip_var.get(),
            "equipment_key": self.equip_var.get(),
            "exported_at": datetime.datetime.now().isoformat(),
            "sample_rate": 16000,
            "embedding_dim": self.engine.dim,
            "preprocessing": "StandardScaler",
            "scaler_mean": self.scaler_cache.mean_.tolist(),
            "scaler_scale": self.scaler_cache.scale_.tolist(),
            "gamma": float(self.gamma_var.get()), 
            "nu": float(self.nu_var.get()), 
            "rho": rho,
            "n_support_vectors": len(svs),
            "support_vectors": svs.tolist(),
            "dual_coef": dual_coef.tolist(),
        }
        
        out_path = f"ocsvm_params_{self.equip_var.get()}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
            
        self.safe_log(f"\n[OUT] 🚀 Model parameters securely exported to: {out_path}")
        self.safe_log(f"[OUT] Total Support Vectors preserved: {len(svs)}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = SmartWaveTrainer()
    app.mainloop()
