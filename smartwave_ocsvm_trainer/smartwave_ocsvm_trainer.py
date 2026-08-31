import os
import json
import time
import datetime
import wave
import threading
import traceback
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import onnxruntime as ort

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Constants
TARGET_AUDIO_LEN = 160000
SAMPLE_RATE = 16000
EMBEDDING_DIM = 512

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
    """Extracts 527-dimensional acoustic embeddings using PANNs CNN10 ONNX model."""
    def __init__(self):
        self.mode = 'cnn10_e2e'
        self.dim = EMBEDDING_DIM
        self.sess = None
        
        onnx_path = '../models_official/smartwave_cnn10_e2e_512.onnx'
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")
            
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    def extract_clip_embedding(self, audio):
        """Extracts embeddings from raw audio. Pads or truncates to TARGET_AUDIO_LEN."""
        if len(audio) < TARGET_AUDIO_LEN:
            audio = np.pad(audio, (0, TARGET_AUDIO_LEN - len(audio)))
        else:
            audio = audio[:TARGET_AUDIO_LEN]
        audio = audio.astype(np.float32)[np.newaxis, :]
        return self.sess.run(None, {'audio': audio})[0][0]

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

class TrainOverlay:
    """Mini overlay for OCSVM Training Labor Illusion"""
    def __init__(self, parent):
        self.popup = tk.Toplevel(parent)
        self.popup.overrideredirect(True)
        self.popup.grab_set() 
        
        pw, ph = 500, 180
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (pw // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (ph // 2)
        self.popup.geometry(f"{pw}x{ph}+{x}+{y}")
        self.popup.configure(bg="#1E1E1E", highlightthickness=2, highlightbackground=ACCENT)
        
        self.lbl_title = tk.Label(self.popup, text="🧠 TRAINING OCSVM MODEL", font=("Segoe UI", 16, "bold"), fg="#FFFFFF", bg="#1E1E1E")
        self.lbl_title.pack(pady=(25, 15))
        
        self.lbl_log = tk.Label(self.popup, text="Initializing...", font=("Segoe UI", 12, "bold"), fg=SUCCESS, bg="#1E1E1E")
        self.lbl_log.pack(pady=(0, 15))
        
        style = ttk.Style()
        style.configure("Mini.Horizontal.TProgressbar", background=ACCENT, troughcolor="#333333", bordercolor="#1E1E1E", thickness=8)
        self.progress_var = tk.DoubleVar()
        self.pbar = ttk.Progressbar(self.popup, variable=self.progress_var, maximum=100, length=400, style="Mini.Horizontal.TProgressbar")
        self.pbar.pack()

    def update_state(self, log_msg, progress):
        self.lbl_log.config(text=log_msg)
        self.progress_var.set(progress)

    def close(self):
        self.popup.grab_release()
        self.popup.destroy()

class SmartWaveTrainer(tk.Tk):
    """Main Application GUI for SmartWave OCSVM Trainer"""
    def __init__(self):
        super().__init__()
        self.title('SmartWave OCSVM Trainer (Pro) - Double Click Charts to Expand')
        self.geometry('1380x880')
        self.configure(bg=BG_COLOR)
        
        self.is_cancelled = False
        self.is_processing = False
        self.X_cache = None
        self.X_abnormal_cache = None
        self.last_audio_cache = None
        self.last_abnormal_audio_cache = None
        self.last_X_scaled_cache = None
        self.scaler_cache = None
        self.ocsvm_cache = None
        
        # UI Initialization
        self._init_engine()
        self._build_ui()
        
    def _init_engine(self):
        """Initializes the PANNs Engine with error handling."""
        try:
            self.engine = PANNsEngine()
            header_text = f"SmartWave OCSVM Trainer - Engine: {self.engine.mode.upper()} ({self.engine.dim}-dim)"
        except FileNotFoundError as e:
            messagebox.showerror("Engine Load Error", str(e) + "\n\nPlease ensure the ONNX model exists before running the trainer.")
            self.engine = None
            header_text = "SmartWave OCSVM Trainer - [ENGINE MISSING]"
            
        tk.Label(self, text=header_text, bg=PRIMARY, fg="white", font=("Segoe UI", 16, "bold"), pady=12).pack(fill=tk.X)
        
    def _build_ui(self):
        """Builds the main user interface."""
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
        
        self._create_label(zone1, "Equipment:", 0, 0)
        self.equip_var = tk.StringVar(value="Motor")
        ttk.Combobox(zone1, textvariable=self.equip_var, values=["Motor", "Conveyor", "Pump", "Fan", "RoboticArm"], font=("Segoe UI", 9), width=15).grid(row=0, column=1, sticky='w', pady=5, padx=10)
        
        self.btn_load = tk.Button(zone1, text="📂 EXTRACT FEATURES", bg=LOAD_BG, fg="white", font=("Segoe UI", 10, "bold"), width=22, cursor="hand2", command=lambda: self.start_load_thread('normal'))
        self.btn_load.grid(row=1, column=0, columnspan=2, pady=5)
        
        if self.engine is None:
            self.btn_load.config(state=tk.DISABLED)
        
        # Zone 2: Hyperparameters
        zone2 = tk.LabelFrame(ctrl_frame, text="2. Hyperparameters", bg=BG_COLOR, font=("Segoe UI", 10, "bold"), padx=15, pady=10)
        zone2.pack(fill=tk.X, pady=(0, 10))
        
        self._create_label(zone2, "Gamma (RBF):", 0, 0)
        self.gamma_var = tk.DoubleVar(value=0.001)
        self._create_slider(zone2, self.gamma_var, 0.0001, 1.0, 0.0001, 0, 1)
        
        self._create_label(zone2, "Nu (Margin):", 1, 0)
        self.nu_var = tk.DoubleVar(value=0.10)
        self._create_slider(zone2, self.nu_var, 0.01, 0.5, 0.01, 1, 1)
        
        # Zone 3: Execution
        zone3 = tk.LabelFrame(ctrl_frame, text="3. Execution", bg=BG_COLOR, font=("Segoe UI", 10, "bold"), padx=15, pady=10)
        zone3.pack(fill=tk.X)
        
        self.btn_train = tk.Button(zone3, text="🧠 FIT OCSVM BOUNDARY", bg=DISABLED_BG, fg=DISABLED_FG, font=("Segoe UI", 10, "bold"), width=22, state=tk.DISABLED, cursor="hand2", command=self.action_train_ocsvm)
        self.btn_train.pack(pady=5)
        
        self.btn_evaluate = tk.Button(zone3, text="🚨 EVALUATE ABNORMAL", bg=DISABLED_BG, fg=DISABLED_FG, font=("Segoe UI", 10, "bold"), width=22, state=tk.DISABLED, cursor="hand2", command=lambda: self.start_load_thread('abnormal'))
        self.btn_evaluate.pack(pady=5)
        
        self.btn_export = tk.Button(zone3, text="🚀 EXPORT TO DEVICE", bg=DISABLED_BG, fg=DISABLED_FG, font=("Segoe UI", 10, "bold"), width=22, state=tk.DISABLED, cursor="hand2", command=self.action_export_model)
        self.btn_export.pack(pady=5)
        
        # Main Log Window
        self.log_text = tk.Text(top_frame, height=16, width=50, font=("Consolas", 10), bg="#1E1E1E", fg="#D4D4D4", padx=10, pady=10)
        self.log_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ---------- BOTTOM HALF (3 Parallel Charts) ----------
        chart_frame = tk.Frame(main_frame, bg=CARD_BG, highlightbackground="#DDDDDD", highlightthickness=1)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        tk.Label(chart_frame, text="💡 TIP: 더블클릭하면 줌(Zoom) 기능이 포함된 큰 팝업으로 차트를 볼 수 있습니다.", bg=CARD_BG, fg=MUTED, font=("Segoe UI", 9, "italic")).pack(pady=(5,0))
        
        self.fig, (self.ax_wave, self.ax_fingerprint, self.ax_pca) = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.patch.set_facecolor(CARD_BG)
        self.ax_wave.set_title('Raw Audio Waveform', fontsize=10, fontweight='bold', color=PRIMARY)
        self.ax_fingerprint.set_title(f'Acoustic Fingerprint ({EMBEDDING_DIM}-Dim)', fontsize=10, fontweight='bold', color=PRIMARY)
        self.ax_pca.set_title('OCSVM Space (PCA 2D)', fontsize=10, fontweight='bold', color=PRIMARY)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Bind double click for interactive chart popup
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        
        if self.engine:
            self.safe_log("[SYS] System Initialized. Awaiting feature extraction.")

    def _on_canvas_click(self, event):
        """Detects double clicks on specific axes to pop up an interactive chart."""
        if event.dblclick:
            if event.inaxes == self.ax_wave:
                self._open_chart_popup('wave')
            elif event.inaxes == self.ax_fingerprint:
                self._open_chart_popup('fingerprint')
            elif event.inaxes == self.ax_pca:
                self._open_chart_popup('pca')

    def _open_chart_popup(self, chart_type):
        """Opens a new Toplevel window with a single interactive matplotlib chart."""
        popup = tk.Toplevel(self)
        popup.title(f"Interactive Chart: {chart_type.upper()}")
        popup.geometry("900x650")
        popup.configure(bg=CARD_BG)
        
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        fig.patch.set_facecolor(CARD_BG)
        
        # Draw the requested chart
        if chart_type == 'wave':
            if self.last_audio_cache is not None:
                plot_len = min(len(self.last_audio_cache), SAMPLE_RATE)
                ax.plot(self.last_audio_cache[:plot_len], color=MUTED, alpha=0.8, linewidth=1.0, label="Normal Wave")
                if self.last_abnormal_audio_cache is not None:
                    ab_plot_len = min(len(self.last_abnormal_audio_cache), SAMPLE_RATE)
                    ax.plot(self.last_abnormal_audio_cache[:ab_plot_len], color=DANGER, alpha=0.6, linewidth=1.0, label="Abnormal Wave")
                ax.set_title('Raw Audio Waveform (Interactive)', fontsize=14, fontweight='bold', color=PRIMARY)
                ax.set_xlabel('Time (Samples)', fontsize=11)
                ax.set_ylabel('Amplitude', fontsize=11)
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.3)
                
        elif chart_type == 'fingerprint':
            if self.X_cache is not None and len(self.X_cache) > 0:
                X_mean = np.mean(self.X_cache, axis=0)
                X_std = np.std(self.X_cache, axis=0)
                bands = np.arange(len(X_mean))
                ax.fill_between(bands, X_mean - X_std, X_mean + X_std, color=MUTED, alpha=0.25, label='Normal Variance')
                ax.plot(bands, X_mean, color=ACCENT, linewidth=2, label='Normal Baseline')
                if self.X_abnormal_cache is not None and len(self.X_abnormal_cache) > 0:
                    Ab_mean = np.mean(self.X_abnormal_cache, axis=0)
                    ax.plot(bands, Ab_mean, color=DANGER, linewidth=2, linestyle='--', label='Anomaly Signature')
                ax.set_title(f'Deep Acoustic Fingerprint ({EMBEDDING_DIM}-Dim)', fontsize=14, fontweight='bold', color=PRIMARY)
                ax.set_xlabel('CNN10 Feature Dimensions', fontsize=11)
                ax.set_ylabel('Normalized Energy', fontsize=11)
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.3)
                
        elif chart_type == 'pca':
            X_scaled = self.last_X_scaled_cache
            if X_scaled is not None and X_scaled.shape[1] >= 2 and self.ocsvm_cache is not None:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                sv_pca = X_pca[self.ocsvm_cache.support_]
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=SUCCESS, alpha=0.6, label='Normal Data')
                ax.scatter(sv_pca[:, 0], sv_pca[:, 1], c=WARNING_COL, edgecolors='white', s=80, label='Support Vectors')
                if self.X_abnormal_cache is not None:
                    Ab_scaled = self.scaler_cache.transform(self.X_abnormal_cache)
                    Ab_pca = pca.transform(Ab_scaled)
                    ax.scatter(Ab_pca[:, 0], Ab_pca[:, 1], c='#000000', marker='x', s=80, label='Abnormal (Anomalies)')
                
                xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min() - 2, X_pca[:, 0].max() + 2, 50),
                                     np.linspace(X_pca[:, 1].min() - 2, X_pca[:, 1].max() + 2, 50))
                grid_2d = np.c_[xx.ravel(), yy.ravel()]
                grid_512 = pca.inverse_transform(grid_2d)
                Z = self.ocsvm_cache.decision_function(grid_512)
                Z = Z.reshape(xx.shape)
                
                ax.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], colors=[DANGER, SUCCESS], alpha=0.15)
                ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors=DANGER)
                ax.set_title('OCSVM Boundary Mapping (PCA 2D)', fontsize=14, fontweight='bold', color=PRIMARY)
                ax.legend(loc='lower right')
                ax.grid(True, linestyle='--', alpha=0.3)
        
        fig.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        
        # Attach Toolbar
        toolbar = NavigationToolbar2Tk(canvas, popup)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Prevent memory leak: close the matplotlib Figure when popup is closed
        popup.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), popup.destroy()))

    def _create_label(self, parent, text, row, col):
        tk.Label(parent, text=text, bg=BG_COLOR, font=("Segoe UI", 9)).grid(row=row, column=col, sticky='w', pady=5)
        
    def _create_slider(self, parent, variable, from_, to_, resolution, row, col):
        scale = tk.Scale(parent, variable=variable, from_=from_, to=to_, resolution=resolution, orient=tk.HORIZONTAL, length=150, bg=BG_COLOR, highlightthickness=0)
        scale.grid(row=row, column=col, padx=10)

    def safe_log(self, msg, is_error=False):
        self.after(0, self._append_log, msg, is_error)

    def _append_log(self, msg, is_error=False):
        if is_error:
            msg = f"❌ {msg}"
            
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

    def start_load_thread(self, target_type='normal'):
        if self.is_processing:
            return
            
        dataset_dir = filedialog.askdirectory(title=f"Select Dataset Folder (Must contain '{target_type}' folder)")
        if not dataset_dir: return
        
        target_dir = os.path.join(dataset_dir, target_type)
        if not os.path.exists(target_dir):
            self.safe_log(f"[ERR] Validation Failed: '{target_type}' directory missing in {dataset_dir}.", is_error=True)
            messagebox.showerror("Folder Error", f"Cannot find '{target_type}' folder inside:\n{dataset_dir}")
            return
            
        self.is_processing = True
        self.is_cancelled = False
        
        if target_type == 'normal':
            self.btn_load.config(state=tk.DISABLED, text="PROCESSING...", bg=MUTED)
        else:
            self.btn_evaluate.config(state=tk.DISABLED, text="EVALUATING...", bg=MUTED)
            
        self.btn_train.config(state=tk.DISABLED, bg=DISABLED_BG, fg=DISABLED_FG)
        self.btn_export.config(state=tk.DISABLED, bg=DISABLED_BG, fg=DISABLED_FG)
        
        self.overlay = LoadingOverlay(self, cancel_callback=self._cancel_process)
        
        thread = threading.Thread(target=self._load_task, args=(target_dir, target_type), daemon=True)
        thread.start()

    def _load_task(self, target_dir, target_type):
        def read_wav(p):
            with wave.open(p, 'rb') as w:
                return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

        self.safe_log(f"\n[SYS] Feature Extraction started from: {target_dir}")
        self.after(0, self.overlay.update_state, f"[PRC] Parsing {target_type} .wav files...", 0)
        
        X = []
        files = [f for f in os.listdir(target_dir) if f.endswith('.wav')]
        total_files = len(files)
        last_audio = None
        
        if total_files == 0:
            self.safe_log(f"[ERR] No .wav files found in {target_dir}.", is_error=True)
            self.after(0, lambda: messagebox.showwarning("Empty Folder", f"No .wav files were found in '{target_type}' folder."))
            self.after(0, self._reset_load_ui, target_type)
            return

        start_time = time.time()
        for i, f in enumerate(files):
            if self.is_cancelled:
                self.safe_log("[WRN] Operation aborted during feature extraction.")
                self.after(0, self._reset_load_ui, target_type)
                return
                
            try:
                audio = read_wav(os.path.join(target_dir, f))
                if i == total_files - 1: last_audio = audio
                emb = self.engine.extract_clip_embedding(audio)
                X.append(emb)
            except Exception as e:
                self.safe_log(f"[ERR] Failed reading {f}: {str(e)}", is_error=True)
                continue 
            
            if i % 3 == 0 or i == total_files - 1:
                short_f = (f[:12] + '..') if len(f) > 14 else f.ljust(14)
                snap = f"{emb[0]:.3f}, {emb[1]:.3f}, {emb[2]:.3f}"
                self.after(0, self.overlay.append_mini_log, f"> {short_f} ➔ [ {snap} ... {EMBEDDING_DIM}-Dim ] ➔ OK")
            
            if i % 5 == 0 or i == total_files - 1:
                progress = (i / total_files) * 100 
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_sec = (total_files - i) / rate if rate > 0 else 0
                
                self.after(0, self.overlay.update_state, f"[PRC] Processing Batch {i}/{total_files}...", progress)
                self.after(0, self.overlay.update_telemetry, i, total_files, rate, eta_sec)

        if len(X) == 0:
            self.safe_log("[ERR] Failed to extract features.", is_error=True)
            self.after(0, self._reset_load_ui, target_type)
            return

        if target_type == 'normal':
            self.X_cache = np.array(X)
            self.last_audio_cache = last_audio
        else:
            self.X_abnormal_cache = np.array(X)
            self.last_abnormal_audio_cache = last_audio
        
        self.safe_log(f"[OUT] Extracted Matrix Shape: {np.array(X).shape}")
        self.after(0, self.overlay.append_mini_log, f"> Matrix perfectly extracted in {time.time()-start_time:.1f}s.")
        time.sleep(0.5)
        
        self.after(0, self._finalize_load, target_type)

    def _reset_load_ui(self, target_type='normal'):
        try: self.overlay.close()
        except: pass
        if target_type == 'normal':
            self.btn_load.config(state=tk.NORMAL, text="📂 EXTRACT FEATURES", bg=LOAD_BG)
        else:
            self.btn_evaluate.config(state=tk.NORMAL, text="🚨 EVALUATE ABNORMAL", bg=DANGER)
            self.btn_train.config(state=tk.NORMAL, bg=ACCENT, fg="white")
            self.btn_export.config(state=tk.NORMAL, bg=SUCCESS, fg="white")
        self.is_processing = False

    def _finalize_load(self, target_type):
        try: self.overlay.close()
        except: pass
        self.is_processing = False
        
        if target_type == 'normal':
            self.btn_load.config(state=tk.NORMAL, text="📂 EXTRACT FEATURES", bg=LOAD_BG)
            self.btn_train.config(state=tk.NORMAL, bg=ACCENT, fg="white")
            self.safe_log("[SYS] Normal features loaded. Ready to fit boundary.")
        else:
            self.btn_evaluate.config(state=tk.NORMAL, text="🚨 EVALUATE ABNORMAL", bg=DANGER)
            self.btn_train.config(state=tk.NORMAL, bg=ACCENT, fg="white")
            self.btn_export.config(state=tk.NORMAL, bg=SUCCESS, fg="white")
            self.safe_log("\n[SYS] Abnormal features loaded. Rendering anomalies on chart...")
            if self.scaler_cache and self.X_cache is not None:
                self._draw_charts(self.scaler_cache.transform(self.X_cache))
        
    def action_train_ocsvm(self):
        if self.is_processing or self.X_cache is None: return
        self.is_processing = True
        
        if len(self.X_cache) < 2:
            self.safe_log("[ERR] Not enough data points.", is_error=True)
            self.is_processing = False
            return
            
        self.btn_train.config(state=tk.DISABLED, text="OPTIMIZING...", bg=MUTED)
        self.train_overlay = TrainOverlay(self)
        
        thread = threading.Thread(target=self._train_task_fake_delay, daemon=True)
        thread.start()

    def _train_task_fake_delay(self):
        try:
            self.after(0, self.train_overlay.update_state, "Step 1: Standardizing Data Matrix...", 15)
            self.safe_log("\n[PRC] Standardizing feature space...")
            time.sleep(1.5)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X_cache)
            
            gamma = self.gamma_var.get()
            nu = self.nu_var.get()
            self.after(0, self.train_overlay.update_state, "Step 2: Optimizing SVM Hyperplanes...", 50)
            self.safe_log("[PRC] Optimizing non-linear hyperplanes (RBF)...")
            time.sleep(2.5)
            
            ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
            ocsvm.fit(X_scaled)
            
            self.after(0, self.train_overlay.update_state, "Step 3: Extracting Support Vectors...", 85)
            self.safe_log("[CHK] Extracting support vectors...")
            time.sleep(1.5)
            
            self.scaler_cache = scaler
            self.ocsvm_cache = ocsvm
            
            self.after(0, self.train_overlay.update_state, "✅ Training Complete!", 100)
            self.safe_log(f"[CHK] Boundary fit complete (Gamma: {gamma}, Nu: {nu}). SVs: {len(ocsvm.support_)}")
            time.sleep(0.5)
            
            self.after(0, self._finalize_train, X_scaled)
        except Exception as e:
            self.safe_log(f"[ERR] Training failed: {str(e)}", is_error=True)
            self.after(0, self.train_overlay.close)
            self.is_processing = False
            self.after(0, lambda: self.btn_train.config(state=tk.NORMAL, text="🧠 FIT OCSVM BOUNDARY", bg=ACCENT))

    def _finalize_train(self, X_scaled):
        try: self.train_overlay.close()
        except: pass
        self.btn_train.config(state=tk.NORMAL, text="🧠 FIT OCSVM BOUNDARY", bg=ACCENT)
        self.btn_evaluate.config(state=tk.NORMAL, bg=DANGER, fg="white")
        self.btn_export.config(state=tk.NORMAL, bg=SUCCESS, fg="white")
        self._draw_charts(X_scaled)
        self.is_processing = False

    def _draw_charts(self, X_scaled=None):
        """Renders the 3 parallel charts and saves scaling for interactive popup."""
        if X_scaled is not None:
            self.last_X_scaled_cache = X_scaled
        else:
            X_scaled = self.last_X_scaled_cache
            
        self.ax_wave.clear()
        if self.last_audio_cache is not None:
            plot_len = min(len(self.last_audio_cache), SAMPLE_RATE)
            self.ax_wave.plot(self.last_audio_cache[:plot_len], color=MUTED, alpha=0.8, linewidth=0.8, label="Normal Wave")
            if self.last_abnormal_audio_cache is not None:
                ab_plot_len = min(len(self.last_abnormal_audio_cache), SAMPLE_RATE)
                self.ax_wave.plot(self.last_abnormal_audio_cache[:ab_plot_len], color=DANGER, alpha=0.6, linewidth=0.8, label="Abnormal Wave")
            self.ax_wave.set_title('Raw Audio Waveform (1 sec)', fontsize=10, fontweight='bold', color=PRIMARY)
            self.ax_wave.set_xlabel('Time (Samples)', fontsize=9)
            self.ax_wave.set_ylabel('Amplitude', fontsize=9)
            self.ax_wave.legend(loc='upper right', fontsize=8)
            self.ax_wave.grid(True, linestyle='--', alpha=0.3)
            
        self.ax_fingerprint.clear()
        if self.X_cache is not None and len(self.X_cache) > 0:
            X_mean = np.mean(self.X_cache, axis=0)
            X_std = np.std(self.X_cache, axis=0)
            bands = np.arange(len(X_mean))
            self.ax_fingerprint.fill_between(bands, X_mean - X_std, X_mean + X_std, color=MUTED, alpha=0.25, label='Normal Variance')
            self.ax_fingerprint.plot(bands, X_mean, color=ACCENT, linewidth=2, label='Normal Baseline')
            if self.X_abnormal_cache is not None and len(self.X_abnormal_cache) > 0:
                Ab_mean = np.mean(self.X_abnormal_cache, axis=0)
                self.ax_fingerprint.plot(bands, Ab_mean, color=DANGER, linewidth=2, linestyle='--', label='Anomaly Signature')
            self.ax_fingerprint.set_title(f'Deep Acoustic Fingerprint ({EMBEDDING_DIM}-Dim)', fontsize=10, fontweight='bold', color=PRIMARY)
            self.ax_fingerprint.set_xlabel('CNN10 Feature Dimensions', fontsize=9)
            self.ax_fingerprint.set_ylabel('Normalized Energy', fontsize=9)
            self.ax_fingerprint.legend(loc='upper right', fontsize=8)
            self.ax_fingerprint.grid(True, linestyle='--', alpha=0.3)
            
        self.ax_pca.clear()
        if X_scaled is not None and X_scaled.shape[1] >= 2 and self.ocsvm_cache is not None:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            sv_pca = X_pca[self.ocsvm_cache.support_]
            self.ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=SUCCESS, alpha=0.6, label='Normal Data')
            self.ax_pca.scatter(sv_pca[:, 0], sv_pca[:, 1], c=WARNING_COL, edgecolors='white', s=60, label='Support Vectors')
            if self.X_abnormal_cache is not None:
                Ab_scaled = self.scaler_cache.transform(self.X_abnormal_cache)
                Ab_pca = pca.transform(Ab_scaled)
                self.ax_pca.scatter(Ab_pca[:, 0], Ab_pca[:, 1], c='#000000', marker='x', s=60, label='Abnormal (Anomalies)')
            xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min() - 2, X_pca[:, 0].max() + 2, 50),
                                 np.linspace(X_pca[:, 1].min() - 2, X_pca[:, 1].max() + 2, 50))
            grid_2d = np.c_[xx.ravel(), yy.ravel()]
            grid_512 = pca.inverse_transform(grid_2d)
            Z = self.ocsvm_cache.decision_function(grid_512)
            Z = Z.reshape(xx.shape)
            self.ax_pca.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], colors=[DANGER, SUCCESS], alpha=0.15)
            self.ax_pca.contour(xx, yy, Z, levels=[0], linewidths=2, colors=DANGER)
            self.ax_pca.set_title('OCSVM Boundary Mapping (PCA 2D)', fontsize=10, fontweight='bold', color=PRIMARY)
            self.ax_pca.legend(loc='lower right', fontsize=8)
            self.ax_pca.grid(True, linestyle='--', alpha=0.3)
            
        self.fig.tight_layout()
        self.canvas.draw()

    def action_export_model(self):
        if self.is_processing or self.ocsvm_cache is None or self.scaler_cache is None: return
        self.is_processing = True
        try:
            rho = float(-self.ocsvm_cache.offset_[0])
            svs = self.scaler_cache.transform(self.X_cache)[self.ocsvm_cache.support_]
            dual_coef = self.ocsvm_cache.dual_coef_[0]
            out = {
                "equipment_name": self.equip_var.get(),
                "equipment_key": self.equip_var.get(),
                "exported_at": datetime.datetime.now().isoformat(),
                "sample_rate": SAMPLE_RATE,
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
            messagebox.showinfo("Export Successful", f"Model parameters saved to:\n{out_path}")
        except Exception as e:
            self.safe_log(f"[ERR] Failed to export model: {str(e)}", is_error=True)
        finally:
            self.is_processing = False

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = SmartWaveTrainer()
    app.mainloop()
