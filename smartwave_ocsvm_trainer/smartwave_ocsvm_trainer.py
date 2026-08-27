import os
import json
import time
import datetime
import wave
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import onnxruntime as ort

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# UI Colors (Samsung One UI Inspired)
PRIMARY = "#000000"
ACCENT = "#0381FE"
SUCCESS = "#10B981"
DANGER = "#F43F5E"
BG_COLOR = "#F7F7F7"
CARD_BG = "#FFFFFF"

# Font handling for Korean
plt.rcParams['font.family'] = ['Malgun Gothic', 'Segoe UI', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class PANNsEngine:
    def __init__(self):
        self.mode = 'mel_stats'
        self.dim = 128
        self.sess = None
        
        # Try ONNX
        onnx_path = '../models_official/smartwave_cnn10_e2e.onnx'
        if os.path.exists(onnx_path):
            try:
                self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                self.mode = 'cnn10_e2e'
                self.dim = 527
                print(f"Loaded ONNX: {onnx_path}")
            except Exception as e:
                print("Failed to load ONNX:", e)
    
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

class SmartWaveTrainer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SmartWave OCSVM Trainer (Pro)')
        self.geometry('1280x850')
        self.configure(bg=BG_COLOR)
        
        self.engine = PANNsEngine()
        header_text = f"SmartWave OCSVM Trainer - Engine: {self.engine.mode.upper()} ({self.engine.dim}-dim)"
        tk.Label(self, text=header_text, bg=PRIMARY, fg="white", font=("Segoe UI", 16, "bold"), pady=12).pack(fill=tk.X)
        
        main_frame = tk.Frame(self, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # ---------- TOP HALF: Controls & Log ----------
        top_frame = tk.Frame(main_frame, bg=BG_COLOR)
        top_frame.pack(fill=tk.X)
        
        # Controls Frame
        ctrl_frame = tk.LabelFrame(top_frame, text="OCSVM Config", bg=BG_COLOR, font=("Segoe UI", 11, "bold"), padx=10, pady=10)
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
        
        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(ctrl_frame, variable=self.progress_var, maximum=100, length=300)
        self.progress_bar.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Log Frame
        self.log_text = tk.Text(top_frame, height=13, width=50, font=("Consolas", 10), bg="#1E1E1E", fg="#D4D4D4", padx=10, pady=10)
        self.log_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ---------- BOTTOM HALF: Charts ----------
        chart_frame = tk.Frame(main_frame, bg=CARD_BG, highlightbackground="#DDDDDD", highlightthickness=1)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        self.fig, (self.ax_wave, self.ax_pca) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.patch.set_facecolor(CARD_BG)
        
        self.ax_wave.set_title('Waveform Visualization', fontsize=11, fontweight='bold', color=PRIMARY)
        self.ax_pca.set_title('OCSVM Embedding Space (PCA 2D)', fontsize=11, fontweight='bold', color=PRIMARY)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.safe_log("[System] Ready. Multi-Threading & Pro UI Enabled.")

    def safe_log(self, msg):
        """Thread-safe logging function"""
        self.after(0, self._append_log, msg)

    def _append_log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        
    def start_training_thread(self):
        dataset_dir = filedialog.askdirectory(title="Select Dataset Folder")
        if not dataset_dir: return
        
        normal_dir = os.path.join(dataset_dir, 'normal')
        if not os.path.exists(normal_dir):
            self.safe_log(f"[Error] 'normal' folder not found in {dataset_dir}")
            return
            
        # Disable button to prevent multiple clicks
        self.train_btn.config(state=tk.DISABLED, text="Processing... Please wait", bg=SAMSUNG_MUTED if 'SAMSUNG_MUTED' in globals() else "#707070")
        self.progress_var.set(0)
        
        # Start background thread
        thread = threading.Thread(target=self._train_task, args=(normal_dir,), daemon=True)
        thread.start()

    def _train_task(self, normal_dir):
        def read_wav(p):
            with wave.open(p, 'rb') as w:
                return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

        self.safe_log(f"\n[1/3] Extracting embeddings (Background Thread)...")
        X = []
        files = [f for f in os.listdir(normal_dir) if f.endswith('.wav')]
        total_files = len(files)
        last_audio = None
        
        if total_files == 0:
            self.safe_log("[Error] No .wav files found in normal directory.")
            self.after(0, self._reset_ui)
            return

        for i, f in enumerate(files):
            # Update Progress Bar smoothly
            if i % 10 == 0 or i == total_files - 1:
                progress_percent = (i / total_files) * 100
                self.after(0, self.progress_var.set, progress_percent)
                if i > 0 and i % 200 == 0:
                    self.safe_log(f"      Processed {i} / {total_files} files...")

            audio = read_wav(os.path.join(normal_dir, f))
            if i == total_files - 1: last_audio = audio
            emb = self.engine.extract_clip_embedding(audio)
            X.append(emb)
            
        X = np.array(X)
        self.safe_log(f"      Feature Matrix Shape: {X.shape}")
        
        # Preprocess
        self.safe_log(f"[2/3] Applying StandardScaler and Training OCSVM...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        gamma = self.gamma_var.get()
        nu = self.nu_var.get()
        
        ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        ocsvm.fit(X_scaled)
        
        # Export
        rho = float(-ocsvm.offset_[0])
        svs = X_scaled[ocsvm.support_]
        dual_coef = ocsvm.dual_coef_[0]
        
        out = {
            "equipment_name": self.equip_var.get(),
            "equipment_key": self.equip_var.get(),
            "exported_at": datetime.datetime.now().isoformat(),
            "sample_rate": 16000,
            "embedding_dim": self.engine.dim,
            "embedding_mode": self.engine.mode,
            "preprocessing": "StandardScaler",
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "gamma": float(gamma),
            "nu": float(nu),
            "rho": rho,
            "n_support_vectors": len(svs),
            "support_vectors": svs.tolist(),
            "dual_coef": dual_coef.tolist(),
        }
        
        out_path = f"ocsvm_params_{self.equip_var.get()}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
            
        self.safe_log(f"[3/3] Export Complete!")
        self.safe_log(f"      Saved to: {out_path}")
        self.safe_log(f"      Support Vectors: {len(svs)}")
        
        # Safely trigger UI updates on the main thread
        self.after(0, self._finalize_train, X_scaled, ocsvm, last_audio)

    def _finalize_train(self, X_scaled, ocsvm, last_audio):
        self.progress_var.set(100)
        self.train_btn.config(state=tk.NORMAL, text="Load Dataset & Train", bg=ACCENT)
        self.safe_log("\n[Notice] Updating Visualization Charts...")
        
        # Draw Charts
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
            
            # Scatter Plot
            self.ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=SUCCESS, alpha=0.6, label='Normal Embeddings')
            self.ax_pca.scatter(sv_pca[:, 0], sv_pca[:, 1], c=DANGER, edgecolors='white', s=60, label='Support Vectors (Boundary)')
            
            # Decision Boundary Contour
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
        self.safe_log("[Notice] Ready for next task.")

    def _reset_ui(self):
        self.progress_var.set(0)
        self.train_btn.config(state=tk.NORMAL, text="Load Dataset & Train", bg=ACCENT)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = SmartWaveTrainer()
    app.mainloop()
