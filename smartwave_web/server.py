import os
import wave
import json
import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog

app = FastAPI(title="SmartWave Web API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TARGET_AUDIO_LEN = 160000
SAMPLE_RATE = 16000
EMBEDDING_DIM = 527

class SmartWaveBackend:
    def __init__(self):
        onnx_path = 'models_official/smartwave_cnn10_e2e.onnx'
        if not os.path.exists(onnx_path):
            onnx_path = '../models_official/smartwave_cnn10_e2e.onnx'
            
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        self.X_normal = None
        self.X_abnormal = None
        self.last_audio = None
        self.last_ab_audio = None
        self.scaler = None
        self.ocsvm = None

    def extract_embedding(self, audio):
        if len(audio) < TARGET_AUDIO_LEN:
            audio = np.pad(audio, (0, TARGET_AUDIO_LEN - len(audio)))
        else:
            audio = audio[:TARGET_AUDIO_LEN]
        audio = audio.astype(np.float32)[np.newaxis, :]
        return self.sess.run(None, {'audio': audio})[0][0]

backend = SmartWaveBackend()

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/ask_folder")
def ask_folder():
    import subprocess
    code = "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.attributes('-topmost', True); root.withdraw(); print(filedialog.askdirectory(title='Select Dataset Folder'))"
    result = subprocess.run(["python", "-c", code], capture_output=True, text=True)
    return {"path": result.stdout.strip()}

@app.post("/api/extract")
def api_extract(target_dir: str = Form(...), type: str = Form(...)):
    if not os.path.exists(target_dir):
        return JSONResponse(status_code=400, content={"error": f"Directory not found: {target_dir}"})
        
    files = [f for f in os.listdir(target_dir) if f.endswith('.wav')]
    if not files:
        return JSONResponse(status_code=400, content={"error": "No .wav files found in directory"})
        
    X = []
    last_audio = None
    for f in files:
        path = os.path.join(target_dir, f)
        try:
            with wave.open(path, 'rb') as w:
                audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
                last_audio = audio
                emb = backend.extract_embedding(audio)
                X.append(emb)
        except Exception:
            continue
            
    if not X:
        return JSONResponse(status_code=400, content={"error": "Extraction failed for all files"})
        
    X_arr = np.array(X)
    
    if type == 'normal':
        backend.X_normal = X_arr
        backend.last_audio = last_audio
    else:
        backend.X_abnormal = X_arr
        backend.last_ab_audio = last_audio
        
    return {"message": "Success", "count": len(X_arr)}

@app.post("/api/train")
def api_train(gamma: float = Form(...), nu: float = Form(...)):
    if backend.X_normal is None:
        return JSONResponse(status_code=400, content={"error": "Normal features must be loaded first."})
        
    backend.scaler = StandardScaler()
    X_scaled = backend.scaler.fit_transform(backend.X_normal)
    
    backend.ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
    backend.ocsvm.fit(X_scaled)
    
    return {"message": "Success"}

@app.get("/api/charts")
def api_charts():
    if backend.X_normal is None or backend.ocsvm is None:
        return JSONResponse(status_code=400, content={"error": "Model not trained"})
        
    ds_factor = 32
    wave_normal = backend.last_audio[:SAMPLE_RATE][::ds_factor].tolist() if backend.last_audio is not None else []
    wave_abnormal = backend.last_ab_audio[:SAMPLE_RATE][::ds_factor].tolist() if backend.last_ab_audio is not None else []
    
    X_mean = np.mean(backend.X_normal, axis=0).tolist()
    X_std = np.std(backend.X_normal, axis=0).tolist()
    Ab_mean = np.mean(backend.X_abnormal, axis=0).tolist() if backend.X_abnormal is not None else []
    
    pca = PCA(n_components=2)
    X_scaled = backend.scaler.transform(backend.X_normal)
    X_pca = pca.fit_transform(X_scaled)
    sv_pca = X_pca[backend.ocsvm.support_].tolist()
    normal_pca = X_pca.tolist()
    
    ab_pca_list = []
    if backend.X_abnormal is not None:
        ab_scaled = backend.scaler.transform(backend.X_abnormal)
        ab_pca = pca.transform(ab_scaled)
        ab_pca_list = ab_pca.tolist()
        
    xx = np.linspace(X_pca[:, 0].min() - 2, X_pca[:, 0].max() + 2, 20)
    yy = np.linspace(X_pca[:, 1].min() - 2, X_pca[:, 1].max() + 2, 20)
    xx_grid, yy_grid = np.meshgrid(xx, yy)
    grid_2d = np.c_[xx_grid.ravel(), yy_grid.ravel()]
    grid_527 = pca.inverse_transform(grid_2d)
    Z = backend.ocsvm.decision_function(grid_527).reshape(xx_grid.shape).tolist()
    
    return {
        "wave_normal": wave_normal,
        "wave_abnormal": wave_abnormal,
        "fp_mean": X_mean,
        "fp_std": X_std,
        "fp_abnormal": Ab_mean,
        "pca_normal": normal_pca,
        "pca_sv": sv_pca,
        "pca_abnormal": ab_pca_list,
        "contour_x": xx.tolist(),
        "contour_y": yy.tolist(),
        "contour_z": Z
    }
