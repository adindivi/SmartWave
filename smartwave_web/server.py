"""
SmartWave Web API Server (FastAPI)
Production-hardened version with error handling and input validation.
"""
import os
import wave
import json
import logging
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartwave")

app = FastAPI(title="SmartWave Web API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TARGET_AUDIO_LEN = 160000
SAMPLE_RATE = 16000
EMBEDDING_DIM = 527
MAX_FILES = 5000  # Safety cap to prevent OOM on huge folders


class SmartWaveBackend:
    """Manages the AI engine state and trained model parameters."""

    def __init__(self):
        self.sess = None
        self.X_normal = None
        self.X_abnormal = None
        self.last_audio = None
        self.last_ab_audio = None
        self.scaler = None
        self.ocsvm = None

        onnx_path = '../models_official/smartwave_cnn10_e2e.onnx'
        if not os.path.exists(onnx_path):
            logger.warning(f"ONNX model not found at {onnx_path}. Engine disabled.")
            return

        try:
            self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            logger.info("PANNs CNN10 ONNX engine loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.sess = None

    def is_ready(self):
        """Returns True if the ONNX engine is loaded and operational."""
        return self.sess is not None

    def extract_embedding(self, audio):
        """Extracts a 527-dim embedding from raw audio. Pads or truncates."""
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


@app.post("/api/extract")
def api_extract(target_dir: str = Form(...), data_type: str = Form(...)):
    """Extracts acoustic embeddings from .wav files in the given directory."""
    # Validate engine
    if not backend.is_ready():
        return JSONResponse(status_code=503, content={"error": "AI engine not loaded. ONNX model file is missing."})

    # Validate directory
    if not target_dir or not os.path.isdir(target_dir):
        return JSONResponse(status_code=400, content={"error": f"Directory not found: {target_dir}"})

    files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith('.wav')])
    if not files:
        return JSONResponse(status_code=400, content={"error": "No .wav files found in directory."})

    # Safety cap
    if len(files) > MAX_FILES:
        logger.warning(f"Capping file count from {len(files)} to {MAX_FILES}")
        files = files[:MAX_FILES]

    X = []
    last_audio = None
    skipped = 0

    for f in files:
        path = os.path.join(target_dir, f)
        try:
            with wave.open(path, 'rb') as w:
                n_frames = w.getnframes()
                if n_frames == 0:
                    skipped += 1
                    continue
                raw = w.readframes(n_frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if len(audio) == 0:
                    skipped += 1
                    continue
                last_audio = audio
                emb = backend.extract_embedding(audio)
                X.append(emb)
        except Exception as e:
            logger.warning(f"Skipping corrupted file {f}: {e}")
            skipped += 1
            continue

    if not X:
        return JSONResponse(status_code=400, content={"error": f"Extraction failed. {skipped} files skipped."})

    X_arr = np.array(X)

    if data_type == 'normal':
        backend.X_normal = X_arr
        backend.last_audio = last_audio
    else:
        backend.X_abnormal = X_arr
        backend.last_ab_audio = last_audio

    logger.info(f"Extracted {len(X_arr)} embeddings ({skipped} skipped) for '{data_type}'")
    return {"message": "Success", "count": len(X_arr), "skipped": skipped}


@app.post("/api/train")
def api_train(gamma: float = Form(...), nu: float = Form(...)):
    """Fits an OCSVM boundary on the extracted normal features."""
    if backend.X_normal is None:
        return JSONResponse(status_code=400, content={"error": "Normal features must be loaded first."})

    if len(backend.X_normal) < 2:
        return JSONResponse(status_code=400, content={"error": "Need at least 2 data points to train."})

    # Validate hyperparameter ranges
    gamma = max(0.0001, min(gamma, 10.0))
    nu = max(0.01, min(nu, 0.99))

    try:
        backend.scaler = StandardScaler()
        X_scaled = backend.scaler.fit_transform(backend.X_normal)

        backend.ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        backend.ocsvm.fit(X_scaled)

        sv_count = len(backend.ocsvm.support_)
        logger.info(f"OCSVM trained: gamma={gamma}, nu={nu}, SVs={sv_count}")
        return {"message": "Success", "support_vectors": sv_count}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return JSONResponse(status_code=500, content={"error": f"Training failed: {str(e)}"})


@app.get("/api/charts")
def api_charts():
    """Returns all chart data as JSON for the frontend to render."""
    if backend.X_normal is None or backend.ocsvm is None:
        return JSONResponse(status_code=400, content={"error": "Model not trained yet."})

    try:
        # 1. Waveform (downsampled)
        ds_factor = 32
        wave_normal = []
        wave_abnormal = []
        if backend.last_audio is not None and len(backend.last_audio) > 0:
            clip = backend.last_audio[:min(len(backend.last_audio), SAMPLE_RATE)]
            wave_normal = clip[::ds_factor].tolist()
        if backend.last_ab_audio is not None and len(backend.last_ab_audio) > 0:
            clip = backend.last_ab_audio[:min(len(backend.last_ab_audio), SAMPLE_RATE)]
            wave_abnormal = clip[::ds_factor].tolist()

        # 2. Fingerprint
        X_mean = np.mean(backend.X_normal, axis=0).tolist()
        X_std = np.std(backend.X_normal, axis=0).tolist()
        Ab_mean = np.mean(backend.X_abnormal, axis=0).tolist() if backend.X_abnormal is not None else []

        # 3. PCA scatter + contour
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

        # Contour grid
        x_min, x_max = X_pca[:, 0].min() - 2, X_pca[:, 0].max() + 2
        y_min, y_max = X_pca[:, 1].min() - 2, X_pca[:, 1].max() + 2
        xx = np.linspace(x_min, x_max, 20)
        yy = np.linspace(y_min, y_max, 20)
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
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        return JSONResponse(status_code=500, content={"error": f"Chart error: {str(e)}"})
