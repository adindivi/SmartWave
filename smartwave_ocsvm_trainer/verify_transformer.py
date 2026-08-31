"""
SmartWave Transformer Verification Script
==========================================
Compares anomaly detection performance:
  A) Baseline:    10s audio → CNN10 → 527-dim (single vector) → OCSVM
  B) Transformer: 10s audio → 1s × 10 chunks → CNN10 × 10 → Transformer → 527-dim → OCSVM

This script verifies whether adding a Transformer encoder layer
to capture temporal context improves anomaly separation.
"""
import os
import sys
import wave
import time
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── Constants ───
TARGET_AUDIO_LEN = 160000   # 10 seconds @ 16kHz
SAMPLE_RATE = 16000
CHUNK_LEN = 16000            # 1 second per chunk
CHUNK_OVERLAP = 8000          # 0.5 second overlap
EMBEDDING_DIM = 527

# ─── Mini Transformer Module ───
class MiniTransformerAggregator(nn.Module):
    """
    Receives a sequence of CNN10 embeddings (one per 1-second chunk)
    and uses a Transformer encoder to capture temporal context,
    then outputs a single aggregated embedding.
    """
    def __init__(self, d_model=527, nhead=17, num_layers=2, dim_feedforward=1024):
        super().__init__()
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,           # 527 / 17 = 31 (must divide evenly → actually 527 is prime!)
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: (batch, seq_len, 527) — sequence of chunk embeddings
        Returns: (batch, 527) — single aggregated embedding
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.norm(x)
        # Aggregate: mean pooling over time
        x = x.mean(dim=1)
        return x


def read_wav(path):
    """Reads a WAV file and returns float32 audio normalized to [-1, 1]."""
    with wave.open(path, 'rb') as w:
        raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def slice_audio_into_chunks(audio, chunk_len=CHUNK_LEN, overlap=CHUNK_OVERLAP):
    """Slices audio into overlapping 1-second chunks."""
    step = chunk_len - overlap
    chunks = []
    for start in range(0, len(audio) - chunk_len + 1, step):
        chunks.append(audio[start:start + chunk_len])
    if not chunks:
        # Audio shorter than 1 chunk — pad
        padded = np.pad(audio, (0, max(0, chunk_len - len(audio))))
        chunks.append(padded[:chunk_len])
    return chunks


def extract_single_embedding(sess, audio):
    """Baseline: extract one embedding from full 10s audio."""
    if len(audio) < TARGET_AUDIO_LEN:
        audio = np.pad(audio, (0, TARGET_AUDIO_LEN - len(audio)))
    else:
        audio = audio[:TARGET_AUDIO_LEN]
    audio = audio.astype(np.float32)[np.newaxis, :]
    return sess.run(None, {'audio': audio})[0][0]


def extract_chunk_embeddings(sess, audio):
    """Transformer path: extract one embedding per 1-second chunk."""
    chunks = slice_audio_into_chunks(audio)
    embeddings = []
    for chunk in chunks:
        if len(chunk) < TARGET_AUDIO_LEN:
            padded = np.pad(chunk, (0, TARGET_AUDIO_LEN - len(chunk)))
        else:
            padded = chunk[:TARGET_AUDIO_LEN]
        inp = padded.astype(np.float32)[np.newaxis, :]
        emb = sess.run(None, {'audio': inp})[0][0]
        embeddings.append(emb)
    return np.array(embeddings)  # (num_chunks, 527)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    onnx_path = '../models_official/smartwave_cnn10_e2e.onnx'
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX model not found: {onnx_path}")
        return
    
    normal_dir = '../dataset/normal'
    abnormal_dir = '../dataset/abnormal'
    
    if not os.path.exists(normal_dir) or not os.path.exists(abnormal_dir):
        print("[ERROR] dataset/normal and dataset/abnormal folders required.")
        return
    
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    normal_files = sorted([f for f in os.listdir(normal_dir) if f.endswith('.wav')])[:50]
    abnormal_files = sorted([f for f in os.listdir(abnormal_dir) if f.endswith('.wav')])[:50]
    
    print(f"Normal files: {len(normal_files)}, Abnormal files: {len(abnormal_files)}")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════
    # A) BASELINE: Single embedding per file
    # ═══════════════════════════════════════════════════════
    print("\n[A] BASELINE (Single 527-dim per file)")
    t0 = time.time()
    
    X_normal_base = []
    for f in normal_files:
        audio = read_wav(os.path.join(normal_dir, f))
        emb = extract_single_embedding(sess, audio)
        X_normal_base.append(emb)
    X_normal_base = np.array(X_normal_base)
    
    X_abnormal_base = []
    for f in abnormal_files:
        audio = read_wav(os.path.join(abnormal_dir, f))
        emb = extract_single_embedding(sess, audio)
        X_abnormal_base.append(emb)
    X_abnormal_base = np.array(X_abnormal_base)
    
    baseline_time = time.time() - t0
    
    # Train OCSVM on normal
    scaler_base = StandardScaler()
    X_n_scaled = scaler_base.fit_transform(X_normal_base)
    ocsvm_base = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.10)
    ocsvm_base.fit(X_n_scaled)
    
    # Score
    scores_normal_base = ocsvm_base.decision_function(X_n_scaled)
    scores_abnormal_base = ocsvm_base.decision_function(
        scaler_base.transform(X_abnormal_base)
    )
    
    pred_normal_base = ocsvm_base.predict(X_n_scaled)
    pred_abnormal_base = ocsvm_base.predict(scaler_base.transform(X_abnormal_base))
    
    acc_normal_base = np.mean(pred_normal_base == 1) * 100
    acc_abnormal_base = np.mean(pred_abnormal_base == -1) * 100
    
    print(f"  Extraction time: {baseline_time:.1f}s")
    print(f"  Normal  scores: mean={scores_normal_base.mean():.4f}, min={scores_normal_base.min():.4f}")
    print(f"  Abnormal scores: mean={scores_abnormal_base.mean():.4f}, max={scores_abnormal_base.max():.4f}")
    print(f"  Score gap (separation): {scores_normal_base.mean() - scores_abnormal_base.mean():.4f}")
    print(f"  Normal correctly classified:   {acc_normal_base:.1f}%")
    print(f"  Abnormal correctly classified: {acc_abnormal_base:.1f}%")
    
    # ═══════════════════════════════════════════════════════
    # B) TRANSFORMER: Chunk embeddings → Transformer → Aggregated
    # ═══════════════════════════════════════════════════════
    print("\n[B] TRANSFORMER (1s chunks → Transformer Encoder → Aggregated 527-dim)")
    t0 = time.time()
    
    # 527 is prime, so nhead must be 1 or we project to a divisible dimension
    # Using nhead=1 for simplicity (still captures temporal attention)
    transformer = MiniTransformerAggregator(d_model=527, nhead=1, num_layers=2)
    transformer.eval()
    
    X_normal_trans = []
    for f in normal_files:
        audio = read_wav(os.path.join(normal_dir, f))
        chunk_embs = extract_chunk_embeddings(sess, audio)  # (N, 527)
        with torch.no_grad():
            seq = torch.tensor(chunk_embs, dtype=torch.float32).unsqueeze(0)  # (1, N, 527)
            agg = transformer(seq).numpy()[0]  # (527,)
        X_normal_trans.append(agg)
    X_normal_trans = np.array(X_normal_trans)
    
    X_abnormal_trans = []
    for f in abnormal_files:
        audio = read_wav(os.path.join(abnormal_dir, f))
        chunk_embs = extract_chunk_embeddings(sess, audio)
        with torch.no_grad():
            seq = torch.tensor(chunk_embs, dtype=torch.float32).unsqueeze(0)
            agg = transformer(seq).numpy()[0]
        X_abnormal_trans.append(agg)
    X_abnormal_trans = np.array(X_abnormal_trans)
    
    trans_time = time.time() - t0
    
    # Train OCSVM
    scaler_trans = StandardScaler()
    X_n_scaled_t = scaler_trans.fit_transform(X_normal_trans)
    ocsvm_trans = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.10)
    ocsvm_trans.fit(X_n_scaled_t)
    
    scores_normal_trans = ocsvm_trans.decision_function(X_n_scaled_t)
    scores_abnormal_trans = ocsvm_trans.decision_function(
        scaler_trans.transform(X_abnormal_trans)
    )
    
    pred_normal_trans = ocsvm_trans.predict(X_n_scaled_t)
    pred_abnormal_trans = ocsvm_trans.predict(scaler_trans.transform(X_abnormal_trans))
    
    acc_normal_trans = np.mean(pred_normal_trans == 1) * 100
    acc_abnormal_trans = np.mean(pred_abnormal_trans == -1) * 100
    
    print(f"  Extraction time: {trans_time:.1f}s (x{trans_time/max(baseline_time,0.1):.1f} slower)")
    print(f"  Chunks per file: ~{len(slice_audio_into_chunks(np.zeros(TARGET_AUDIO_LEN)))}")
    print(f"  Normal  scores: mean={scores_normal_trans.mean():.4f}, min={scores_normal_trans.min():.4f}")
    print(f"  Abnormal scores: mean={scores_abnormal_trans.mean():.4f}, max={scores_abnormal_trans.max():.4f}")
    print(f"  Score gap (separation): {scores_normal_trans.mean() - scores_abnormal_trans.mean():.4f}")
    print(f"  Normal correctly classified:   {acc_normal_trans:.1f}%")
    print(f"  Abnormal correctly classified: {acc_abnormal_trans:.1f}%")
    
    # ═══════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    gap_base = scores_normal_base.mean() - scores_abnormal_base.mean()
    gap_trans = scores_normal_trans.mean() - scores_abnormal_trans.mean()
    
    print(f"{'Metric':<35} {'Baseline':>12} {'Transformer':>12}")
    print("-" * 60)
    print(f"{'Extraction Time':<35} {baseline_time:>10.1f}s {trans_time:>10.1f}s")
    print(f"{'Score Gap (larger = better)':<35} {gap_base:>12.4f} {gap_trans:>12.4f}")
    print(f"{'Normal Accuracy':<35} {acc_normal_base:>11.1f}% {acc_normal_trans:>11.1f}%")
    print(f"{'Abnormal Detection Rate':<35} {acc_abnormal_base:>11.1f}% {acc_abnormal_trans:>11.1f}%")
    
    if gap_trans > gap_base:
        print(f"\n>>> Transformer improved separation by {((gap_trans/max(gap_base,0.0001))-1)*100:.1f}%")
    else:
        print(f"\n>>> Baseline performed better. Transformer needs fine-tuning or more data.")


if __name__ == '__main__':
    main()
