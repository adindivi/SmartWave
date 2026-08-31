"""
SmartWave Temporal Model Comparison (512-dim)
=============================================
Compares LSTM vs Transformer for temporal sequence learning on 512-dim features.
"""
import os
import time
import wave
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ─── Constants ───
TARGET_AUDIO_LEN = 160000
CHUNK_LEN = 16000
CHUNK_OVERLAP = 8000
EMBEDDING_DIM = 512
MASK_RATIO = 0.3
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4

# ─── Models ───
class LSTMAggregator(nn.Module):
    def __init__(self, d_model=512, hidden_dim=256, num_layers=2):
        super().__init__()
        # Bidirectional LSTM: 256 * 2 = 512 output dimension
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.recon_head = nn.Linear(512, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(self, x, mask_indices=None):
        B, S, D = x.shape
        if mask_indices is not None:
            mask_expanded = mask_indices.unsqueeze(-1).expand_as(x)
            x = torch.where(mask_expanded, self.mask_token.expand(B, S, D), x)
        
        out, _ = self.lstm(x) # (B, S, 512)
        reconstructed = self.recon_head(out)
        aggregated = out.mean(dim=1)
        return reconstructed, aggregated

    def aggregate(self, x):
        out, _ = self.lstm(x)
        return out.mean(dim=1)

class TransformerAggregator(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=2):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.recon_head = nn.Linear(d_model, d_model)

    def forward(self, x, mask_indices=None):
        B, S, D = x.shape
        x = x + self.pos_embedding[:, :S, :]
        if mask_indices is not None:
            mask_expanded = mask_indices.unsqueeze(-1).expand_as(x)
            x = torch.where(mask_expanded, self.mask_token.expand(B, S, D), x)
        
        encoded = self.transformer(x)
        encoded = self.norm(encoded)
        reconstructed = self.recon_head(encoded)
        aggregated = encoded.mean(dim=1)
        return reconstructed, aggregated

    def aggregate(self, x):
        B, S, D = x.shape
        x = x + self.pos_embedding[:, :S, :]
        encoded = self.transformer(x)
        encoded = self.norm(encoded)
        return encoded.mean(dim=1)

# ─── Dataset & Utils ───
class ChunkDataset(Dataset):
    def __init__(self, seqs): self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx): return torch.tensor(self.seqs[idx], dtype=torch.float32)

def collate_fn(batch):
    max_len = max(s.size(0) for s in batch)
    padded = torch.zeros(len(batch), max_len, EMBEDDING_DIM)
    lengths = []
    for i, s in enumerate(batch):
        padded[i, :s.size(0)] = s
        lengths.append(s.size(0))
    return padded, lengths

def read_wav(path):
    with wave.open(path, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

def extract_chunks(sess, audio):
    step = CHUNK_LEN - CHUNK_OVERLAP
    chunks = [audio[s:s+CHUNK_LEN] for s in range(0, len(audio)-CHUNK_LEN+1, step)]
    if not chunks: chunks.append(np.pad(audio, (0, max(0, CHUNK_LEN-len(audio))))[:CHUNK_LEN])
    
    embs = []
    for c in chunks:
        if len(c) < TARGET_AUDIO_LEN: c = np.pad(c, (0, TARGET_AUDIO_LEN-len(c)))
        embs.append(sess.run(None, {'audio': c.astype(np.float32)[np.newaxis, :]})[0][0])
    return np.array(embs)

# ─── Main ───
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = '../models_official/smartwave_cnn10_e2e_512.onnx' # 512-dim ONNX 사용!
    normal_dir, abnormal_dir = '../dataset/normal', '../dataset/abnormal'

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # 빠른 검증을 위해 정상 300개, 불량 50개만 사용
    norm_files = sorted([f for f in os.listdir(normal_dir) if f.endswith('.wav')])[:300]
    abnorm_files = sorted([f for f in os.listdir(abnormal_dir) if f.endswith('.wav')])[:50]

    print("[1] Extracting 512-dim Chunks...")
    seq_norm = [extract_chunks(sess, read_wav(os.path.join(normal_dir, f))) for f in norm_files]
    seq_abnorm = [extract_chunks(sess, read_wav(os.path.join(abnormal_dir, f))) for f in abnorm_files]

    # Baseline (단순 평균)
    X_base_n = np.array([s.mean(axis=0) for s in seq_norm])
    X_base_a = np.array([s.mean(axis=0) for s in seq_abnorm])

    loader = DataLoader(ChunkDataset(seq_norm), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    def train_model(model, name):
        print(f"\n[2] Training {name} ({EPOCHS} Epochs)...")
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        model.train()
        for ep in range(EPOCHS):
            for bx, lengths in loader:
                B, S, _ = bx.shape
                mask = torch.rand(B, S) < MASK_RATIO
                for b in range(B):
                    mask[b, lengths[b]:] = False
                    if mask[b, :lengths[b]].all(): mask[b, 0] = False
                target = bx.clone()
                recon, _ = model(bx, mask)
                mask_exp = mask.unsqueeze(-1).expand_as(target)
                loss = criterion(recon[mask_exp], target[mask_exp])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()

    def eval_model(X_n, X_a, name):
        sc = StandardScaler()
        Xn_s = sc.fit_transform(X_n)
        svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.10).fit(Xn_s)
        sn = svm.decision_function(Xn_s)
        sa = svm.decision_function(sc.transform(X_a))
        gap = sn.mean() - sa.mean()
        acc = np.mean(svm.predict(Xn_s) == 1) * 100
        print(f"  {name:15s} | Score Gap: {gap:7.4f} | Normal Acc: {acc:.1f}% | Min Score: {sn.min():.4f}")
        return gap, acc

    print("\n[3] Evaluation Results (512-dim features)")
    print("-" * 65)
    eval_model(X_base_n, X_base_a, "Baseline (Avg)")

    # LSTM
    lstm_model = LSTMAggregator()
    train_model(lstm_model, "LSTM")
    with torch.no_grad():
        X_lstm_n = np.array([lstm_model.aggregate(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).numpy()[0] for s in seq_norm])
        X_lstm_a = np.array([lstm_model.aggregate(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).numpy()[0] for s in seq_abnorm])
    eval_model(X_lstm_n, X_lstm_a, "LSTM")

    # Transformer
    tf_model = TransformerAggregator(nhead=4) # 512 is divisible by 4
    train_model(tf_model, "Transformer")
    with torch.no_grad():
        X_tf_n = np.array([tf_model.aggregate(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).numpy()[0] for s in seq_norm])
        X_tf_a = np.array([tf_model.aggregate(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).numpy()[0] for s in seq_abnorm])
    eval_model(X_tf_n, X_tf_a, "Transformer")

if __name__ == '__main__':
    main()
