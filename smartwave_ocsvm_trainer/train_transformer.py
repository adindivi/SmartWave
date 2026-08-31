"""
SmartWave Transformer Self-Supervised Training (Stage 1)
========================================================
Train a mini Transformer on NORMAL data only using masked reconstruction.

Strategy:
  1. Extract chunk embeddings from all normal audio (cache to disk)
  2. Train Transformer: mask 30% of chunks → reconstruct them
  3. Compare: Baseline vs Random Transformer vs TRAINED Transformer
"""
import os
import sys
import wave
import time
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
SAMPLE_RATE = 16000
CHUNK_LEN = 16000
CHUNK_OVERLAP = 8000
EMBEDDING_DIM = 527
MASK_RATIO = 0.3
EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-4


# ─── Transformer Model ───
class MaskedTransformerAggregator(nn.Module):
    """Transformer that learns normal temporal patterns via masked reconstruction."""

    def __init__(self, d_model=527, nhead=1, num_layers=2, dim_feedforward=1024):
        super().__init__()
        self.d_model = d_model

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, d_model) * 0.02)
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        # Reconstruction head
        self.recon_head = nn.Linear(d_model, d_model)

    def forward(self, x, mask_indices=None):
        """
        x: (batch, seq_len, 527)
        mask_indices: (batch, seq_len) bool tensor, True = masked
        Returns: reconstructed (batch, seq_len, 527), aggregated (batch, 527)
        """
        B, S, D = x.shape
        x = x + self.pos_embedding[:, :S, :]

        # Apply masking during training
        if mask_indices is not None:
            mask_expanded = mask_indices.unsqueeze(-1).expand_as(x)
            x = torch.where(mask_expanded, self.mask_token.expand(B, S, D), x)

        encoded = self.transformer(x)
        encoded = self.norm(encoded)

        reconstructed = self.recon_head(encoded)
        aggregated = encoded.mean(dim=1)

        return reconstructed, aggregated

    def aggregate(self, x):
        """Inference: just aggregate, no masking."""
        B, S, D = x.shape
        x = x + self.pos_embedding[:, :S, :]
        encoded = self.transformer(x)
        encoded = self.norm(encoded)
        return encoded.mean(dim=1)


# ─── Dataset ───
class ChunkEmbeddingDataset(Dataset):
    """Dataset of chunk embedding sequences for self-supervised training."""

    def __init__(self, sequences):
        self.sequences = sequences  # list of (num_chunks, 527) arrays

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.float32)


def collate_fn(batch):
    """Pad sequences to the same length in a batch."""
    max_len = max(s.size(0) for s in batch)
    padded = torch.zeros(len(batch), max_len, EMBEDDING_DIM)
    lengths = []
    for i, s in enumerate(batch):
        padded[i, :s.size(0)] = s
        lengths.append(s.size(0))
    return padded, lengths


# ─── Helpers ───
def read_wav(path):
    with wave.open(path, 'rb') as w:
        raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def slice_chunks(audio):
    step = CHUNK_LEN - CHUNK_OVERLAP
    chunks = []
    for start in range(0, len(audio) - CHUNK_LEN + 1, step):
        chunks.append(audio[start:start + CHUNK_LEN])
    if not chunks:
        padded = np.pad(audio, (0, max(0, CHUNK_LEN - len(audio))))
        chunks.append(padded[:CHUNK_LEN])
    return chunks


def extract_single(sess, audio):
    if len(audio) < TARGET_AUDIO_LEN:
        audio = np.pad(audio, (0, TARGET_AUDIO_LEN - len(audio)))
    else:
        audio = audio[:TARGET_AUDIO_LEN]
    return sess.run(None, {'audio': audio.astype(np.float32)[np.newaxis, :]})[0][0]


def extract_chunks(sess, audio):
    chunks = slice_chunks(audio)
    embs = []
    for c in chunks:
        if len(c) < TARGET_AUDIO_LEN:
            c = np.pad(c, (0, TARGET_AUDIO_LEN - len(c)))
        embs.append(sess.run(None, {'audio': c.astype(np.float32)[np.newaxis, :]})[0][0])
    return np.array(embs)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    onnx_path = '../models_official/smartwave_cnn10_e2e.onnx'
    normal_dir = '../dataset/normal'
    abnormal_dir = '../dataset/abnormal'

    for p in [onnx_path, normal_dir, abnormal_dir]:
        if not os.path.exists(p):
            print(f"[ERROR] Not found: {p}")
            return

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    normal_files = sorted([f for f in os.listdir(normal_dir) if f.endswith('.wav')])
    abnormal_files = sorted([f for f in os.listdir(abnormal_dir) if f.endswith('.wav')])
    print(f"Normal: {len(normal_files)} files, Abnormal: {len(abnormal_files)} files")

    # ═══════════════════════════════════════════════════════
    # STEP 1: Extract & Cache All Chunk Embeddings
    # ═══════════════════════════════════════════════════════
    print("\n[Step 1] Extracting chunk embeddings from all files...")
    t0 = time.time()

    normal_seqs = []
    normal_singles = []
    for i, f in enumerate(normal_files):
        audio = read_wav(os.path.join(normal_dir, f))
        normal_singles.append(extract_single(sess, audio))
        normal_seqs.append(extract_chunks(sess, audio))
        if (i + 1) % 50 == 0:
            print(f"  Normal: {i+1}/{len(normal_files)}")

    abnormal_seqs = []
    abnormal_singles = []
    for i, f in enumerate(abnormal_files):
        audio = read_wav(os.path.join(abnormal_dir, f))
        abnormal_singles.append(extract_single(sess, audio))
        abnormal_seqs.append(extract_chunks(sess, audio))
        if (i + 1) % 50 == 0:
            print(f"  Abnormal: {i+1}/{len(abnormal_files)}")

    extract_time = time.time() - t0
    print(f"  Extraction complete: {extract_time:.1f}s")
    print(f"  Chunks per normal file: {normal_seqs[0].shape[0]}")

    # ═══════════════════════════════════════════════════════
    # STEP 2: Self-Supervised Training (Masked Reconstruction)
    # ═══════════════════════════════════════════════════════
    print(f"\n[Step 2] Training Transformer ({EPOCHS} epochs, mask={MASK_RATIO*100:.0f}%)...")
    t0 = time.time()

    model = MaskedTransformerAggregator(d_model=527, nhead=1, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    dataset = ChunkEmbeddingDataset(normal_seqs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        count = 0
        for batch_x, lengths in loader:
            B, S, D = batch_x.shape

            # Create random mask
            mask = torch.rand(B, S) < MASK_RATIO
            # Don't mask padding
            for b in range(B):
                mask[b, lengths[b]:] = False
                # Ensure at least 1 unmasked
                if mask[b, :lengths[b]].all():
                    mask[b, 0] = False

            target = batch_x.clone()
            reconstructed, _ = model(batch_x, mask_indices=mask)

            # Loss only on masked positions
            mask_expanded = mask.unsqueeze(-1).expand_as(target)
            loss = criterion(
                reconstructed[mask_expanded],
                target[mask_expanded]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {total_loss/count:.6f}")

    train_time = time.time() - t0
    print(f"  Training complete: {train_time:.1f}s")

    # ═══════════════════════════════════════════════════════
    # STEP 3: Evaluate All Three Methods
    # ═══════════════════════════════════════════════════════
    print("\n[Step 3] Evaluating 3 methods...")
    model.eval()

    # A) Baseline
    X_n_base = np.array(normal_singles)
    X_a_base = np.array(abnormal_singles)
    sc_base = StandardScaler()
    Xn_s = sc_base.fit_transform(X_n_base)
    ocsvm_base = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.10)
    ocsvm_base.fit(Xn_s)
    sn_base = ocsvm_base.decision_function(Xn_s)
    sa_base = ocsvm_base.decision_function(sc_base.transform(X_a_base))
    acc_n_base = np.mean(ocsvm_base.predict(Xn_s) == 1) * 100
    acc_a_base = np.mean(ocsvm_base.predict(sc_base.transform(X_a_base)) == -1) * 100

    # B) Trained Transformer
    with torch.no_grad():
        X_n_trans = []
        for seq in normal_seqs:
            t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            agg = model.aggregate(t).numpy()[0]
            X_n_trans.append(agg)
        X_n_trans = np.array(X_n_trans)

        X_a_trans = []
        for seq in abnormal_seqs:
            t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            agg = model.aggregate(t).numpy()[0]
            X_a_trans.append(agg)
        X_a_trans = np.array(X_a_trans)

    sc_trans = StandardScaler()
    Xn_t = sc_trans.fit_transform(X_n_trans)
    ocsvm_trans = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.10)
    ocsvm_trans.fit(Xn_t)
    sn_trans = ocsvm_trans.decision_function(Xn_t)
    sa_trans = ocsvm_trans.decision_function(sc_trans.transform(X_a_trans))
    acc_n_trans = np.mean(ocsvm_trans.predict(Xn_t) == 1) * 100
    acc_a_trans = np.mean(ocsvm_trans.predict(sc_trans.transform(X_a_trans)) == -1) * 100

    # ═══════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════
    gap_base = sn_base.mean() - sa_base.mean()
    gap_trans = sn_trans.mean() - sa_trans.mean()

    print("\n" + "=" * 70)
    print("FINAL RESULTS: Baseline vs Trained Transformer")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Baseline':>12} {'Trained TF':>12}")
    print("-" * 60)
    print(f"{'Normal Score Mean':<35} {sn_base.mean():>12.4f} {sn_trans.mean():>12.4f}")
    print(f"{'Normal Score Min':<35} {sn_base.min():>12.4f} {sn_trans.min():>12.4f}")
    print(f"{'Abnormal Score Mean':<35} {sa_base.mean():>12.4f} {sa_trans.mean():>12.4f}")
    print(f"{'Score Gap (larger=better)':<35} {gap_base:>12.4f} {gap_trans:>12.4f}")
    print(f"{'Normal Accuracy':<35} {acc_n_base:>11.1f}% {acc_n_trans:>11.1f}%")
    print(f"{'Abnormal Detection Rate':<35} {acc_a_base:>11.1f}% {acc_a_trans:>11.1f}%")

    improvement = ((gap_trans / max(gap_base, 0.0001)) - 1) * 100
    print(f"\n{'Score Gap Change:':<35} {improvement:>+.1f}%")

    if acc_n_trans > acc_n_base:
        print(f"{'Normal Accuracy Change:':<35} {acc_n_trans - acc_n_base:>+.1f}% (IMPROVED)")
    else:
        print(f"{'Normal Accuracy Change:':<35} {acc_n_trans - acc_n_base:>+.1f}%")

    if gap_trans > gap_base and acc_n_trans >= acc_n_base:
        print("\n>>> VERDICT: Trained Transformer WINS on both metrics!")
    elif gap_trans > gap_base:
        print("\n>>> VERDICT: Trained Transformer has better separation but needs more tuning for normal accuracy.")
    else:
        print("\n>>> VERDICT: Baseline still superior. More training data or epochs may help.")

    # Save model weights
    save_path = 'transformer_weights.pt'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved to: {save_path}")


if __name__ == '__main__':
    main()
