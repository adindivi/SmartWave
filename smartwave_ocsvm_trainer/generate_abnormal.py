import numpy as np
import wave
import os
import sys

# Create abnormal folder
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/abnormal'))
os.makedirs(output_dir, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 10

def generate_abnormal_wave(filename):
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, False)
    
    # 1. Base motor sound (50 Hz hum)
    base = 0.2 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))
    
    # 2. ANOMALY: High-frequency mechanical grinding/whining noise (1200 Hz modulated)
    # This simulates a broken bearing or severe friction in the motor.
    grinding = 0.4 * np.sin(2 * np.pi * 1200 * t) * np.sin(2 * np.pi * 5 * t)
    
    signal = base + grinding
    
    # Normalize to -1.0 ~ 1.0
    signal = signal / np.max(np.abs(signal))
    
    # Convert to 16-bit PCM
    signal_int16 = (signal * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2) # 2 bytes = 16 bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(signal_int16.tobytes())

print(f"Generating 100 dummy ABNORMAL (grinding noise) samples into:\n{output_dir}")
for i in range(100):
    filepath = os.path.join(output_dir, f'dummy_abnormal_{i:04d}.wav')
    generate_abnormal_wave(filepath)
    if (i+1) % 20 == 0:
        print(f"[{i+1}/100] Generated...")

print("✅ Complete! You can now use the 'EVALUATE ABNORMAL' button in the trainer.")
