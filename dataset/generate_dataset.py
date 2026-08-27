import os
import wave
import struct
import numpy as np

def generate_wave(path, is_abnormal=False, ab_type=None):
    sr = 16000
    duration = 10
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Base motor sound (120-180 Hz)
    f0 = np.random.uniform(120, 180)
    sig = np.sin(2 * np.pi * f0 * t)
    sig += 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
    sig += 0.25 * np.sin(2 * np.pi * 3 * f0 * t)
    
    # Amplitude modulation
    mod = 1.0 + 0.2 * np.sin(2 * np.pi * 5 * t)
    sig *= mod
    
    if is_abnormal:
        if ab_type == 'A':
            # Impact noise
            clicks = np.zeros_like(sig)
            click_indices = np.random.choice(len(sig), size=20, replace=False)
            for idx in click_indices:
                if idx < len(sig) - 100:
                    clicks[idx:idx+100] = np.random.randn(100) * 2.0
            sig += clicks
        elif ab_type == 'B':
            # Highlight 2nd harmonic
            sig += 2.0 * np.sin(2 * np.pi * 2 * f0 * t)
        elif ab_type == 'C':
            # Subharmonic
            sig += 1.0 * np.sin(2 * np.pi * (f0 / 2) * t)
            
    # Add noise
    sig += np.random.randn(len(sig)) * 0.1
    
    # Normalize
    sig = sig / np.max(np.abs(sig))
    sig = (sig * 32767).astype(np.int16)
    
    with wave.open(path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(sig.tobytes())

def main():
    os.makedirs('normal', exist_ok=True)
    os.makedirs('abnormal', exist_ok=True)
    
    print("Generating normal data (1200)...")
    for i in range(1200):
        if i % 100 == 0: print(i)
        generate_wave(f'normal/norm_{i:04d}.wav')
        
    print("Generating abnormal data (300)...")
    for i in range(100): generate_wave(f'abnormal/abn_A_{i:04d}.wav', True, 'A')
    for i in range(100): generate_wave(f'abnormal/abn_B_{i:04d}.wav', True, 'B')
    for i in range(100): generate_wave(f'abnormal/abn_C_{i:04d}.wav', True, 'C')
    print("Done!")

if __name__ == '__main__':
    main()
