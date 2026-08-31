import onnxruntime as ort
import numpy as np
import librosa
import sys

def prepare_audio(wav_path, target_length=160000, sr=16000):
    # Load audio
    audio, _ = librosa.load(wav_path, sr=sr)
    
    # Pad or truncate to target_length
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    elif len(audio) > target_length:
        audio = audio[:target_length]
        
    # Add batch dimension
    audio_batch = np.expand_dims(audio, axis=0)
    
    # Ensure float32
    return audio_batch.astype(np.float32)

def run_inference(onnx_path, audio_input):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: audio_input})
    return outputs[0]

if __name__ == "__main__":
    wav_path = "c:/Users/user/Desktop/SmartWave/models_official/norm_0001.wav"
    model1_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e.onnx"
    model2_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e_512.onnx"
    
    print("Preparing audio...")
    audio_input = prepare_audio(wav_path)
    print(f"Audio input shape: {audio_input.shape}")
    
    print(f"\nRunning inference on {model1_path}...")
    out1 = run_inference(model1_path, audio_input)
    print(f"Output shape: {out1.shape}")
    print(f"First 10 values: {out1[0][:10]}")
    
    print(f"\nRunning inference on {model2_path}...")
    out2 = run_inference(model2_path, audio_input)
    print(f"Output shape: {out2.shape}")
    print(f"First 10 values: {out2[0][:10]}")
    
    print("\nComparison:")
    print(f"Model 1 output max: {np.max(out1):.4f}, min: {np.min(out1):.4f}, mean: {np.mean(out1):.4f}")
    print(f"Model 2 output max: {np.max(out2):.4f}, min: {np.min(out2):.4f}, mean: {np.mean(out2):.4f}")
