import onnxruntime as ort
import numpy as np
import librosa
import json
import sys

def run_inference(onnx_path, wav_path):
    print(f"Loading {onnx_path}...")
    session = ort.InferenceSession(onnx_path)
    
    input_details = session.get_inputs()
    output_details = session.get_outputs()
    
    print(f"Inputs:")
    for i in input_details:
        print(f"  Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
        
    print(f"Outputs:")
    for o in output_details:
        print(f"  Name: {o.name}, Shape: {o.shape}, Type: {o.type}")

if __name__ == "__main__":
    print("--- smartwave_cnn10_e2e.onnx ---")
    run_inference("c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e.onnx", "c:/Users/user/Desktop/SmartWave/models_official/norm_0001.wav")
    print("\n--- smartwave_cnn10_e2e_512.onnx ---")
    run_inference("c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e_512.onnx", "c:/Users/user/Desktop/SmartWave/models_official/norm_0001.wav")
