import onnxruntime as ort
import numpy as np
import librosa

def prepare_audio(wav_path, target_length=160000, sr=16000):
    audio, _ = librosa.load(wav_path, sr=sr)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    elif len(audio) > target_length:
        audio = audio[:target_length]
    audio_batch = np.expand_dims(audio, axis=0)
    return audio_batch.astype(np.float32)

if __name__ == "__main__":
    wav_path = "c:/Users/user/Desktop/SmartWave/models_official/norm_0001.wav"
    audio_input = prepare_audio(wav_path)
    
    # Run 512 model
    sess512 = ort.InferenceSession("c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e_512.onnx")
    out512 = sess512.run(None, {sess512.get_inputs()[0].name: audio_input})[0]
    
    # Run modified model
    sess_mod = ort.InferenceSession("c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e_modified.onnx")
    outputs = sess_mod.run(None, {sess_mod.get_inputs()[0].name: audio_input})
    
    out_512_from_original = None
    for i, out_desc in enumerate(sess_mod.get_outputs()):
        if out_desc.name == "/Relu_output_0":
            out_512_from_original = outputs[i]
            
    # Calculate ratios where out512 > 0
    mask = out512 > 1e-6
    ratios = out_512_from_original[mask] / out512[mask]
    
    print(f"Number of non-zero elements: {np.sum(mask)}")
    print(f"Ratios: {ratios[:10]}")
    print(f"Mean ratio: {np.mean(ratios):.6f}")
    print(f"Std ratio: {np.std(ratios):.6e}")
