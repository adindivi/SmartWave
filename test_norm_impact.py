import onnxruntime as ort
import numpy as np
import onnx
from onnx import numpy_helper

def test_classifier_with_normalized_input():
    model_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e.onnx"
    model = onnx.load(model_path)
    
    # Extract weights and bias of fc_audioset
    W = None
    b = None
    for init in model.graph.initializer:
        if init.name == 'fc_audioset.weight':
            W = numpy_helper.to_array(init)
        elif init.name == 'fc_audioset.bias':
            b = numpy_helper.to_array(init)
            
    if W is None or b is None:
        print("Could not find weights or bias")
        return
        
    print(f"W shape: {W.shape}, b shape: {b.shape}")
    
    # Get the un-normalized 512 feature from earlier (using our modified model)
    sess_mod = ort.InferenceSession("c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e_527_with_512_layer.onnx")
    
    import librosa
    audio, _ = librosa.load("c:/Users/user/Desktop/SmartWave/models_official/norm_0001.wav", sr=16000)
    audio = np.pad(audio, (0, 160000 - len(audio)), mode='constant') if len(audio) < 160000 else audio[:160000]
    audio_input = np.expand_dims(audio, axis=0).astype(np.float32)
    
    outputs = sess_mod.run(None, {sess_mod.get_inputs()[0].name: audio_input})
    
    out_527 = None
    unnormalized_512 = None
    for i, out_desc in enumerate(sess_mod.get_outputs()):
        if out_desc.name == '/Relu_output_0':
            unnormalized_512 = outputs[i]
        elif out_desc.name == 'embedding':
            out_527 = outputs[i]
            
    # Calculate original logits
    # The original graph: Gemm(A, B, C) where A=unnormalized_512, B=W.T, C=b (usually W is stored as (out, in), so we use W.T)
    logits_unnorm = np.dot(unnormalized_512, W.T) + b
    # Original final output applies L2 norm
    final_output_from_unnorm = logits_unnorm / np.linalg.norm(logits_unnorm, axis=1, keepdims=True)
    
    # Calculate logits with normalized 512 input
    norm_factor = np.linalg.norm(unnormalized_512, axis=1, keepdims=True)
    normalized_512 = unnormalized_512 / norm_factor
    
    logits_norm = np.dot(normalized_512, W.T) + b
    # And if we L2 normalize that...
    final_output_from_norm = logits_norm / np.linalg.norm(logits_norm, axis=1, keepdims=True)
    
    print("\nCompare final outputs:")
    diff = np.abs(final_output_from_unnorm - final_output_from_norm)
    print(f"Max diff: {np.max(diff):.6f}")
    print(f"Mean diff: {np.mean(diff):.6f}")

    print("\nSanity check vs ONNX output:")
    diff_onnx = np.abs(final_output_from_unnorm - out_527)
    print(f"Diff between calculated unnorm and ONNX output: {np.max(diff_onnx):.6e}")
    
if __name__ == "__main__":
    test_classifier_with_normalized_input()
