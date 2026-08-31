import onnx
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

def extract_intermediate_and_compare():
    model_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e.onnx"
    model512_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e_512.onnx"
    wav_path = "c:/Users/user/Desktop/SmartWave/models_official/norm_0001.wav"

    audio_input = prepare_audio(wav_path)
    
    sess512 = ort.InferenceSession(model512_path)
    out512 = sess512.run(None, {sess512.get_inputs()[0].name: audio_input})[0]

    print("Loading ONNX model to modify outputs...")
    model = onnx.load(model_path)
    
    print("\nLooking for Gemm/MatMul nodes near the end...")
    target_node = None
    for node in model.graph.node:
        if node.op_type in ["Gemm", "MatMul"]:
            print(f"Found {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")
            target_node = node
            
    # target_node should now be the last Gemm/MatMul.
    # Its first input is usually the features from the previous layer.
    intermediate_tensor_name = target_node.input[0]
    print(f"\nIdentifying intermediate tensor: {intermediate_tensor_name}")
    
    inferred_model = onnx.shape_inference.infer_shapes(model)
    
    intermediate_info = None
    for vi in inferred_model.graph.value_info:
        if vi.name == intermediate_tensor_name:
            intermediate_info = vi
            break
            
    if not intermediate_info:
        intermediate_info = onnx.helper.make_tensor_value_info(
            intermediate_tensor_name,
            onnx.TensorProto.FLOAT,
            [None, 512]
        )
        
    inferred_model.graph.output.append(intermediate_info)
    
    modified_model_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e_modified.onnx"
    onnx.save(inferred_model, modified_model_path)
    
    print("\nRunning inference on modified model...")
    sess_mod = ort.InferenceSession(modified_model_path)
    outputs = sess_mod.run(None, {sess_mod.get_inputs()[0].name: audio_input})
    
    out_512_from_original = None
    for i, out_desc in enumerate(sess_mod.get_outputs()):
        if out_desc.name == intermediate_tensor_name:
            out_512_from_original = outputs[i]
            
    if out_512_from_original is None:
        out_512_from_original = outputs[-1]

    print(f"\nExtracted shape: {out_512_from_original.shape}")
    print(f"512-model shape: {out512.shape}")
    
    print("\nFirst 10 values (Original model's intermediate 512):")
    print(out_512_from_original[0][:10])
    
    print("\nFirst 10 values (Cut model's 512 output):")
    print(out512[0][:10])
    
    diff = np.abs(out_512_from_original - out512)
    print(f"\nMax difference between the two 512 outputs: {np.max(diff):.6e}")
    print(f"Mean difference: {np.mean(diff):.6e}")

if __name__ == "__main__":
    extract_intermediate_and_compare()
