import onnx
import onnxruntime as ort
import numpy as np
import librosa
import sys

def prepare_audio(wav_path, target_length=160000, sr=16000):
    audio, _ = librosa.load(wav_path, sr=sr)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    elif len(audio) > target_length:
        audio = audio[:target_length]
    audio_batch = np.expand_dims(audio, axis=0)
    return audio_batch.astype(np.float32)

def main():
    model_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e.onnx"
    model512_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e_512.onnx"
    wav_path = "c:/Users/user/Desktop/SmartWave/models_official/norm_0001.wav"

    audio_input = prepare_audio(wav_path)

    # 1. 기존 527 모델을 로드하여 512차원 출력(Relu_output_0)을 그래프 출력에 추가
    model = onnx.load(model_path)
    
    intermediate_tensor_name = '/Relu_output_0' # fc_audioset 바로 직전의 512차원 특징 텐서
    
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
    
    modified_model_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e_527_with_512_layer.onnx"
    onnx.save(inferred_model, modified_model_path)
    
    # 2. 수정된 모델로 추론
    sess_mod = ort.InferenceSession(modified_model_path)
    outputs = sess_mod.run(None, {sess_mod.get_inputs()[0].name: audio_input})
    
    out_527 = None
    out_512_from_527 = None
    for i, out_desc in enumerate(sess_mod.get_outputs()):
        if out_desc.name == intermediate_tensor_name:
            out_512_from_527 = outputs[i]
        elif out_desc.name == 'embedding':
            out_527 = outputs[i]
            
    # 3. 잘려진 512 전용 모델 추론
    sess512 = ort.InferenceSession(model512_path)
    out_512_standalone = sess512.run(None, {sess512.get_inputs()[0].name: audio_input})[0]
    
    # 결과 비교 출력
    print("=== 모델 출력 비교 ===")
    print(f"\n1. 기존 527 모델에서 뽑아낸 512차원 특징 (Shape: {out_512_from_527.shape})")
    print(out_512_from_527[0][:10])
    
    print(f"\n2. 잘려진 512 모델의 512차원 출력 (Shape: {out_512_standalone.shape})")
    print(out_512_standalone[0][:10])
    
    print("\n=== 차이점 분석 ===")
    # L2 정규화 검증
    l2_norm = np.linalg.norm(out_512_from_527)
    print(f"527 모델에서 뽑은 512 특징의 L2 Norm 값: {l2_norm:.4f}")
    
    normalized_512 = out_512_from_527 / l2_norm
    diff = np.abs(normalized_512 - out_512_standalone)
    
    print("\n527 모델의 512 특징을 L2 정규화(L2 Norm으로 나눔) 한 후의 첫 10개 값:")
    print(normalized_512[0][:10])
    print(f"정규화된 값과 512 모델 출력의 최대 차이: {np.max(diff):.6e} (거의 0에 수렴)")

if __name__ == "__main__":
    main()
