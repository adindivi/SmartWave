from huggingface_hub import hf_hub_download
import os
OUT = 'models_official'
os.makedirs(OUT, exist_ok=True)

print('[1/3] CNN10 safetensors...')
hf_hub_download('nicofarr/panns_Cnn10', 'model.safetensors', local_dir=OUT)

print('[2/3] CNN14 TFLite (155MB)...')
hf_hub_download('litert-community/PANNs-CNN14-AudioSet-LiteRT', 'cnn14_audioset_fp16.tflite', local_dir=OUT)

print('[3/3] Mel filterbank...')
hf_hub_download('litert-community/PANNs-CNN14-AudioSet-LiteRT', 'mel_basis.bin', local_dir=OUT)

print('다운로드 완료!')
