import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.numpy import load_file
import onnxruntime as ort
import numpy as np
import os

class STFT(nn.Module):
    def __init__(self):
        super(STFT, self).__init__()
        self.conv_real = nn.Conv1d(1, 513, 1024, stride=320, padding=0, bias=False)
        self.conv_imag = nn.Conv1d(1, 513, 1024, stride=320, padding=0, bias=False)
    
    def forward(self, x):
        # x: (batch, time)
        x = x.unsqueeze(1) # (batch, 1, time)
        x = F.pad(x, pad=(512, 512), mode='reflect')
        real = self.conv_real(x)
        imag = self.conv_imag(x)
        power = real**2 + imag**2
        power = power.unsqueeze(1).transpose(2, 3) # (batch, 1, time_steps, 513)
        return power

class LogMelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.melW = nn.Parameter(torch.empty(513, 64))
        
    def forward(self, x):
        mel = torch.matmul(x, self.melW)
        mel = torch.clamp(mel, min=1e-10)
        logmel = 10.0 * torch.log10(mel)
        return logmel # (batch, 1, time_steps, 64)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 2)
        return x

class Cnn10(nn.Module):
    def __init__(self):
        super(Cnn10, self).__init__()
        self.stft = STFT()
        self.logmel = LogMelExtractor()
        
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, 527, bias=True)
        
    def forward(self, audio):
        x = self.stft(audio)
        x = self.logmel(x) 
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        x = F.relu(self.fc1(x))
        x = self.fc_audioset(x)
        
        x = F.normalize(x, p=2, dim=1)
        return x

def build():
    model = Cnn10()
    model.eval()

    d = load_file('models_official/model.safetensors')
    state_dict = model.state_dict()

    mapping = {
        'stft.conv_real.weight': 'backbone.spectrogram_extractor.stft.conv_real.weight',
        'stft.conv_imag.weight': 'backbone.spectrogram_extractor.stft.conv_imag.weight',
        'logmel.melW': 'backbone.logmel_extractor.melW',
    }

    for name in state_dict.keys():
        if name in mapping:
            safetensor_key = mapping[name]
        else:
            safetensor_key = 'backbone.' + name
        
        if safetensor_key in d:
            state_dict[name].copy_(torch.from_numpy(d[safetensor_key]))
        else:
            print("Missing in safetensors:", safetensor_key)

    print("Exporting ONNX...")
    audio = torch.randn(1, 160000)
    torch.onnx.export(
        model, audio, "models_official/smartwave_cnn10_e2e.onnx",
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['embedding'],
        dynamic_axes={'audio': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
    )

    print("Verifying ONNX...")
    sess = ort.InferenceSession('models_official/smartwave_cnn10_e2e.onnx', providers=['CPUExecutionProvider'])
    audio_np = (np.random.randn(1, 160000).astype(np.float32) * 0.05)
    emb = sess.run(None, {'audio': audio_np})[0][0]
    print(f'[OK] shape={emb.shape}, norm={np.linalg.norm(emb):.4f}')

if __name__ == '__main__':
    build()
