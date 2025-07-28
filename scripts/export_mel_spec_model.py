import torch
import torch.nn as nn
from convmelspec.stft import ConvertibleSpectrogram as Spectrogram

class MelSpectrogramModel(nn.Module):
    def __init__(self, sample_rate=48000, n_fft=2048, n_mels=128, hop_length=172):
        super().__init__()

        self.mel_spectrogram = Spectrogram(
            sr=sample_rate,
            n_fft=n_fft,
            hop_size=hop_length,
            n_mel=n_mels,
            power=2.0,
            spec_mode="DFT",
                
        )

    def forward(self, waveform):
        return self.mel_spectrogram(waveform)

# Instantiate your model
model = MelSpectrogramModel()
model.eval() # Set to evaluation mode

# Create a dummy input (e.g., a batch of waveforms)
# Shape: (batch_size, num_samples)
dummy_input = torch.randn(1, 24000)  # 0.5 seconds of audio at 48kHz

# Export the model to ONNX
try:
    torch.onnx.export(
        model,
        dummy_input,
        "mel_spectrogram_model.onnx",
        input_names=["waveform"],
        output_names=["mel_spec"],
    )
    print("Model exported to mel_spectrogram_model.onnx successfully!")
except Exception as e:
    print(f"Error during ONNX export: {e}")

# test load and run

# import onnxruntime as ort
# import numpy as np

# ort_session = ort.InferenceSession("mel_spectrogram_model.onnx")
# inputs = {"waveform": dummy_input.numpy()}
# outputs = ort_session.run(None, inputs)

# print(outputs)