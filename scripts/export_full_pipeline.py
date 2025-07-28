import torch
import torch.nn as nn
from convmelspec.stft import ConvertibleSpectrogram as Spectrogram
from SimpleCNN import SimpleCNN  # or SmallerCNN

class CombinedAudioClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.mel = Spectrogram(
            sr=48000,
            n_fft=2048,
            hop_size=172,
            n_mel=128,
            power=2.0,
            spec_mode="DFT",
        )

        self.cnn = SimpleCNN(num_classes)

    def forward(self, waveform):
        # waveform: (batch_size, num_samples)
        mel = self.mel(waveform)  # (batch_size, n_mels, time_frames)
        mel = mel.unsqueeze(1)    # Add channel dim → (batch_size, 1, n_mels, time_frames)
        out = self.cnn(mel)
        return out


def export_combined_model_to_onnx(model_path, onnx_path, input_length=24000, num_classes=6):
    # 1. Instantiate the combined model
    model = CombinedAudioClassifier(num_classes=num_classes)

    # 2. Load the CNN weights (assuming you trained SimpleCNN)
    model.cnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 3. Create dummy input: batch of raw audio
    dummy_input = torch.randn(1, input_length)

    # 4. Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["waveform"],
        output_names=["class_scores"],
        dynamic_axes={"waveform": {1: "num_samples"}},  # optional, allows variable-length audio
        opset_version=16
    )

    print(f"✅ Combined model exported to: {onnx_path}")


export_combined_model_to_onnx(
    model_path="models/torch/simple_spectrogram_cnn_model_100_epoch_bs_32.pth",
    onnx_path="models/onnx/audio_to_class.onnx",
    input_length=24000,  # 0.5 seconds
    num_classes=6
)
