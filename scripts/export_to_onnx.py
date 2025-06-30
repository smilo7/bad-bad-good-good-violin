import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SmallerCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallerCNN, self).__init__()
        # Input shape: (Batch, 1, n_mels, target_spectrogram_width) e.g., (Batch, 1, 128, 128)
        # Reduce filters and layers

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1) # Reduced filters
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (Batch, 8, 64, 64)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1) # Reduced filters
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (Batch, 16, 32, 32)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # Reduced filters
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (Batch, 32, 16, 16)

        self._to_linear = 32 * 16 * 16 # Match output of pool3

        # Reduce the size of the fully connected layer
        self.fc1 = nn.Linear(self._to_linear, 128) # Reduced units in FC layer
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes) # Match the output of fc1

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(-1, self._to_linear)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Input shape: (Batch, 1, n_mels, target_spectrogram_width) e.g., (Batch, 1, 128, 128)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # Add Batch Normalization
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (Batch, 16, 64, 64) assuming 128x128 input

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32) # Add Batch Normalization
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (Batch, 32, 32, 32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64) # Add Batch Normalization
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (Batch, 64, 16, 16)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128) # Add Batch Normalization
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (Batch, 128, 8, 8)

        self._to_linear = 128 * 8 * 8 # This needs to match the actual output size

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.dropout = nn.Dropout(0.5) # Add dropout for regularization
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten the output for the fully connected layers
        x = x.view(-1, self._to_linear) # -1 means infer batch size

        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Apply dropout
        x = self.fc2(x) # No activation here, CrossEntropyLoss expects raw scores

        return x

def export_model_to_onnx(model_path, onnx_path, num_classes, input_shape=(1, 1, 128, 128)):
    """
    Exports a trained PyTorch model to ONNX format.

    Args:
        model_path (str): Path to the saved PyTorch model (.pth).
        onnx_path (str): Desired path for the ONNX output file.
        num_classes (int): Number of output classes for your model.
        input_shape (tuple): Example input shape for the model (Batch, Channels, Height, Width).
                             Defaults to (1, 1, 128, 128) which is typical for single-channel
                             mel-spectrograms of 128x128.
    """
    # 1. Instantiate the model
    # model = SmallerCNN(num_classes=num_classes)
    model = SimpleCNN(num_classes=num_classes)  # Use the SimpleCNN or SmallerCNN as needed

    # 2. Load the trained weights
    # Make sure to map to CPU if your model was trained on GPU but you're exporting on CPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # 3. Set the model to evaluation mode
    model.eval()

    # 4. Create a dummy input tensor
    # This is crucial for ONNX export to trace the model's operations.
    # The batch size can be 1 for a single inference, or variable if you want to support
    # dynamic batching in ONNX (though typically fixed batch size is easier for web).
    dummy_input = torch.randn(input_shape, requires_grad=False)

    # 5. Define input and output names (optional but good practice)
    input_names = ["input"]
    output_names = ["output"]

    # 6. Export the model
    try:
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          verbose=True, # Set to True for more detailed output during export
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=11, # Common opset version, adjust if needed
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}} # Allows dynamic batch size
                         )
        print(f"Model successfully exported to {onnx_path}")
    except Exception as e:
        print(f"Error exporting model: {e}")

if __name__ == "__main__":
    # Replace with your actual path to the trained PyTorch model
    trained_model_file = "models/simple_spectrogram_cnn_model_100_epoch_bs_32.pth"  # Path to your trained model file
    output_onnx_file = "simple_cnn.onnx"
    num_classes = 6 # Replace with the actual number of classes your model predicts

    # Create a dummy trained model file for demonstration if it doesn't exist
    if not os.path.exists(trained_model_file):
        print(f"Creating a dummy model file for demonstration at {trained_model_file}")
        model = SmallerCNN(num_classes=num_classes)
        # Train your model and save its state_dict like this:
        # torch.save(model.state_dict(), trained_model_file)
        # For demonstration, just saving an untrained model's state_dict
        torch.save(model.state_dict(), trained_model_file)
        print("Dummy model file created. In a real scenario, this would be your *trained* model.")

    # Call the export function
    export_model_to_onnx(trained_model_file, output_onnx_file, num_classes)

    # verify the ONNX model (optional, requires onnx library)
    try:
        import onnx
        onnx_model = onnx.load(output_onnx_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check successful!")
    except ImportError:
        print("ONNX library not installed. Cannot perform ONNX model check.")
        print("Install with: pip install onnx")
    except Exception as e:
        print(f"ONNX model check failed: {e}")