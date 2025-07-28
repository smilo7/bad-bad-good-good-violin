import torch.nn as nn
import torch.nn.functional as F

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