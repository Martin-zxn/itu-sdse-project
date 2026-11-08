"""The convolutional model used by the vial classification notebook."""

from pathlib import Path
from typing import Union

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """CNN for binary classification of 64x64 RGB vial images."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.pool(self.relu(self.conv1(inputs)))
        inputs = self.pool(self.relu(self.conv2(inputs)))
        inputs = inputs.flatten(start_dim=1)
        inputs = self.relu(self.fc1(inputs))
        inputs = self.dropout(inputs)
        return self.fc2(inputs)


def load_model(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> SimpleCNN:
    """Load a SimpleCNN from the checkpoint format used by the notebook."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Model checkpoint must contain a state dictionary.")

    architecture = checkpoint.get("architecture", "SimpleCNN")
    if architecture != "SimpleCNN":
        raise ValueError(f"Unsupported model architecture: {architecture}")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Model checkpoint has no valid model state dictionary.")

    model = SimpleCNN()
    model.load_state_dict(state_dict)
    model.to(map_location)
    model.eval()
    return model
