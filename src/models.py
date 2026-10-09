"""Model definitions: a small CNN trained from scratch and a fine-tuned ResNet-18."""
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """Three blocks of [conv3x3 - BN - ReLU] x 2 + max-pool, then dropout and a linear layer.

    Spatial size: 32x32 -> 16x16 -> 8x8 -> 4x4, with 32 -> 64 -> 128 channels.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(num_classes)
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # new 10-class head
        return model
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
