from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


# ImageFolder 的实际索引顺序，和训练脚本中未使用的 classes 列表不同。
CLASSES = (
    ("Cr", "Crazing", "龟裂"),
    ("In", "Inclusion", "夹杂"),
    ("PS", "Pitted surface", "麻点"),
    ("Pa", "Patches", "斑块"),
    ("RS", "Rolled-in scale", "轧入氧化皮"),
    ("Sc", "Scratches", "划痕"),
)

PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1)
        self.conv3 = (
            nn.Conv2d(input_channels, output_channels, 1, stride)
            if stride != 1
            else None
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        residual = x if self.conv3 is None else self.conv3(x)
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + residual)


def build_model():
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(1, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        ),
        nn.Sequential(ResBlock(64, 64), ResBlock(64, 64)),
        nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128)),
        nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256)),
        nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512)),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, len(CLASSES)),
    )


class SteelClassifier:
    def __init__(self, weights: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image):
        batch = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(self.device)
        scores = self.model(batch).softmax(dim=1)[0].tolist()
        return sorted(zip(CLASSES, scores), key=lambda result: result[1], reverse=True)
