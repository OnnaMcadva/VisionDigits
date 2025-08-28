import torch
import torch.nn as nn

class DummyNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(28*28, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class ModelWrapper:
    def __init__(self, path: str):
        # в реальном проекте torch.load(path)
        self.model = DummyNet(num_classes=3)
        self.labels = ["arrow", "circle", "square"]

    def predict(self, image_tensor):
        with torch.no_grad():
            outputs = self.model(image_tensor.unsqueeze(0))
            _, predicted = torch.max(outputs, 1)
        return self.labels[predicted.item()]