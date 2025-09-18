import torch
from app.simple_cnn import SimpleCNN

# import torch.nn as nn

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64*5*5, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = torch.flatten(x, 1)
#         x = torch.relu(self.fc1(x))
#         return self.fc2(x)


class ModelWrapper:
    def __init__(self, path: str, num_classes: int = 10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        except FileNotFoundError:
            raise RuntimeError(f"Model weights file not found: {path}")
        self.model.eval()
        self.labels = [str(i) for i in range(num_classes)]

    def predict(self, image_tensor):
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # (1,1,28,28)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        return self.labels[predicted.item()]
