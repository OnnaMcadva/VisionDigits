import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Простая сверточная нейронная сеть для распознавания цифр.
    Архитектура: Conv2d -> Conv2d -> Conv2d -> Dropout -> FC -> FC
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Первый сверточный блок
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Второй сверточный блок
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Третий сверточный блок (новый)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.25)
        
        # Полносвязные слои
        # После трех сверточных слоев и пулингов размер: 128*1*1 = 128
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Первый сверточный блок
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Второй сверточный блок
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Третий сверточный блок
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Преобразование в одномерный тензор
        x = torch.flatten(x, 1)
        
        # Dropout перед полносвязными слоями
        x = self.dropout(x)
        
        # Полносвязные слои
        x = torch.relu(self.fc1(x))
        return self.fc2(x)