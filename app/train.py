import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.simple_cnn import SimpleCNN


def calculate_accuracy(model, data_loader, device):
    """
    Вычисление точности модели на датасете.
    
    Args:
        model: Модель для оценки
        data_loader: DataLoader с данными
        device: Устройство для вычислений
        
    Returns:
        float: Точность в процентах
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


# Параметры обучения
batch_size = 64
epochs = 10  # Увеличено количество эпох
lr = 0.001

# Преобразования для данных
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Загрузка данных
print("📁 Загрузка данных MNIST...")
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"📊 Размер обучающей выборки: {len(train_dataset)}")
print(f"📊 Размер тестовой выборки: {len(test_dataset)}")

# Настройка устройства и модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Используется устройство: {device}")

model = SimpleCNN(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

print("🚀 Начало обучения...")
print(f"🔢 Параметры: epochs={epochs}, batch_size={batch_size}, lr={lr}")

# Обучение модели
model.train()
for epoch in range(epochs):
    total_loss = 0
    
    # Прогресс-бар для батчей
    train_pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}")
    
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Обновление прогресс-бара
        train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Вычисление метрик
    avg_loss = total_loss / len(train_loader)
    train_accuracy = calculate_accuracy(model, train_loader, device)
    test_accuracy = calculate_accuracy(model, test_loader, device)
    
    print(f"Эпоха {epoch+1}/{epochs}")
    print(f"  📉 Loss: {avg_loss:.4f}")
    print(f"  🎯 Train Accuracy: {train_accuracy:.2f}%")
    print(f"  🎯 Test Accuracy: {test_accuracy:.2f}%")
    print("-" * 50)

# Проверка существования файла модели перед сохранением
model_path = "model/sketch_cnn.pt"
if os.path.exists(model_path):
    print(f"⚠️  Файл {model_path} уже существует!")
    response = input("Перезаписать? (y/N): ").lower().strip()
    if response != 'y':
        print("❌ Сохранение отменено")
        exit(0)

# Создание директории, если она не существует
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Сохранение модели
torch.save(model.state_dict(), model_path)
print(f"✅ Модель сохранена в {model_path}")

# Финальная оценка
final_train_accuracy = calculate_accuracy(model, train_loader, device)
final_test_accuracy = calculate_accuracy(model, test_loader, device)
print(f"🏆 Финальная точность на обучающей выборке: {final_train_accuracy:.2f}%")
print(f"🏆 Финальная точность на тестовой выборке: {final_test_accuracy:.2f}%")