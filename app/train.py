import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from app.simple_cnn import SimpleCNN
import os
from tqdm import tqdm

batch_size = 64
epochs = 10  # увеличено количество эпох
lr = 0.001
dropout = 0.3

transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10, dropout=dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

model.train()
for epoch in range(epochs):
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_loss = total_loss/len(train_loader)
    acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Test Accuracy: {acc*100:.2f}%")

save_path = "model/sketch_cnn.pt"
if os.path.exists(save_path):
    import datetime
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model/sketch_cnn_{dt}.pt"
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to {save_path}")
