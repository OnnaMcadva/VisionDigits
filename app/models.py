import torch
import os
from .simple_cnn import SimpleCNN


class ModelWrapper:
    """
    Обертка для модели с обработкой ошибок при загрузке весов.
    """
    def __init__(self, path: str, num_classes: int = 10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        self.labels = [str(i) for i in range(num_classes)]
        
        # Обработка ошибок при загрузке весов модели
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл модели не найден: {path}")
            
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print(f"✅ Модель успешно загружена из {path}")
            
        except FileNotFoundError as e:
            print(f"❌ Ошибка: {e}")
            print("💡 Запустите 'python app/train.py' для обучения модели")
            raise
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            raise

    def predict(self, image_tensor):
        """
        Предсказание класса для изображения.
        
        Args:
            image_tensor: Тензор изображения размером (1, 28, 28)
            
        Returns:
            str: Предсказанный класс (цифра)
        """
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # (1,1,28,28)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        return self.labels[predicted.item()]
