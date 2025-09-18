from PIL import Image
import io
import torch
import torchvision.transforms as T
import base64


def read_imagefile(file, grayscale=True) -> torch.Tensor:
    """
    Преобразует файл изображения в тензор PyTorch.
    
    Args:
        file: Байтовые данные изображения
        grayscale: Если True, конвертирует в оттенки серого, иначе сохраняет RGB
        
    Returns:
        torch.Tensor: Тензор изображения размером (C, 28, 28), где C=1 для grayscale или C=3 для RGB
        
    Raises:
        ValueError: Если файл не является изображением или не может быть обработан
    """
    try:
        # Попытка открыть изображение
        image = Image.open(io.BytesIO(file))
        
        # Проверка, что файл действительно является изображением
        if not hasattr(image, 'format') or image.format is None:
            raise ValueError("Файл не является корректным изображением")
        
        # Конвертация в нужный формат
        if grayscale:
            image = image.convert("L")  # grayscale
            channels = 1
        else:
            image = image.convert("RGB")  # RGB
            channels = 3
            
        # Применение преобразований
        transform = T.Compose([
            T.Resize((28, 28)),
            T.ToTensor(),
        ])
        
        tensor = transform(image)
        
        # Проверка размерности тензора
        expected_shape = (channels, 28, 28)
        if tensor.shape != expected_shape:
            raise ValueError(f"Неожиданный размер тензора: {tensor.shape}, ожидался: {expected_shape}")
            
        return tensor
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Ошибка при обработке изображения: {str(e)}")


def decode_base64_image(base64_string: str, grayscale=True) -> torch.Tensor:
    """
    Декодирует изображение из base64 строки и преобразует в тензор.
    
    Args:
        base64_string: Base64 строка изображения (может содержать data URL префикс)
        grayscale: Если True, конвертирует в оттенки серого, иначе сохраняет RGB
        
    Returns:
        torch.Tensor: Тензор изображения
        
    Raises:
        ValueError: Если строка base64 некорректна или не содержит изображение
    """
    try:
        # Удаление префикса data URL, если он есть
        if base64_string.startswith('data:image'):
            # Формат: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
            base64_string = base64_string.split(',', 1)[1]
        
        # Декодирование base64
        image_bytes = base64.b64decode(base64_string)
        
        # Использование основной функции для обработки
        return read_imagefile(image_bytes, grayscale=grayscale)
        
    except Exception as e:
        raise ValueError(f"Ошибка при декодировании base64 изображения: {str(e)}")