from PIL import Image, UnidentifiedImageError
import io
import torch
import torchvision.transforms as T
import base64

def read_imagefile(file, mode='L') -> torch.Tensor:
    """
    file: bytes (или base64-строка)
    mode: 'L' — grayscale, 'RGB' — цветное
    """
    try:
        if isinstance(file, str):  # если base64-строка
            file = base64.b64decode(file.split(',')[-1])
        image = Image.open(io.BytesIO(file)).convert(mode)
    except (UnidentifiedImageError, ValueError, TypeError, OSError) as e:
        raise ValueError("Failed to open the image. Please check the file format or the base64 string.") from e
    transform = T.Compose([
        T.Resize((28, 28)),
        T.ToTensor(),
    ])
    if mode == 'RGB':
        image = image.convert('RGB')
        tensor = transform(image)
        tensor = tensor.mean(dim=0, keepdim=True)  # приводим к (1,28,28)
    else:
        tensor = transform(image)
    return tensor
