from PIL import Image
import io
import torch
import torchvision.transforms as T

# преобразуем картинку в тензор
def read_imagefile(file) -> torch.Tensor:
    image = Image.open(io.BytesIO(file)).convert("L")  # grayscale
    transform = T.Compose([
        T.Resize((28, 28)),
        T.ToTensor(),
    ])
    return transform(image)