from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.models import ModelWrapper
from app.utils import read_imagefile, decode_base64_image
from pydantic import BaseModel
import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VisionDigits API",
    description="API для распознавания рукописных цифр. Поддерживает загрузку файлов и изображения из canvas.",
    version="1.0.0"
)

# Инициализация модели с обработкой ошибок
try:
    model = ModelWrapper("model/sketch_cnn.pt", num_classes=10)
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {e}")
    model = None


class Base64Image(BaseModel):
    """Модель для принятия base64 изображений из canvas"""
    image: str


@app.get("/")
def root():
    """Главная страница API"""
    return {"message": "VisionDigits API is running 🎯"}


@app.get("/health")
def health_check():
    """Проверка состояния API и модели"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Модель не загружена"}
        )
    return {"status": "ok", "message": "API и модель работают корректно"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Предсказание цифры из загруженного файла изображения.
    
    Args:
        file: Загруженный файл изображения (JPEG, PNG, etc.)
        
    Returns:
        dict: Результат предсказания с цифрой
        
    Raises:
        HTTPException: При ошибках обработки файла или предсказания
    """
    # Проверка доступности модели
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Обратитесь к администратору."
        )
    
    try:
        # Проверка типа файла
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый тип файла: {file.content_type}. Загрузите изображение (JPEG, PNG, etc.)"
            )
        
        # Чтение и обработка файла
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Загруженный файл пуст"
            )
        
        # Преобразование изображения в тензор
        image_tensor = read_imagefile(file_content, grayscale=True)
        
        # Предсказание
        result = model.predict(image_tensor)
        
        logger.info(f"Успешное предсказание: {result} для файла {file.filename}")
        
        return {
            "prediction": result,
            "filename": file.filename,
            "content_type": file.content_type,
            "status": "success"
        }
        
    except ValueError as e:
        # Ошибки валидации изображения
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки изображения: {str(e)}"
        )
    except Exception as e:
        # Неожиданные ошибки
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера при обработке изображения"
        )


@app.post("/predict/canvas/")
async def predict_canvas(image_data: Base64Image):
    """
    Предсказание цифры из изображения canvas (base64).
    
    Args:
        image_data: Base64 строка изображения из canvas
        
    Returns:
        dict: Результат предсказания с цифрой
        
    Raises:
        HTTPException: При ошибках обработки изображения или предсказания
    """
    # Проверка доступности модели
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Обратитесь к администратору."
        )
    
    try:
        # Декодирование и обработка base64 изображения
        image_tensor = decode_base64_image(image_data.image, grayscale=True)
        
        # Предсказание
        result = model.predict(image_tensor)
        
        logger.info(f"Успешное предсказание из canvas: {result}")
        
        return {
            "prediction": result,
            "source": "canvas",
            "status": "success"
        }
        
    except ValueError as e:
        # Ошибки валидации изображения
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки изображения из canvas: {str(e)}"
        )
    except Exception as e:
        # Неожиданные ошибки
        logger.error(f"Ошибка при предсказании из canvas: {e}")
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера при обработке изображения из canvas"
        )

# python3 -m venv venv
# source venv/bin/activate
# uvicorn app.main:app --reload

# pip cache purge ???
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pip install fastapi uvicorn
# pip install python-multipart
# python app/train.py
# uvicorn app.main:app --reload


# http://127.0.0.1:8000
#  → там будет {"message": "Sketch2Action API is running 🚀"}

# http://127.0.0.1:8000/docs
#  → интерактивная Swagger-документация, там можно тестировать загрузку файлов прямо из браузера.