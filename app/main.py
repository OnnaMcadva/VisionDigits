from fastapi import FastAPI, UploadFile, File, HTTPException
from app.models import ModelWrapper
from app.utils import read_imagefile
from app.schemas import PredictRequest, PredictResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="VisionDigits API")

app.mount("/static", StaticFiles(directory="static"), name="static")

model = ModelWrapper("model/sketch_cnn.pt", num_classes=10)

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/")
def root():
    return {"message": "VisionDigits API is running 🦄"}

@app.post("/predict_file/", response_model=PredictResponse)
async def predict_file(
    file: UploadFile = File(...),
    color: bool = False
):
    """
    Accepts an image file (e.g., PNG/JPEG) via multipart/form-data.
    color: True if the image is in color (e.g., drawn on a browser canvas).
    """
    try:
        data = await file.read()
        image = read_imagefile(data, mode='RGB' if color else 'L')
        result = model.predict(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")
    return PredictResponse(prediction=result)

@app.post("/predict/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Accepts a JSON object with base64img (a base64-encoded string) and color (a boolean).
    """
    try:
        image = read_imagefile(request.base64img, mode='RGB' if request.color else 'L')
        result = model.predict(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")
    return PredictResponse(prediction=result)

# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# python -m app.train
# uvicorn app.main:app ## --reload

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