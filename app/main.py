from fastapi import FastAPI, UploadFile, File
from app.models import ModelWrapper
from app.utils import read_imagefile
import torch

app = FastAPI(title="Sketch2Action API")

model = ModelWrapper("model/dummy_model.pt")

@app.get("/")
def root():
    return {"message": "Sketch2Action API is running 🦄"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    result = model.predict(image)
    return {"prediction": result}

# python3 -m venv venv
# source venv/bin/activate
# uvicorn app.main:app --reload

# pip cache purge
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pip install fastapi uvicorn
# pip install python-multipart
# uvicorn app.main:app --reload



# http://127.0.0.1:8000
#  → там будет {"message": "Sketch2Action API is running 🚀"}

# http://127.0.0.1:8000/docs
#  → интерактивная Swagger-документация, там можно тестировать загрузку файлов прямо из браузера.