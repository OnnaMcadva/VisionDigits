from pydantic import BaseModel

class PredictRequest(BaseModel):
    base64img: str
    color: bool = False

class PredictResponse(BaseModel):
    prediction: str