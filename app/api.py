from fastapi import APIRouter, FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from app.inference import predict_genre
from app.model import load_model
router = APIRouter()

MODEL = load_model()
class PredictionResponse(BaseModel):
    genre: str
    confidence: float

class SpectrogramResponse(BaseModel):
    image: str

@router.post("/predict/", response_model=PredictionResponse)
async def predict_genre_endpoint(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return await predict_genre(file, MODEL)
