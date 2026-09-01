from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .inference import LexiconModel, SentimentService


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    label: str
    score: float


service: SentimentService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = SentimentService(LexiconModel())
    yield
    service = None


app = FastAPI(title="Sentiment ML Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": service.model.name if service else "starting"}


@app.post("/v1/predict", response_model=list[PredictionResponse])
def predict(request: PredictRequest):
    if service is None:
        raise RuntimeError("Service is not ready")
    return service.predict(request.texts)
