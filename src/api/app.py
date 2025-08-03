from fastapi import FastAPI
from pydantic import BaseModel
import os

from utils.infer_utils import load_model, load_meta, predict_url

MODEL_PATH = "../models/best_model_3.pt"
META_PATH = "../artifacts/preprocess_meta_3.json"

app = FastAPI(title="Phishing URL Detection API")

meta = load_meta(META_PATH)
char2idx = meta["char2idx"]
vocab_size = meta["maxlen"]

model = load_model(MODEL_PATH, vocab_size)

class URLRequest(BaseModel):
    url: str

@app.post("/predict")
async def predict_url_api(request: URLRequest):
    url = request.url
    label, confidence = predict_url(model, url, char2idx, maxlen)
    label_name = "phishing" if label == 1 else "benign"
    return {
        "url": url,
        "prediction": label_name,
        "confidence": round(confidence)
    }