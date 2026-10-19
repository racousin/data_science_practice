import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

model = joblib.load("model.joblib")     # the whole pipeline: scaler + model
app = FastAPI(title="Penguin species")


class Penguin(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(penguin: Penguin):
    X = pd.DataFrame([penguin.model_dump()])
    return {"species": model.predict(X)[0]}
