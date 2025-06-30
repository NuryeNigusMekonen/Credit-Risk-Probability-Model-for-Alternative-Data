from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import pandas as pd
import joblib

app = FastAPI()

# Update to the correct path to your model
MODEL_PATH = "/app/models/random_forest_model.pkl"
model = joblib.load(MODEL_PATH)

THRESHOLD = 0.5

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(customer: CustomerFeatures):
    data = pd.DataFrame([customer.dict()])
    probability = float(model.predict_proba(data)[:, 1][0])  # Risk score
    is_high_risk = int(probability >= THRESHOLD)  # Risk label
    return {
        "risk_probability": probability,
        "is_high_risk": is_high_risk
    }
