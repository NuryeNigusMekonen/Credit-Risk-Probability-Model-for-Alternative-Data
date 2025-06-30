# src/api/main.py
import joblib
import pandas as pd
from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI()

# Match the path inside the container
MODEL_PATH = "/app/models/random_forest_model.pkl"
model = joblib.load(MODEL_PATH)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(customer: CustomerFeatures):
    try:
        data = pd.DataFrame([customer.dict()])
        probability = model.predict_proba(data)[0][1]
        return {"risk_probability": float(probability)}
    except Exception as e:
        import traceback
        traceback.print_exc()  # Prints full error to Docker logs
        return {"error": str(e)}
