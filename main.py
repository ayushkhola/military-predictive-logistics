from enum import Enum
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

app = FastAPI(title="Military Predictive Logistics API")

# Load models on startup
xgb_model = joblib.load('xgb_risk_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
lstm_model = load_model('lstm_demand_model.h5')

# Define clean Enum classes for dropdown selection in Swagger UI
class SectorEnum(str, Enum):
    NORTHERN = "Northern Command"
    EASTERN = "Eastern Command"
    WESTERN = "Western Command"
    SOUTHERN="Southern command"
class SupplyCategoryEnum(str, Enum):
    FUEL = "Fuel"
    AMMUNITION = "Ammunition"
    RATIONS = "Rations"
    MEDICAL = "Medical"

# Define request body using the Enums
class ConvoyData(BaseModel):
    current_stock_days: float
    daily_consumption_rate: float
    weather_severity_index: float
    terrain_difficulty_score: float
    route_disruption_active: int
    transport_capacity_util_pct: float

    sector: SectorEnum = Field(..., description="Select the active military sector command")
    supply_category: SupplyCategoryEnum = Field(..., description="Select the category of supplies being transported")

@app.post("/predict-risk")
def predict_convoy_risk(data: ConvoyData):
    # Convert input to dictionary (Enum values automatically serialize cleanly)
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)

    risk_prob = float(xgb_model.predict_proba(input_encoded)[:, 1][0])
    risk_label = int(xgb_model.predict(input_encoded)[0])

    status = "HIGH RISK - Immediate Resupply Required" if risk_label == 1 else "ON TRACK"

    return {
        "risk_probability": round(risk_prob, 2),
        "status": status,
        "action_recommended": "Reroute convoy or dispatch air drop" if risk_label == 1 else "Maintain routine schedule"
    }

@app.post("/forecast-demand")
def forecast_demand(past_consumption_values: list[float]):
    if len(past_consumption_values) < 14:
        return {"error": "Please provide at least 14 days of historical data."}

    recent_data = np.array(past_consumption_values[-14:]).reshape(-1, 1)
    scaled_input = recent_data / 500.0
    X_input = scaled_input.reshape(1, 14, 1)

    predicted_scaled = lstm_model.predict(X_input)
    predicted_value = float(predicted_scaled[0][0]) * 500.0

    return {
        "forecasted_consumption_next_day": round(predicted_value, 2)
    }
