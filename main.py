%%writefile main.py
from enum import Enum
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

app = FastAPI(title="Military Predictive Logistics API")

# Load models and column schema on startup
xgb_model = joblib.load('xgb_risk_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
lstm_model = load_model('lstm_demand_model.h5')

with open("columns.txt", "r") as f:
    MODEL_COLUMNS = [line.strip() for line in f.readlines()]

# Define clean Enum classes matching your training dataset's exact category names
class SectorEnum(str, Enum):
    NORTHERN = "High Altitude (Eastern Ladakh)"
    WESTERN = "Desert Sector (Rajasthan)"
    SIACHEN = "High Altitude (Siachen Logistics Node)"

class SupplyCategoryEnum(str, Enum):
    FUEL = "POL (Fuel, Oil, Lubricants)"
    MEDICAL = "Medical Supplies & Spare Parts"
    RATIONS = "Rations & Water"

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
    # Create a zero-filled DataFrame matching the exact columns the model was trained on
    input_encoded = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
    
    # Fill in numerical inputs safely
    if 'current_stock_days' in input_encoded.columns:
        input_encoded['current_stock_days'] = data.current_stock_days
    if 'daily_consumption_rate' in input_encoded.columns:
        input_encoded['daily_consumption_rate'] = data.daily_consumption_rate
    if 'weather_severity_index' in input_encoded.columns:
        input_encoded['weather_severity_index'] = data.weather_severity_index
    if 'terrain_difficulty_score' in input_encoded.columns:
        input_encoded['terrain_difficulty_score'] = data.terrain_difficulty_score
    if 'route_disruption_active' in input_encoded.columns:
        input_encoded['route_disruption_active'] = data.route_disruption_active
    if 'transport_capacity_util_pct' in input_encoded.columns:
        input_encoded['transport_capacity_util_pct'] = data.transport_capacity_util_pct
        
    # Activate the exact one-hot encoded categorical columns matching user selection
    sector_col = f"sector_{data.sector.value}"
    if sector_col in input_encoded.columns:
        input_encoded[sector_col] = 1
        
    category_col = f"supply_category_{data.supply_category.value}"
    if category_col in input_encoded.columns:
        input_encoded[category_col] = 1
    
    # Run prediction
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
        return {"error": "Please provide at least 14 days of historical data"}
    
    arr = np.array(past_consumption_values[-14:]).reshape(1, 14, 1)
    scaled_input = scaler.transform(arr.reshape(-1, 1)).reshape(1, 14, 1)
    
    pred_scaled = lstm_model.predict(scaled_input)
    pred_value = scaler.inverse_transform(pred_scaled)[0][0]
    
    return {
        "forecasted_demand_next_day": round(float(pred_value), 2)
    }
