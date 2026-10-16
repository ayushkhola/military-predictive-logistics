from enum import Enum
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
import traceback

app = FastAPI(title="Military Predictive Logistics API")

# Load models on startup
try:
    xgb_model = joblib.load('xgb_risk_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    lstm_model = load_model('lstm_demand_model.h5')
    
    # EXACT 12 COLUMNS - Completely ignoring columns.txt
    MODEL_COLUMNS = [
        'current_stock_days', 'daily_consumption_rate', 'weather_severity_index',
        'terrain_difficulty_score', 'route_disruption_active', 'transport_capacity_util_pct',
        'sector_High Altitude (Eastern Ladakh)', 'sector_Desert Sector (Rajasthan)', 'sector_High Altitude (Siachen Logistics Node)',
        'supply_category_POL (Fuel, Oil, Lubricants)', 'supply_category_Medical Supplies & Spare Parts', 'supply_category_Rations & Water'
    ]
except Exception as e:
    print(f"Error loading files: {e}")
    MODEL_COLUMNS = []

class SectorEnum(str, Enum):
    NORTHERN = "High Altitude (Eastern Ladakh)"
    WESTERN = "Desert Sector (Rajasthan)"
    SIACHEN = "High Altitude (Siachen Logistics Node)"

class SupplyCategoryEnum(str, Enum):
    FUEL = "POL (Fuel, Oil, Lubricants)"
    MEDICAL = "Medical Supplies & Spare Parts"
    RATIONS = "Rations & Water"

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
    try:
        # 1. Create a DataFrame with exactly the 12 columns
        input_encoded = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
        
        # 2. Fill numericals
        input_encoded['current_stock_days'] = data.current_stock_days
        input_encoded['daily_consumption_rate'] = data.daily_consumption_rate
        input_encoded['weather_severity_index'] = data.weather_severity_index
        input_encoded['terrain_difficulty_score'] = data.terrain_difficulty_score
        input_encoded['route_disruption_active'] = data.route_disruption_active
        input_encoded['transport_capacity_util_pct'] = data.transport_capacity_util_pct
            
        # 3. Fill categoricals safely
        target_sector = str(data.sector.value).lower()
        for col in input_encoded.columns:
            if "sector_" in col.lower() and target_sector in col.lower():
                input_encoded[col] = 1
                break
                
        target_category = str(data.supply_category.value).lower()
        for col in input_encoded.columns:
            if "supply_category_" in col.lower() and target_category in col.lower():
                input_encoded[col] = 1
                break
        
        # 4. Predict using raw array to prevent feature name mismatches
        input_array = input_encoded.values
        probs = xgb_model.predict_proba(input_array)
        risk_prob = float(probs[0][1] if len(probs.shape) == 2 and probs.shape[1] > 1 else probs[0])
        
        preds = xgb_model.predict(input_array)
        risk_label = int(preds[0])
        
        status = "HIGH RISK - Immediate Resupply Required" if risk_label == 1 else "ON TRACK"
        
        return {
            "risk_probability": round(risk_prob, 2),
            "status": status,
            "action_recommended": "Reroute convoy or dispatch air drop" if risk_label == 1 else "Maintain routine schedule"
        }
        
    except Exception as e:
        return {
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/forecast-demand")
def forecast_demand(past_consumption_values: list[float]):
    try:
        if len(past_consumption_values) < 14:
            return {"error": "Please provide at least 14 days of historical data"}
        
        arr = np.array(past_consumption_values[-14:]).reshape(1, 14, 1)
        scaled_input = scaler.transform(arr.reshape(-1, 1)).reshape(1, 14, 1)
        
        pred_scaled = lstm_model.predict(scaled_input)
        pred_value = scaler.inverse_transform(pred_scaled)[0][0]
        
        return {
            "forecasted_demand_next_day": round(float(pred_value), 2)
        }
    except Exception as e:
        return {
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
