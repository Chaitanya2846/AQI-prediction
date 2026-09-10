from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

try:
    with open('aqi_models.pkl', 'rb') as f:
        artifacts = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file 'aqi_models.pkl' not found.")

app = FastAPI(title="AQI Prediction API v1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic schema matching your React frontend
class AQIInput(BaseModel):
    City: str
    PM2_5: float
    PM10: float
    NO2: float
    SO2: float
    CO: float
    O3: float
    Year: int
    Month: int
    Day: int

@app.post("/predict")
def predict_aqi(data: AQIInput):
    try:
        # Safely handle unseen cities
        encoder = artifacts['city_encoder']
        if data.City in encoder.classes_:
            city_encoded = encoder.transform([data.City])[0]
        else:
            city_encoded = 0  # Default fallback for unknown cities
        
        # Map incoming data to exact feature names
        input_data = {
            'PM2.5': data.PM2_5, 'PM10': data.PM10, 'NO2': data.NO2, 
            'SO2': data.SO2, 'CO': data.CO, 'O3': data.O3,
            'Year': data.Year, 'Month': data.Month, 'Day': data.Day, 
            'City_Encoded': city_encoded
        }
        
        input_features = pd.DataFrame([input_data])[artifacts['features']]
        
        # Hardcode the numerical columns to match the old .pkl file
        num_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        input_features[num_cols] = artifacts['scaler'].transform(input_features[num_cols])
        
        # Run predictions
        aqi_value = artifacts['regressor'].predict(input_features)[0]
        category_encoded = artifacts['classifier'].predict(input_features)[0]
        category_label = artifacts['aqi_encoder'].inverse_transform([category_encoded])[0]
        
        return {
            "predicted_aqi": round(float(aqi_value), 2),
            "predicted_category": category_label
        }
    except Exception as e:
        print("VALIDATION/PREDICTION ERROR:", str(e))
        raise HTTPException(status_code=400, detail=str(e))