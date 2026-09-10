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

app = FastAPI(title="AQI Prediction API v2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AQIInput(BaseModel):
    City: str
    PM2_5: float
    PM10: float
    NO2: float
    SO2: float
    CO: float
    O3: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Crop_Burning: int
    Festival: int
    Year: int
    Month: int
    Day: int

@app.post("/predict")
def predict_aqi(data: AQIInput):
    try:
        encoder = artifacts['city_encoder']
        city_encoded = encoder.transform([data.City])[0] if data.City in encoder.classes_ else 0
        
        input_data = {
            'PM2.5': data.PM2_5, 'PM10': data.PM10, 'NO2': data.NO2, 
            'SO2': data.SO2, 'CO': data.CO, 'O3': data.O3,
            'Temperature': data.Temperature, 'Humidity': data.Humidity, 'Wind_Speed': data.Wind_Speed,
            'Crop_Burning': data.Crop_Burning, 'Festival': data.Festival,
            'Year': data.Year, 'Month': data.Month, 'Day': data.Day, 
            'City_Encoded': city_encoded
        }
        
        input_features = pd.DataFrame([input_data])[artifacts['features']]
        
        num_cols = artifacts['numerical_features']
        input_features[num_cols] = artifacts['scaler'].transform(input_features[num_cols])
        
        aqi_value = artifacts['regressor'].predict(input_features)[0]
        category_encoded = artifacts['classifier'].predict(input_features)[0]
        category_label = artifacts['aqi_encoder'].inverse_transform([category_encoded])[0]
        
        return {
            "predicted_aqi": round(float(aqi_value), 2),
            "predicted_category": category_label
        }
    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(status_code=400, detail=str(e))