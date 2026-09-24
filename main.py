from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from pathlib import Path
import os
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import json
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "aqi_models_v4.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file 'aqi_models_v4.pkl' not found. Run train_model.py first.")
if artifacts.get("version") != 4:
    raise RuntimeError("Unsupported model artifact version. Run train_model.py to create a v4 model.")

app = FastAPI(title="AQI Prediction API v4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
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

def openweather(path, params):
    url = f"https://api.openweathermap.org{path}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=12) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            provider_message = json.loads(exc.read().decode("utf-8")).get("message", "")
        except (ValueError, UnicodeDecodeError):
            provider_message = ""
        if exc.code == 404:
            raise HTTPException(status_code=404, detail="Location not found by OpenWeatherMap.")
        if exc.code in (401, 403):
            detail = "OpenWeatherMap rejected the API key or this API is not enabled for it."
        elif exc.code == 429:
            detail = "OpenWeatherMap request quota was exceeded."
        else:
            detail = f"OpenWeatherMap returned HTTP {exc.code}."
        if provider_message:
            detail = f"{detail} Provider message: {provider_message}"
        raise HTTPException(status_code=502, detail=detail)
    except (URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", None)
        raise HTTPException(status_code=502, detail=f"Could not reach OpenWeatherMap: {reason or 'connection timed out'}.")

# US EPA AQI breakpoints. OpenWeatherMap gives pollutant concentrations in
# µg/m³; gases are converted to the units used by EPA before interpolation.
EPA_BREAKPOINTS = {
    "PM2.5": [(0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)],
    "PM10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
    "NO2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
    "SO2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500)],
    "CO": [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)],
    "O3": [(0, .054, 0, 50), (.055, .070, 51, 100), (.071, .085, 101, 150), (.086, .105, 151, 200), (.106, .200, 201, 300)],
}

def epa_subindex(pollutant, concentration):
    for low, high, index_low, index_high in EPA_BREAKPOINTS[pollutant]:
        if concentration <= high:
            return round((index_high - index_low) / (high - low) * (concentration - low) + index_low)
    return 500 if pollutant != "O3" else 300

def calculate_live_aqi(components):
    # EPA concentration precision: PM2.5 to 0.1 µg/m³, PM10 to whole µg/m³,
    # CO to 0.1 ppm, and the other gases to whole ppb.
    def trunc(value, decimals):
        factor = 10 ** decimals
        return int(value * factor) / factor

    concentrations = {
        "PM2.5": trunc(components["pm2_5"], 1),
        "PM10": int(components["pm10"]),
        "NO2": int(components["no2"] * 24.45 / 46.0055),
        "SO2": int(components["so2"] * 24.45 / 64.066),
        "CO": trunc(components["co"] * 24.45 / (28.01 * 1000), 1),
        "O3": trunc(components["o3"] * 24.45 / (48.00 * 1000), 3),
    }
    subindices = {key: epa_subindex(key, value) for key, value in concentrations.items()}
    dominant = max(subindices, key=subindices.get)
    return max(subindices.values()), dominant, subindices

@app.get("/live-conditions/{city}")
def live_conditions(city: str):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Set OPENWEATHER_API_KEY in the project .env file.")
    place = openweather("/geo/1.0/direct", {"q": f"{city},IN", "limit": 1, "appid": api_key})
    if not place:
        raise HTTPException(status_code=404, detail=f"Could not find {city} in India.")
    lat, lon = place[0]["lat"], place[0]["lon"]
    weather = openweather("/data/2.5/weather", {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"})
    pollution = openweather("/data/2.5/air_pollution", {"lat": lat, "lon": lon, "appid": api_key})
    components = pollution["list"][0]["components"]
    actual_aqi, dominant_pollutant, subindices = calculate_live_aqi(components)
    return {
        "city": place[0]["name"], "country": place[0].get("country", "IN"),
        "observed_at": weather["dt"], "provider_aqi": pollution["list"][0]["main"]["aqi"],
        "actual_aqi": actual_aqi, "actual_aqi_category": aqi_category(actual_aqi),
        "actual_aqi_scale": "US EPA AQI estimate",
        "dominant_pollutant": dominant_pollutant, "pollutant_subindices": subindices,
        "inputs": {
            "PM2_5": components["pm2_5"], "PM10": components["pm10"],
            "NO2": components["no2"], "SO2": components["so2"],
            "CO": components["co"], "O3": components["o3"],
            "Temperature": weather["main"]["temp"], "Humidity": weather["main"]["humidity"],
            "Wind_Speed": weather["wind"]["speed"] * 3.6,
        },
    }

def aqi_category(value):
    if value <= 50: return "Good"
    if value <= 100: return "Moderate"
    if value <= 150: return "Unhealthy for Sensitive Groups"
    if value <= 200: return "Unhealthy"
    if value <= 300: return "Very Unhealthy"
    return "Hazardous"

@app.get("/evaluation")
def get_evaluation():
    report_path = ROOT / "evaluation.json"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Evaluation report not found. Run train_model.py first.")
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read evaluation report: {exc}")

@app.post("/predict")
def predict_aqi(data: AQIInput):
    try:
        input_data = {
            "City": data.City.strip().lower(),
            "PM2.5": data.PM2_5, "PM10": data.PM10, "NO2": data.NO2,
            "SO2": data.SO2, "CO": data.CO, "O3": data.O3,
            "Temperature": data.Temperature,
            "Humidity": data.Humidity, "Wind_Speed": data.Wind_Speed,
            "Crop_Burning": data.Crop_Burning, "Festival": data.Festival,
            "Year": data.Year, "Month": data.Month, "Day": data.Day,
        }
        input_features = pd.DataFrame([input_data])[artifacts["features"]]
        aqi_value = artifacts["regression_model"].predict(input_features)[0]
        category_label = artifacts["classification_model"].predict(input_features)[0]
        return {"predicted_aqi": round(float(aqi_value)), "predicted_category": str(category_label)}
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid prediction input: {exc}") from exc
