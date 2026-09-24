# AeroSense — AQI Prediction

AeroSense is a React and FastAPI application that compares a live air-quality estimate with a machine-learning prediction. It fetches weather and pollutant readings from OpenWeatherMap and includes a dashboard for model evaluation.

## Quick start

The project needs Python 3.11+, Node.js/npm, an OpenWeatherMap API key, and the historical Kaggle dataset. The dataset and trained model are not stored in this repository.

1. Create a Python environment and install dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root and open it:

   ```powershell
   New-Item .env -ItemType File
   notepad .env
   ```

   Add your OpenWeatherMap key, save the file, and keep it private:

   ```text
   OPENWEATHER_API_KEY=your_key
   ```
3. Download `INDIA_AQI_COMPLETE_20251126.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025), then train:

   ```powershell
   python train_model.py --data ".\data\INDIA_AQI_COMPLETE_20251126.csv"
   ```

4. Start the API from the project root:

   ```powershell
   uvicorn main:app --reload
   ```

5. In a second terminal, start the frontend:

   ```powershell
   cd aqi-frontend
   npm ci
   npm start
   ```

Open `http://localhost:3000`. The API docs are at `http://127.0.0.1:8000/docs`.

## What the app shows

- **Actual live AQI:** an estimate calculated from current OpenWeatherMap pollutant concentrations using US EPA breakpoints. It is not an official station reading.
- **Predicted AQI:** the locally trained Random Forest model's output from the weather, pollutant, and calendar inputs.
- **Evaluation:** historical holdout metrics, candidate-model comparisons, and charts. These are not the accuracy of an individual live reading.

For detailed dataset information, pipeline, model rationale, metrics, full setup steps, API endpoints, and troubleshooting, see [project_summary.md](project_summary.md). The evaluation methodology is also documented in [MODEL_EVALUATION.md](MODEL_EVALUATION.md).
