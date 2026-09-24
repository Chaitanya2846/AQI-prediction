# AQI model training and evaluation

`train_model.py` is the current training workflow. It uses the dataset's measured `US_AQI` target and refuses to invent target values from PM2.5. The earlier notebook workflow remains in `02_recent_data.ipynb` for reference; its randomized split and pre-split scaling produced the older metrics and should not be used to refresh the app report.

## Retrain

Install the project dependencies, then run:

```powershell
pip install -r requirements.txt
python train_model.py --data "C:\path\to\INDIA_AQI_COMPLETE_20251126.csv"
```

The dataset is not included in the repository. If it is already in KaggleHub's default cache, the `--data` argument can be omitted. Training writes `aqi_models_v4.pkl` and `evaluation.json` to the project root. FastAPI loads the model at startup and serves the evaluation report at `/evaluation` for the frontend dashboard.

## Evaluation design

- Records are sorted by timestamp; the earliest 80% of timestamps are used for training and the latest 20% are held out for the final score.
- A separate expanding-window, three-fold `TimeSeriesSplit` compares HistGradientBoosting, Extra Trees, Random Forest, XGBoost, and LightGBM regressors. The best configuration from each model family is retrained and scored on the same chronological holdout. Per the project request, the deployed model is the lowest-MAE model on that shared holdout. Since this selects by the holdout, these scores are model-selection metrics; a new future time window is needed for an unbiased final estimate.
- A median-only regression baseline and most-frequent-category baseline show how much value the model adds over simple predictions.
- The report includes MAE, median absolute error, MAPE, RMSE, R², category MAE/RMSE, category accuracy, balanced accuracy, macro precision/recall/F1, a confusion matrix, and a sample of held-out actual/predicted values.
- Classification candidates include unweighted and class-balanced models; selection uses macro F1 so rare severe categories count during tuning.
- Numeric features are imputed inside the model pipeline, and city names are one-hot encoded. No scaler is fit using test data.

## Latest run

The latest run used `INDIA_AQI_COMPLETE_20251126.csv`, which contains 842,160 source rows; 839,644 rows had both a numeric AQI and category label for training. The chronological holdout contains 166,581 rows from March 30 through November 26, 2025. Random Forest had the lowest holdout MAE and is the selected regressor; the selected classifier was class-balanced Extra Trees. LightGBM narrowly led validation MAE (13.70 versus 13.88 for Random Forest), but Random Forest was better on the shared holdout (MAE 13.67 versus 14.60, RMSE 22.05 versus 23.13, R² 0.789 versus 0.773). The app therefore uses Random Forest based on the requested holdout comparison.

- Regression MAE: 13.67 AQI points; median absolute error and category error statistics are in `evaluation.json`; RMSE: 22.05; R²: 0.789; MAPE: 16.92%.
- Median baseline MAE: 36.47, so the selected model reduced MAE by 62.5% on this holdout.
- Category accuracy: 70.9%; balanced accuracy: 65.1%; macro F1: 60.5%. The most-frequent-category baseline accuracy was 48.2%.

Class balancing raised macro F1 from 58.3% and balanced accuracy from 56.2% in the unweighted run, while overall accuracy decreased from 73.9% to 70.9%. The rare Hazardous class remains difficult: its 976 test examples had regression MAE of about 141 AQI points. More representative severe-pollution examples are the clearest remaining data improvement.

## Live AQI comparison

The model target is US AQI, so the live comparison now uses US EPA pollutant breakpoints. The live value is an estimate from OpenWeatherMap's current pollutant concentrations; those instantaneous readings do not provide the full averaging windows used for regulatory AQI reporting. The model and live value therefore share the US AQI scale, but the live value is not a certified station reading.
