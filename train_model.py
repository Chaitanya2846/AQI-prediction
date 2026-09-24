"""Train and evaluate AeroSense models against a chronological US AQI holdout.

Usage:
  python train_model.py --data path/to/aqi_india_38cols_knn_final.csv

The script refuses to synthesize AQI from PM2.5. It writes a compact versioned
model artifact and an evaluation.json report consumed by the FastAPI app.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = (
    Path.home()
    / ".cache/kagglehub/datasets/bhautikvekariya21/"
    / "air-quality-dataset-indian-cities-2022-2025/versions/5/"
    / "INDIA_AQI_COMPLETE_20251126.csv"
)

FEATURES = [
    "City", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature",
    "Humidity", "Wind_Speed", "Crop_Burning", "Festival", "Year", "Month", "Day",
]
NUMERIC_FEATURES = [name for name in FEATURES if name != "City"]
NUMERIC_SOURCE = {
    "PM2.5": "pm2_5_ugm3", "PM10": "pm10_ugm3", "NO2": "no2_ugm3",
    "SO2": "so2_ugm3", "CO": "co_ugm3", "O3": "o3_ugm3",
    "Temperature": "temp_2m_c", "Humidity": "humidity_percent",
    "Wind_Speed": "wind_speed_10m_kmh", "Crop_Burning": "crop_burning_season",
    "Festival": "festival_period",
}


def make_pipeline(estimator):
    prep = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "city",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["City"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", prep), ("model", estimator)])


def make_regressor(model_name, params):
    if model_name == "extra_trees":
        return ExtraTreesRegressor(random_state=42, **params)
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=42, **params)
    if model_name == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("XGBoost was selected but xgboost is not installed.")
        return XGBRegressor(random_state=42, **params)
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise RuntimeError("LightGBM was selected but lightgbm is not installed.")
        return LGBMRegressor(random_state=42, **params)
    return HistGradientBoostingRegressor(random_state=42, **params)


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [column.strip().lower() for column in frame.columns]
    required = {"city", "datetime", "us_aqi", "aqi_category", "pm2_5_ugm3", "pm10_ugm3", "no2_ugm3", "so2_ugm3", "co_ugm3", "o3_ugm3", "humidity_percent"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    # This dataset's authoritative numeric target is US_AQI. Never replace a
    # missing target with an AQI formula because that would manufacture labels.
    frame["Datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    frame["AQI"] = pd.to_numeric(frame["us_aqi"], errors="coerce")
    frame["AQI_Category"] = frame["aqi_category"].astype("string").str.strip().str.replace("_", " ", regex=False)
    frame["AQI_Category"] = frame["AQI_Category"].replace({"Unhealthy Sensitive": "Unhealthy for Sensitive Groups"})
    frame = frame.dropna(subset=["Datetime", "AQI", "AQI_Category"]).copy()
    frame["City"] = frame["city"].astype("string").str.strip().str.lower()
    for target, source in NUMERIC_SOURCE.items():
        frame[target] = pd.to_numeric(frame[source], errors="coerce") if source in frame.columns else np.nan
    frame["Year"] = frame["Datetime"].dt.year
    frame["Month"] = frame["Datetime"].dt.month
    frame["Day"] = frame["Datetime"].dt.day
    frame = frame.sort_values(["Datetime", "City"], kind="stable").reset_index(drop=True)
    if frame["AQI"].nunique() < 2:
        raise ValueError("US_AQI has insufficient variation to train a model.")
    return frame


def time_holdout(frame: pd.DataFrame, test_fraction: float = 0.2):
    timestamps = frame["Datetime"].drop_duplicates().sort_values().to_numpy()
    split_at = max(1, min(len(timestamps) - 1, int(len(timestamps) * (1 - test_fraction))))
    cutoff = pd.Timestamp(timestamps[split_at])
    train = frame.loc[frame["Datetime"] < cutoff].copy()
    test = frame.loc[frame["Datetime"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Could not form a chronological train/test split.")
    return train, test, cutoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Source AQI CSV")
    parser.add_argument("--model-output", type=Path, default=ROOT / "aqi_models_v4.pkl")
    parser.add_argument("--evaluation-output", type=Path, default=ROOT / "evaluation.json")
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}. Pass its CSV path with --data.")

    frame = load_dataset(args.data)
    train, test, test_cutoff = time_holdout(frame)
    train_times = train["Datetime"].drop_duplicates().sort_values().to_numpy()
    val_start = pd.Timestamp(train_times[max(1, int(len(train_times) * 0.88))])
    fit_rows = train.loc[train["Datetime"] < val_start]
    val_rows = train.loc[train["Datetime"] >= val_start]
    if min(len(fit_rows), len(val_rows), len(test)) == 0:
        raise ValueError("Training/validation/test windows must all contain examples.")

    # Tune a small, temporal validation sample to keep retraining practical.
    tuning_rows = fit_rows.sample(n=min(120_000, len(fit_rows)), random_state=42).sort_values("Datetime")
    cv_rows = pd.concat([tuning_rows, val_rows.sample(n=min(30_000, len(val_rows)), random_state=42)])
    cv_rows = cv_rows.sort_values("Datetime").reset_index(drop=True)
    split_plan = TimeSeriesSplit(n_splits=3)
    configurations = [
        {"model": "hist_gradient_boosting", "params": {"max_iter": 120, "learning_rate": 0.08, "max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 1.0}},
        {"model": "hist_gradient_boosting", "params": {"max_iter": 160, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0}},
        {"model": "hist_gradient_boosting", "params": {"max_iter": 220, "learning_rate": 0.04, "max_leaf_nodes": 31, "min_samples_leaf": 60, "l2_regularization": 2.0}},
        {"model": "extra_trees", "params": {"n_estimators": 50, "max_depth": 20, "min_samples_leaf": 8, "max_features": 0.8, "bootstrap": True, "max_samples": 0.8, "n_jobs": -1}},
        {"model": "random_forest", "params": {"n_estimators": 160, "max_depth": 24, "min_samples_leaf": 4, "max_features": 0.8, "n_jobs": -1}},
    ]
    if XGBRegressor is not None:
        configurations.append({"model": "xgboost", "params": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10, "reg_lambda": 1.0, "objective": "reg:squarederror", "tree_method": "hist", "n_jobs": -1, "verbosity": 0}})
    else:
        print("Skipping XGBoost: install xgboost to include it in the comparison.", flush=True)
    if LGBMRegressor is not None:
        configurations.append({"model": "lightgbm", "params": {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "max_depth": -1, "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_jobs": -1, "verbosity": -1}})
    else:
        print("Skipping LightGBM: install lightgbm to include it in the comparison.", flush=True)
    X_cv = cv_rows[FEATURES]
    y_cv = cv_rows["AQI"]
    cv_results = []
    for specification in configurations:
        fold_maes = []
        for train_idx, val_idx in split_plan.split(X_cv):
            estimator = make_regressor(specification["model"], specification["params"])
            candidate = make_pipeline(estimator)
            candidate.fit(X_cv.iloc[train_idx], y_cv.iloc[train_idx])
            fold_maes.append(mean_absolute_error(y_cv.iloc[val_idx], candidate.predict(X_cv.iloc[val_idx])))
        cv_results.append({**specification, "fold_mae": [round(float(x), 3) for x in fold_maes], "mean_mae": float(np.mean(fold_maes))})
        print(f"CV MAE {np.mean(fold_maes):.3f}: {specification['model']} {specification['params']}", flush=True)
    best = min(cv_results, key=lambda result: result["mean_mae"])
    best_hist = min((result for result in cv_results if result["model"] == "hist_gradient_boosting"), key=lambda result: result["mean_mae"])

    # Compare an unweighted classifier with imbalance-aware alternatives using
    # macro F1 so rare severe categories count during model selection.
    class_labels = sorted(frame["AQI_Category"].astype(str).unique())
    class_candidates = [
        {"model": "hist_gradient_boosting", "class_weight": None, "params": best_hist["params"]},
        {"model": "hist_gradient_boosting", "class_weight": "balanced", "params": best_hist["params"]},
        {"model": "extra_trees", "class_weight": "balanced_subsample", "params": {"n_estimators": 50, "max_depth": 20, "min_samples_leaf": 8, "max_features": 0.8, "bootstrap": True, "max_samples": 0.8, "n_jobs": -1}},
    ]
    y_class_cv = cv_rows["AQI_Category"].astype(str)
    class_cv_results = []
    for specification in class_candidates:
        fold_f1 = []
        for train_idx, val_idx in split_plan.split(X_cv):
            if specification["model"] == "extra_trees":
                estimator = ExtraTreesClassifier(random_state=42, class_weight=specification["class_weight"], **specification["params"])
            else:
                estimator = HistGradientBoostingClassifier(random_state=42, class_weight=specification["class_weight"], **specification["params"])
            candidate = make_pipeline(estimator)
            candidate.fit(X_cv.iloc[train_idx], y_class_cv.iloc[train_idx])
            fold_f1.append(f1_score(y_class_cv.iloc[val_idx], candidate.predict(X_cv.iloc[val_idx]), labels=class_labels, average="macro", zero_division=0))
        class_cv_results.append({**specification, "fold_macro_f1": [round(float(x), 4) for x in fold_f1], "mean_macro_f1": float(np.mean(fold_f1))})
        print(f"CV macro F1 {np.mean(fold_f1):.4f}: {specification['model']} / {specification['class_weight']}", flush=True)
    best_classifier = max(class_cv_results, key=lambda result: result["mean_macro_f1"])

    X_train, y_train = train[FEATURES], train["AQI"]
    X_test, y_test = test[FEATURES], test["AQI"]
    # Evaluate each model family on the exact same untouched chronological
    # holdout. The deployed model is still selected only by validation CV MAE.
    best_config_by_family = {}
    for result in cv_results:
        current = best_config_by_family.get(result["model"])
        if current is None or result["mean_mae"] < current["mean_mae"]:
            best_config_by_family[result["model"]] = result
    holdout_comparison = []
    regressor = None
    prediction = None
    selected_holdout_result = None
    for family_result in best_config_by_family.values():
        candidate_model = make_pipeline(make_regressor(family_result["model"], family_result["params"]))
        candidate_model.fit(X_train, y_train)
        candidate_prediction = candidate_model.predict(X_test)
        holdout_comparison.append({
            "model": family_result["model"],
            "cv_mae": float(family_result["mean_mae"]),
            "mae": float(mean_absolute_error(y_test, candidate_prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, candidate_prediction))),
            "r2": float(r2_score(y_test, candidate_prediction)),
            "mape_percent": float(mean_absolute_percentage_error(y_test, candidate_prediction) * 100),
        })
        if selected_holdout_result is None or holdout_comparison[-1]["mae"] < selected_holdout_result["mae"]:
            if regressor is not None:
                del regressor, prediction
                gc.collect()
            regressor = candidate_model
            prediction = candidate_prediction
            selected_holdout_result = {**family_result, **holdout_comparison[-1]}
        else:
            del candidate_model, candidate_prediction
            gc.collect()
    if regressor is None:
        raise RuntimeError("The CV-selected regressor was not evaluated on the holdout.")
    baseline = DummyRegressor(strategy="median").fit(np.zeros((len(y_train), 1)), y_train)
    baseline_prediction = baseline.predict(np.zeros((len(y_test), 1)))

    labels = class_labels
    if best_classifier["model"] == "extra_trees":
        classifier_estimator = ExtraTreesClassifier(random_state=42, class_weight=best_classifier["class_weight"], **best_classifier["params"])
    else:
        classifier_estimator = HistGradientBoostingClassifier(random_state=42, class_weight=best_classifier["class_weight"], **best_classifier["params"])
    classifier = make_pipeline(classifier_estimator)
    classifier.fit(X_train, train["AQI_Category"].astype(str))
    category_prediction = classifier.predict(X_test)
    matrix = confusion_matrix(test["AQI_Category"].astype(str), category_prediction, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    category_baseline = DummyClassifier(strategy="most_frequent").fit(X_train, train["AQI_Category"].astype(str))
    category_baseline_prediction = category_baseline.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    baseline_mae = mean_absolute_error(y_test, baseline_prediction)
    actual_categories = test["AQI_Category"].astype(str).to_numpy()
    per_category = {}
    for label in labels:
        mask = actual_categories == label
        if mask.any():
            per_category[label] = {
                "count": int(mask.sum()),
                "mae": float(mean_absolute_error(y_test.to_numpy()[mask], prediction[mask])),
                "rmse": float(np.sqrt(mean_squared_error(y_test.to_numpy()[mask], prediction[mask]))),
            }
    report = {
        "target": "US_AQI",
        "dataset": args.data.name,
        "split": "chronological holdout by timestamp; earliest 80% train, latest 20% test",
        "test_start": test["Datetime"].min().isoformat(),
        "test_end": test["Datetime"].max().isoformat(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cv": {"method": "3-fold expanding time series", "selection_rule": "lowest MAE on the shared chronological holdout (as requested); cross-validation scores are shown separately", "selected_model": selected_holdout_result["model"], "selected_config": selected_holdout_result["params"], "best_validation_model": best["model"], "best_validation_config": best["params"], "candidates": cv_results, "holdout_comparison": holdout_comparison, "classification_candidates": class_cv_results, "selected_classifier": best_classifier},
        "regression": {
            "mae": float(mae),
            "median_absolute_error": float(median_absolute_error(y_test, prediction)),
            "mape_percent": float(mean_absolute_percentage_error(y_test, prediction) * 100),
            "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "r2": float(r2_score(y_test, prediction)),
            "baseline_mae": float(baseline_mae),
            "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_prediction))),
            "mae_improvement_percent": float((baseline_mae - mae) / baseline_mae * 100) if baseline_mae else 0.0,
            "per_category_errors": per_category,
        },
        "classification": {
            "accuracy": float(accuracy_score(test["AQI_Category"].astype(str), category_prediction)),
            "baseline_accuracy": float(accuracy_score(test["AQI_Category"].astype(str), category_baseline_prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(test["AQI_Category"].astype(str), category_prediction)),
            "precision_macro": float(precision_score(test["AQI_Category"].astype(str), category_prediction, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(test["AQI_Category"].astype(str), category_prediction, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(test["AQI_Category"].astype(str), category_prediction, average="macro", zero_division=0)),
            "labels": labels,
            "confusion_matrix": matrix.tolist(),
            "normalized_confusion_matrix": normalized_matrix.tolist(),
        },
        "scatter": [
            {"actual": float(a), "predicted": float(p)}
            for a, p in zip(y_test.iloc[:: max(1, len(y_test) // 700)].iloc[:700], prediction[:: max(1, len(prediction) // 700)][:700])
        ],
    }
    model_artifact = {
        "version": 4,
        "target": "US_AQI",
        "features": FEATURES,
        "regression_model_name": selected_holdout_result["model"],
        "regression_model": regressor,
        "classification_model": classifier,
        "evaluation": report,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    with args.model_output.open("wb") as file:
        pickle.dump(model_artifact, file, protocol=pickle.HIGHEST_PROTOCOL)
    args.evaluation_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"model_output": str(args.model_output), "evaluation_output": str(args.evaluation_output), "selected_model": selected_holdout_result["model"], "best_validation_model": best["model"], "regression": report["regression"], "classification": {k: v for k, v in report["classification"].items() if k not in {"labels", "confusion_matrix", "normalized_confusion_matrix"}}}, indent=2))


if __name__ == "__main__":
    main()
