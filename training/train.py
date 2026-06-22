"""
Trains a scikit-learn pipeline on the synthetic Stockholm housing dataset
and saves it to models/housing-price-pipeline.joblib.

Pipeline:
  ColumnTransformer
    └─ OneHotEncoder  →  area (categorical)
    └─ passthrough    →  rooms, size, monthlyFee (numeric)
  RandomForestRegressor

Evaluation: R² and RMSE on a held-out 20 % test split.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH  = Path(__file__).parent.parent / "data"   / "stockholm-housing.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "housing-price-pipeline.joblib"

CATEGORICAL = ["area"]
NUMERIC     = ["rooms", "size", "monthlyFee"]
FEATURES    = CATEGORICAL + NUMERIC
TARGET      = "price"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    return df[FEATURES], df[TARGET]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> None:
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pct  = rmse / y_true.mean() * 100
    print(f"  R2:   {r2:.4f}")
    print(f"  RMSE: {rmse:>12,.0f} SEK  ({pct:.1f} % of mean price)")


if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH}")
    X, y = load_data()
    print(f"  {len(X):,} rows, {X.shape[1]} features\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train: {len(X_train):,} rows   Test: {len(X_test):,} rows\n")

    pipeline = build_pipeline()

    print("Training...")
    pipeline.fit(X_train, y_train)

    print("\nTest-set evaluation:")
    y_pred = pipeline.predict(X_test)
    evaluate(y_test, y_pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")
