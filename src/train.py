# src/train.py

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/term-deposit-marketing-2020.csv"
MODEL_PATH = "models/hgb_model.joblib"
META_PATH = "models/metadata.json"
MODEL_VERSION = "1.0.0"


# -----------------------------
# Build Pipeline
# -----------------------------
def build_pipeline(df):

    drop_cols = ["duration", "month", "day"]
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=["y"])
    y = df["y"].map({"yes": 1, "no": 0})

    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline, X, y, categorical_cols, numeric_cols


# -----------------------------
# Cross-Validation
# -----------------------------
def evaluate_model(pipeline, X, y):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
    }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    train_results = {
        metric: float(np.mean(cv_results[f"train_{metric}"]))
        for metric in scoring.keys()
    }

    test_results = {
        metric: float(np.mean(cv_results[f"test_{metric}"]))
        for metric in scoring.keys()
    }

    return train_results, test_results


# -----------------------------
# Save Metadata
# -----------------------------
def save_metadata(train_res, test_res, df, categorical_cols, numeric_cols):

    class_distribution = df["y"].value_counts(normalize=True).to_dict()

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "HistGradientBoostingClassifier",
        "training_timestamp": datetime.now().isoformat(),
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "target_column": "y",
        "excluded_columns": ["duration", "month", "day"],
        "categorical_features": categorical_cols,
        "numeric_features": numeric_cols,
        "class_distribution": class_distribution,
        "cv_train_metrics": train_res,
        "cv_validation_metrics": test_res,
        "hyperparameters": {
            "learning_rate": 0.05,
            "max_iter": 300,
            "random_state": 42
        }
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {META_PATH}")


# -----------------------------
# Main
# -----------------------------
def main():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    pipeline, X, y, cat_cols, num_cols = build_pipeline(df)

    print("\nRunning 5-Fold Cross Validation...\n")
    train_res, test_res = evaluate_model(pipeline, X, y)

    print("Training Results:", train_res)
    print("Validation Results:", test_res)

    print("\nTraining final model on full dataset...")
    pipeline.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")

    save_metadata(train_res, test_res, df, cat_cols, num_cols)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()