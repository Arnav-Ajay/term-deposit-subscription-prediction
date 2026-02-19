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
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, train_test_split
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/term-deposit-marketing-2020.csv"
MODEL_PATH = "models/hgb_model.joblib"
META_PATH = "models/metadata.json"
MODEL_VERSION = "1.0.0"

LEARNING_RATE = 0.03
MAX_ITER = 200
MIN_SAMPLE_LEAF = 50
MAX_DEPTH = None
MAX_BINS = 64
L2_REG = 0.1

# -----------------------------
# Build Pipeline
# -----------------------------
def build_pipeline(df):

    drop_cols = ["duration"]
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=["y"])
    # X["housing"].map({"yes": 1, "no": 0})
    # X["loan"].map({"yes": 1, "no": 0})
    y = df["y"].map({"yes": 1, "no": 0})

    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = HistGradientBoostingClassifier(
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITER,
        random_state=42,
        min_samples_leaf=MIN_SAMPLE_LEAF,
        max_depth= MAX_DEPTH,
        max_bins= MAX_BINS,
        l2_regularization= L2_REG,
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

    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)

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
        "excluded_columns": ["duration"],
        "categorical_features": categorical_cols,
        "numeric_features": numeric_cols,
        "class_distribution": class_distribution,
        "cv_train_metrics": train_res,
        "cv_validation_metrics": test_res,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "max_iter": MAX_ITER,
            "random_state": 42,
            "min_samples_leaf": MIN_SAMPLE_LEAF,
            "max_depth": MAX_DEPTH,
            "max_bins": MAX_BINS,
            "l2_regularization": L2_REG
        }
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {META_PATH}")

# -----------------------------
# Lift Calculation
# -----------------------------
def calculate_lift(pipeline, X, y, top_percents=[0.1, 0.2, 0.3, 0.5]):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Out-of-fold predicted probabilities
    probs = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    df_lift = pd.DataFrame({
        "y": y,
        "prob": probs
    })

    # Sort descending by predicted probability
    df_lift = df_lift.sort_values("prob", ascending=False).reset_index(drop=True)

    total_positives = df_lift["y"].sum()

    print("\n------ Lift Analysis ------\n")

    for pct in top_percents:
        cutoff = int(len(df_lift) * pct)
        top_slice = df_lift.iloc[:cutoff]

        captured = top_slice["y"].sum()
        capture_rate = captured / total_positives
        lift = capture_rate / pct

        print(f"Top {int(pct*100)}%:")
        print(f"  Capture Rate: {capture_rate:.4f}")
        print(f"  Lift: {lift:.2f}")
        print("")

def create_stratified_holdout(df, holdout_size=500, random_state=42):

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        stratify=df["y"],
        random_state=random_state
    )

    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df_full = pd.read_csv(DATA_PATH)

    train_df, holdout_df = create_stratified_holdout(df_full, holdout_size=500)

    print(f"Training rows: {train_df.shape[0]}")
    print(f"Holdout rows: {holdout_df.shape[0]}")

    # Save holdout set
    os.makedirs("data", exist_ok=True)
    holdout_df.to_csv("data/holdout_500.csv", index=False)

    df = train_df

    pipeline, X, y, cat_cols, num_cols = build_pipeline(df)

    print("\nRunning 5-Fold Cross Validation...\n")
    train_res, test_res = evaluate_model(pipeline, X, y)

    print("Training Results:", train_res)
    print("Validation Results:", test_res)

    print("\nTraining final model on full dataset...")
    pipeline.fit(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    roc_scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    mean_roc = np.mean(roc_scores["test_score"])
    print("Mean ROC-AUC (CV):", mean_roc)

    # Calculate Lift
    calculate_lift(pipeline, X, y)


    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")

    save_metadata(train_res, test_res, df, cat_cols, num_cols)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()