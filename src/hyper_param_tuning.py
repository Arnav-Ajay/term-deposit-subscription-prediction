# src/tune.py

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/term-deposit-marketing-2020.csv"


def build_pipeline(df):

    # Drop leakage features
    df = df.drop(columns=["duration"], errors="ignore")

    y = df["y"].map({"yes": 1, "no": 0})
    X = df.drop(columns=["y"])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    model = HistGradientBoostingClassifier(
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipeline, X, y


def main():
    df = pd.read_csv(DATA_PATH)
    pipeline, X, y = build_pipeline(df)

    param_dist = {
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_iter": [100, 200, 300, 500],
        "model__max_depth": [None, 3, 5, 7],
        "model__min_samples_leaf": [10, 20, 30, 50],
        "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
        "model__max_bins": [255, 128, 64]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    print("Running Hyperparameter Search...")
    search.fit(X, y)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest ROC-AUC:")
    print(search.best_score_)


if __name__ == "__main__":
    main()
