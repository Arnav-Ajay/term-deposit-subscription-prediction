# src/predict.py

import os
import argparse
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Default Paths
# -----------------------------
DEFAULT_MODEL_PATH = "models/hgb_model.joblib"
DEFAULT_INPUT_PATH = "data/holdout_500.csv"
DEFAULT_OUTPUT_PATH = "data/predictions_output.csv"
DEFAULT_THRESHOLD = 0.5


# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return joblib.load(model_path)


# -----------------------------
# Load Input File
# -----------------------------
def load_input(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV and Excel files are supported.")

    return df


# -----------------------------
# Predict
# -----------------------------
def predict(model, df, threshold=0.5):

    # Remove duration if user accidentally includes it
    df = df.drop(columns=["duration"], errors="ignore")

    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)

    df_result = df.copy()
    df_result["predicted_probability"] = probs
    df_result["predicted_label"] = preds
    df_result["predicted_label"] = df_result["predicted_label"].map({1: "yes", 0: "no"})

    return df_result


# -----------------------------
# CLI
# -----------------------------
def main():

    parser = argparse.ArgumentParser(description="Predict term deposit subscription")

    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model (.joblib)"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to input CSV/Excel file"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save predictions"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Classification threshold (default=0.5)"
    )

    parser.add_argument(
        "--save_output",
        action="store_true",
        help="Save predictions to output file"
    )

    args = parser.parse_args()

    print("\nLoading model...")
    model = load_model(args.model_path)

    print("Loading input data...")
    df_input = load_input(args.input_path)

    print("Generating predictions...")
    df_predictions = predict(model, df_input, threshold=args.threshold)

    print("\nPrediction Results:\n")
    print(df_predictions.head())

    if args.save_output:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        df_predictions.to_csv(args.output_path, index=False)
        print(f"\nPredictions saved to: {args.output_path}")

    print("\nPrediction complete.")


if __name__ == "__main__":
    main()