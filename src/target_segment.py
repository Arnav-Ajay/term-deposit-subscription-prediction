# src/target_segment.py

import pandas as pd


def identify_target_segment(predictions_csv, top_percent=0.1):
    df = pd.read_csv(predictions_csv)

    df_sorted = df.sort_values(
        "predicted_probability",
        ascending=False
    ).reset_index(drop=True)

    cutoff = int(len(df_sorted) * top_percent)

    target_segment = df_sorted.head(cutoff)

    print(f"Targeting top {int(top_percent*100)}% customers")
    print(f"Estimated segment size: {len(target_segment)}")

    return target_segment


if __name__ == "__main__":
    segment = identify_target_segment(
        "data/predictions_output.csv",
        top_percent=0.1
    )

    segment.to_csv("data/target_segment.csv", index=False)
    print("Target segment exported.")
