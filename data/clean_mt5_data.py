import os

import pandas as pd


def clean_mt5_csv(input_path: str, output_path: str):
    """
    Cleans raw MT5 exported data into the standard format required by the AI.
    Converts MT5 '<DATE>' and '<TIME>' into a unified Pandas DatetimeIndex.
    """
    print(f"Cleaning {input_path}...")

    # 1. Read the raw MT5 data (MT5 exports are tab-separated)
    try:
        df = pd.read_csv(input_path, sep="\t")
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        return

    # 2. Clean headers: MT5 outputs '<DATE>', '<TIME>', etc. We need 'date', 'time'.
    df.columns = [
        col.replace("<", "").replace(">", "").lower().strip() for col in df.columns
    ]

    # 3. Merge and format Date & Time
    if "date" in df.columns and "time" in df.columns:
        # MT5 date format is usually YYYY.MM.DD, replace dots with dashes
        df["date"] = df["date"].str.replace(".", "-")
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

        # Drop the old date, time, and unused volume/spread columns to save memory
        cols_to_drop = ["date", "time", "tickvol", "vol", "spread"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # Rename the unified datetime column to 'time' and set it as the index
        df = df.rename(columns={"datetime": "time"})
        df = df.set_index("time")
    else:
        print("Warning: Could not find 'date' and 'time' columns. Found:", df.columns)
        return

    # 4. Filter for price action only and ensure chronological order
    essential_cols = ["open", "high", "low", "close"]
    df = df[[c for c in essential_cols if c in df.columns]].sort_index()

    # 5. Save the processed file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Successfully saved to {output_path} | Total Rows: {len(df)}")


if __name__ == "__main__":
    # Define your paths based on your actual raw data locations
    # Adjust these filenames to match exactly what you moved into your raw folder

    # Define Inputs
    raw_xau_path = "raw/XAUUSDr_M1_202602241013_202606052358.csv"
    raw_dxy_path = "raw/USDIndex_M1_202602031250_202606052359.csv"

    # Define Outputs
    clean_xau_path = "processed/xauusd_m1_clean.csv"
    clean_dxy_path = "processed/dxy_m1_clean.csv"

    # Execute the cleaning
    clean_mt5_csv(raw_xau_path, clean_xau_path)
    clean_mt5_csv(raw_dxy_path, clean_dxy_path)
