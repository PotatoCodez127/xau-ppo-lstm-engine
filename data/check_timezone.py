import pandas as pd


def find_broker_timezone(csv_path: str):
    df = pd.read_csv(csv_path, index_col="time", parse_dates=True)

    # Extract the hour of every row in the dataset
    df["hour"] = df.index.hour

    # Count how many total minutes of data exist for each hour (0 to 23)
    hour_counts = df.groupby("hour").size()

    # The hour with the absolute lowest count (usually 0) is the daily close
    dead_hour = hour_counts.idxmin()

    print("\n--- TIMEZONE DETECTED ---")
    print(f"The 1-hour Gold market close happens at: {dead_hour}:00 Broker Time.")

    if dead_hour == 23 or dead_hour == 0:
        print("Your broker is using standard EET (Eastern European Time).")
        print("Set London Session to: '09:00' to '14:00'")
        print("Set New York Session to: '14:00' to '18:00'")
    elif dead_hour == 21 or dead_hour == 22:
        print("Your broker is using GMT.")
        print("Set London Session to: '08:00' to '13:00'")
        print("Set New York Session to: '13:00' to '17:00'")
    else:
        print(
            f"Unusual timezone detected. You need to offset EST by ({dead_hour} - 17) hours."
        )


if __name__ == "__main__":
    find_broker_timezone("processed/xauusd_m1_clean.csv")
