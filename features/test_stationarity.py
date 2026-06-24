import pandas as pd


def run_stationarity_test(csv_path: str):
    print(f"--- Running Stationarity Unit Test on {csv_path} ---")
    df = pd.read_csv(csv_path, index_col="time")
    failed = False

    for col in df.columns:
        # We ignore the environment's raw price columns
        if col.startswith("env_"):
            continue

        max_val = df[col].max()
        min_val = df[col].min()

        if max_val > 0.5 or min_val < -0.5:
            print(
                f"❌ FAIL: AI Feature '{col}' leaked raw data! (Min: {min_val:.2f},"
                f" Max: {max_val:.2f})"
            )
            failed = True
        else:
            print(
                f"✅ PASS: '{col}' is stationary. (Min: {min_val:.5f}, Max: {max_val:.5f})"
            )

    if not failed:
        print("\n🏆 ALL TESTS PASSED! The AI is completely blind to raw prices.")
    else:
        print("\n⚠️ WARNING: Raw prices are still leaking into the dataset.")


if __name__ == "__main__":
    run_stationarity_test("../data/processed/test_features_15m.csv")
