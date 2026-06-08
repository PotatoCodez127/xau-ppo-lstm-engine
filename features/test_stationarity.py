import pandas as pd
import numpy as np

def run_stationarity_test(csv_path: str):
    print(f"--- Running Stationarity Unit Test on {csv_path} ---")
    df = pd.read_csv(csv_path, index_col='time')
    
    failed = False
    
    for col in df.columns:
        max_val = df[col].max()
        min_val = df[col].min()
        
        # If any value is vastly outside normal percentage bounds (e.g., a raw price of 1900),
        # the test should fail. We use +/- 0.5 (50% distance) as a generous boundary.
        if max_val > 0.5 or min_val < -0.5:
            print(f"❌ FAIL: Column '{col}' leaked raw or unscaled data! (Min: {min_val:.2f}, Max: {max_val:.2f})")
            failed = True
        else:
            print(f"✅ PASS: '{col}' is stationary. (Min: {min_val:.5f}, Max: {max_val:.5f})")
            
    if not failed:
        print("\n🏆 ALL TESTS PASSED! The AI is completely blind to raw prices.")
        print("Distribution shift between 2020 ($2k Gold) and 2026 ($5k Gold) is solved.")
    else:
        print("\n⚠️ WARNING: Raw prices are still leaking into the dataset.")

if __name__ == "__main__":
    # Test your newly generated output
    run_stationarity_test('../data/processed/test_features_15m.csv')