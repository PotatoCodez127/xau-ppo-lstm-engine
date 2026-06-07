import pandas as pd
import numpy as np
import os
from datetime import timedelta

def generate_synthetic_gold(input_csv: str, output_csv: str, target_years: int = 4):
    """
    Expands a small XAUUSD dataset into years of synthetic data using 24-Hour Block Bootstrapping.
    Preserves Candlestick shapes, volatility, and exact time-of-day session behaviors.
    """
    print(f"Loading base data from {input_csv}...")
    df = pd.read_csv(input_csv, index_col='time', parse_dates=True)
    
    # 1. Calculate intra-candle and inter-candle percentage returns
    # This prevents chart gaps when we stitch different days together
    df['prev_close'] = df['close'].shift(1)
    df = df.dropna()
    
    df['ret_open'] = (df['open'] / df['prev_close']) - 1
    df['ret_high'] = (df['high'] / df['prev_close']) - 1
    df['ret_low'] = (df['low'] / df['prev_close']) - 1
    df['ret_close'] = (df['close'] / df['prev_close']) - 1

    # 2. Extract strictly 24-hour Daily Blocks
    print("Extracting 24-Hour continuous blocks...")
    df['date_only'] = df.index.date
    unique_days = df['date_only'].unique()
    
    daily_blocks = []
    for day in unique_days:
        day_data = df[df['date_only'] == day]
        # Only keep days that have roughly a full 24-hour cycle (at least 1400 minutes)
        # This drops partial weekend/holiday sessions that would ruin the time alignment
        if len(day_data) > 1200: 
            daily_blocks.append(day_data[['ret_open', 'ret_high', 'ret_low', 'ret_close']].copy())
            
    print(f"Successfully extracted {len(daily_blocks)} perfect 24-hour blocks.")

    # 3. Define the synthetic timeline
    # 4 years * 252 trading days roughly equals 1008 days
    target_days = target_years * 252 
    print(f"Synthesizing {target_days} continuous trading days ({target_years} years)...")

    # 4. Bootstrapping (The Shuffle)
    np.random.seed(42) # For reproducibility 
    # Pick random index numbers instead of passing the whole DataFrames to numpy
    random_indices = np.random.choice(len(daily_blocks), size=target_days, replace=True)
    synthetic_blocks = [daily_blocks[i] for i in random_indices]
    
    # Concatenate the shuffled returns into one massive column
    synth_df = pd.concat(synthetic_blocks, ignore_index=True)
    
    # 5. Reconstruct the Absolute Prices seamlessly
    print("Reconstructing absolute continuous prices...")
    start_price = 2000.00 # Arbitrary starting point for the synthetic reality
    
    # Calculate continuous close first
    close_multipliers = 1 + synth_df['ret_close']
    cumulative_multipliers = close_multipliers.cumprod()
    synth_df['close'] = start_price * cumulative_multipliers
    
    # Shift to get the running previous close
    synth_df['synth_prev_close'] = synth_df['close'].shift(1).fillna(start_price)
    
    # Reconstruct O, H, L based on the running previous close
    synth_df['open'] = synth_df['synth_prev_close'] * (1 + synth_df['ret_open'])
    synth_df['high'] = synth_df['synth_prev_close'] * (1 + synth_df['ret_high'])
    synth_df['low'] = synth_df['synth_prev_close'] * (1 + synth_df['ret_low'])
    
    # 6. Generate continuous synthetic timestamps (ignoring weekends for pure sequential logic)
    start_time = pd.Timestamp("2020-01-01 00:00:00")
    # Generating purely continuous minute timestamps
    synth_timestamps = [start_time + timedelta(minutes=i) for i in range(len(synth_df))]
    synth_df['time'] = synth_timestamps
    
    # Clean up and save
    final_cols = ['time', 'open', 'high', 'low', 'close']
    final_df = synth_df[final_cols].set_index('time')
    
    # Round to MT5 standard decimals (2 for Gold)
    final_df = final_df.round(2)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv)
    print(f"Synthetic Masterpiece Saved! -> {output_csv} | Total Rows: {len(final_df)}")

if __name__ == "__main__":
    generate_synthetic_gold(
        input_csv='processed/xauusd_m1_clean.csv', 
        output_csv='processed/xauusd_m1_synthetic_4yrs.csv',
        target_years=4
    )