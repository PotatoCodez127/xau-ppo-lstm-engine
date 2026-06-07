import pandas as pd
from build_pivots import calculate_daily_levels
from build_zones import calculate_wick_zones, add_15m_ema

def resample_hloc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resamples M1 data to higher timeframes safely."""
    return df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

def build_master_dataset(xau_m1_path: str, dxy_m1_path: str) -> pd.DataFrame:
    """
    Orchestrates the creation of the ML Feature state.
    Base execution timeframe: 15m.
    """
    print("Loading raw M1 data...")
    # Load M1 XAUUSD (Assuming datetime index)
    df_m1 = pd.read_csv(xau_m1_path, index_col='time', parse_dates=True).sort_index()
    
    # 1. Calculate Daily Levels on M1
    df_m1 = calculate_daily_levels(df_m1)
    
    # 2. Build Base Execution Timeframe (15m)
    print("Building 15m Execution Base & 50 EMA...")
    df_15m = resample_hloc(df_m1, '15min')
    df_15m = add_15m_ema(df_15m, period=50)
    df_15m = calculate_wick_zones(df_15m, window=5)
    
    # Rename 15m specific columns to avoid confusion
    df_15m = df_15m.rename(columns=lambda x: f"{x}_15m" if 'zone' in x or 'rolling' in x else x)
    
    # 3. Build Higher Timeframe Context (30m, 4h)
    print("Mapping 30m and 4H structural zones...")
    df_30m = resample_hloc(df_m1, '30min')
    df_30m = calculate_wick_zones(df_30m, window=5)
    df_30m = df_30m.rename(columns=lambda x: f"{x}_30m" if 'zone' in x else x)
    
    df_4h = resample_hloc(df_m1, '4h')
    df_4h = calculate_wick_zones(df_4h, window=5)
    df_4h = df_4h.rename(columns=lambda x: f"{x}_4h" if 'zone' in x else x)
    
    # 4. Process DXY Context
    print("Processing DXY correlation metrics...")
    try:
        dxy_m1 = pd.read_csv(dxy_m1_path, index_col='time', parse_dates=True).sort_index()
        dxy_15m = resample_hloc(dxy_m1, '15min')
        dxy_15m['dxy_pct_change_15m'] = dxy_15m['close'].pct_change()
    except FileNotFoundError:
        print("Warning: DXY file not found. Generating zeros for testing purposes.")
        dxy_15m = pd.DataFrame(index=df_15m.index)
        dxy_15m['dxy_pct_change_15m'] = 0.0

    # 5. Stitch it all together using Merge AsOf (Zero Lookahead Bias)
    print("Stitching timelines together...")
    
    # Grab daily levels from M1 and snap them to the 15m index
    daily_cols = df_m1[['daily_eq', 'pivot', 'R1', 'S1']].resample('15min').last().ffill()
    master = pd.merge_asof(df_15m, daily_cols, left_index=True, right_index=True)
    
    # Merge HTF Zones (We only want the zone math, not the HTF candle prices)
    htf_cols_30m = [c for c in df_30m.columns if 'zone' in c]
    master = pd.merge_asof(master, df_30m[htf_cols_30m], left_index=True, right_index=True)
    
    htf_cols_4h = [c for c in df_4h.columns if 'zone' in c]
    master = pd.merge_asof(master, df_4h[htf_cols_4h], left_index=True, right_index=True)
    
    # Merge DXY
    master = pd.merge_asof(master, dxy_15m[['dxy_pct_change_15m']], left_index=True, right_index=True)
    # FIX: Fill missing DXY values with 0.0 (neutral momentum) to protect synthetic rows
    master['dxy_pct_change_15m'] = master['dxy_pct_change_15m'].fillna(0.0)
    # Cleanup NaN values resulting from the multi-timeframe S&R rolling windows
    master = master.dropna()
    
    # Cleanup NaN values resulting from the rolling windows
    master = master.dropna()
    print("Master Feature Dataset Complete.")
    
    return master

if __name__ == "__main__":
    print("Starting Master Processor for FORWARD TESTING...")
    # Point back to the REAL 2026 MT5 data we cleaned earlier
    df_master = build_master_dataset(
        '../data/processed/xauusd_m1_clean.csv', 
        '../data/processed/dxy_m1_clean.csv'
    )
    # Save as a distinct test file so we don't overwrite our training set
    df_master.to_csv('../data/processed/test_features_15m.csv')
    print("Successfully saved test_features_15m.csv")

# ==========================================================================
# ===============================FOR TRAINING===============================
# ==========================================================================

# if __name__ == "__main__":
#     print("Starting Master Processor...")
#     # Point to the 4-year synthetic data
#     df_master = build_master_dataset(
#         '../data/processed/xauusd_m1_synthetic_4yrs.csv', 
#         '../data/processed/dxy_m1_clean.csv'
#     )
#     # Save the output
#     df_master.to_csv('../data/processed/master_features_15m.csv')
#     print("Successfully saved master_features_15m.csv")

# ==========================================================================
# ===============================FOR TRAINING===============================
# ==========================================================================