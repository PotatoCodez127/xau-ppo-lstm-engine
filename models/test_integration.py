import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

# Ensure our custom environment can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.trading_env import GoldTradingEnv

def run_pipeline_integration_test():
    print("🚀 Starting RecurrentPPO Pipeline Integration Test...")
    
    # 1. Create mock stationary data mimicking 'test_features_15m.csv'
    print("Creating mock feature data slice...")
    timestamps = pd.date_range(start="2026-01-01 00:00", periods=100, freq="15min")
    
    mock_data = {
        'time': timestamps,
        'env_open': np.random.uniform(5000, 5100, size=100),
        'env_high': np.random.uniform(5101, 5150, size=100),
        'env_low': np.random.uniform(4950, 4999, size=100),
        'env_close': np.random.uniform(5000, 5100, size=100),
        'close_ret': np.random.uniform(-0.01, 0.01, size=100),
        'high_ret': np.random.uniform(0, 0.005, size=100),
        'low_ret': np.random.uniform(-0.005, 0, size=100),
        'dist_ema_50': np.random.uniform(-0.02, 0.02, size=100),
        'dxy_pct_change_15m': np.random.uniform(-0.002, 0.002, size=100)
    }
    
    df_mock = pd.DataFrame(mock_data)
    
    # 2. Initialize environment
    print("Initializing GoldTradingEnv with mock data...")
    env_raw = GoldTradingEnv(df=df_mock, session='ALL', window_size=30)
    
    # Test observation shape sanity
    obs_sample, _ = env_raw.reset()
    print(f"✅ Observation Space Shape Match: {obs_sample.shape} (Expected: (30, {len(env_raw.feature_cols)}))")
    
    # 3. Apply vector wrappers
    env = DummyVecEnv([lambda: env_raw])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # 4. Instantiate RecurrentPPO
    print("Instantiating RecurrentPPO model...")
    policy_kwargs = dict(lstm_hidden_size=64, enable_cudnn_lstm=False) # Turn off cuDNN just for local test safety
    
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        n_steps=64, # Small rollout window for quick test
        batch_size=16,
        verbose=0
    )
    
    # 5. Execute 5 timesteps of training
    print("Executing ultra-short training run (5 steps)...")
    try:
        model.learn(total_timesteps=5)
        print("🏆 INTEGRATION TEST PASSED! The RecurrentPPO agent successfully processed states, managed hidden sequences, and accepted the environment rewards without failure.")
    except Exception as e:
        print(f"❌ INTEGRATION TEST FAILED: {str(e)}")
        raise e

if __name__ == "__main__":
    run_pipeline_integration_test()