# models/test_integration.py

import os
import sys

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure our custom environment can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.trading_env import GoldTradingEnv


def run_pipeline_integration_test():
    print("🚀 Starting RecurrentPPO Pipeline Integration Test...")

    # 1. Create highly truncated mock stationary data to optimize CPU execution speed
    # Engineering Guardrail: Truncate periods <= 50 to avoid runner queue timeouts
    print("Creating miniature mock feature data slice...")
    timestamps = pd.date_range(start="2026-01-01 00:00", periods=45, freq="15min")

    mock_data = {
        "time": timestamps,
        "env_open": np.random.uniform(5000, 5100, size=45),
        "env_high": np.random.uniform(5101, 5150, size=45),
        "env_low": np.random.uniform(4950, 4999, size=45),
        "env_close": np.random.uniform(5000, 5100, size=45),
        "close_ret": np.random.uniform(-0.01, 0.01, size=45),
        "high_ret": np.random.uniform(0, 0.005, size=45),
        "low_ret": np.random.uniform(-0.005, 0, size=45),
        "dist_ema_50": np.random.uniform(-0.02, 0.02, size=45),
        "dxy_pct_change_15m": np.random.uniform(-0.002, 0.002, size=45),
    }

    df_mock = pd.DataFrame(mock_data)

    # 2. Initialize environment
    print("Initializing GoldTradingEnv with mock data...")
    env_raw = GoldTradingEnv(df=df_mock, session="ALL", window_size=30)

    # Test observation shape sanity dynamically
    obs_sample, _ = env_raw.reset()
    expected_shape = env_raw.observation_space.shape

    if obs_sample.shape == expected_shape:
        print(
            f"✅ Observation Space Shape Match: {obs_sample.shape} "
            f"(Dynamic Target verified: {expected_shape})"
        )
    else:
        print(f"❌ Shape Mismatch! Got: {obs_sample.shape}, Expected: {expected_shape}")
        sys.exit(1)

    # 3. Apply vector wrappers
    env = DummyVecEnv([lambda: env_raw])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 4. Instantiate RecurrentPPO with highly restricted batch parameters
    # Engineering Guardrail: Enforce batch_size <= 5 to bypass CPU un-mocked bottlenecks
    print("Instantiating RecurrentPPO model...")
    policy_kwargs = dict(lstm_hidden_size=32)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=32,  # Mini rollout execution layer
        batch_size=4,  # Optimized for cloud environments
        verbose=0,
    )

    # 5. Execute 2 timesteps of training
    print("Executing ultra-short training run (2 steps)...")
    try:
        model.learn(total_timesteps=2)
        print(
            "🏆 INTEGRATION TEST PASSED! The RecurrentPPO agent successfully processed states, "
            "managed hidden sequences, and accepted the environment rewards without failure."
        )
    except Exception as e:
        print(f"❌ INTEGRATION TEST FAILED: {str(e)}")
        raise e


if __name__ == "__main__":
    run_pipeline_integration_test()
