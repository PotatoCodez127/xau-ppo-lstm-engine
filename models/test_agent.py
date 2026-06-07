import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # NEW IMPORT

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.trading_env import GoldTradingEnv

def run_forward_test(model_path: str, test_data_path: str, session: str):
    print(f"Loading Test Data from {test_data_path}...")
    df_test = pd.read_csv(test_data_path)
    
    print("Initializing Environment in Evaluation Mode...")
    # 1. Base Environment
    env_raw = GoldTradingEnv(df=df_test, session=session, window_size=30)
    
    # 2. Vectorize and Load the Normalization Scaler
    env = DummyVecEnv([lambda: env_raw])
    env = VecNormalize.load(f"vec_normalize_{session.lower()}.pkl", env)
    
    # 3. CRITICAL: Disable training mode on the scaler so it doesn't leak future test data
    env.training = False
    env.norm_reward = False
    
    print(f"Loading Trained AI Brain from {model_path}...")
    model = PPO.load(model_path)
    
    # VecEnv reset and info extraction
    obs = env.reset()
    info = env.env_method("_get_info")[0]
    
    equity_curve = [info['equity']]
    trade_journal = []
    
    print("Starting Forward Simulation (Inference Mode)...")
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        # Step through the Vectorized Environment
        obs, rewards, dones, infos = env.step(action)
        
        info = infos[0]
        done = dones[0]
        
        equity_curve.append(info['equity'])
        
        action_val = int(action[0])
        action_name = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}[action_val]
        
        if action_name != "HOLD":
            trade_journal.append({
                "step": info['step'],
                "action": action_name,
                "position_state": info['position'],
                "balance": info['balance'],
                "equity": info['equity']
            })

    print("\n--- SIMULATION COMPLETE ---")
    initial_balance = env.get_attr("initial_balance")[0]
    print(f"Starting Balance: ${initial_balance}")
    print(f"Ending Balance:   ${info['balance']:.2f}")
    print(f"Ending Equity:    ${info['equity']:.2f}")
    
    journal_df = pd.DataFrame(trade_journal)
    journal_path = "forward_trading_journal.csv"
    journal_df.to_csv(journal_path, index=False)
    print(f"Trading Journal saved to {journal_path}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='AI Account True Equity', color='gold')
    plt.axhline(y=initial_balance, color='red', linestyle='--', label='Breakeven')
    plt.title(f"XAUUSD AI Forward Test - {session} Session")
    plt.xlabel("Timesteps (15m Candles)")
    plt.ylabel("Account Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_forward_test(
        model_path='lstm_ppo_gold_london_final.zip',
        test_data_path='../data/processed/test_features_15m.csv',
        # test_data_path='../data/processed/master_features_15m.csv',
        session='LONDON'
    )