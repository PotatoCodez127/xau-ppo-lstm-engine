import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.trading_env import GoldTradingEnv

def run_forward_test(model_path: str, test_data_path: str, session: str):
    """
    Loads a trained PPO model and tests it on unseen data.
    Generates a trading journal and an equity curve.
    """
    print(f"Loading Test Data from {test_data_path}...")
    df_test = pd.read_csv(test_data_path)
    
    print("Initializing Environment in Evaluation Mode...")
    # We use the exact same environment, but with the test dataset
    env = GoldTradingEnv(df=df_test, session=session, window_size=30)
    
    print(f"Loading Trained AI Brain from {model_path}...")
    model = PPO.load(model_path)
    
    # Reset the environment to start at candle 0
    obs, info = env.reset()
    
    # Tracking variables for our Trading Journal
    equity_curve = [info['balance']]
    trade_journal = []
    
    print("Starting Forward Simulation (Inference Mode)...")
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # The AI looks at the observation and decides the absolute best action
        # deterministic=True disables random exploration
        action, _states = model.predict(obs, deterministic=True)
        
        # We step the environment forward 1 candle based on the AI's action
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        # Log the equity curve
        equity_curve.append(info['balance'])
        
        # Log the actions for the journal
        action_name = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}[int(action)]
        if action_name != "HOLD":
            trade_journal.append({
                "step": info['step'],
                "action": action_name,
                "position_state": info['position'],
                "balance": info['balance']
            })

    print("\n--- SIMULATION COMPLETE ---")
    print(f"Starting Balance: ${env.initial_balance}")
    print(f"Ending Balance:   ${info['balance']:.2f}")
    
    # 1. Save Trading Journal
    journal_df = pd.DataFrame(trade_journal)
    journal_path = "forward_trading_journal.csv"
    journal_df.to_csv(journal_path, index=False)
    print(f"Trading Journal saved to {journal_path}")
    
    # 2. Plot the Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='AI Account Balance', color='gold')
    plt.axhline(y=env.initial_balance, color='red', linestyle='--', label='Breakeven')
    plt.title(f"XAUUSD AI Forward Test - {session} Session")
    plt.xlabel("Timesteps (15m Candles)")
    plt.ylabel("Account Balance ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Ensure these paths match what you generated
    run_forward_test(
        model_path='lstm_ppo_gold_london.zip',
        test_data_path='../data/processed/test_features_15m.csv',
        session='LONDON'
    )