import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO  # CRITICAL FIX: Upgraded to RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.trading_env import GoldTradingEnv


def run_forward_test(model_path: str, test_data_path: str, session: str):
    print(f"Loading Test Data from {test_data_path}...")
    df_test = pd.read_csv(test_data_path)

    print("Initializing Environment in Evaluation Mode...")
    env_raw = GoldTradingEnv(df=df_test, session=session, window_size=30)

    env = DummyVecEnv([lambda: env_raw])
    env = VecNormalize.load(f"vec_normalize_{session.lower()}.pkl", env)
    env.training = False
    env.norm_reward = False

    # --- REALISTIC SPREAD INJECTION ---
    # Override the environment's default $0.30 spread to a realistic tight $0.05 spread
    env.set_attr("spread", 0.05)
    print("Injected realistic tight spread ($0.05) into testing environment.")

    print(f"Loading Trained AI Brain from {model_path}...")
    model = RecurrentPPO.load(model_path)

    # --- LSTM STATE TRACKING ---
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    obs = env.reset()
    info = env.env_method("_get_info")[0]
    initial_balance = env.get_attr("initial_balance")[0]

    # --- ADVANCED METRICS TRACKING ---
    equity_curve = [initial_balance]
    trade_journal = []

    peak_equity = initial_balance
    max_drawdown_pct = 0.0

    active_trade_entry_step = None
    entry_balance = None
    holding_times = []
    winning_trades = 0
    losing_trades = 0

    print("\n" + "=" * 40)
    print("   STARTING LIVE MARKET SIMULATION")
    print("=" * 40)

    done = False

    while not done:
        # 1. Feed the current "Live" Candle and LSTM state to the brain
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )

        # 2. Step the market forward by one candle
        obs, rewards, dones, infos = env.step(action)

        info = infos[0]
        done = dones[0]
        episode_starts = dones  # Store for next bar's memory

        current_equity = info["equity"]
        current_balance = info["balance"]
        current_step = info["step"]
        position = info["position"]

        # --- DRAWDOWN TRACKING ---
        if current_equity > peak_equity:
            peak_equity = current_equity

        current_dd = (peak_equity - current_equity) / peak_equity
        if current_dd > max_drawdown_pct:
            max_drawdown_pct = current_dd

        equity_curve.append(current_equity)

        # --- ACTION & HOLDING TIME LOGIC ---
        action_val = int(action[0])
        action_name = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}[action_val]

        if action_name != "HOLD":
            # Simulate real-time terminal output
            print(
                f"[LIVE STREAM] Candle {current_step} | Action: {action_name:<5} | "
                f"Pos: {position:>2} | Equity: ${current_equity:.2f}"
            )

            # Logic for recording trade metrics
            if action_name in ["BUY", "SELL"] and active_trade_entry_step is None:
                # Trade Opened
                active_trade_entry_step = current_step
                entry_balance = current_balance

            elif action_name == "CLOSE" or (
                action_name in ["BUY", "SELL"] and active_trade_entry_step is not None
            ):
                # Trade Closed or Reversed
                if active_trade_entry_step is not None:
                    bars_held = current_step - active_trade_entry_step
                    holding_times.append(bars_held)

                    # Check Win/Loss based on realized balance increase
                    if current_balance > entry_balance:
                        winning_trades += 1
                    elif current_balance < entry_balance:
                        losing_trades += 1

                    if action_name == "CLOSE":
                        active_trade_entry_step = None
                    else:
                        # Agent reversed position instantly without closing first
                        active_trade_entry_step = current_step
                        entry_balance = current_balance

            trade_journal.append(
                {
                    "step": current_step,
                    "action": action_name,
                    "position_state": position,
                    "balance": current_balance,
                    "equity": current_equity,
                    "drawdown_pct": round(current_dd * 100, 4),
                }
            )

    # --- ADVANCED REPORT GENERATION ---
    print("\n" + "=" * 40)
    print("      SIMULATION COMPLETE REPORT")
    print("=" * 40)

    total_trades = winning_trades + losing_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    avg_hold = sum(holding_times) / len(holding_times) if holding_times else 0.0
    net_profit = current_equity - initial_balance

    print(f"Initial Balance:    ${initial_balance:,.2f}")
    print(f"Ending Equity:      ${current_equity:,.2f}")
    print(
        f"Net Realized PnL:   ${net_profit:,.2f} ({(net_profit/initial_balance)*100:.2f}%)"
    )
    print("-" * 40)
    print(f"Total Trades Taken: {total_trades}")
    print(f"Win Rate:           {win_rate:.2f}%")
    print(
        f"Avg Holding Time:   {avg_hold:.1f} candles ({avg_hold * 15 / 60:.2f} hours)"
    )
    print(f"Maximum Drawdown:   {max_drawdown_pct * 100:.2f}%")
    print("=" * 40)

    journal_df = pd.DataFrame(trade_journal)
    journal_path = "advanced_forward_journal.csv"
    journal_df.to_csv(journal_path, index=False)
    print(f"\nDetailed Trading Journal saved to {journal_path}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="AI True Equity", color="gold", linewidth=1.5)
    plt.axhline(y=initial_balance, color="red", linestyle="--", label="Initial Balance")
    plt.fill_between(
        range(len(equity_curve)),
        initial_balance,
        equity_curve,
        where=(np.array(equity_curve) >= initial_balance),
        color="green",
        alpha=0.1,
    )
    plt.fill_between(
        range(len(equity_curve)),
        initial_balance,
        equity_curve,
        where=(np.array(equity_curve) < initial_balance),
        color="red",
        alpha=0.1,
    )
    plt.title("XAUUSD AI Strategy Forward Test (5M Steps, Tight Spread: $0.05)")
    plt.xlabel("Timesteps (15m Candles)")
    plt.ylabel("Account Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_forward_test(
        model_path="lstm_ppo_gold_london_final.zip",
        test_data_path="../data/processed/test_features_15m.csv",
        session="LONDON",
    )
