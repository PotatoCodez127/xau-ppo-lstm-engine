# 🧠 SYSTEM PROMPT & PROJECT MEMORY: PROJECT AURELIUS

**Developer:** Jeandre
**OS/Environment:** Windows (`C:\Users\Jeandre\Desktop\XAU_RL_Engine`)
**Core Objective:** Develop, train, and forward-test a Deep Reinforcement Learning (DRL) algorithmic trading bot for Gold (XAUUSD) using Proximal Policy Optimization (PPO), an LSTM feature extractor, and multi-timeframe structural data.

## 🛠️ Tech Stack & Architecture

* **Libraries:** Python 3.11+, PyTorch, Stable-Baselines3 (PPO), Gymnasium, Pandas, NumPy, Matplotlib.
* **Network Architecture:** Actor-Critic MLP with an LSTM temporal feature extractor.
* **Optimizer:** Adam.
* **Target Asset:** XAUUSD (Primary) with DXY (US Dollar Index) for intermarket momentum correlation.
* **Timeframes:** 15m execution base, heavily relying on multi-timeframe alignment (30m and 4H structural wick-based S&R zones, Daily EQ, Standard Pivots, 15m 50 EMA).
* **Broker Timezone Alignment:** Standard MT5 EET (Eastern European Time).

## 🗄️ Data Engineering State

* **Synthetic Bootstrapping:** A custom Block Bootstrapping engine (`bootstrap_data.py`) takes ~4 months of clean MT5 data and synthesizes it into 4 years of continuous 15m sequential data. This uses a strict 1200-minute daily cycle to perfectly preserve London (09:00-14:00 EET) and NY (14:00-18:00 EET) volume profiles.
* **Missing Features:** Synthetic timestamps from 2020 do not overlap with real DXY data from 2026. The `main_processor.py` resolves this via `fillna(0.0)` for DXY, feeding the network a "neutral momentum" signal for synthetic timelines to prevent `dropna()` from annihilating the dataset.

## 🎮 The Gymnasium Environment (`trading_env.py`)

* **Observation Space:** A rolling window (`window_size=30`) of raw numerical features.
* **Feature Scaling [CRITICAL]:** The raw environment is wrapped in Stable-Baselines3's `VecNormalize` inside `lstm_ppo.py`. This prevents the PyTorch LSTM's `tanh` activation functions from instantly saturating at `1.0` (which previously caused mathematical blindness).
* **Action Space:** `Discrete(4)` -> [0: HOLD, 1: BUY, 2: SELL, 3: CLOSE].
* **Reward Function (Asymmetric Realized PnL):** After fighting "Reward Hacking" (where the AI milked floating profit or refused to trade out of cowardice), the current psychological model is:
* *Realized Wins:* `trade_profit * 5.0` (Massive dopamine for executing a profitable `CLOSE`).
* *Realized Losses:* `trade_profit * 1.0` (Normal pain, preventing trauma).
* *Floating Drawdown:* `unrealized_pnl * 0.1` (Softened pain to allow the bot to breathe through normal Gold volatility wicks).
* *Breadcrumbs:* `+0.05` per step for floating in profit (encourages holding winners).
* *Inactivity Tax:* `-0.001` for sitting flat (prevents permanent hibernation without forcing bad trades).



## 🚀 Training & Inference Pipeline

* **Training (`lstm_ppo.py`):** Runs the PPO loop, outputs training metrics (Value Loss, Entropy, Approx KL) via terminal, logs granular step math to `logs/training_step_metrics.csv`, and auto-saves `.zip` models and `vec_normalize_london.pkl` scaler data.
* **Inference (`test_agent.py`):** Operates on unseen out-of-sample data. Loads the `.zip` brain and the `.pkl` scaler (`env.training = False`). Executes actions deterministically. Tracks **True Equity** (Balance + Floating PnL) to prevent "Perma-Bear/Bull" illusions, outputs a `forward_trading_journal.csv`, and plots a Matplotlib equity curve.

## 📍 EXACT CURRENT STATUS & NEXT ACTION

* **The Bug We Just Fixed:** The AI was previously running on an outdated reward function that caused the "Milking Exploit" (holding trades forever without closing) and lacked the `VecNormalize` scaler.
* **Pending User Action:** The developer needs to wipe the corrupted brains (`del *.zip`, `del *.pkl`, `rmdir /s /q saved_models`) and launch a fresh training run using the newly patched `trading_env.py` and scaled `lstm_ppo.py`.
* **Next Goal:** Evaluate the subsequent `forward_trading_journal.csv` to ensure the bot is successfully utilizing the `CLOSE` action to realize profits dynamically based on the S&R zones.