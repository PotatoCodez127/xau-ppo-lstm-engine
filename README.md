```
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```
Project_Aurelius/
│
├── data/
│   ├── raw/                 # Raw tick/M1 data for XAUUSD and DXY
│   └── processed/           # Data after S&R and feature engineering
│
├── features/
│   ├── build_zones.py       # Logic to calculate 15m/30m/4h wick-based S&R
│   ├── build_pivots.py      # Logic to calculate Daily EQ and Pivots
│   └── merge_dxy.py         # Logic to align DXY timestamps with XAU
│
├── environment/
│   ├── trading_env.py       # The OpenAI Gym environment (the rules of the game)
│   └── reward_funcs.py      # The math that punishes drawdowns and rewards profit
│
├── models/
│   ├── lstm_ppo_london.py   # Training script for London session
│   ├── lstm_ppo_ny.py       # Training script for NY session
│   └── atr_regime_test.py   # Script to test the ATR dynamic filter
│
├── xagusd_sandbox/          # Completely separate app for Silver transfer learning
│   └── ... (mirrors structure above)
│
├── .gitignore
├── requirements.txt
└── README.md
```