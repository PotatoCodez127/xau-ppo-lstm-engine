import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class GoldTradingEnv(gym.Env):
    """
    Custom Trading Environment for XAUUSD.
    Designed specifically to interface with an LSTM-PPO agent.
    """
    metadata = {'render_modes': ['human', 'system']}

    def __init__(self, df: pd.DataFrame, session: str = 'ALL', window_size: int = 30, initial_balance: float = 10000.0):
        super(GoldTradingEnv, self).__init__()
        
        # 1. Session Filtering
        self.df = self._filter_session(df.copy(), session)
        self.window_size = window_size
        
        # We need continuous indices for stepping through the environment
        self.df = self.df.reset_index() 
        
        # Extract purely numerical features for the AI's observation
        # (Exclude timestamps and purely informational strings)
        self.feature_cols = [c for c in self.df.columns if self.df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        self.data = self.df[self.feature_cols].values
        
        # 2. Spaces definition
        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell, 3 = Close Position
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: A 2D array representing (Window_Size x Number_of_Features)
        # This is strictly required for the LSTM to process sequences.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, len(self.feature_cols)), 
            dtype=np.float32
        )
        
        # Account variables
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.position = 0 # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0
        
    def _filter_session(self, df: pd.DataFrame, session: str) -> pd.DataFrame:
        """Filters the dataframe to strictly trade within specific regimes."""
        # Force the 'time' column to be actual Datetime objects, not strings
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        # If time is already the index but still a string, convert it
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Using detected EET Broker Time
        if session == 'LONDON':
            return df.between_time('09:00', '14:00')
        elif session == 'NY':
            return df.between_time('14:00', '18:00')
        return df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        
        obs = self._next_observation()
        info = self._get_info()
        return obs, info

    def _next_observation(self):
        # Returns the trailing window of features up to the current step
        obs = self.data[self.current_step - self.window_size : self.current_step]
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        # Calculate live floating PnL
        current_price = self.df.loc[self.current_step, 'close']
        unrealized_pnl = (current_price - self.entry_price) * self.position if self.position != 0 else 0.0
        
        return {
            "balance": self.balance,
            "equity": self.balance + unrealized_pnl, # True account value
            "position": self.position,
            "step": self.current_step
        }

    def step(self, action):
        self.current_step += 1
        
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        current_price = self.df.loc[self.current_step, 'close']
        reward = 0.0
        
        # --- 1. ACTION EXECUTION (Asymmetric Realized Rewards) ---
        if action == 1: # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1: 
                trade_profit = self.entry_price - current_price
                self.balance += trade_profit
                self.position = 1
                self.entry_price = current_price
                # 5x Dopamine hit for winning, normal 1x pain for losing
                reward += (trade_profit * 5.0) if trade_profit > 0 else trade_profit
                
        elif action == 2: # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1: 
                trade_profit = current_price - self.entry_price
                self.balance += trade_profit
                self.position = -1
                self.entry_price = current_price
                reward += (trade_profit * 5.0) if trade_profit > 0 else trade_profit
                
        elif action == 3: # CLOSE
            if self.position == 1:
                trade_profit = current_price - self.entry_price
                self.balance += trade_profit
                self.position = 0
                reward += (trade_profit * 5.0) if trade_profit > 0 else trade_profit
            elif self.position == -1:
                trade_profit = self.entry_price - current_price
                self.balance += trade_profit
                self.position = 0
                reward += (trade_profit * 5.0) if trade_profit > 0 else trade_profit
                
        # --- 2. FLOATING PENALTIES & BREADCRUMBS ---
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position
            
            if unrealized_pnl < 0:
                reward += (unrealized_pnl * 0.1) 
            else:
                reward += 0.05 
                
        if self.position == 0:
            reward -= 0.001

        # Blowout check
        if self.balance <= 0:
            terminated = True
            reward -= 10000

        obs = self._next_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info