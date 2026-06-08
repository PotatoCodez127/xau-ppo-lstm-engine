import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class GoldTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'system']}

    def __init__(self, df: pd.DataFrame, session: str = 'ALL', window_size: int = 30, initial_balance: float = 10000.0):
        super(GoldTradingEnv, self).__init__()
        
        self.window_size = window_size
        self.df = self._filter_session(df.copy(), session).reset_index() 
        
        self.feature_cols = [
            c for c in self.df.columns 
            if self.df[c].dtype in [np.float64, np.float32, np.int64, np.int32] 
            and not c.startswith('env_')
        ]
        self.data = self.df[self.feature_cols].values
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, len(self.feature_cols)), 
            dtype=np.float32
        )
        
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.position = 0 
        self.entry_price = 0.0
        
        # --- INTRADAY RISK MECHANICS ---
        self.spread = 0.15       # Realistic standard account spread
        self.commission = 0.10   # Fixed commission per trade (forces seeking larger moves)
        self.bars_held = 0 
        self.max_drawdown_pct = 0.90
        
    def _filter_session(self, df: pd.DataFrame, session: str) -> pd.DataFrame:
        """Filters the dataframe, ensuring we keep all columns (including env_ prefixed ones)."""
        # Ensure time is handled correctly
        if 'time' in df.columns:
            df = df.set_index('time')
        elif not isinstance(df.index, pd.DatetimeIndex):
            # If the CSV didn't have a time column, we might be in trouble, 
            # but usually it's the index.
            pass
            
        # Convert index to datetime if it isn't already
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Filter rows
        if session == 'LONDON':
            df = df.between_time('09:00', '14:00')
        elif session == 'NY':
            df = df.between_time('14:00', '18:00')
            
        return df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.bars_held = 0
        return self._next_observation(), self._get_info()

    def _next_observation(self):
        obs = self.data[self.current_step - self.window_size : self.current_step]
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        # Using the hidden raw price for actual PnL
        current_price = self.df.loc[self.current_step, 'env_close']
        unrealized_pnl = (current_price - self.entry_price) * self.position if self.position != 0 else 0.0
        return {
            "balance": self.balance,
            "equity": self.balance + unrealized_pnl,
            "position": self.position,
            "step": self.current_step
        }

    def step(self, action):
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        current_price = self.df.loc[self.current_step, 'env_close']
        reward = 0.0
        
        if self.position != 0:
            self.bars_held += 1
        else:
            self.bars_held = 0

        # --- ACTION EXECUTION ---
        if action == 1: # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price + self.spread
                # Pay commission upfront
                self.balance -= self.commission 
                reward -= self.commission
            elif self.position == -1: 
                trade_profit = self.entry_price - (current_price + self.spread)
                self.balance += trade_profit - self.commission # Net profit minus next entry comm
                reward += trade_profit - self.commission
                self.position = 1
                self.entry_price = current_price + self.spread
                
        elif action == 2: # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price - self.spread
                # Pay commission upfront
                self.balance -= self.commission 
                reward -= self.commission
            elif self.position == 1: 
                trade_profit = (current_price - self.spread) - self.entry_price
                self.balance += trade_profit - self.commission
                reward += trade_profit - self.commission
                self.position = -1
                self.entry_price = current_price - self.spread
                
        elif action == 3: # CLOSE
            if self.position == 1:
                trade_profit = (current_price - self.spread) - self.entry_price
                self.balance += trade_profit
                reward += trade_profit
                self.position = 0
            elif self.position == -1:
                trade_profit = self.entry_price - (current_price + self.spread)
                self.balance += trade_profit
                reward += trade_profit
                self.position = 0
                
        # --- PENALTIES ---
        # REMOVED: Inactivity tax. The bot is now free to wait hours for the perfect setup.
            
        # Intraday Time decay: Punish dead trades held longer than 8 hours (32 bars)
        if self.bars_held > 32:
            reward -= 0.01 * (self.bars_held - 32)

        # --- CIRCUIT BREAKER ---
        info = self._get_info()
        if info["equity"] <= (self.initial_balance * self.max_drawdown_pct):
            terminated = True
            reward -= 1000 

        return self._next_observation(), reward, terminated, truncated, info