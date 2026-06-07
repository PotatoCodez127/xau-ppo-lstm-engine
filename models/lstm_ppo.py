import torch
import torch.nn as nn
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

# Import our custom environment (Adjust path if running from a different directory)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.trading_env import GoldTradingEnv

class TradingLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int = 5000, log_dir: str = "./logs", verbose=0):
        super(TradingLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.history = []
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Retrieve the environment's current step info
        # info is packed as a list of dicts because SB3 vectorizes environments
        env_info = self.training_env.env_method('_get_info')[0]
        
        # Access the last reward given to the agent
        last_reward = self.locals['rewards'][0]
        
        # Save historical snapshots of data points
        self.history.append({
            "total_timesteps": self.num_timesteps,
            "env_step": env_info['step'],
            "position": env_info['position'],
            "balance": env_info['balance'],
            "equity": env_info['equity'],
            "reward": last_reward
        })

        # Periodically flush history data to disk to avoid consuming excessive RAM
        if self.num_timesteps % self.check_freq == 0 and self.history:
            df_log = pd.DataFrame(self.history)
            log_file = os.path.join(self.log_dir, "training_step_metrics.csv")
            
            # Append if file exists, write header if it doesn't
            if not os.path.exists(log_file):
                df_log.to_csv(log_file, index=False)
            else:
                df_log.to_csv(log_file, mode='a', header=False, index=False)
            
            self.history = [] # Clear temporary buffer memory
        return True

class LSTMExtractor(BaseFeaturesExtractor):
    """
    Custom PyTorch Feature Extractor for Stable-Baselines3.
    Replaces the standard Multi-Layer Perceptron with a Recurrent Neural Network
    so the AI can perceive sequence and momentum over time.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # We must call the parent constructor
        super(LSTMExtractor, self).__init__(observation_space, features_dim)
        
        # Extract sequence length (window size) and number of features from the Gym environment
        self.seq_len = observation_space.shape[0]
        self.num_features = observation_space.shape[1]

        # The LSTM Network
        # batch_first=True means the input tensor is shaped (Batch, Sequence, Features)
        self.lstm = nn.LSTM(
            input_size=self.num_features, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True
        )
        
        # A linear layer to map the LSTM's final hidden state to the required feature dimension for PPO
        self.linear = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Gym/SB3 sometimes flattens multi-dimensional boxes. 
        # We ensure it's reshaped correctly for the LSTM: (Batch, Sequence, Features)
        if len(observations.shape) == 2:
            observations = observations.view(-1, self.seq_len, self.num_features)
            
        # Run through the LSTM
        lstm_out, (hidden_state, cell_state) = self.lstm(observations)
        
        # We only care about the very last output of the sequence (the most recent thought)
        last_out = lstm_out[:, -1, :]
        
        # Pass through the linear layer to match PPO's expected input size
        return self.linear(last_out)

def train_session_model(csv_path: str, session: str, total_timesteps: int = 100000):
    """
    Initializes the environment and trains the PPO agent for a specific session.
    """
    print(f"Loading data from {csv_path} for {session} session...")
    df = pd.read_csv(csv_path)
    
    # Initialize the custom Gym environment
    env = GoldTradingEnv(df=df, session=session, window_size=30)
    
    # Tell Stable-Baselines3 to use our custom LSTM instead of standard Linear layers
    policy_kwargs = dict(
        features_extractor_class=LSTMExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    print(f"Initializing PPO Agent for {session}...")
    model = PPO(
        "MlpPolicy", # 'MlpPolicy' is used, but we've overridden the extractor with our LSTM
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99, # Discount factor (focus on long-term reward)
        ent_coef=0.01, # Entropy coefficient (forces the agent to explore new actions)
        verbose=1,
        device="auto" # Will automatically use GPU if PyTorch detects one
    )
    
    print(f"Beginning Training: {total_timesteps} timesteps...")
    
    # 1. Your custom CSV logger
    logger_callback = TradingLoggerCallback(check_freq=5000, log_dir="./logs")
    
    # 2. NEW: Auto-save the brain every 50,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path="./saved_models/", 
        name_prefix=f"ppo_gold_{session.lower()}"
    )

    # Pass both callbacks as a list
    model.learn(
        total_timesteps=total_timesteps,
        callback=[logger_callback, checkpoint_callback]
    )
    
    save_path = f"lstm_ppo_gold_{session.lower()}_final.zip"
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    # Train the London session model
    train_session_model('../data/processed/master_features_15m.csv', session='LONDON', total_timesteps=100000)