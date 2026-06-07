import torch
import torch.nn as nn
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

# Import our custom environment (Adjust path if running from a different directory)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.trading_env import GoldTradingEnv

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
    model.learn(total_timesteps=total_timesteps)
    
    save_path = f"lstm_ppo_gold_{session.lower()}.zip"
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    # Train the London session model
    train_session_model('../data/processed/master_features_15m.csv', session='LONDON', total_timesteps=100000)