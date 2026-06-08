import torch
import pandas as pd
import os
import sys

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

# Import our custom environment 
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
        env_info = self.training_env.env_method('_get_info')[0]
        last_reward = self.locals['rewards'][0]
        
        self.history.append({
            "total_timesteps": self.num_timesteps,
            "env_step": env_info['step'],
            "position": env_info['position'],
            "balance": env_info['balance'],
            "equity": env_info['equity'],
            "reward": last_reward
        })

        if self.num_timesteps % self.check_freq == 0 and self.history:
            df_log = pd.DataFrame(self.history)
            log_file = os.path.join(self.log_dir, "training_step_metrics.csv")
            
            if not os.path.exists(log_file):
                df_log.to_csv(log_file, index=False)
            else:
                df_log.to_csv(log_file, mode='a', header=False, index=False)
            
            self.history = [] 
        return True

def train_session_model(csv_path: str, session: str, total_timesteps: int = 100000):
    print(f"Loading data from {csv_path} for {session} session...")
    df = pd.read_csv(csv_path)
    
    # Initialize custom Gym environment
    env_raw = GoldTradingEnv(df=df, session=session, window_size=30)
    
    env = DummyVecEnv([lambda: env_raw])
    # CRITICAL FIX: norm_reward=True stabilizes PPO gradients when using real symmetric PnL
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Configure the native SB3-Contrib LSTM architecture
    policy_kwargs = dict(
        lstm_hidden_size=128,
    )
    
    print(f"Initializing Recurrent PPO Agent for {session}...")
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99, 
        ent_coef=0.01, 
        verbose=1,
        device="auto" 
    )
    
    print(f"Beginning Training: {total_timesteps} timesteps...")
    
    logger_callback = TradingLoggerCallback(check_freq=5000, log_dir="./logs")
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path="./saved_models/", 
        name_prefix=f"ppo_gold_{session.lower()}"
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[logger_callback, checkpoint_callback]
    )
    
    save_path = f"lstm_ppo_gold_{session.lower()}_final.zip"
    model.save(save_path)
    env.save(f"vec_normalize_{session.lower()}.pkl")
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train_session_model('../data/processed/master_features_15m.csv', session='LONDON', total_timesteps=5000000)