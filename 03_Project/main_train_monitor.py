import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from src.env_V36_PPO import CryptoEnv


def main():
    # Callback class to monitor training
    class SaveOnBestTrainingRewardCallback(BaseCallback):
        """
        Callback for saving a model (the check is done every ``check_freq`` steps)
        based on the training reward (in practice, we recommend using ``EvalCallback``).

        :param check_freq:
        :param log_dir: Path to the folder where the model will be saved.
        It must contains the file created by the ``Monitor`` wrapper.
        :param verbose: Verbosity level.
        """
        def __init__(self, check_freq: int, log_dir_model: str, verbose: int = 1):
            super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.log_dir = log_dir_model
            self.save_path = f"{log_dir_model}/best_model"
            self.best_mean_reward = -np.inf

        def _init_callback(self) -> None:
            # Create folder if needed
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self) -> bool:
            if self.n_calls % self.check_freq == 0:

                # Retrieve training reward
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    # Mean training reward over the last 25 episodes
                    mean_reward = np.mean(y[-25:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # Example for saving best model
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.save_path}")
                        self.model.save(f"{self.save_path}/model")

                return True

    def moving_average(values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')


    def plot_results(log_folder, algo_train, TIMESTEPS, title='Learning Curve'):
        """
        plot the results

        :param log_folder: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        y = moving_average(y, window=50)
        # Truncate x
        x = x[len(x) - len(y):]

        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        plt.savefig(f'models/{algo_train}/Timesteps_{TIMESTEPS}/Timesteps_{TIMESTEPS}_performance_1.png', bbox_inches='tight')
        plt.show()

    # Define timesteps and path names
    algo = "PPO V36"
    algo_train = "PPO_env_V36_train" # DQN, PPO, A2C
    algo_train_eval = "PPO_env_V36_train_eval" # DQN, PPO, A2C
    TIMESTEPS = 100000

    log_dir_model_train = f"models/{algo_train}/Timesteps_{TIMESTEPS}"
    log_dir_tb_train = f"models/{algo_train}/Timesteps_{TIMESTEPS}/logs"

    log_dir_model_train_eval = f"models/{algo_train_eval}/Timesteps_{TIMESTEPS}"
    log_dir_tb_train_eval = f"models/{algo_train_eval}/Timesteps_{TIMESTEPS}/logs"

    # Create log and model dir
    if not os.path.exists(log_dir_model_train):
        os.makedirs(log_dir_model_train)

    if not os.path.exists(log_dir_tb_train):
        os.makedirs(log_dir_tb_train)

    if not os.path.exists(log_dir_model_train_eval):
        os.makedirs(log_dir_model_train_eval)

    if not os.path.exists(log_dir_tb_train_eval):
        os.makedirs(log_dir_tb_train_eval)

    # import data for training
    df_train = pd.read_csv('data/data_processed_extended/df_processed_train.csv', index_col = 0)
    df_train_stat = pd.read_csv('data/data_processed_extended_stat/df_processed_train.csv', index_col = 0)

    # import data for evaluation
    df_train_eval = pd.read_csv('data/data_processed_extended/df_processed_20.csv', index_col = 0)
    df_train_eval_stat = pd.read_csv('data/data_processed_extended_stat/df_processed_20.csv', index_col = 0)

    # Define model parameter
    model_params = model_params()

    # Create train and evaluate environment (serial=False: varying windows; serial=True: same window)
    env = CryptoEnv(df_train, df_train_stat, algo_train, TIMESTEPS, serial=False)
    env_eval = CryptoEnv(df_train_eval, df_train_eval_stat, algo_train_eval, TIMESTEPS, serial=False)

    # Wrap the environment
    env = Monitor(env, log_dir_model_train)

    # Create model
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir_tb_train, device="cuda", seed=123)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir_model=log_dir_model_train)
    # Monitor training time
    start_time = time.monotonic()
    # Train the agent with evaluation every 5000 steps
    model.learn(total_timesteps=int(TIMESTEPS), reset_num_timesteps=False, tb_log_name=f"{algo_train}", eval_env=env_eval,
            eval_freq=50000, callback=callback)

    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))

    env.render()

    plot_results(log_dir_model_train, algo_train, TIMESTEPS)
    plt.savefig(f'models/{algo_train}/Timesteps_{TIMESTEPS}/Timesteps_{TIMESTEPS}_performance_training.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

# Tensorboard
# http://localhost:6006/
# tensorboard --logdir ./models/PPO_env_V36_train/Timesteps_1000000/logs/PPO_env_V36_train_0/
