import os
import time
import gym
import optuna
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import BaseCallback
from src.env_V36_PPO_tune import CryptoEnv


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

    # Optuna: Optimization of hyperparameter
    def optimize_ppo(trial):
        """ Learning hyperparamters we want to optimise"""
        return {
            # 'gamma': 0.9874443977030081,
            # 'learning_rate': 8.399142379136018e-05,
            # 'ent_coef': 0.00010562664299918551,
            # 'gae_lambda': 0.9891372650095183,
            # 'clip_range': 0.206583,
            # 'n_epochs': 18,
            # 'target_kl': 0.008076189050086552,
            # 'vf_coef': 0.9757916583346159,
            # 'max_grad_norm': 0.34805599525792064
            'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
            'learning_rate': trial.suggest_loguniform('learning_rate', 5e-6, 0.003),
            # PPO: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
            # 'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
            'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
            'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 1.),
            'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.3),
            'n_epochs': int(trial.suggest_loguniform('n_epochs', 3, 30)),
            'target_kl': trial.suggest_loguniform('target_kl', 0.003, 0.03),
            'vf_coef': trial.suggest_loguniform('vf_coef', 0.5, 1),
            'max_grad_norm': trial.suggest_loguniform('max_grad_norm', 0.3, 0.7)
            # A2C
            ## 'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
            # 'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
            # 'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 1.),
            # 'vf_coef': trial.suggest_loguniform('vf_coef', 0.5, 1),
            # 'max_grad_norm': trial.suggest_loguniform('max_grad_norm', 0.3, 0.7),
            # 'rms_prop_eps': trial.suggest_loguniform('rms_prop_eps', 1e-8, 1e-2)
            # DQN
            # 'batch_size': trial.suggest_uniform('batch_size', batch_size),
            # 'tau': trial.suggest_uniform('tau', 0, 1),
            # 'exploration_fraction': trial.suggest_uniform('exploration_fraction', 0.1, 0.3)
        }

    def optimize_agent(trial):
        """ Train the model and optimise
            Optuna maximises the negative log likelihood, so we
            need to negate the reward here
        """
        # Define timesteps and path names
        algo = "PPO V36_tune"
        algo_train = "PPO_env_V36_train_tune" # DQN, PPO, A2C
        algo_train_eval = "PPO_env_V36_train_eval_tune" # DQN, PPO, A2C
        algo_test = "PPO_env_V36_test_tune" # DQN, PPO, A2C
        TIMESTEPS = 500000

        log_dir_model_train = f"models/{algo_train}/Timesteps_{TIMESTEPS}"
        log_dir_tb_train = f"models/{algo_train}/Timesteps_{TIMESTEPS}/logs"

        log_dir_model_train_eval = f"models/{algo_train_eval}/Timesteps_{TIMESTEPS}"
        log_dir_tb_train_eval = f"models/{algo_train_eval}/Timesteps_{TIMESTEPS}/logs"

        log_dir_model_test = f"models/{algo_test}/Timesteps_{TIMESTEPS}"
        log_dir_tb_test = f"models/{algo_test}/Timesteps_{TIMESTEPS}/logs"

        path_list = [log_dir_model_train, log_dir_tb_train, log_dir_model_train_eval, log_dir_tb_train_eval,
            log_dir_model_test, log_dir_tb_test]

        final_path = []
        idx = str(0)
        for path in path_list:
        # Create log and model dir
            if not os.path.exists(path):
                f_path = f"{path}/0"
                os.makedirs(f_path)
                final_path.append(f_path)
            else:
                if os.path.isdir(path):
                    list_dir = os.listdir(path)
                    if 'logs' in list_dir:
                        number_of_dirs = int(len(os.listdir(path))-1)
                    else:
                        number_of_dirs = int(len(os.listdir(path)))
                    f_path = f"{path}/{str(number_of_dirs)}"
                    os.makedirs(f_path)
                    final_path.append(f_path)
                    idx = str(number_of_dirs)

        # import data for training
        df_train = pd.read_csv('data/data_processed_extended/df_processed_train.csv', index_col = 0)
        df_train_stat = pd.read_csv('data/data_processed_extended/df_processed_train.csv', index_col = 0)

        # import data for evaluation
        df_train_eval = pd.read_csv('data/data_processed_extended/df_processed_20.csv', index_col = 0)
        df_train_eval_stat = pd.read_csv('data/data_processed_extended/df_processed_20.csv', index_col = 0)

        # import data for testing
        df_test = pd.read_csv('data/data_processed_extended/df_processed_test.csv', index_col = 0)
        df_test_stat = pd.read_csv('data/data_processed_extended/df_processed_test.csv', index_col = 0)

        # Initialize parameter
        seed = 123

        # Define model parameter
        model_params = optimize_ppo(trial)

        # Create train and evaluate environment
        env = CryptoEnv(df_train, df_train_stat, algo_train, TIMESTEPS, idx, serial=True)
        env_eval = CryptoEnv(df_train_eval, df_train_eval_stat, algo_train_eval, TIMESTEPS, idx, serial=True)
        env_test = CryptoEnv(df_test, df_test_stat, algo_test, TIMESTEPS, idx, serial=True)

        # Wrap the environment
        env = Monitor(env, final_path[0])
        env_test = Monitor(env_test)

        # Create model for training
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=final_path[1], device="cuda", seed=seed, **model_params)
        # Create the callback: check every 1000 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir_model=final_path[0])
        # Monitor training time
        start_time = time.monotonic()
        # Train the agent with evaluation every 5000 steps
        model.learn(total_timesteps=int(TIMESTEPS), reset_num_timesteps=False, tb_log_name=f"{algo_train}", eval_env=env_eval,
            eval_freq=50000, callback=callback)

        end_time = time.monotonic()
        print(timedelta(seconds=end_time - start_time))

        env.render()

        # Test model with test dataset
        model_test = PPO.load(f"{final_path[0]}/best_model/model", env=env_test, print_system_info=True, seed=123)

        rewards = []
        n_episodes, reward_sum = 0, 0.0

        obs = env_test.reset()
        while n_episodes < 4:
            action, _ = model_test.predict(obs)
            obs, reward, done, _ = env_test.step(action)
            reward_sum += reward

            if done:
                rewards.append(reward_sum)
                reward_sum = 0.0
                n_episodes += 1
                obs = env_test.reset()

        last_reward = np.mean(rewards)
        trial.report(-1 * last_reward, n_episodes-1)

        return -1 * last_reward

    # create optuna study (DB required)
    study = optuna.create_study(study_name='PPO_V36_tune', storage='mysql+pymysql://<user>:@<host>/<dbname>', load_if_exists=True,
                sampler=TPESampler(seed=123))

    study.optimize(optimize_agent, n_trials=10, n_jobs=1)


if __name__ == '__main__':
    main()

