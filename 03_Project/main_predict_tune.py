import gym
import os
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from src.env_V36_PPO_tune import CryptoEnv


def main():

    for num in range(10):
        algo_train = "PPO_env_V36_tune_train" # folder name, where the trained model is saved
        # algo_test = "PPO_env_V36_tune_pred_21" # folder name to save the predictions on the year 2021
        algo_test = "PPO_env_V36_tune_pred_bench" # folder name to save the predictions on the benchmark period
        TIMESTEPS = 500000 # Timesteps used during training
        idx = num

        models_dir_train = f"models/{algo_train}/Timesteps_{TIMESTEPS}/{idx}" # path to get saved model

        models_dir_test = f"models/{algo_test}/Timesteps_{TIMESTEPS}/{idx}" # path to save predictions
        log_dir_test = f"models/{algo_test}/Timesteps_{TIMESTEPS}/{idx}/logs" # path to save logs of predictions (optional)

        if not os.path.exists(models_dir_test):
            os.makedirs(models_dir_test)

        if not os.path.exists(log_dir_test):
            os.makedirs(log_dir_test)

        df_test = pd.read_csv('data/data_processed_extended/df_processed_benchmark.csv', index_col = 0) # benchmark period
        df_test_stat = pd.read_csv('data/data_processed_extended_stat/df_processed_benchmark.csv', index_col = 0) # benchmark period
        df_test21 = pd.read_csv('data/data_processed_extended/df_processed_test.csv', index_col = 0) # year 2021
        df_test21_stat = pd.read_csv('data/data_processed_extended_stat/df_processed_test.csv', index_col = 0) # year 2021

        env_eval = DummyVecEnv([lambda: CryptoEnv(df_test, df_test_stat, algo_test, TIMESTEPS, idx=idx, serial=True)])

        model = PPO.load(f"{models_dir_train}/best_model/model", env=env_eval, print_system_info=True, seed=123)

        episodes = 1 # number of runs
        for ep in range(episodes):
            obs = env_eval.reset()
            done = False

            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env_eval.step(action)

        env_eval.render()


if __name__ == '__main__':
    main()
