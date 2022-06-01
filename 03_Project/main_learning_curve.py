import gym
import numpy as np
from openpyxl import Workbook
import matplotlib.pyplot as plt
from datetime import timedelta
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


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
        print(x,y)
        y = moving_average(y, window=10)
        print(y)
        # Truncate x
        x = x[len(x) - len(y):]
        print(x)

        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        plt.savefig(f'models/{algo_train}/Timesteps_{TIMESTEPS}/Timesteps_{TIMESTEPS}_performance_last100000000.png', bbox_inches='tight')
        plt.show()

algo_train = "PPO_env_V36_train" # folder name, where the trained model is saved
TIMESTEPS = 1000000 # Timesteps used during training

log_dir_model_train = f"models/{algo_train}/Timesteps_{TIMESTEPS}" # path to save graph

plot_results(log_dir_model_train, algo_train, TIMESTEPS)
