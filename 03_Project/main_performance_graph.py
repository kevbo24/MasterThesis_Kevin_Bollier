import matplotlib.pyplot as plt
import pandas as pd


def save_plot(p_profit, p_benchmark_profit, reward, filename, path=''):

    filename = f'{path}{filename}.png'

    # Plot the graph of the reward
    fig = plt.figure()
    if 'train' in filename:
        fig.suptitle('Training graph')
    elif 'test' in filename:
        fig.suptitle('Testing graph')
    else:
        fig.suptitle('Graph')

    high = plt.subplot(2, 1, 1)
    high.set(ylabel='Gain in %')
    plt.plot(p_profit, label='Bot profit in %')
    plt.plot(p_benchmark_profit, label='Bitcoin profit in %')
    high.legend(loc='upper left')

    low = plt.subplot(2, 1, 2)
    low.set(xlabel='Episode', ylabel='Reward')
    plt.plot(reward, label='reward')

    plt.savefig(filename, bbox_inches='tight')
    # plt.show()

algo_train = "PPO_env_V30_11_1_train_24" # folder name, where the trained model is saved
TIMESTEPS = 1000000 # Timesteps used during training

log_dir_model_train = f"models/{algo_train}/Timesteps_{TIMESTEPS}" # path to save graph

df_rewards = pd.read_csv(f"{log_dir_model_train}/Timesteps_{TIMESTEPS}_performance.csv", index_col = None)

graph_p_profit = df_rewards['Profit_Percent']
graph_p_benchmark_profit = df_rewards['Benchmark_Profit_Percent']
graph_overall_reward = df_rewards['Overall_Reward']

save_plot(graph_p_profit, graph_p_benchmark_profit, graph_overall_reward,
                            'performance', f'models/{algo_train}/Timesteps_{TIMESTEPS}/Timesteps_{TIMESTEPS}_new_')

