import csv
import pandas as pd
import matplotlib.pyplot as plt

def dict_to_csv(list, filename, path=''):

    # get keys of the first dict in the list for column names (all following dicts have the same structure)
    keys = list[0].keys()

    filename = f'{path}{filename}.csv'

    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list)


def lists_to_csv(filename, path='', *args):

    filename = f'{path}{filename}.csv'

    columns = ['Profit', 'Benchmark_Profit', 'Profit_Percent', 'Benchmark_Profit_Percent', 'Model_Reward', 'Overall_Reward']
    df = pd.DataFrame(list(zip(*args)), columns=columns)

    df.to_csv(filename, index=None)  


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

