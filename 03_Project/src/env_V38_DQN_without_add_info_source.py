import gym
import random
import numpy as np
from gym import spaces
from src.data_enums import Scaler, Trading_conditions
from utils.model_helper_abV20 import dict_to_csv, lists_to_csv, save_plot

class CryptoEnv(gym.Env):
    def __init__(self, df, df_stat, algo, timesteps, lookback_window_size = 50,
                 initial_balance = 10000,
                 serial = False,
                 scaler = Scaler.MINMAX):
        super(CryptoEnv, self).__init__()

        # Standard parameter
        self.df = df
        self.df_stat = df_stat  # stationary data frame
        self.algo = algo # folder name to save everything
        self.timesteps = timesteps
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.serial = serial # False = works with random data points, random sequences of data / True = Times series
        self.scaler = scaler
        self.amount_div = 10
        self.obs_amount = self.amount_div + 1
        self.trades = []

        # Additional Parameter to evaluate and enhance the model
        self.reset_count = -1 # Used to save all trades before resetting the environment
        self.profit = [] # for evaluation
        self.benchmark_profit = [] # for evaluation
        self.graph_p_profit = [] # for evaluation and creating performance graph
        self.graph_p_benchmark_profit = [] # for evaluation and creating performance graph
        self.graph_reward = [] # for evaluation
        self.graph_overall_reward = [] # for evaluation and creating performance graph

        # 3 actions (buy, hold, sell); trades always with the full amount
        self.action_space = spaces.Discrete(3)

        # Observes the OHCLV data of crypto and S&P500, Twitter tweets, google trends, net worth and trade history
        # shape(number of columns, number of rows - history how many past data points should be passed to the model)
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (20, lookback_window_size + 1), dtype = np.float16)


    def reset(self):
        # Used to save all trades before resetting the environment
        self.reset_count += 1
        if len(self.trades) > 0:
            dict_to_csv(self.trades, 'trades', f'models/{self.algo}/Timesteps_{self.timesteps}/Timesteps_{self.timesteps}_{self.reset_count}_')

        # Standard Parameter
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.crypto_holding = 0
        self.total_fees = 0
        self.total_volume_traded = 0
        self.current_step = 0
        self.trades = []

        # Additional Parameter to enhance the model
        self.current_buy_costs = 0
        self.avg_buy_price = 0
        self.price_invest = 0
        self.hold_penalty = 0
        self.net_worth_highest = self.net_worth
        self.overall_reward = 0
        self.price_previous = 0
        self.original_price = 0

        # Parameter which have to be given to the observation space in addition to the ones coming from the dataset
        # includes net_worth, amount of crypto bought, costs of this purchase, amount of crypto sold, sale amount
        # self.lookback_window_size + 1: +1 accounts for the current step. That means, current step + 50 entires of history
        self.account_history = np.repeat([[self.net_worth], [self.balance], [self.crypto_holding], [self.total_fees], [0], [0], [0], [0], [0], [0]], 
                                    self.lookback_window_size + 1, axis = 1)

        # reset the dataframe that should be used
        self._reset_active_df()

        # initialize first observation
        return self._next_observation()


    def _reset_active_df(self):
        # work with serial datapoints (kind of a time series)
        if self.serial:
            self.steps_left = len(self.df) - self.lookback_window_size - 1
            self.frame_start = self.lookback_window_size
        # work with varying time window (random start point)
        else:
            self.steps_left = np.random.randint(1, (len(self.df) - self.lookback_window_size))
            self.frame_start = np.random.randint(self.lookback_window_size, len(self.df) - self.steps_left)

        # data frame for normal data
        self.active_df = self.df[self.frame_start - self.lookback_window_size:self.frame_start + self.steps_left + 1]
        # data frame for stationary data
        self.active_df_stat = self.df_stat[self.frame_start - self.lookback_window_size:self.frame_start + self.steps_left + 1]
    
    # set observation space with all required records from data sources
    def _next_observation(self):
        # define the last record of the observation space
        self.end_active_df = self.current_step + self.lookback_window_size + 1

        obs = np.array([
            self.active_df['Open'].values[self.current_step:self.end_active_df],
            self.active_df['High'].values[self.current_step:self.end_active_df],
            self.active_df['Low'].values[self.current_step:self.end_active_df],
            self.active_df['Close'].values[self.current_step:self.end_active_df],
            self.active_df_stat['Close'].values[self.current_step:self.end_active_df],
            self.active_df['Volume'].values[self.current_step:self.end_active_df],
            self.active_df['Volatility'].values[self.current_step:self.end_active_df],
            self.active_df['pct_returns'].values[self.current_step:self.end_active_df],
            self.active_df_stat['pct_returns'].values[self.current_step:self.end_active_df],
            self.active_df['returns_flag'].values[self.current_step:self.end_active_df]
        ])
        # scale data according to the scaling method
        obs = self._scaling(obs, False)
        acc_history = self._scaling(self.account_history, False)
        # append together as input for the neural network
        obs = np.append(obs, acc_history[:, -(self.lookback_window_size + 1):], axis=0)
        return obs

        return obs

    def _take_action(self, action):
        # Standard Parameter
        action_type = action # action type: buy, sell, hold
        amount = 1 # 100% of the amount
        crypto_bought = 0
        crypto_sold = 0
        cost = 0
        sale = 0
        crypto_fees_sold = 0
        crypto_fees_bought = 0

        # Additional Parameter
        self.trade_profit = 0
        self.min_trade_penalty = False
        self.trend_reward = 0
        self.slippage = random.uniform(0.001, 0.002)
        self.profit_reward = 0
        crypto_holding_previous = self.crypto_holding

        # The current price is at the end since table is structured in the way that older data is on top of the table and new one is appended at the end
        self.current_price_loc = self.current_step + self.lookback_window_size

        # Set the highest net_worth achieved
        if self.net_worth_highest < self.net_worth:
            self.net_worth_highest = self.net_worth

        # Set the previous price 
        if self.current_step == 0:
            self.price_previous = self.active_df.iloc[self.current_price_loc, self.active_df.columns.get_loc('Close')]
        else:
            self.price_previous = self.original_price

        # get the current close price
        self.original_price = self.active_df.iloc[self.current_price_loc, self.active_df.columns.get_loc('Close')]

        # Buy
        if action_type == 0:
            # calculate buy price when slippage is considered
            self.current_price = self.original_price + (self.original_price * self.slippage)
            crypto_bought = self.balance * amount / self.current_price
            crypto_fees_bought = crypto_bought * self.current_price * Trading_conditions.MAKER_FEE.value
            # repeating step because negative balance is not possible (fees need to be paid with account balance)
            # Binance allows to pay fees with their own currency BNB (which is not taking into account in this study)
            crypto_bought = (self.balance - crypto_fees_bought) * amount / self.current_price
            cost = crypto_bought * self.current_price + crypto_fees_bought
            # everything under USD 10 cannot be executed by the exchange
            if cost < 10:
                self.min_trade_penalty = True
                self.current_price = self.original_price
                crypto_bought = 0
                crypto_fees_bought = 0
                cost = 0
            # Clearing 
            self.crypto_holding += crypto_bought
            self.total_fees += crypto_fees_bought
            self.total_volume_traded += crypto_bought * self.current_price
            self.balance -= cost
            self.current_buy_costs += cost
            # Clearing: calculate the new average buy price
            if self.crypto_holding > 0:
                self.avg_buy_price = self.current_buy_costs / self.crypto_holding
                self.price_invest = self.original_price
            else:
                self.avg_buy_price = 0
                self.price_invest = 0

        # Sell
        if action_type == 1:
            # calculate sell price when slippage is considered
            self.current_price = self.original_price - (self.original_price * self.slippage)
            crypto_sold = self.crypto_holding * amount
            crypto_fees_sold = crypto_sold * self.current_price * Trading_conditions.TAKER_FEE.value
            sale = crypto_sold * self.current_price - crypto_fees_sold
            # everything under USD 10 cannot be executed by the exchange
            if sale < 10:
                self.min_trade_penalty = True
                self.current_price = self.original_price
                crypto_sold = 0
                crypto_fees_sold = 0
                sale = 0
            # Clearing 
            self.crypto_holding -= crypto_sold
            self.total_fees += crypto_fees_sold
            self.total_volume_traded += crypto_sold * self.current_price
            self.balance += sale
            self.trade_profit = sale - (crypto_sold * self.avg_buy_price)
            self.current_buy_costs -= crypto_sold * self.avg_buy_price
            # Clearing: calculate the new average buy price
            if self.crypto_holding > 0:
                self.avg_buy_price = self.current_buy_costs / self.crypto_holding
                self.price_invest = self.original_price
            else:
                self.avg_buy_price = 0
                self.price_invest = 0

        # Hold
        if action_type == 2:
            self.current_price = self.original_price

        # calculate total account value
        self.net_worth = self.balance + self.crypto_holding * self.original_price

        # calculate Bitcoin profit if start account balance had been invested (used for tracking activities)
        self.bench_profit = ((self.initial_balance / self.active_df.iloc[self.lookback_window_size, self.active_df.columns.get_loc('Close')] * 
                                self.active_df.iloc[self.current_price_loc, self.active_df.columns.get_loc('Close')]) - 
                                self.initial_balance)

        # Set the net worth when last buy
        if action_type == 0 and crypto_bought > 0:
            self.net_worth_last_buy = self.net_worth

        if self.net_worth > self.net_worth_last_buy and crypto_holding_previous > 0:
            self.extra_reward = 1
            if action_type == 1:
                self.net_worth_last_buy = 0
        else:
            self.extra_reward = 0

        # Flag whether current net_worth is higher or lower than initial balance
        if self.net_worth > self.initial_balance:
            net_worth_q = 1
        else:
            net_worth_q = 0

        # reward depends on the bitcoins held at the previous step and the number held now
        # if no bitcoins were held in the previous step and no buy action was taken in this step:
        # a downtrend leads to a positive reward as it is good not to invest and vice versa
        if crypto_holding_previous == 0:
            self.trend_reward = ((self.original_price - self.price_previous) / self.price_previous) * -1
        else:
            self.trend_reward = (((self.crypto_holding * self.original_price) - (crypto_holding_previous * self.price_previous)) / 
                                (crypto_holding_previous * self.price_previous))

        # if action is "buy" = no reward
        if action_type == 0: 
            self.trend_reward = 0

        # if action "buy" or "sell" below minimum order amount = penalty
        if self.min_trade_penalty:
            self.trend_reward = -1

        # calculate overall reward
        self.overall_reward += self.trend_reward

        # track activites during training
        self.trades.append({
            'trade_row_loc': self.current_price_loc,
            'original_price': self.original_price,
            'trade_price': self.current_price,
            'avg_buy_price': self.avg_buy_price,
            'trade_slippage': self.slippage,
            'trade_amount': crypto_bought if crypto_bought > 0 else crypto_sold,
            'trade_total': cost if crypto_bought > 0 else sale,
            'trade_fee': crypto_fees_bought if crypto_bought > 0 else crypto_fees_sold,
            'trade_profit': self.trade_profit,
            'trade_type': "buy" if action_type == 0 else "sell" if action_type == 1 else "hold",
            'trade_min_trade_penalty': self.min_trade_penalty,
            'trade_hold_penalty': self.hold_penalty,
            'trend_reward': self.trend_reward,
            'profit_reward': self.profit_reward,
            'overall_reward ': self.overall_reward,
            'net_worth': self.net_worth,
            'benchmark': self.bench_profit,
            'total_fees': self.total_fees,
            'Balance': self.balance,
            'crypto_holding': self.crypto_holding,
            'crypto_worth': self.crypto_holding * self.original_price,
            'total_volume_traded': self.total_volume_traded,
            'highest_net_worth' : self.net_worth_highest
            })

        # additional variables created by trading
        self.account_history = np.append(self.account_history, [
            [self.net_worth],
            [self.balance],
            [self.crypto_holding],
            [self.total_fees],
            [self.avg_buy_price],
            [crypto_bought],
            [cost],
            [crypto_sold],
            [sale],
            [net_worth_q]
            ], axis=1)


    def step(self, action, end=True):
        # take action
        self._take_action(action)

        self.current_step += 1

        # calculate bot profit 
        profit = self.net_worth - self.initial_balance
        profit_percent = profit / self.initial_balance * 100
        # calculate Bitcoin profit if start account balance had been invested
        benchmark_profit = ((self.initial_balance / self.active_df.iloc[self.lookback_window_size, self.active_df.columns.get_loc('Close')] *
                                self.active_df.iloc[self.current_price_loc, self.active_df.columns.get_loc('Close')]) -
                                self.initial_balance) * 100
        benchmark_profit_percent = benchmark_profit / self.initial_balance

        reward = self.trend_reward
        
        # Check if observations can still be made based on the df-length
        if self.current_price_loc + 1 >= len(self.active_df):
            end = True
        else:
            end = False

        done = self.net_worth <= 0 or self.crypto_holding < 0 or end

        # save profits and rewards after each episode
        if done and end:
            self.profit.append(profit)
            self.benchmark_profit.append(benchmark_profit)
            self.graph_p_profit.append(profit_percent)
            self.graph_p_benchmark_profit.append(benchmark_profit_percent)
            self.graph_reward.append(reward)
            self.graph_overall_reward.append(self.overall_reward)

        # if done, no more observations are possible
        if done:
            obs = 0
        else:
            obs = self._next_observation()

        # {} needed because gym wants 4 args
        return obs, reward, done, {}

    # for evaluation to create performance graph and save key figures of each episode
    def render(self, mode='human'):
        lists_to_csv('performance', f'models/{self.algo}/Timesteps_{self.timesteps}/Timesteps_{self.timesteps}_',
                            self.profit, self.benchmark_profit, self.graph_p_profit,
                            self.graph_p_benchmark_profit, self.graph_reward, self.graph_overall_reward)

        save_plot(self.graph_p_profit, self.graph_p_benchmark_profit, self.graph_overall_reward,
                            'performance', f'models/{self.algo}/Timesteps_{self.timesteps}/Timesteps_{self.timesteps}_')

    # Scale data
    def _scaling(self, data, iscolumn):
        # use defined scaler 
        scaler = self.scaler.value

        if iscolumn:
            return scaler.fit_transform(data) # scaling based on record
        else:
            return scaler.fit_transform(data.T).T # scaling based on feature