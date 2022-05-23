from enum import Enum
import os
import glob
import pandas as pd
import numpy as np
from datetime import timedelta
from pandas.core.frame import DataFrame
from utils.data_prep_helper import find_delimiter, split_datetime
from statsmodels.tsa.stattools import adfuller


class Preprocessing():
    def __init__(self, scaler: Enum) -> None:

        self.dir_original: str = r'./data/data_original/'
        self.dir_processed: str = r'./data/data_processed_extended_stat/'
        self.all_files: str = glob.glob(self.dir_original + "*.csv")
        self.all_dfs_processed: list[DataFrame] = []
        self.df_processed: DataFrame = None

        self.scaler = scaler


    def save_df_processed(self, process_type = '') -> None:

        filename = process_type + 'df_processed.csv'
        file_merged = os.path.join(self.dir_processed, filename)
        self.df_processed.to_csv(file_merged, index = False)

    def get_df_processed(self) -> None:

        return self.df_processed


    # Format csv files depending on the structure
    def format_csv(self) -> None:

        for filename in self.all_files:
            processed = True
            delimiter = find_delimiter(filename) # finds the separator/delimiter of the specified csv-file
            df = pd.read_csv(filename, index_col = None, header = 0, delimiter = delimiter) # index_col None or False does not use the index as first column

            # Crypto data is in the timezone UTC (which is the desired timezone - no change required for crypto data)
            if 'crypto' in filename.lower():
                # Split the time column into two separate columns (from date with time to a column Date and column time)
                df = split_datetime(df, col_to_split = 'Open time', cols = ['Date', 'Open_Time'])

                # define columns which should be kept (in this way get rid of useless columns) and create a dataframe accordingly
                col_list = ['Date', 'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume'] # 'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume'
                df = df[col_list]
                df['Moving_avg_close'] = self.moving_average(df['Close'], 168)
                df['Moving_avg_close'] = df['Moving_avg_close'].fillna(method = 'bfill')

            elif 'google' in filename.lower():
                # rename columns, which are needed later
                df.rename(columns = {'date': 'Date', 'adjusted.hits.daily': 'Google_adj_hits_daily'}, inplace=True)

                # define columns which should be kept (in this way get rid of useless columns) and create a dataframe accordingly
                col_list = ['Date', 'Google_adj_hits_daily']
                df = df[col_list]

            # Stock data is in the timezone GMT-6
            elif 'stocks' in filename.lower():
                # define columns names since the dataset comes without
                df.columns = ['Date', 'Open_Time', 'Open_S', 'High_S', 'Low_S', 'Close_S', 'Volume_S']

                # Combine Date and time column in order to convert from GMT-6 to UTC
                df['Date_time'] = pd.to_datetime(df['Date'] + ' ' + df['Open_Time'], dayfirst = True)

                # Add 6 hours (from GMT-6 to GMT) to convert it to UTC since GMT = UTC
                df['Date_time'] = pd.to_datetime(df['Date_time'] + timedelta(hours = 6))

                # Split the time column into two separate columns (from date with time to a column Date and column time)
                df = split_datetime(df, col_to_split = 'Date_time', cols = ['Date', 'Open_Time'])

                # Drop the merged column (used for the GMT-6 to GMT=UTC conversion) as it is not needed anymore
                df.drop('Date_time', axis = 1, inplace = True)

            elif 'twitter' in filename.lower():
                # define columns names since the dataset comes without
                df.columns = ['Date', 'Twitter_tweets']
                df['Twitter_tweets'] = pd.to_numeric(df['Twitter_tweets'], errors='coerce').astype('Int64')

            else:
                processed = False
                print(filename, ' does not match and is therefore not processed - Please check again')

            # convert the Date column to a datetime object with day on first position, ensuring the same format. Important for merging
            df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)

            # Increase date by one day since tweets/trends are on a daily basis and should be one day delayed to the hourly prices
            if 'google' in filename.lower() or 'twitter' in filename.lower():
                df['Date'] = df['Date'] + timedelta(days = 1)

            if processed:
                # Convert Open_time column to string. Important for merging
                if 'Open_Time' in df.columns:
                    df['Open_Time'] = df['Open_Time'].astype('string')

                # append each df to the list
                self.all_dfs_processed.append(df)


    def merge_csv(self) -> None:

        # Get first element in the list - one df is needed as starting position
        self.df_processed = next(iter(self.all_dfs_processed))

        # Merge all data frames together based on either "Date" or "Date" AND "Open_Time", then save as csv
        for i in range(1, len(self.all_dfs_processed)):
            df = self.all_dfs_processed[i]

            if 'Open_Time' in df.columns:
                self.df_processed = self.df_processed.merge(df, how='left', on = ['Date', 'Open_Time'])
            else:
                self.df_processed = self.df_processed.merge(df, how='left', on = 'Date')

        # Merge Date and time into one column and remove the two redundant ones (no need to have date and time separated)
        self.df_processed.insert (0, 'Date_time', pd.to_datetime(self.df_processed['Date'].astype('string') + ' ' + self.df_processed['Open_Time'], dayfirst = True))
        self.df_processed.drop(['Date', 'Open_Time'], axis = 1, inplace = True)
        self.df_processed = self.volatility(self.df_processed)
        self.df_processed = self.returns(self.df_processed)


    # Checks data for NAs
    def check_data(self, df) -> None:

        print(df.info())
        print('\nNumber rows: ', len(df))
        print('\nSum of NAs: \n', df.isnull().sum(axis = 0))


    # deals with NAs
    def handle_NA(self) -> None:

        # All -1 values in Twitter columns are replaced first by 'nan', afterwards replaced by the mean/average of neigboring cells which are not -1
        self.df_processed['Twitter_tweets'].replace(to_replace = -1, value = np.nan, inplace = True)
        self.df_processed['Twitter_tweets'] = (self.df_processed['Twitter_tweets'].fillna(method = 'ffill')
                                                + self.df_processed['Twitter_tweets'].fillna(method = 'bfill')) / 2

        # Only the last columns showed NA's (related to stock prices) because of weekends or holidays where no trading took place / was recorded
        # These "empty" hours will be filled with the last known stock price
        self.df_processed.iloc[: , -8:-3] = self.df_processed.iloc[: , -8:-3].fillna(method = 'ffill')

        self.df_processed.iloc[: , -3:] = self.df_processed.iloc[: , -3:].fillna(method = 'bfill')
        self.df_processed = self.stationary(self.df_processed)


    # Scale data
    def scaling(self) -> str:

        scaler = self.scaler.value
        # transform data from the thrid column to the end since first two columns are Date and Open_time
        self.df_processed.iloc[:, 1:] = scaler.fit_transform(self.df_processed.iloc[:, 1:])

        return str(self.scaler.name).lower()

    def stationary(self, df_in):

        df_date = df_in.iloc[:,0]
        df_list = [df_date]
        for column in (column for column in df_in if column not in ['Date_time', 'returns_flag']):
            df_col = df_in[column]
            # take log and first order difference
            df_stat = np.log(df_col).diff(1)
            df_stat.replace(-np.Inf, np.nan, inplace=True)
            df_stat.replace(np.Inf, np.nan, inplace=True)
            df_stat = df_stat.fillna(method = 'bfill')
            # for all which could not be filled with backwards fill
            df_stat = df_stat.fillna(method = 'ffill')
            # Augmented Dickey-Fuller Test
            t_stat, p_value, _, _, critical_values, _  = adfuller(df_stat.values, autolag='AIC')
            print(f'ADF Statistic: {t_stat:.2f}')
            print(f'p-value: {p_value:.2f}')
            for key, value in critical_values.items():
                print('Critical Values:')
                print(f'   {key}, {value:.2f}')

            df_list.append(df_stat)

        df_out = pd.concat(df_list, axis=1)

        return df_out

    def volatility(self, data):

        # take volatility of Bitcoin closing price
        pct = data.filter(['Close'], axis=1)
        # percentage change between hourly prices
        pct = pct.Close.pct_change()
        # moving average over the last day with standard deviation to calculate volatility
        df_volatility = pct.rolling(window=24).std()
        df = data.merge(df_volatility.rename('Volatility'), left_index=True, right_index=True)

        return df

    def returns(self, data):

        # percentage return between hourly prices
        pct = data.filter(['Close'], axis=1)
        df_returns = pct.Close.pct_change()
        df_returns = df_returns.fillna(method = 'bfill')
        df = data.merge(df_returns.rename('pct_returns'), left_index=True, right_index=True)
        df['returns_flag'] = np.where(df['pct_returns'] > 0, 1, 0)

        return df

    def moving_average(self, data, window):

        return data.rolling(window).mean()

    def prep_data (self) -> None:
        self.format_csv()
        self.merge_csv()
        self.check_data(self.get_df_processed())
        self.handle_NA()
        # # self.save_df_processed()
        self.check_data(self.get_df_processed())
        # self.stationary()
        # s = self.scaling()
        self.save_df_processed()


class Train_Test_Split():
    def __init__(self) -> None:

        self.train_filename = 'df_processed_train.csv'
        self.test_filename = 'df_processed_test.csv'
        self.path_save = r'./data/data_processed_extended_stat/'
        self.path_train = os.path.join(self.path_save, self.train_filename)
        self.path_test = os.path.join(self.path_save, self.test_filename)
        self.df = pd.read_csv('data/data_processed_extended_stat/df_processed.csv', index_col = None)


    def train_test_split(self):
        self.df['Date_time'] = pd.to_datetime(self.df['Date_time'])
        df_train = self.df[pd.DatetimeIndex(self.df['Date_time']).year != 2021]
        df_20 = self.df[pd.DatetimeIndex(self.df['Date_time']).year == 2020]
        df_19 = self.df[pd.DatetimeIndex(self.df['Date_time']).year == 2019]
        df_1817 = self.df.loc[((self.df['Date_time']).dt.year == 2018) | ((self.df['Date_time']).dt.year == 2017)]
        df_test = self.df[pd.DatetimeIndex(self.df['Date_time']).year == 2021]

        df_train.to_csv(self.path_train, index=None)
        df_20.to_csv(f'{self.path_save}df_processed_20.csv', index=None)
        df_19.to_csv(f'{self.path_save}df_processed_19.csv', index=None)
        df_1817.to_csv(f'{self.path_save}df_processed_1817.csv', index=None)
        df_test.to_csv(self.path_test, index=None)

