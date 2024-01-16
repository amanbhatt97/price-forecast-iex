import pandas as pd
import numpy as np
import math
from functools import reduce


class FeatureEngineering:
    def __init__(self, PROJECT_PATH):
        self.PROJECT_PATH = PROJECT_PATH

    def shift_date(self, df, n):
        """ This method shifts dataframe df by n days downwards"""
        shifted_df = df.copy()
        try:
            shifted_df['datetime'] += pd.DateOffset(days=n)
        except:
            shifted_df['date'] += pd.DateOffset(days=n)

        return shifted_df

    def merge_dataframes(self, dfs, on_column='datetime'):
        """This method merges all the dataframes in a list dfs"""
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=on_column, how='left'), dfs)
        merged_df.dropna(inplace=True)
        return merged_df

    def _capping(self, data):
        conditions = [
            (data['datetime'] <= '2022-04-02 23:45:00'),
            (data['datetime'] <= '2023-04-03 23:45:00'),
            (data['datetime'] > '2023-04-03 23:45:00')
        ]
        capping_values = [20000, 12000, 10000]
        data['capping'] = np.select(conditions, capping_values)
        return data

    def _datetime_features(self, df):
        df['date'] = pd.to_datetime(df['datetime'].dt.date)
        df['hour'] = df['datetime'].dt.hour + 1
        df['dom'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['dow'] = df['datetime'].dt.dayofweek
        df['doy'] = df['datetime'].dt.dayofyear
        df['tb'] = ((df['datetime'].dt.hour * 60 + df['datetime'].dt.minute) // 15 + 1)
        df['next_day_sunday'] = np.where(df['dow'] == 5, 1, 0)
        df['hour_dow'] = df['hour'] * df['dow']

        df['isMorning'] = ((df['datetime'].dt.hour >= 6) & (df['datetime'].dt.hour <= 9)).astype(int)
        df['isDay'] = ((df['datetime'].dt.hour >= 10) & (df['datetime'].dt.hour <= 16)).astype(int)
        df['isEvening'] = ((df['datetime'].dt.hour >= 17) & (df['datetime'].dt.hour <= 23)).astype(int)
        df['isNight'] = ((df['datetime'].dt.hour == 24) | ((df['datetime'].dt.hour >= 1) & (df['datetime'].dt.hour <= 5))).astype(int)

        df['winter'] = ((df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2)).astype(int)
        df['summer'] = ((df['month'] >= 3) & (df['month'] <= 6)).astype(int)
        df['monsoon'] = ((df['month'] >= 7) & (df['month'] <= 8)).astype(int)
        df['autumn'] = ((df['month'] >= 9) & (df['month'] == 11)).astype(int)

        return df

    def _target(self, data, market_type):
        data['target'] = data[f'mcp_{market_type}'].shift(-96) if market_type == 'dam' else data[f'mcp_{market_type}'].shift(-96 * 2)
        return data

    def _lags(self, df):
        for column in df.columns[1:11]:
            shift_columns = [df[column].shift(96 * i) for i in range(1, 6)]
            df = pd.concat([df] + shift_columns, axis=1)
            df.columns = [*df.columns[:-5], *[f'change_in_{column}_wrt_day_{i}' for i in range(1, 6)]]

            for i in [1, 2, 4, 8, 12, 18]:
                df[f'{column}_lag_{i}h'] = df[column].shift(i * 4)

        return df

    def _min_max(self, df):
        for column in df.columns[1:11]:
            hour_mean = df.groupby(['date', 'hour']).mean()[column].reset_index().rename(columns={column: f'hour_mean_{column}'})
            df = pd.merge(df, hour_mean, on=['date', 'hour'], how='left')

            daily_mean = df.groupby(['date']).mean()[column].reset_index().rename(columns={column: f'daily_mean_{column}'})
            df = pd.merge(df, daily_mean, on=['date'], how='left')

            daily_min = df.groupby(['date']).min()[column].reset_index().rename(columns={column: f'daily_min_{column}'})
            df = pd.merge(df, daily_min, on=['date'], how='left')

            daily_max = df.groupby(['date']).max()[column].reset_index().rename(columns={column: f'daily_max_{column}'})
            df = pd.merge(df, daily_max, on=['date'], how='left')

        return df

    def _ema(self, df, market_type):
        for i in [1, 3, 6, 12]:
            df[f'ewma_{i}h'] = df[f'mcp_{market_type}'].ewm(span=4 * i).mean()

        for i in [1, 3, 5]:
            df[f'ewma_{i}d'] = df[f'mcp_{market_type}'].ewm(span=96 * i).mean()
        return df

    def _mean(self, df, market_type):
        window_sizes = [2, 3, 5]
        for window_size in window_sizes:
            df[f'mcp_{market_type}_mean_{window_size}d'] = df[f'mcp_{market_type}'].rolling(window_size * 96).mean()
        return df

    def _interaction(self, df, market_type):
        df[f'mcp_{market_type}_with_dom'] = df[f'mcp_{market_type}'] * df['dom']
        df[f'mcp_{market_type}_with_month'] = df[f'mcp_{market_type}'] * df['month']
        df[f'mcp_{market_type}_with_dow'] = df[f'mcp_{market_type}'] * df['dow']
        df[f'mcp_{market_type}_with_doy'] = df[f'mcp_{market_type}'] * df['doy']
        df[f'mcp_{market_type}_with_tb'] = df[f'mcp_{market_type}'] * df['tb']
        return df

    def _covid(self, df):
        df.loc[(df.year == 2020) & (df.month.isin([3, 4, 5, 6, 7, 8, 9])), 'covid_first_wave'] = 1
        df['covid_first_wave'] = df['covid_first_wave'].replace(np.nan, 0)

        df.loc[(df.year == 2021) & (df.month.isin([3, 4, 5, 6])), 'covid_second_wave'] = 1
        df['covid_second_wave'] = df['covid_second_wave'].replace(np.nan, 0)
        return df

    def _cyclic(self, df, feature):
        df['norm'] = 2 * math.pi * df[f"{feature}"] / df[f"{feature}"].max()
        df[f"cos_{feature}"] = np.cos(df["norm"])
        df[f"sin_{feature}"] = np.sin(df["norm"])
        df.drop('norm', axis=1, inplace=True)
        return df

    def _price_features(self, data, market_type, task):
        data = self._capping(data)
        data = self._datetime_features(data)
        if task == 'train':
            data = self._target(data, market_type)
        data = self._lags(data)
        data = self._min_max(data)
        data = self._ema(data, market_type)
        data = self._mean(data, market_type)
        data = self._interaction(data, market_type)
        data = self._cyclic(data, 'tb')
        data = self._cyclic(data, 'hour')
        data = self._cyclic(data, 'dow')
        data = self._cyclic(data, 'doy')
        return data

    def _weather_features(self, data, weather):
        for column in weather.columns[1:65]:
            daily_mean = data.groupby(['date']).mean()[column].reset_index().rename(columns={column: f'daily_mean_{column}'})
            data = pd.merge(data, daily_mean, on=['date'], how='left')

            for i in range(1, 4):
                data[f'change_in_{column}_wrt_day_{i}'] = (data[column] - data[column].shift(96 * i))

            prec_columns = data.filter(regex='^prec_')
            data['prec_tb'] = prec_columns.sum(axis=1)
        return data

    def _interaction_features(self, data, weather, market_type):
        for i in weather.columns[1:65]:
            data[f'mcp_{market_type}_with_{i}'] = data[f'mcp_{market_type}'] * data[i]
        return data

    def _get_features(self, data, weather, market_type, task='train'):
        data = self._price_features(data, market_type, task)
        data = self._weather_features(data, weather)
        data = self._interaction_features(data, weather, market_type)
        data = data.drop('date', axis=1)
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data
