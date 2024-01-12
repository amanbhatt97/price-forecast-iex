from functools import reduce
import pandas as pd
import numpy as np
import sys, os
import math

ROOT_PATH = os.getenv('ROOT_PATH')
sys.path.append(ROOT_PATH)

def merge_dataframes(dfs, on_column='datetime'):
    """This method merges all the dataframes in a list dfs"""
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=on_column, how='left'), dfs)
    merged_df.dropna(inplace=True)
    return merged_df

def datetime_features(df):
    data = df.copy()
    # time related features
    data['date'] = data.datetime.dt.date     
    data['date'] = pd.to_datetime(data['date'])
    data['hour'] = data.datetime.dt.hour + 1   
    data['dom' ] = data.datetime.dt.day        
    data['month'] = data.datetime.dt.month     
    data['year'] = data.datetime.dt.year       
    data['dow' ] = data.datetime.dt.dayofweek  
    data['doy' ] = data.datetime.dt.dayofyear  
    data['tb'] = data.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1)) 
    data['next_day_sunday'] = np.where(data['dow']==5, 1,0) 
    data['hour_dow'] = data['hour'] * data['dow']

    # time of day features
    data['isMorning'] = ((data['datetime'].dt.hour>=6) &  (data['datetime'].dt.hour<=9) ).astype(int)
    data['isDay'] = ((data['datetime'].dt.hour>=10) &  (data['datetime'].dt.hour<=16) ).astype(int)
    data['isEvening'] = ((data['datetime'].dt.hour>=17) &  (data['datetime'].dt.hour<=23) ).astype(int)
    data['isNight'] = ((data['datetime'].dt.hour==24) | ((data['datetime'].dt.hour>=1) & (data['datetime'].dt.hour<=5)) ).astype(int)

    # season features
    data['winter'] = ((data['month'] == 12) | (data['month'] == 1) | (data['month'] == 2)).astype(int)
    data['summer'] = ((data['month'] >= 3) & (data['month'] <= 6)).astype(int)
    data['monsoon'] = ((data['month'] >= 7) & (data['month'] <= 8)).astype(int)
    data['autumn'] = ((data['month'] >= 9) & (data['month'] == 11)).astype(int)

    return data


def target(data, data_type):
    if data_type == 'dam':
        data['target'] = data[f'mcp_{data_type}'].shift(-96)
    elif data_type == 'rtm':
        data['target'] = data[f'mcp_{data_type}'].shift(-96*2)
    else:
        print('Chose dam or rtm.')
    return data


def lags(df):
    data = df.copy()

    for column in df.columns[1:11]:  
        shift_columns = [data[column].shift(96 * i) for i in range(1, 6)]
        data = pd.concat([data] + shift_columns, axis=1)
        data.columns = [*data.columns[:-5], *[f'change_in_{column}_wrt_day_{i}' for i in range(1, 6)]]

        for i in [1, 2, 4, 8, 12, 18]: 
            # hour lags feature
            data[f'{column}_lag_{i}h'] = data[column].shift(i*4)

    return data

def min_max(df):
    data = df.copy()

    for column in df.columns[1:11]:
        # hour mean
        hour_mean = data.groupby(['date', 'hour']).mean()[column].reset_index().rename(columns={column: f'hour_mean_{column}'})
        data = pd.merge(data, hour_mean, on=['date', 'hour'], how='left')    
        
        # daily mean
        daily_mean = data.groupby(['date']).mean()[column].reset_index().rename(columns={column: f'daily_mean_{column}'})
        data = pd.merge(data, daily_mean, on=['date'], how='left')
        
        # daily minimum
        daily_min = data.groupby(['date']).min()[column].reset_index().rename(columns={column: f'daily_min_{column}'})
        data = pd.merge(data, daily_min, on=['date'], how='left')
        
        # daily maximum
        daily_max = data.groupby(['date']).max()[column].reset_index().rename(columns={column: f'daily_max_{column}'})
        data = pd.merge(data, daily_max, on=['date'], how='left')

    return data


# exponential weighted moving averages
def ema(df, data_type):
    data = df.copy()
    for i in [1,3,6,12]:
        data[f'ewma_{i}h'] = data[f'mcp_{data_type}'].ewm(span=4*i).mean() 
        
    for i in [1,3,5]:
        data[f'ewma_{i}d'] = data[f'mcp_{data_type}'].ewm(span=96*i).mean()
    return data


# mean
def mean(df, data_type):
    data = df.copy()
    window_sizes = [2, 3, 5]
    for window_size in window_sizes:
        data[f'mcp_{data_type}_mean_{window_size}d'] = data[f'mcp_{data_type}'].rolling(window_size * 96).mean()
    return data


# interaction with datetime
def interaction(df, data_type):
    data = df.copy()
    data[f'mcp_{data_type}_with_dom'] = data[f'mcp_{data_type}'] * data['dom'] 
    data[f'mcp_{data_type}_with_month'] = data[f'mcp_{data_type}'] * data['month']
    data[f'mcp_{data_type}_with_dow'] = data[f'mcp_{data_type}'] * data['dow']
    data[f'mcp_{data_type}_with_doy'] = data[f'mcp_{data_type}'] * data['doy']
    data[f'mcp_{data_type}_with_tb'] = data[f'mcp_{data_type}'] * data['tb']
    return data


# covid features
def covid(df):
    data = df.copy()
    data.loc[(data.year == 2020)&(data.month.isin([3,4,5,6,7,8,9])),'covid_first_wave'] = 1
    data['covid_first_wave'] = data['covid_first_wave'].replace(np.nan, 0)

    # covid 2nd wave
    data.loc[(data.year == 2021)&(data.month.isin([3,4,5,6])),'covid_second_wave'] = 1
    data['covid_second_wave'] = data['covid_second_wave'].replace(np.nan, 0)
    return data


# cyclic features
def cyclic(df,feature):
    data = df.copy()
    data['norm'] = 2 * math.pi * data[f"{feature}"] / data[f"{feature}"].max()
    data[f"cos_{feature}"] = np.cos(data["norm"])
    data[f"sin_{feature}"] = np.sin(data["norm"])
    data.drop('norm',axis=1,inplace=True)
    return data


def price_features(data, data_type):
    data = datetime_features(data)
    data = target(data, data_type)
    data = lags(data)
    data = min_max(data)
    data = ema(data, data_type)
    data = mean(data, data_type)
    data = interaction(data, data_type)
    data = cyclic(data,'tb')
    data = cyclic(data,'hour')
    data = cyclic(data,'dow')
    data = cyclic(data,'doy')

    return data


def weather_features(data, weather):
    for column in weather.columns[1:65]:
        # daily mean weather
        daily_mean = data.groupby(['date']).mean()[column].reset_index().rename(columns = {column:f'daily_mean_{column}'})
        data = pd.merge(data, daily_mean, on=['date'], how='left')

        # difference in weather features
        for i in range(1, 4):
                data[f'change_in_{column}_wrt_day_{i}'] = (data[column] - data[column].shift(96*i))

        # total precipitation in each timeblock
        prec_columns = data.filter(regex='^prec_')
        data['prec_tb'] = prec_columns.sum(axis=1)
    
    return data


def interaction_features(data, weather, data_type):
    for i in weather.columns[1:65]:
        data[f'mcp_{data_type}_with_{i}'] = data[f'mcp_{data_type}'] * data[i]
    return data
    

def feature_engineering(data, weather, data_type):
    data = price_features(data, data_type)
    data = weather_features(data, weather)
    data = interaction_features(data, weather, data_type)
    data = data.drop('date', axis = 1)    
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data


def save_features(data, location):
    data.to_pickle(location)