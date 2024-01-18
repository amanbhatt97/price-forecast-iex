# %%
"""
Script to forecast day-ahead market values.

Author: Aman Bhatt
"""
import time
start_time = time.time()
import os, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

os.environ['TZ'] = 'Asia/Calcutta'
time.tzset()

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# visualization
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)

# %%
from src.data_ingestion.iex_data import IexDataFetcher
from src.data_ingestion.weather_data import WeatherDataFetcher
from src.feature_engineering.build_features import FeatureEngineering
from src.model_building.forecast_model import ModelForecaster
from src.db_insertion.db_insertion import DAMInsertion
from src.utils import *
from config.paths import *

forecasting_logs = configure_logger(LOGS_PATH, 'forecasting.log')
# %%
market_type = 'dam'
forecasting_logs.info('%s forecasting script running.', market_type)
# %%
# creating instances
iex_data = IexDataFetcher()
weather_data = WeatherDataFetcher()

featured_data = FeatureEngineering(PROJECT_PATH)
forecasting = ModelForecaster(MODELS_PATH, market_type) 
db_insert = DAMInsertion() 

# %%
# dam = iex_data._get_processed_data('dam')
# rtm = iex_data._get_processed_data('rtm')
# weather = weather_data._get_processed_weather('weather')
# wind = weather_data._get_processed_weather('wind')
# hydro = weather_data._get_processed_weather('hydro')
# solar = weather_data._get_processed_weather('solar')

# %%
dam = load_pickle(PROCESSED_DATA_PATH, 'dam_data')
rtm = load_pickle(PROCESSED_DATA_PATH, 'rtm_data')
weather = load_pickle(PROCESSED_DATA_PATH, 'weather_data')
wind = load_pickle(PROCESSED_DATA_PATH, 'wind_data')
hydro = load_pickle(PROCESSED_DATA_PATH, 'hydro_data')
solar = load_pickle(PROCESSED_DATA_PATH, 'solar_data')
holidays = featured_data.process_holidays(load_pickle(EXTERNAL_DATA_PATH, 'holidays_data'))
print('Data loaded.')
forecasting_logs.info('Data loaded.')
# %%
rtm = featured_data.shift_date(rtm, 1) 
weather = featured_data.shift_date(weather, -1)
hydro = featured_data.shift_date(hydro, -1) 
solar = featured_data.shift_date(solar, -1) 
wind = featured_data.shift_date(wind, -1)
holidays = featured_data.shift_date(holidays, -1) 
holidays = holidays.rename(columns = {'holiday': 'holiday_next_day'})

data = featured_data.merge_dataframes([dam, rtm, weather, hydro, solar, wind])

# %%
data = featured_data._get_features(data, weather, market_type, task = 'inference')

# %%
print(f'Features created for {market_type}.')
forecasting_logs.info('Features created for %s.', market_type)
# %%
forecast_date = forecasting.forecasting_date(data, market_type)

# %%
print('Forecasting date: ', forecast_date)
forecasting_logs.info('Forecasting date: %s.', forecast_date)
# %%
forecast = forecasting.create_forecast(data, forecast_date, market_type)

# %%
print(f'{market_type} forecast created.')
forecasting_logs.info('%s forecast created.', market_type)
# %%
db_insert.save_forecast(forecast, forecast_date, f'{market_type}_forecast')

# %%
db_insert.save_forecast(forecast,  forecast_date, 'lower_bound')

# %%
db_insert.save_forecast(forecast,  forecast_date, 'upper_bound')

# %%
end_time = time.time()
total_time = (end_time - start_time)/60
print(f'Forecasting time: {total_time:.2f} minutes.')
forecasting_logs.info('Forecasting time: %.2f minutes.', total_time)
forecasting_logs.info('**********************************************\n')