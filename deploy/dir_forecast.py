# %%
"""
Script to forecast directional market values.

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
PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

os.environ['TZ'] = 'Asia/Calcutta'
time.tzset()

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
from src.db_insertion.db_insertion import DirInsertion
from src.utils import *
from config.paths import *

forecasting_logs = configure_logger(LOGS_PATH, 'forecasting.log')
# %%
market_type = 'rtm'
forecasting_logs.info('%s forecasting script running.', market_type)
# %%
# creating instances
iex_data = IexDataFetcher()
weather_data = WeatherDataFetcher()

featured_data = FeatureEngineering(PROJECT_PATH)
forecasting = ModelForecaster(MODELS_PATH, market_type) 
db_insert = DirInsertion() 

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
rtm = rtm[rtm['datetime'] < datetime.now().strftime('%Y-%m-%d')]

# %%
dam = featured_data.shift_date(dam, 1) 
weather = featured_data.shift_date(weather, 2)
hydro = featured_data.shift_date(hydro, 2) 
solar = featured_data.shift_date(solar, 2) 
wind = featured_data.shift_date(wind, 2)
holidays = featured_data.shift_date(holidays, -2) 
holidays = holidays.rename(columns = {'holiday': 'holiday_next_day'})

data = featured_data.merge_dataframes([rtm, dam, weather, hydro, solar, wind])

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
dam_forecast = load_pickle(DAM_FORECAST_PATH, f'dam_forecast_{forecast_date}')
rtm_forecast = load_pickle(DIR_FORECAST_PATH, f'{market_type}_forecast_{forecast_date}')

# %%
dir_rtm = featured_data.merge_dataframes([dam_forecast, rtm_forecast])

# %%
forecasts = dir_rtm.copy()
forecasts['dam_rtm_diff'] = forecasts['dam_forecast'] - forecasts[f'{market_type}_forecast']

# if dam > rtm, make dam_greater = 1, elif if dam < rtm, make dam_greater = 0 else dam 
forecasts.loc[(forecasts['dam_rtm_diff'] > 0), 'dam_greater'] = 1
forecasts.loc[(forecasts['dam_rtm_diff'] < 0), 'dam_greater'] = 0
forecasts.loc[(forecasts['dam_rtm_diff'] == 0), 'dam_greater'] = -1

forecasts = forecasts[['datetime', 'dam_greater']]
forecasts = forecasts.rename(columns = {'dam_greater': f'dir_forecast'})

save_pickle(forecasts, DIR_FORECAST_PATH, f'dir_forecast_{forecast_date}')
save_excel(forecasts, DIR_FORECAST_PATH, f'dir_forecast_{forecast_date}')
print('Directional forecast created.')
forecasting_logs.info('Directional forecast created.')
# %%
dir_rtm.set_index('datetime')[['dam_forecast', f'{market_type}_forecast']].plot()

# %%
db_insert.save_forecast(forecasts, forecast_date, 'dir')

# %%
end_time = time.time()
total_time = (end_time - start_time)/60
print(f'Forecasting time: {total_time:.2f} minutes.')
forecasting_logs.info('Forecasting time: %.2f minutes.', total_time)
forecasting_logs.info('**********************************************')
forecasting_logs.info('**********************************************\n')