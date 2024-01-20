# %%
"""
Script to fetch data for day-ahead, real-time market prices, weather, power stations

Author: Aman Bhatt
"""
import time, sys, os
start_time = time.time()
os.environ['TZ'] = 'Asia/Calcutta'
time.tzset()

from dotenv import load_dotenv
load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

# %%
from src.data_ingestion.iex_data import IexDataFetcher
from src.data_ingestion.weather_data import WeatherDataFetcher
from config.paths import LOGS_PATH
from src.utils import *

data_logs = configure_logger(LOGS_PATH, 'data.log')

iex_data = IexDataFetcher()
weather_data = WeatherDataFetcher()

# %%
dam = iex_data._get_processed_data('dam')
data_logs.info('dam data uptated upto: %s', dam['datetime'].iloc[-1])

rtm = iex_data._get_processed_data('rtm')
data_logs.info('rtm data uptated upto: %s', rtm['datetime'].iloc[-1])

weather = weather_data._get_processed_weather('weather')
wind = weather_data._get_processed_weather('wind')
hydro = weather_data._get_processed_weather('hydro')
solar = weather_data._get_processed_weather('solar')


# %%
end_time = time.time()
total_time = (end_time - start_time)/60
print(f'Time to fetch data: {total_time:.2f} minutes.')
data_logs.info('Time to fetch data: %.2f minutes.', total_time)
data_logs.info('**********************************************\n')