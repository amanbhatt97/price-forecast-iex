# %%
import time, sys, os
start_time = time.time()

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

# %%
from src.data_ingestion.iex_data import IexDataFetcher
from src.data_ingestion.weather_data import WeatherDataFetcher

# %%
iex_data = IexDataFetcher()
weather_data = WeatherDataFetcher()

# %%
dam = iex_data._get_processed_data('dam')
rtm = iex_data._get_processed_data('rtm')
weather = weather_data._get_processed_weather('weather')
wind = weather_data._get_processed_weather('wind')
hydro = weather_data._get_processed_weather('hydro')
solar = weather_data._get_processed_weather('solar')

# %%
end_time = time.time()
total_time = (end_time - start_time)/60
print(f'Time to fetch data: {total_time:.2f} minutes.')


