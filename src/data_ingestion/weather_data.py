import os
import time
import sys
import yaml
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access credentials
api_key = os.getenv('meteoblue_api_key')

ROOT_PATH = os.getenv('ROOT_PATH')
sys.path.append(ROOT_PATH)

from config.paths import raw_data_path, processed_data_path


class WeatherDataFetcher:
    def __init__(self):
        # Access credentials
        self.api_key = os.getenv('meteoblue_api_key')
        self.ROOT_PATH = os.getenv('ROOT_PATH')
        sys.path.append(self.ROOT_PATH)

    def _get_weather_response(self, lat, lon, asl, name):
        try:
            url = f"http://my.meteoblue.com/packages/basic-1h?name={name}&lat={lat}&lon={lon}&asl={asl}&tz=UTC&apikey={self.api_key}"
            response = requests.get(url)
            data = response.json()
            return data
        except Exception as e:
            print("Error occurred while retrieving data:", str(e))

    def _get_solar_response(self, lat, lon, asl, name):
        try:
            url = f"http://my.meteoblue.com/packages/solar-15min?name={name}&lat={lat}&lon={lon}&asl={asl}&tz=UTC&apikey={self.api_key}"
            response = requests.get(url)
            data = response.json()
            return data

        except Exception as e:
            print("Error occurred while retrieving data:", str(e))

    def _fetch_weather_data(self, locations, required_features, location_type):
        df_combined = pd.DataFrame()

        for location in locations:
            lat, lon, asl, name = location
            if location_type == 'solar':
                data_dict = self._get_solar_response(lat, lon, asl, name)
                df = pd.DataFrame.from_dict(data_dict['data_xmin'])
            else:
                data_dict = self._get_weather_response(lat, lon, asl, name)
                df = pd.DataFrame.from_dict(data_dict['data_1h'])

            df = self._preprocess_weather_data(df, required_features, name)
            df_combined = pd.concat([df_combined, df], ignore_index=True)
        df_combined = self._resample_and_interpolate(df_combined)
        df_combined.to_pickle(os.path.join(raw_data_path, f'{location_type}'))
        return df_combined

    def _preprocess_weather_data(self, df, required_features, name):
        cols = required_features
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df = df.sort_values('time').drop_duplicates(subset=['time'], keep='first')
        df = df.rename(columns={'time': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'] + timedelta(hours=5.5)
        df = df[['datetime'] + required_features]
        df = df.rename(columns={'windspeed': 'ws', 'ghi_instant': 'ghi',
                                'relativehumidity': 'rh', 'winddirection': 'wd',
                                'temperature': 'temp', 'felttemperature': 'atemp', 'precipitation': 'prec'})
        df['datetime'] = df['datetime'].astype(str)
        df['location'] = name
        return df

    def _resample_and_interpolate(self, df):
        df = df.pivot(index='datetime', columns='location')
        df.columns = [f"{col[0]}_{col[1][:3]}" for col in df.columns]
        df.reset_index(inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').resample('15MIN').interpolate(method='linear').reset_index()
        return df

    def _load_locations(self, file_path, location_type):
        with open(file_path, 'r') as file:
            locations = yaml.safe_load(file)
        return locations[location_type]

    def _get_processed_weather(self, location_type):
        # Load weather locations from YAML file
        locations = self._load_locations(os.path.join(self.ROOT_PATH, 'config', 'locations.yaml'), location_type)

        # Get required features based on location type
        required_features = locations['required_features']

        raw_df = self._fetch_weather_data(locations['locations'], required_features, location_type)

        start_date = raw_df['datetime'].iloc[0]
        data_historical = pd.read_pickle(os.path.join(processed_data_path, f'{location_type}_data'))
        data_historical = data_historical[data_historical['datetime'] < start_date]
        final_df = pd.concat([data_historical, raw_df]).reset_index(drop=True)
        final_df.to_pickle(os.path.join(processed_data_path, f'{location_type}_data'))

        return final_df
