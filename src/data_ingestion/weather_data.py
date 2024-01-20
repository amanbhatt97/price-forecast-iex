'''
This script fetches weather data from the Meteoblue API, processes it, and stores it in pickle files.
It includes a class `WeatherDataFetcher` with methods to retrieve data, process it, and save the results.

Author: Aman Bhatt
'''

# Import necessary libraries
import os
import sys
import yaml
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access Meteoblue API key from environment variables
api_key = os.getenv('meteoblue_api_key')

# Access project paths
PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

# Import utility functions
from config.paths import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.utils import *
from config.paths import LOGS_PATH

data_logs = configure_logger(LOGS_PATH, 'data.log')

class WeatherDataFetcher:
    def __init__(self):
        '''
        Initializes the WeatherDataFetcher class with necessary credentials and paths.
        '''
        # Access credentials
        self.api_key = os.getenv('meteoblue_api_key')
        self.PROJECT_PATH = os.getenv('PROJECT_DIR')
        sys.path.append(self.PROJECT_PATH)

    def _get_weather_response(self, lat, lon, asl, name):
        '''
        Sends a request to the Meteoblue API for basic 1-hour weather data.

        Args:
        - lat: Latitude of the location
        - lon: Longitude of the location
        - asl: Altitude above sea level of the location
        - name: Name of the location

        Returns:
        - Weather data in JSON format
        '''
        try:
            url = f"http://my.meteoblue.com/packages/basic-1h?name={name}&lat={lat}&lon={lon}&asl={asl}&tz=UTC&apikey={self.api_key}"
            response = requests.get(url)
            data = response.json()
            return data
        except Exception as e:
            print("Error occurred while retrieving json data:", str(e))
            data_logs.error("Error occurred while retrieving the json data: ", str(e))

    def _get_solar_response(self, lat, lon, asl, name):
        '''
        Sends a request to the Meteoblue API for solar 15-minute weather data.

        Args:
        - lat: Latitude of the location
        - lon: Longitude of the location
        - asl: Altitude above sea level of the location
        - name: Name of the location

        Returns:
        - Solar weather data in JSON format
        '''
        try:
            url = f"http://my.meteoblue.com/packages/solar-15min?name={name}&lat={lat}&lon={lon}&asl={asl}&tz=UTC&apikey={self.api_key}"
            response = requests.get(url)
            data = response.json()
            return data

        except Exception as e:
            print("Error occurred while retrieving json data:", str(e))
            data_logs.error("Error occurred while retrieving json data:", str(e)) 

    def _fetch_weather_data(self, locations, required_features, location_type):
        '''
        Fetches weather data for multiple locations, preprocesses it, and combines into a DataFrame.

        Args:
        - locations: List of location coordinates and names
        - required_features: List of features to be included in the processed data
        - location_type: Type of location data ('solar' or other)

        Returns:
        - Combined DataFrame with weather data
        '''
        try:
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
            save_pickle(df_combined, RAW_DATA_PATH, f'{location_type}')
            return df_combined
        except Exception as e:
            print(f'Error while retrieving {location_type} data: ', str(e))
            data_logs.error('Errorwhile retrieving %s data: ', location_type, str(e))

    def _preprocess_weather_data(self, df, required_features, name):
        '''
        Preprocesses raw weather data by converting columns to numeric, handling duplicates, and renaming columns.

        Args:
        - df: Raw weather data DataFrame
        - required_features: List of features to be included in the processed data
        - name: Name of the location

        Returns:
        - Preprocessed weather data DataFrame
        '''
        try:
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
        except Exception as e:
            print(f'Error during preprocessing: ', str(e))
            data_logs.error('Error during preprocessing: ', str(e)) 

    def _resample_and_interpolate(self, df):
        '''
        Resamples and interpolates weather data to a 15-minute interval.

        Args:
        - df: Preprocessed weather data DataFrame

        Returns:
        - Resampled and interpolated weather data DataFrame
        '''
        try:
            df = df.pivot(index='datetime', columns='location')
            df.columns = [f"{col[0]}_{col[1][:3]}" for col in df.columns]
            df.reset_index(inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').resample('15MIN').interpolate(method='linear').reset_index()
            return df
        except Exception as e:
            print(f'Error during resampling and interpolation: ', str(e))
            data_logs.error('Error during resampling and interpolation: ', str(e))

    def _load_locations(self, file_path, location_type):
        '''
        Loads location coordinates and names from a YAML file.

        Args:
        - file_path: Path to the YAML file
        - location_type: Type of location data ('solar' or other)

        Returns:
        - Dictionary containing location information
        '''
        try:
            with open(file_path, 'r') as file:
                locations = yaml.safe_load(file)
            return locations[location_type]
        except Exception as e:
            print(f'Error while fetching {location_type} locations: ',str(e))
            data_logs.error("Error while fetching %s locations: %s", location_type, str(e)) 

    def _get_processed_weather(self, location_type):
        '''
        Retrieves processed weather data, updates historical data, and saves the final DataFrame.

        Args:
        - location_type: Type of location data ('solar' or other)

        Returns:
        - Final DataFrame with updated weather data
        '''
        try:
            # Load weather locations from YAML file
            locations = self._load_locations(os.path.join(self.PROJECT_PATH, 'config', 'locations.yaml'), location_type)

            # Get required features based on location type
            required_features = locations['required_features']

            raw_df = self._fetch_weather_data(locations['locations'], required_features, location_type)

            start_date = raw_df['datetime'].iloc[0]
            data_historical = pd.read_pickle(os.path.join(PROCESSED_DATA_PATH, f'{location_type}_data'))
            data_historical = data_historical[data_historical['datetime'] < start_date]
            final_df = pd.concat([data_historical, raw_df]).reset_index(drop=True)
            save_pickle(final_df, PROCESSED_DATA_PATH, f'{location_type}_data')
            print(f'{location_type} data updated.')
            data_logs.info('%s data updated.', location_type)
            return final_df
        except Exception as e:
            print(f'Error during {location_type} processing: ', str(e))
            data_logs.error('Error during %s processing: ', location_type, str(e))