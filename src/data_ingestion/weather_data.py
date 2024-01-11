import os, time, sys
import yaml
import requests
import pandas as pd
from datetime import datetime, timedelta
os.environ['TZ'] = 'Asia/Calcutta'
time.tzset()

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access credentials
api_key = os.getenv('meteoblue_api_key')

ROOT_PATH = os.getenv('ROOT_PATH')
sys.path.append(ROOT_PATH)

from config.utils import *

def get_weather_response(lat, lon, asl, name):
    try:
        url = f"http://my.meteoblue.com/packages/basic-1h?name={name}&lat={lat}&lon={lon}&asl={asl}&tz=UTC&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        print("Error occurred while retrieving data:", str(e))


def fetch_weather_data(locations, required_features, para='weather'):
    df_combined = pd.DataFrame()

    for location in locations:
        lat, lon, asl, name = location
        data_dict = get_weather_response(lat, lon, asl, name)
        df = pd.DataFrame.from_dict(data_dict['data_1h'])
        df.to_pickle(os.path.join(data_path, 'raw', 'weather'))
        df = preprocess_weather_data(df, required_features, name)
        df_combined = pd.concat([df_combined, df], ignore_index=True)

    df_combined = resample_and_interpolate(df_combined)
    return df_combined


def preprocess_weather_data(df, required_features, name):
    cols = required_features
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.sort_values('time').drop_duplicates(subset=['time'], keep='first')
    df = df.rename(columns={'time': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'] + timedelta(hours=5.5)
    df = df[['datetime'] + required_features]
    df = df.rename(columns={ 'windspeed': 'ws', 'ghi_instant': 'ghi',
                            'relativehumidity': 'rh', 'winddirection': 'wd',
                            'temperature': 'temp', 'felttemperature': 'atemp', 'precipitation': 'prec'})
    df['datetime'] = df['datetime'].astype(str)
    df['location'] = name
    return df


def resample_and_interpolate(df):
    df = df.pivot(index='datetime', columns='location')
    df.columns = [f"{col[0]}_{col[1][:3]}" for col in df.columns]
    df.reset_index(inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').resample('15MIN').interpolate(method='linear').reset_index()
    return df


def load_locations(file_path):
    with open(file_path, 'r') as file:
        locations = yaml.safe_load(file)
    return locations['weather_locations']


def get_weather():
    # Load weather locations from YAML file
    locations = load_locations(os.path.join(ROOT_PATH, 'config', 'locations.yaml'))

    required_features = ['temperature', 'felttemperature', 'relativehumidity', 'precipitation']
    weather_df = fetch_weather_data(locations, required_features)

    start_date = weather_df['datetime'].iloc[0]
    weather_historical = pd.read_pickle(os.path.join(data_path, 'processed', 'weather_data'))
    weather_historical = weather_historical[weather_historical['datetime'] < start_date]
    weather = pd.concat([weather_historical, weather_df]).reset_index(drop=True)

    weather.to_pickle(os.path.join(data_path, 'processed', 'weather_data'))

    last_date = weather['datetime'].iloc[-1]
    return weather