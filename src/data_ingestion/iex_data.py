import os, time, sys
import requests
import pandas as pd
from datetime import datetime, timedelta
os.environ['TZ'] = 'Asia/Calcutta'
time.tzset()

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access credentials
base_url = os.getenv('base_url')
user_email = os.getenv('email')
user_password = os.getenv('password')

# Parent directory
ROOT_PATH = os.getenv('ROOT_PATH')
sys.path.append(ROOT_PATH)

# Custom modules
from config.paths import *
from config.utils import get_datetime_variables

def get_token(base_url):
    """
    Fetches the access token for the specified data API.

    Args:
        base_url (str): The base URL for the API.

    Returns:
        str: Access token for authentication.
    """
    try:
        url = base_url + 'login'
        email = user_email
        password = user_password
        data = {'email': email, 'password': password}
        req = requests.post(url=url, data=data, verify=True).json()
        return req['access_token']
    except Exception as e:
        print("Error occurred while retrieving the access token:", str(e))


def get_market_data(start_date_str, end_date_str, token, market_type):
    """
    Makes a GET request to the specified market API endpoint.

    Parameters:
        start_date_str (str): Start date for data retrieval in string format (format: "dd-mm-yyyy").
        end_date_str (str): End date for data retrieval in string format (format: "dd-mm-yyyy").
        token (str): Access token for API authentication.
        market_type (str): Market type ('DAM' or 'RTM').

    Returns:
        dict: JSON response containing market data.
    """
    try:
        endpoint = 'getMarketVolume' if market_type == 'DAM' else 'getRTMMarketVolume'
        url = base_url + endpoint
        headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
        params = {"start_date": start_date_str, "end_date": end_date_str}
        r = requests.get(url=url, headers=headers, params=params)
        return r.json()
    except Exception as e:
        print.info("Error occurred while retrieving data:", str(e))

    
def get_raw_data(data_type):
    """
    Fetches DAM (Day-Ahead Market) actual data and logs information.

    Returns:
        pd.DataFrame: DataFrame containing the fetched DAM data.
    """
    try:
        token = get_token(base_url)
        start_date, end_date, start_date_str, end_date_str, data_historical = get_datetime_variables(data_type)
        data_dict = get_market_data(start_date_str, end_date_str, token, data_type)
        raw_data = pd.DataFrame(data_dict['data'])
        if raw_data.empty:
            print('Data is already updated upto: ', data_historical['datetime'].iloc[-1])
            return pd.DataFrame()
        else:
            print('Data updated upto: ', raw_data['date'].iloc[-1])
            raw_data.to_pickle(os.path.join(data_path, 'raw', f'{data_type}'))
            return raw_data
    except Exception as e:
        print("Error in fetching data:", str(e)) 


def get_processed_data(data_type):
    """
    Perform preprocessing on market data.

    Args:
        raw_data (pd.DataFrame): Raw market data.
        historical_data (pd.DataFrame): Historical market data.
        start_date (pd.Timestamp): Start date for data processing.
        end_date (pd.Timestamp): End date for data processing.
        data_type (str): Type of market data (e.g., 'dam' or 'rtm').

    Returns:
        pd.DataFrame: Processed market data.
    """
    raw_data = get_raw_data(data_type)
    start_date, end_date, start_date_str, end_date_str, data_historical = get_datetime_variables(data_type)

    if data_type == 'dam': 
        if not raw_data.empty:
            # removing reduntant columns
            df = raw_data[['mcp', 'mcv', 'purchase_bid', 'sell_bid']]

            # converting to numeric type
            for column in df.columns:
                df.loc[:, column] = pd.to_numeric(df[column], errors='coerce').copy()

            # renaming
            df = df.rename(columns={'mcp': 'mcp_dam', 'mcv': 'clearedvolume_dam',
                                'purchase_bid': 'pb_dam', 'sell_bid': 'sb_dam'})

            # creating difference feature
            df['diff_sb_pb_dam'] = df['pb_dam'] - df['sb_dam']

            # creating daterange and merging to raw data
            dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='15min')[:-1]
            current_dam = pd.DataFrame({'datetime': dates})
            current_dam = pd.concat([current_dam, df], axis=1).dropna()

            # merging with historical data
            dam = pd.concat([data_historical, current_dam]).reset_index(drop=True)

            # saving data
            dam.to_pickle(os.path.join(data_path, 'processed', f'{data_type}_data'))

        else:
            return data_historical


    elif data_type == 'rtm':
        if not raw_data.empty:
            df = raw_data.copy()
            # Additional preprocessing steps for RTM data
            df['datetime'] = pd.to_datetime(df['date'], format='%d-%m-%Y') + pd.to_timedelta(df['time_block'].str.split('-').str[0] + ':00')
            df = df[df['datetime'] >= start_date]
            df = df[['datetime', 'mcp', 'mcv', 'purchase_bid', 'sell_bid']]

            for column in df.columns[1:]:
                df[column] = pd.to_numeric(df[column])

            df = df.rename(columns={'mcp': 'mcp_rtm', 'mcv': 'clearedvolume_rtm',
                                    'purchase_bid': 'pb_rtm', 'sell_bid': 'sb_rtm'})
            df['diff_sb_pb_rtm'] = df['pb_rtm'] - df['sb_rtm']

            # Merging with historical data
            current_data = df.copy()
            processed_data = pd.concat([data_historical, current_data]).reset_index(drop=True)

            # Saving data
            processed_data.to_pickle(os.path.join(data_path, 'processed', f'{data_type}_data'))

            last_date = processed_data['datetime'].iloc[-1].strftime('%d-%m-%Y %H:%M')
            return processed_data
        
        else:
            return data_historical 
        
    else:
        print('Use either "dam" or "rtm" to fetch data.')