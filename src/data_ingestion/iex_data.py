'''
This script fetches market data from the IEX API, processes it, and stores it in pickle files.
It includes a class `IexDataFetcher` with methods to retrieve raw market data (dam and rtm), process it, and save the results.

Author: Aman Bhatt
'''

import os
import time
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Parent directory
PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

# Custom modules
from config.paths import *
from src.utils import *

data_logs = configure_logger(LOGS_PATH, 'data.log')

class IexDataFetcher:
    def __init__(self):
        # api credentials
        self.base_url = os.getenv('base_url')
        self.user_email = os.getenv('email')
        self.user_password = os.getenv('password')

    def _get_token(self):
        """
        Retrieves the access token by sending a POST request to the IEX API login endpoint.
        """
        try:
            url = self.base_url + 'login'
            data = {'email': self.user_email, 'password': self.user_password}
            req = requests.post(url=url, data=data, verify=True).json()
            return req['access_token']
        except Exception as e:
            print("Error occurred while retrieving the access token:", str(e))
            data_logs.error("Error in retrieving the access token: %s", str(e))

    def _get_market_data(self, start_date_str, end_date_str, token, market_type):
        """
        Fetches market data from the IEX API based on specified parameters.
        
        Args:
            start_date_str (str): Start date string.
            end_date_str (str): End date string.
            token (str): Access token.
            market_type (str): Type of market data ('dam' or 'rtm').
        """
        try:
            endpoint = 'getMarketVolume' if market_type == 'dam' else 'getRTMMarketVolume'
            url = self.base_url + endpoint
            headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
            params = {"start_date": start_date_str, "end_date": end_date_str}
            r = requests.get(url=url, headers=headers, params=params)
            return r.json()
        except Exception as e:
            print(f"Error occurred while retrieving json data for {market_type}:", str(e))
            data_logs.error("Error occurred while retrieving json data for %s: %s", market_type, str(e))

    def _get_datetime_variables(self, market_type):
        """
        Gets datetime variables for fetching market data.
        
        Args:
            market_type (str): Type of market data ('DAM' or 'RTM').
            
        Returns:
            Tuple: Tuple containing start date, end date, start date string,
                   end date string, and historical data.
        """
        try:
            data_historical = load_pickle(PROCESSED_DATA_PATH, f'{market_type}_data')
            start_date = data_historical['datetime'].iloc[-1] + timedelta(hours=0.25)
            end_date = start_date + timedelta(days=30)
            start_date_str = start_date.strftime("%d-%m-%Y")
            end_date_str = end_date.strftime("%d-%m-%Y")
            return start_date, end_date, start_date_str, end_date_str, data_historical
        except Exception as e:
            print(f"Error while creating datetime variables for {market_type} data.:", str(e))
            data_logs.error("Error while creating datetime variables for %s data: %s", market_type, str(e))

    def _get_raw_data(self, market_type):
        """
        Fetches raw market data, processes it, and saves it in pickle files.
        
        Args:
            market_type (str): Type of market data ('DAM' or 'RTM').
            
        Returns:
            pd.DataFrame: Raw market data.
        """
        try:
            token = self._get_token()
            start_date, end_date, start_date_str, end_date_str, data_historical = self._get_datetime_variables(market_type)
            data_dict = self._get_market_data(start_date_str, end_date_str, token, market_type)
            raw_data = pd.DataFrame(data_dict['data'])
            if raw_data.empty:
                print(f'{market_type} data is already updated up to: ', data_historical['datetime'].iloc[-1])
                return pd.DataFrame()
            else:
                save_pickle(raw_data, RAW_DATA_PATH, f'{market_type}')
                return raw_data
        except Exception as e:
            print("Error in fetching data:", str(e))
            data_logs.error("Error in fetching raw %s data: %s", market_type, str(e))

    def _get_processed_data(self, market_type):
        """
        Processes raw market data, creates additional features, and saves the results.
        
        Args:
            market_type (str): Type of market data ('DAM' or 'RTM').
            
        Returns:
            pd.DataFrame: Processed market data.
        """
        try:
            raw_data = self._get_raw_data(market_type)
            start_date, end_date, start_date_str, end_date_str, data_historical = self._get_datetime_variables(market_type)

            if market_type == 'dam':
                # Processing steps for DAM data
                if not raw_data.empty:
                    df = raw_data[['mcp', 'mcv', 'purchase_bid', 'sell_bid']]
                    for column in df.columns:
                        df.loc[:, column] = pd.to_numeric(df[column], errors='coerce').copy()
                    df = df.rename(columns={'mcp': 'mcp_dam', 'mcv': 'clearedvolume_dam',
                                            'purchase_bid': 'pb_dam', 'sell_bid': 'sb_dam'})
                    df['diff_sb_pb_dam'] = df['pb_dam'] - df['sb_dam']
                    dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='15min')[:-1]
                    current_dates = pd.DataFrame({'datetime': dates})
                    current_data = pd.concat([current_dates, df], axis=1).dropna()
                    processed_data = pd.concat([data_historical, current_data]).reset_index(drop=True)
                    last_date = processed_data['datetime'].iloc[-1].strftime('%d-%m-%Y %H:%M')
                    print(f'{market_type} data updated up to: ', last_date)
                    save_pickle(processed_data, PROCESSED_DATA_PATH, f'{market_type}_data')
                    return processed_data
                else:
                    return data_historical

            elif market_type == 'rtm':
                # Processing steps for RTM data
                if not raw_data.empty:
                    df = raw_data.copy()
                    df['datetime'] = pd.to_datetime(df['date'], format='%d-%m-%Y') + pd.to_timedelta(df['time_block'].str.split('-').str[0] + ':00')
                    df = df[df['datetime'] >= start_date]
                    df = df[['datetime', 'mcp', 'mcv', 'purchase_bid', 'sell_bid']]
                    for column in df.columns[1:]:
                        df[column] = pd.to_numeric(df[column])
                    df = df.rename(columns={'mcp': 'mcp_rtm', 'mcv': 'clearedvolume_rtm',
                                            'purchase_bid': 'pb_rtm', 'sell_bid': 'sb_rtm'})
                    df['diff_sb_pb_rtm'] = df['pb_rtm'] - df['sb_rtm']
                    current_data = df.copy()
                    processed_data = pd.concat([data_historical, current_data]).reset_index(drop=True)
                    save_pickle(processed_data, PROCESSED_DATA_PATH, f'{market_type}_data') 
                    last_date = processed_data['datetime'].iloc[-1].strftime('%d-%m-%Y %H:%M')
                    print(f'{market_type} data updated up to: ', last_date)
                    return processed_data
                else:
                    last_date = data_historical['datetime'].iloc[-1].strftime('%d-%m-%Y %H:%M')
                    return data_historical 
                
            else:
                print('Use either "dam" or "rtm" to fetch data.')
                data_logs.warning('Use either "dam" or "rtm" to fetch data.')
        except Exception as e:
            print("Error in processing data:", str(e)) 
            data_logs.error('Error while processing %s data: %s', market_type, str(e))
            