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
ROOT_PATH = os.getenv('ROOT_PATH')
sys.path.append(ROOT_PATH)

# Custom modules
from config.paths import *

class IexDataFetcher:
    def __init__(self):
        # Access credentials
        self.base_url = os.getenv('base_url')
        self.user_email = os.getenv('email')
        self.user_password = os.getenv('password')

    def _get_token(self):
        try:
            url = self.base_url + 'login'
            data = {'email': self.user_email, 'password': self.user_password}
            req = requests.post(url=url, data=data, verify=True).json()
            return req['access_token']
        except Exception as e:
            print("Error occurred while retrieving the access token:", str(e))

    def _get_market_data(self, start_date_str, end_date_str, token, market_type):
        try:
            endpoint = 'getMarketVolume' if market_type == 'DAM' else 'getRTMMarketVolume'
            url = self.base_url + endpoint
            headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
            params = {"start_date": start_date_str, "end_date": end_date_str}
            r = requests.get(url=url, headers=headers, params=params)
            return r.json()
        except Exception as e:
            print("Error occurred while retrieving data:", str(e))

    def _get_raw_data(self, data_type):
        try:
            token = self._get_token()
            start_date, end_date, start_date_str, end_date_str, data_historical = self._get_datetime_variables(data_type)
            data_dict = self._get_market_data(start_date_str, end_date_str, token, data_type)
            raw_data = pd.DataFrame(data_dict['data'])
            if raw_data.empty:
                print('Data is already updated up to: ', data_historical['datetime'].iloc[-1])
                return pd.DataFrame()
            else:
                print('Data updated up to: ', raw_data['date'].iloc[-1])
                raw_data.to_pickle(os.path.join(raw_data_path, f'{data_type}'))
                return raw_data
        except Exception as e:
            print("Error in fetching data:", str(e))
    
    def _get_processed_data(self, data_type):
        raw_data = self._get_raw_data(data_type)
        start_date, end_date, start_date_str, end_date_str, data_historical = self._get_datetime_variables(data_type)

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
                dam.to_pickle(os.path.join(processed_data_path, f'{data_type}_data'))

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
                processed_data.to_pickle(os.path.join(processed_data_path, f'{data_type}_data'))

                last_date = processed_data['datetime'].iloc[-1].strftime('%d-%m-%Y %H:%M')
                return processed_data
            
            else:
                return data_historical 
            
        else:
            print('Use either "dam" or "rtm" to fetch data.')

    
    def _get_datetime_variables(self, data_type):
        data_historical = pd.read_pickle(os.path.join(processed_data_path, f'{data_type}_data'))

        # Determine the start date based on data type
        if data_type == 'dam':
            start_date = data_historical['datetime'].iloc[-1] + timedelta(days=1)
        elif data_type == 'rtm':
            start_date = data_historical['datetime'].iloc[-1] + timedelta(hours=0.25)

        # Set the end date to 30 days from the start date
        end_date = start_date + timedelta(days=30)

        # Dates in string format
        start_date_str = start_date.strftime("%d-%m-%Y")
        end_date_str = end_date.strftime("%d-%m-%Y")

        return start_date, end_date, start_date_str, end_date_str, data_historical