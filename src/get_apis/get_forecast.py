# %%
'''
This script fetches forecasted market data from the IEX API, processes it in order to find accuracy.

Author: Aman Bhatt
'''
import os
import pandas as pd
import sys
import requests

from dotenv import load_dotenv
load_dotenv()

# Parent directory
PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

# %%
from src.data_ingestion.iex_data import IexDataFetcher
from src.utils import *
from config.paths import *

accuracy_logs = configure_logger(LOGS_PATH, 'accuracy.log')
# %%
iex_data = IexDataFetcher()

# %%
class IexForecast:
    def __init__(self):
        # Access credentials
        self.base_url = os.getenv('base_url')
    
    def _get_forecast_json(self, start_date_str, end_date_str, market_type):
        """
        Fetches forecast data from the IEX API based on specified parameters.
        
        Args:
            start_date_str (str): Start date string.
            end_date_str (str): End date string.
            market_type (str): Type of market data ('dam' or 'rtm').
            
        Returns:
            dict: JSON response from the API.
        """
        try:
            token = iex_data._get_token()
            endpoint = 'getPriceForecast' if market_type == 'dam' else 'getRTMPriceForecast'
            url = self.base_url + endpoint
            headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
            params = {"start_date": start_date_str, "end_date": end_date_str}
            r = requests.get(url=url, headers=headers, params=params)
            return r.json()
        except Exception as e:
            print(f"Error occurred while retrieving {market_type} json data:", str(e))
            accuracy_logs.error('Error occurred while retrieving %s json data: %s', market_type, str(e))

    def _get_forecast_dict(self, start_date_str, end_date_str, market_type):
        """
        Extracts the forecast data from the JSON response.
        
        Args:
            start_date_str (str): Start date string.
            end_date_str (str): End date string.
            market_type (str): Type of market data ('dam' or 'rtm').
            
        Returns:
            dict: Forecast data in dictionary format.
        """
        try:
            forecast_dict = self._get_forecast_json(start_date_str, end_date_str, market_type)['data']
            return forecast_dict
        except Exception as e:
            print(f"Error occurred while retrieving {market_type} dictionary:", str(e))
            accuracy_logs.error('Error occurred while retrieving %s dictionary: %s', market_type, str(e))
    
    def _get_forecast_df(self, start_date_str, end_date_str, market_type):
        """
        Converts the forecast dictionary to a Pandas DataFrame.
        
        Args:
            start_date_str (str): Start date string.
            end_date_str (str): End date string.
            market_type (str): Type of market data ('dam' or 'rtm').
            
        Returns:
            pd.DataFrame: Processed forecast data.
        """
        try:
            forecast_df = pd.DataFrame(self._get_forecast_dict(start_date_str, end_date_str, market_type))
            if market_type == 'dam':
                forecast_df = forecast_df[['date', 'time_block', 'price', 'label']]
            else:
                forecast_df = forecast_df.sort_values(by=['date', 'time_block', 'revision'], \
                                        ascending=[True, True, False])\
                                            .drop_duplicates(subset=['date', 'time_block'], keep='first')
                forecast_df = forecast_df[['date', 'time_block', 'price']]
            return forecast_df
        except Exception as e:
            print(f"Error occurred while getting {market_type} dataframe:", str(e))
            accuracy_logs.error('Error occurred while getting %s dataframe: %s', market_type, str(e)) 
    
    def _get_processed_forecast(self, start_date_str, end_date_str, market_type):
        """
        Processes and cleans the forecast data, returning a DataFrame with relevant columns.
        
        Args:
            start_date_str (str): Start date string.
            end_date_str (str): End date string.
            market_type (str): Type of market data ('dam' or 'rtm').
            
        Returns:
            pd.DataFrame: Processed forecast data with datetime and price columns.
        """
        try:
            df = self._get_forecast_df(start_date_str, end_date_str, market_type)
            if market_type == 'dam':
                df = df[df['label'] == 'forecast']
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time_block'].str.split('-').str[0])
            df = df[['datetime', 'price']]
            df = df.rename(columns={'price': 'forecast'})
            print(f'{market_type} forecast updated up to: ', df['datetime'].max())
            accuracy_logs.info('%s forecast updated up to: %s.', market_type, df['datetime'].max())
            return df
        except Exception as e:
            print(f'{market_type} for selected dates not found.')
            accuracy_logs.warning('%s for selected dates not found: %s', market_type, str(e))

