# import pandas as pd
# import numpy as np
# import requests
# import json
# import os, sys
# from datetime import datetime

# PROJECT_PATH = os.getenv('PROJECT_DIR')
# sys.path.append(PROJECT_PATH)

# from src.data_ingestion.iex_data import IexDataFetcher

# class DAMInsertion:
#     def __init__(self):
#         # Access credentials
#         self.base_url = os.getenv('base_url')
#         self.user_email = os.getenv('email')
#         self.user_password = os.getenv('password')
#         self.iex_data = IexDataFetcher()

#     def forecast_dict(self, forecasts, forecasting_date, revision):
#         df = pd.DataFrame()

#         # creating block number
#         l = [i for i in range(1, 25) for _ in range(4)]
#         df['block_no'] = pd.Series(l)

#         # create time_block
#         df['start'] = pd.date_range(start=f'{forecasting_date} 00:00', end=f'{forecasting_date} 23:45', freq='15min').strftime('%H:%M')
#         df['end'] = df['start'].shift(-1).replace(np.nan, '00:00')
#         df['time_block'] = df[['start', 'end']].agg('-'.join, axis=1)

#         # create label
#         if revision == 0:
#             df['label'] = 'forecast'  
#             df['MCP'] = forecasts['forecast']
#         elif revision == 1:
#             df['label'] = 'lower_bound'
#             df['MCP'] = forecasts['lower_bound']
#         else:
#             df['label'] = 'upper_bound'
#             df['MCP'] = forecasts['upper_bound']
            
#         # Drop the 'start' and 'end' columns
#         df.drop(['start', 'end'], axis=1, inplace=True)

#         # Create a dictionary to store the result
#         result_dict = {
#             'date': datetime.strptime(forecasting_date, '%Y-%m-%d').strftime('%d-%m-%Y'),
#             'revision': revision,
#             'data': {'MCP': {}}
#         }

#         # Group the DataFrame by 'block_no'
#         grouped = df.groupby('block_no')

#         # Iterate over each group and create the nested dictionary structure
#         for block_no, group in grouped:
#             nested_dict = [
#                 {
#                     'time_block': row['time_block'],
#                     'price': row['MCP'],
#                     'label': row['label']
#                 }
#                 for _, row in group.iterrows()
#             ]
#             result_dict['data']['MCP'][block_no] = nested_dict

#         return result_dict

#     def save_forecast(self, forecast_data, revision):
#         url = self.base_url + 'savePriceForecast'
#         headers = {'Authorization': 'Bearer ' + str(self.iex_data._get_token()), 'Content-Type': 'application/json'}
#         response = requests.post(url=url, data=json.dumps(forecast_data), headers=headers)
#         print(response.json)
#         print(response.content)
#         if response.json()['status'] == 'success':
#             print(f'DAM forecast data inserted for revision {revision}')
#         else:
#             print(f'DAM forecast data not inserted for revision {revision}')

import pandas as pd
import numpy as np
import requests
import json
import os, sys
from datetime import datetime

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

from src.data_ingestion.iex_data import IexDataFetcher

class DAMInsertion:
    def __init__(self):
        # Access credentials
        self.base_url = os.getenv('base_url')
        self.user_email = os.getenv('email')
        self.user_password = os.getenv('password')
        self.iex_data = IexDataFetcher()

    def forecast_dict(self, forecasts, forecasting_date, forecast_type):
        df = pd.DataFrame()

        # creating block number
        l = [i for i in range(1, 25) for _ in range(4)]
        df['block_no'] = pd.Series(l)

        # create time_block
        df['start'] = pd.date_range(start=f'{forecasting_date} 00:00', end=f'{forecasting_date} 23:45', freq='15min').strftime('%H:%M')
        df['end'] = df['start'].shift(-1).replace(np.nan, '00:00')
        df['time_block'] = df[['start', 'end']].agg('-'.join, axis=1)

        # create label
        df['label'] = forecast_type  
        df['MCP'] = forecasts[forecast_type]
            
        # Drop the 'start' and 'end' columns
        df.drop(['start', 'end'], axis=1, inplace=True)

        # Calculate revision based on forecast_type
        revision = {'forecast': 0, 'lower_bound': 1, 'upper_bound': 2}.get(forecast_type, -1)

        # Create a dictionary to store the result
        result_dict = {
            'date': datetime.strptime(forecasting_date, '%Y-%m-%d').strftime('%d-%m-%Y'),
            'revision': revision,
            'data': {'MCP': {}}
        }

        # Group the DataFrame by 'block_no'
        grouped = df.groupby('block_no')

        # Iterate over each group and create the nested dictionary structure
        for block_no, group in grouped:
            nested_dict = [
                {
                    'time_block': row['time_block'],
                    'price': row['MCP'],
                    'label': row['label']
                }
                for _, row in group.iterrows()
            ]
            result_dict['data']['MCP'][block_no] = nested_dict

        return result_dict

    def save_forecast(self, forecasts, forecasting_date, forecast_type):
        forecast_data = self.forecast_dict(forecasts, forecasting_date, forecast_type)  
        print(forecast_data)
        url = self.base_url + 'savePriceForecast'
        headers = {'Authorization': 'Bearer ' + str(self.iex_data._get_token()), 'Content-Type': 'application/json'}
        response = requests.post(url=url, data=json.dumps(forecast_data), headers=headers)
        if response.json()['status'] == 'success':
            print(f'{forecast_type} data inserted successfully')
        else:
            print(f'{forecast_type} data not inserted')
