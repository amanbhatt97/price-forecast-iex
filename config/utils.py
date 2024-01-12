""" 
Date handling for the project.

Author: Aman Bhatt 
"""

import os, time, sys
import pandas as pd
import numpy as np
from datetime import timedelta

# Set the timezone to Asia/Calcutta
os.environ['TZ'] = 'Asia/Calcutta'
time.tzset()

ROOT_PATH = os.getenv('ROOT_PATH')
sys.path.append(ROOT_PATH)

# Custom modules
from config.paths import *


def get_datetime_variables(data_type):
    # Load processed data
    data_historical = pd.read_pickle(os.path.join(data_path, 'processed', f'{data_type}_data'))
    
    # Get the last date in the historical data
    last_date = data_historical['datetime'].iloc[-1].strftime('%d-%m-%Y %H:%M:%S') 

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


def shift_date(df, n):
    """ This method shifts dataframe df by n days downwards"""
    shifted_df = df.copy()
    try:
        shifted_df['datetime'] = shifted_df['datetime'] + pd.DateOffset(days=n)
    except:
        shifted_df['date'] = shifted_df['date'] + pd.DateOffset(days=n) 

    return shifted_df


# capping conditions
def capping(data):
    conditions = [
        (data['datetime'] <= '2022-04-02 23:45:00'),     # Up to 02-04-2022 23:45:00
        (data['datetime'] <= '2023-04-03 23:45:00'),     # From 02-04-2022 23:45:01 to 03-04-2023 23:45:00
        (data['datetime'] > '2023-04-03 23:45:00')       # After 03-04-2023 23:45:00
        # add more cappings if there in future
    ]

    # corresponding capping values
    capping_values = [20000, 12000, 10000]
    data['capping'] = np.select(conditions, capping_values)
    return data