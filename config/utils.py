""" 
Date handling for the project.

Author: Aman Bhatt 
"""

import os, time, sys
import pandas as pd
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