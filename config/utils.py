import os, time
# import requests
import pandas as pd
from datetime import timedelta
os.environ['TZ'] = 'Asia/Calcutta'
time.tzset()


# Parent directory
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)

# Custom modules
from config.paths import *


def get_datetime_variables(data_type):
    data_historical = pd.read_pickle(os.path.join(data_path, 'processed', f'{data_type}_data'))
    last_date = data_historical['datetime'].iloc[-1].strftime('%d-%m-%Y %H:%M:%S') 

    # Fetch data from last available data upto latest data available
    if data_type == 'dam':
        start_date = data_historical['datetime'].iloc[-1] + timedelta(days=1)
    elif data_type == 'rtm':
        start_date = data_historical['datetime'].iloc[-1] + timedelta(hours=0.25)

    end_date = start_date + timedelta(days=30)
    # Dates in string format
    start_date_str = start_date.strftime("%d-%m-%Y")
    end_date_str = end_date.strftime("%d-%m-%Y")

    return start_date, end_date, start_date_str, end_date_str, data_historical