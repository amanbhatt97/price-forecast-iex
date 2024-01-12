""" 
Date handling for the project.

Author: Aman Bhatt 
"""

import os, time, sys
import pandas as pd
import numpy as np
from functools import reduce
from datetime import timedelta

# Set the timezone to Asia/Calcutta
os.environ['TZ'] = 'Asia/Calcutta'
time.tzset()

ROOT_PATH = os.getenv('ROOT_PATH')
sys.path.append(ROOT_PATH)

# Custom modules
from config.paths import *


def shift_date(df, n):
    """ This method shifts dataframe df by n days downwards"""
    shifted_df = df.copy()
    try:
        shifted_df['datetime'] = shifted_df['datetime'] + pd.DateOffset(days=n)
    except:
        shifted_df['date'] = shifted_df['date'] + pd.DateOffset(days=n) 

    return shifted_df


def merge_dataframes(dfs, on_column='datetime'):
    """This method merges all the dataframes in a list dfs"""
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=on_column, how='left'), dfs)
    merged_df.dropna(inplace=True)
    return merged_df