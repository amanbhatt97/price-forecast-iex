"""
Script to find accuracy of forecasted day-ahead market prices.

Author: Aman Bhatt
"""
import time
start_time = time.time()
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)

os.environ['TZ'] = 'Asia/Calcutta'
time.tzset()

import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

# %%
from src.data_ingestion.iex_data import IexDataFetcher
from src.get_apis.get_forecast import IexForecast
from src.feature_engineering.build_features import FeatureEngineering
from src.utils import *
from config.paths import *

accuracy_logs = configure_logger(LOGS_PATH, 'accuracy.log')
# %%
iex_data = IexDataFetcher()
iex_forecast = IexForecast()
featured_data = FeatureEngineering(PROJECT_PATH)

# %%
market_type = 'dam'

accuracy_logs.info('%s accuracy script running.', market_type)
# %%
actual = iex_data._get_processed_data(f'{market_type}')[['datetime', f'mcp_{market_type}']]

# %%
acc_report = load_pickle(REPORTS_PATH, f'{market_type}_accuracy_report')
acc_start_date = (datetime.strptime(acc_report['Date'].iloc[-1], '%d-%m-%Y')\
                   + timedelta(days=1)).strftime('%d-%m-%Y')

# %%
try:
    sdt = acc_start_date
    tdt = (datetime.now() + timedelta(days=28)).strftime('%d-%m-%Y')
    forecast = iex_forecast._get_processed_forecast(sdt, tdt, market_type)
    df = featured_data.merge_dataframes([forecast, actual])

# %%
    unique_dates = df['datetime'].dt.date.unique()
    acc_data = []

    # %%
    night_hours = [[x for x in range(0, 6*4)] + [y for y in range(23*4, 24*4)]]
    morning_hours = [[x for x in range(6*4, 10*4)]]
    day_hours = [[x for x in range(10*4, 17*4)]]
    evening_hours = [[x for x in range(17*4, 23*4)]]

    hours = morning_hours + day_hours + evening_hours + night_hours

    # %%
    for curr_date in unique_dates:
        curr_date_str = curr_date.strftime("%d-%m-%Y")
        datas = [curr_date.strftime("%d-%m-%Y")]

        filtered_df = df[df['datetime'].dt.date == curr_date]
        filtered_df = filtered_df.reset_index(drop=True)

        MAEs = []
        for x in hours:
            mae = np.round(np.abs(filtered_df.loc[x]['forecast'] - filtered_df.loc[x]['mcp_dam']).mean(), 2)
            MAEs.append(mae)

        # Total MAE for the day
        day_MAE = np.round(np.abs(filtered_df['forecast'] - filtered_df['mcp_dam']).mean(), 2)

        # MAPE of the forecast
        MAPE = np.round((np.abs(filtered_df['forecast'] - filtered_df['mcp_dam']) / filtered_df['mcp_dam']).mean() * 100, 2)

        datas.extend([day_MAE] + MAEs + [MAPE])
        acc_data.append(datas)

    # %%
    curr_acc = pd.DataFrame(data = acc_data, columns = ['Date','MAE','Morning_MAE','Day_MAE',
                                                    'Evening_MAE','Night_MAE','MAPE'])
    curr_acc

    # %%
    acc = pd.concat([acc_report, curr_acc], ignore_index = True)

    # %%
    save_pickle(acc, REPORTS_PATH, f'{market_type}_accuracy_report')
    save_excel(acc, REPORTS_PATH, f'{market_type}_accuracy_report')

    # %%
    print(f'Accuracy report for {market_type} generated.')
    accuracy_logs.info('Accuracy report for %s generated.', market_type)
    end_time = time.time()
    total_time = (end_time - start_time)/60
    print(f'Run time: {total_time:.2f} minutes.')
    accuracy_logs.info('Run time: %.2f minutes.', total_time)
    accuracy_logs.info('**********************************************\n')
    
except:
    print(f'{market_type} accuracy report already updated.')
    accuracy_logs.info('%s accuracy report already updated.', market_type)
# %%
