"""
Script to find accuracy of directional forecasts.

Author: Aman Bhatt
"""
import time
start_time = time.time()
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta

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
market_type = 'rtm'
accuracy_logs.info('%s accuracy script running.', market_type)
# %%
dam_actual = iex_data._get_processed_data('dam')[['datetime', f'mcp_dam']]
rtm_actual = iex_data._get_processed_data('rtm')[['datetime', f'mcp_rtm']]

# %%
dam_rtm_actual = featured_data.merge_dataframes([rtm_actual, dam_actual])
dam_rtm_actual['actual'] = -1  # Initialize with -1 for cases where mcp_dam is equal to mcp_rtm

# Update 'actual' based on conditions
dam_rtm_actual['actual'] = dam_rtm_actual.apply(lambda row: 1 if row['mcp_dam'] > row['mcp_rtm'] else (0 if row['mcp_dam'] < row['mcp_rtm'] else -1), axis=1)

# %%
dir_actual = dam_rtm_actual[['datetime', 'actual']]

# %%
acc_report = load_pickle(REPORTS_PATH, 'dir_accuracy_report')
acc_start_date = (acc_report['Date'].iloc[-1]\
                + timedelta(days=1)).strftime('%d-%m-%Y')

# %%
try:
    sdt = acc_start_date
    tdt = (datetime.now() + timedelta(days=30)).strftime('%d-%m-%Y')
    forecast = iex_forecast._get_processed_forecast(sdt, tdt, market_type)
except:
    print(f'{market_type} accuracy report already updated.')
    accuracy_logs.info('%s accuracy report already updated.', market_type)
# %%
df = featured_data.merge_dataframes([forecast, dir_actual])

# %%
df['Accuracy'] = (df['actual'] == df['forecast']).astype(int)
df['Date'] = df['datetime'].dt.date

# %%
curr_acc = df.groupby(df['Date'])['Accuracy'].sum().reset_index()

# %%
acc = pd.concat([acc_report, curr_acc], ignore_index = True)

# %%
save_pickle(acc, REPORTS_PATH, 'dir_accuracy_report')
save_excel(acc, REPORTS_PATH, 'dir_accuracy_report')
# %%
print(f'Accuracy report for directional generated.')
accuracy_logs.info('Accuracy report for %s generated.', market_type)
end_time = time.time()
total_time = (end_time - start_time)/60
print(f'Run time: {total_time:.2f} minutes.')
accuracy_logs.info('Run time: %.2f minutes.', total_time)
accuracy_logs.info('**********************************************')
accuracy_logs.info('**********************************************\n')