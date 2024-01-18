# %%
"""
Script to train model for real-time market prices.

Author: Aman Bhatt
"""
import time
start_time = time.time()
import os, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

os.environ['TZ'] = 'Asia/Calcutta'
time.tzset()

from dotenv import load_dotenv
load_dotenv()
PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# visualization
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)

from config.paths import LOGS_PATH
from src.utils import *

training_logs = configure_logger(LOGS_PATH, 'training.log')
# %%
# custom modules
from src.data_ingestion.iex_data import IexDataFetcher
from src.data_ingestion.weather_data import WeatherDataFetcher
from src.feature_engineering.build_features import FeatureEngineering
from src.model_building.train_model import ModelTraining
from src.model_building.eval_model import ModelEvaluator
from src.utils import *
from config.paths import *

# %%
# creating instances
iex_data = IexDataFetcher()
weather_data = WeatherDataFetcher()

featured_data = FeatureEngineering(PROJECT_PATH) 
build_model = ModelTraining(PROJECT_PATH)

# %%
market_type = 'rtm'
n = 5   # number of days for which evaluation is reqd

training_logs.info('%s training script running.', market_type)
# %% [markdown]
# ### Data Ingestion

# %%
# dam = iex_data._get_processed_data('dam')
# rtm = iex_data._get_processed_data('rtm')
# weather = weather_data._get_processed_weather('weather')
# wind = weather_data._get_processed_weather('wind')
# hydro = weather_data._get_processed_weather('hydro')
# solar = weather_data._get_processed_weather('solar')

# %%
dam = load_pickle(PROCESSED_DATA_PATH, 'dam_data')
rtm = load_pickle(PROCESSED_DATA_PATH, 'rtm_data')
weather = load_pickle(PROCESSED_DATA_PATH, 'weather_data')
wind = load_pickle(PROCESSED_DATA_PATH, 'wind_data')
hydro = load_pickle(PROCESSED_DATA_PATH, 'hydro_data')
solar = load_pickle(PROCESSED_DATA_PATH, 'solar_data')
holidays = featured_data.process_holidays(load_pickle(EXTERNAL_DATA_PATH, 'holidays_data'))
print('Data loaded.')
training_logs.info('Data loaded.')
# %%
rtm = rtm[rtm['datetime'] < datetime.now().strftime('%Y-%m-%d')]

# %% [markdown]
# ### Feature Engineering

# %%
dam = featured_data.shift_date(dam, 1) 
weather = featured_data.shift_date(weather, 2)
hydro = featured_data.shift_date(hydro, 2) 
solar = featured_data.shift_date(solar, 2) 
wind = featured_data.shift_date(wind, 2)
holidays = featured_data.shift_date(holidays, -2) 
holidays = holidays.rename(columns = {'holiday': 'holiday_next_day'})

data = featured_data.merge_dataframes([rtm, dam, weather, hydro, solar, wind])

# %%
training_data = featured_data._get_features(data, weather, market_type)

# %%
print('Features created.')
training_logs.info('Features created for %s.', market_type)
# %% [markdown]
# ### Features & Parameters

# %%
# trail and error
n_trials = 50
n_features = 10

# %%
best_features, best_params = build_model._features_n_params(training_data, n_trials, n_features)
print('Best features: ', best_features)
training_logs.info('Best features: %s', best_features)
# %%
print('Best Features and Parameters found.')
training_logs.info('Best Features and Parameters found.')
# %% [markdown]
# ### Model Training & Evaluation

# %%
# training upto this date
training_upto = training_data['datetime'][::96].iloc[-n-1].strftime('%Y-%m-%d')
validation_upto = datetime.now().date().strftime('%Y-%m-%d')
X_train, y_train, X_test, y_test, X_valid, y_valid = build_model._split_data(training_data, training_upto, validation_upto)

# %%
model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'regression')

# %%
if n > X_test[::96].shape[0]:
    n = X_test[::96].shape[0]

# %%
evaluator = ModelEvaluator(model, best_features)
print('Model Evaluation:')
training_logs.info('Model Evaluation:')
evaluator.evaluate_on_data(X_test, y_test, n, market_type)

# %% [markdown]
# ### Final Model

# %%
# training upto this date
training_upto = datetime.now().date().strftime('%Y-%m-%d')
validation_upto = datetime.now().date().strftime('%Y-%m-%d')
X_train, y_train, X_test, y_test, X_valid, y_valid = build_model._split_data(training_data, training_upto, validation_upto)

# %%
model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'regression')
save_pickle(model, MODELS_PATH, f'{market_type}_forecast')
print(f'{market_type}_forecast model saved.')
training_logs.info('%s_forecast model saved.', market_type)
# %%
end_time = time.time()
total_time = (end_time - start_time)/60
print(f'Training time: {total_time:.2f} minutes.')
training_logs.info('Time to train %s model: %.2f minutes.', market_type, total_time)
training_logs.info('**********************************************')
training_logs.info('**********************************************\n')
