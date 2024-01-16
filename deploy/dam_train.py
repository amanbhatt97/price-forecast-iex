# %%
import time
start_time = time.time()
import os, sys
import pandas as pd
import numpy as np
from datetime import datetime
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
market_type = 'dam'
n = 5   # number of days for which evaluation is reqd

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
print('Data loaded.')

# %% [markdown]
# ### Feature Engineering

# %%
rtm = featured_data.shift_date(rtm, 1) 
weather = featured_data.shift_date(weather, -1)
hydro = featured_data.shift_date(hydro, -1) 
solar = featured_data.shift_date(solar, -1) 
wind = featured_data.shift_date(wind, -1)

data = featured_data.merge_dataframes([dam, rtm, weather, hydro, solar, wind])

# %%
training_data = featured_data._get_features(data, weather, market_type)

# %%
print('Features created.')

# %% [markdown]
# ### Features & Parameters

# %%
# trail and error
n_trials = 50
n_features = 10

# %%
best_features, best_params = build_model._features_n_params(training_data, n_trials, n_features)

# %%
print('Best Features and Parameters found.')

# %% [markdown]
# ### Model Evaluation

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

# %%
lower_model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'quantile', alpha = 0.1)
save_pickle(lower_model, MODELS_PATH, f'{market_type}_lower')
print(f'{market_type}_lower model saved.')

# %%
upper_model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'quantile', alpha = 0.9)
save_pickle(upper_model, MODELS_PATH, f'{market_type}_upper')
print(f'{market_type}_upper model saved.')

# %%
end_time = time.time()
total_time = (end_time - start_time)/60
print(f'Training time: {total_time:.2f} minutes.')