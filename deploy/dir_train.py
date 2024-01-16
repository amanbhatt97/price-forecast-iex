# %%
import os, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
market_type = 'rtm'

# %% [markdown]
# ### Data Ingestion

# %%
dam = iex_data._get_processed_data('dam')
rtm = iex_data._get_processed_data('rtm')
# weather = weather_data._get_processed_weather('weather')
# wind = weather_data._get_processed_weather('wind')
# hydro = weather_data._get_processed_weather('hydro')
# solar = weather_data._get_processed_weather('solar')

# %%
weather = load_pickle(PROCESSED_DATA_PATH, 'weather_data')
wind = load_pickle(PROCESSED_DATA_PATH, 'wind_data')
hydro = load_pickle(PROCESSED_DATA_PATH, 'hydro_data')
solar = load_pickle(PROCESSED_DATA_PATH, 'solar_data')

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

data = featured_data.merge_dataframes([rtm, dam, weather, hydro, solar, wind])

# %%
training_data = featured_data._get_features(data, weather, market_type)

# %%
save_pickle(training_data, PROCESSED_DATA_PATH, f'{market_type}_training_data')

# %%
training_data

# %% [markdown]
# ### Model Building

# %%
training_data = load_pickle(PROCESSED_DATA_PATH, f'{market_type}_training_data')

# %%
# trail and error
n_trials = 50
n_features = 25

# %%
best_features, best_params = build_model._features_n_params(training_data, n_trials, n_features)

# %% [markdown]
# ### Model Training & Evaluation

# %%
n = 5   # number of days for which evaluation is reqd

# %%
# training upto this date
training_upto = training_data['datetime'][::96].iloc[-n].strftime('%Y-%m-%d')
validation_upto = datetime.now().date().strftime('%Y-%m-%d')
X_train, y_train, X_test, y_test, X_valid, y_valid = build_model._split_data(training_data, training_upto, validation_upto)

# %%
model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'regression')

# %%
if n > X_test[::96].shape[0]:
    n = X_test[::96].shape[0]

# %%
evaluator = ModelEvaluator(model, best_features)
evaluator.evaluate_on_data(X_test, y_test, n, market_type)

# %% [markdown]
# ### Final Model

# %%
# training upto this date
training_upto = datetime.now().date().strftime('%Y-%m-%d')
validation_upto = datetime.now().date().strftime('%Y-%m-%d')
X_train, y_train, X_test, y_test, X_valid, y_valid = build_model._split_data(training_data, training_upto, validation_upto)

# %%
# training upto this date
training_upto = datetime.now().date().strftime('%Y-%m-%d')
validation_upto = datetime.now().date().strftime('%Y-%m-%d')
X_train, y_train, X_test, y_test, X_valid, y_valid = build_model._split_data(training_data, training_upto, validation_upto)

# %%
model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'regression')
save_pickle(model, MODELS_PATH, f'{market_type}_forecast')


