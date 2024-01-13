# %%
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
weather = load_pickle(processed_data_path, 'weather_data')
wind = load_pickle(processed_data_path, 'wind_data')
hydro = load_pickle(processed_data_path, 'hydro_data')
solar = load_pickle(processed_data_path, 'solar_data')

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
training_data = featured_data._get_features(data, weather, data_type = 'dam')

# %%
save_pickle(training_data, processed_data_path, 'training_data')

# %% [markdown]
# ### Model Building

# %%
training_data = load_pickle(processed_data_path, 'training_data')

# %%
# Set the desired cutoff dates
training_upto = training_data.iloc[int(training_data.shape[0]*0.9)]['datetime'].strftime('%Y-%m-%d')      # 80% data for training
validation_upto = training_data.iloc[int(training_data.shape[0]*0.95)]['datetime'].strftime('%Y-%m-%d')        # last day data for testing

# Split the data
X_train, y_train, X_valid, y_valid, X_test, y_test = build_model._split_data(training_data, training_upto, validation_upto)

# %%
# trail and error
n_trials = 20
n_features = 20

# %%
best_features, best_params = build_model._features_n_params(X_train, y_train, X_valid, y_valid, n_trials, n_features)

# %% [markdown]
# ### Model Training

# %%
# training upto this date
training_upto = training_data[::96]['datetime'].iloc[-1].strftime('%Y-%m-%d')
validation_upto = datetime.now().date().strftime('%Y-%m-%d')
X_train, y_train, X_test, y_test, X_valid, y_valid = build_model._split_data(training_data, training_upto, validation_upto)

# %%
model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'regression')

# %%
save_pickle(model, models_path, 'dam_forecast')

# %%
lower_model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'quantile', alpha = 0.1)

# %%
save_pickle(lower_model, models_path, 'dam_lower')

# %%
upper_model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'quantile', alpha = 0.9)

# %%
save_pickle(upper_model, models_path, 'dam_upper')

# %% [markdown]
# ### Model Evaluation

# %%
evaluator = ModelEvaluator(model, best_features)

# %%
evaluator.evaluate_on_data(X_train, y_train, 'train', 5)

# %%
evaluator.evaluate_on_data(X_test, y_test, 'test', 1)

# %%


# %%


# %%



