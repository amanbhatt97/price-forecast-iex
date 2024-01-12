# %%
# dependencies
import os, sys
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.getenv('ROOT_PATH')
sys.path.append(ROOT_PATH)

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# visualization
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)

from sklearn.metrics import r2_score, mean_absolute_percentage_error

# %%
# custom modules
from src.data_ingestion.iex_data import IexDataFetcher
from src.data_ingestion.weather_data import WeatherDataFetcher
from src.feature_engineering.build_features import FeatureEngineering
from src.model_building.train_model import ModelTraining
from src.model_building.eval_model import ModelEvaluator

from config.paths import *
from config.utils import shift_date, merge_dataframes

# %%
# creating instances
iex_data = IexDataFetcher()
weather_data = WeatherDataFetcher()

featured_data = FeatureEngineering(ROOT_PATH) 
build_model = ModelTraining(ROOT_PATH)

# %% [markdown]
# ### Data Ingestion

# %%
dam = iex_data._get_processed_data('dam')
rtm = iex_data._get_processed_data('rtm')
weather = weather_data._get_processed_weather('weather')
wind = weather_data._get_processed_weather('wind')
hydro = weather_data._get_processed_weather('hydro')
solar = weather_data._get_processed_weather('solar')

# %% [markdown]
# ### Feature Engineering

# %%
rtm = shift_date(rtm, 1) 
weather = shift_date(weather, -1)
hydro = shift_date(hydro, -1) 
solar = shift_date(solar, -1) 
wind = shift_date(wind, -1)

data = merge_dataframes([dam, rtm, weather, hydro, solar, wind])

# %%
data_for_training = featured_data._get_features(data, weather, data_type = 'dam')
featured_data._save_features(data_for_training, os.path.join(processed_data_path, 'training_data'))

# %% [markdown]
# ### Model Building

# %%
data = pd.read_pickle(os.path.join(processed_data_path, 'training_data'))

# %%
# Set the desired cutoff dates
training_upto = data.iloc[int(data.shape[0]*0.7)]['datetime'].strftime('%Y-%m-%d')      # 70% data for training
validation_upto = data.iloc[-96]['datetime'].strftime('%Y-%m-%d')        # last day data for testing

# Split the data
X_train, y_train, X_valid, y_valid, X_test, y_test = build_model._split_data(data, training_upto, validation_upto)

# %%
# trail and error
n_trials = 50
n_features = 10

# %%
best_features, best_params = build_model._features_n_params(X_train, y_train, X_valid, y_valid, n_trials, n_features)

# %% [markdown]
# ### Model Training

# %%
# training upto this date
training_upto = data[::96]['datetime'].iloc[-1].strftime('%Y-%m-%d')
validation_upto = datetime.now().date().strftime('%Y-%m-%d')
X_train, y_train, X_test, y_test, X_valid, y_valid = build_model._split_data(data, training_upto, validation_upto)

# %%
model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'regression', model_type = 'dam_forecast')

# %%
lower_model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'quantile', model_type = 'dam_lower', alpha = 0.1)

# %%
upper_model = build_model._train_model(X_train, y_train, best_params, best_features, objective = 'quantile', model_type = 'dam_upper', alpha = 0.9)

# %% [markdown]
# ### Model Evaluation

# %%
evaluator = ModelEvaluator(model, best_features)

# %%
evaluator.evaluate_on_data(X_train, y_train, 'train', 5)

# %%
evaluator.evaluate_on_data(X_test, y_test, 'test', 5)

# %%


# %%


# %%



