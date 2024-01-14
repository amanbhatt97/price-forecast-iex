'''
Author: Aman Bhatt
'''

# Import necessary libraries and modules
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

# Custom Modules
from src.data_ingestion.iex_data import IexDataFetcher
from src.data_ingestion.weather_data import WeatherDataFetcher
from src.feature_engineering.build_features import FeatureEngineering
from src.model_building.train_model import ModelTraining
from src.model_building.eval_model import ModelEvaluator
from src.utils import *
from config.paths import *

# Load environment variables
load_dotenv()

# Set project path and add it to sys.path
PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set visualization configurations
plt.rcParams['figure.figsize'] = (12, 6)

# Instantiate data fetchers, feature engineering, and model building
iex_data = IexDataFetcher()
weather_data = WeatherDataFetcher()
featured_data = FeatureEngineering(PROJECT_PATH)
build_model = ModelTraining(PROJECT_PATH)

# Fetch and load processed data
dam = iex_data._get_processed_data('dam')
rtm = iex_data._get_processed_data('rtm')
weather = load_pickle(PROCESSED_DATA_PATH, 'weather_data')
wind = load_pickle(PROCESSED_DATA_PATH, 'wind_data')
hydro = load_pickle(PROCESSED_DATA_PATH, 'hydro_data')
solar = load_pickle(PROCESSED_DATA_PATH, 'solar_data')

# Shift date for temporal alignment
rtm = featured_data.shift_date(rtm, 1)
weather = featured_data.shift_date(weather, -1)
hydro = featured_data.shift_date(hydro, -1)
solar = featured_data.shift_date(solar, -1)
wind = featured_data.shift_date(wind, -1)

# Merge dataframes
data = featured_data.merge_dataframes([dam, rtm, weather, hydro, solar, wind])

# Get features for training data
training_data = featured_data._get_features(data, weather, data_type='dam')
save_pickle(training_data, PROCESSED_DATA_PATH, 'training_data')

# Load training data
training_data = load_pickle(PROCESSED_DATA_PATH, 'training_data')

# Set desired cutoff dates for training and validation
training_upto = training_data.iloc[int(training_data.shape[0] * 0.9)]['datetime'].strftime('%Y-%m-%d')  # 90% data for training
validation_upto = training_data.iloc[int(training_data.shape[0] * 0.95)]['datetime'].strftime('%Y-%m-%d')  # 5% data for validation

# Split the data
X_train, y_train, X_valid, y_valid, X_test, y_test = build_model._split_data(training_data, training_upto, validation_upto)

# Hyperparameter tuning
n_trials = 50
n_features = 100
best_features, best_params = build_model._features_n_params(X_train, y_train, X_valid, y_valid, n_trials, n_features)

# Set training and validation cutoff dates for final model training
training_upto = training_data[::96]['datetime'].iloc[-1].strftime('%Y-%m-%d')
validation_upto = datetime.now().date().strftime('%Y-%m-%d')
X_train, y_train, X_test, y_test, X_valid, y_valid = build_model._split_data(training_data, training_upto, validation_upto)

# Train the main model and save
model = build_model._train_model(X_train, y_train, best_params, best_features, objective='regression')
save_pickle(model, MODELS_PATH, 'dam_forecast')

# Train lower quantile model and save
lower_model = build_model._train_model(X_train, y_train, best_params, best_features, objective='quantile', alpha=0.1)
save_pickle(lower_model, MODELS_PATH, 'dam_lower')

# Train upper quantile model and save
upper_model = build_model._train_model(X_train, y_train, best_params, best_features, objective='quantile', alpha=0.9)
save_pickle(upper_model, MODELS_PATH, 'dam_upper')

# Evaluate the model on training and test data
evaluator = ModelEvaluator(model, best_features)
evaluator.evaluate_on_data(X_train, y_train, 'train', 5)
evaluator.evaluate_on_data(X_test, y_test, 'test', 1)