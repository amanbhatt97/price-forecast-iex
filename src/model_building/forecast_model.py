import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

from src.utils import *
from config.paths import *

class ModelForecaster:
    def __init__(self, models_path):
        """
        Initialize the LightGBMForecaster.

        Args:
            models_path (str): Path to the directory containing model files.
        """
        self.models_path = models_path
        self.best_features, self.model, self.lower, self.upper = self.load_model()

    def load_model(self):
        """
        Load LightGBM model and its best features.

        Returns:
            tuple: Tuple containing the loaded model and best features.
        """
        best_features = load_pickle(self.models_path, 'dam_forecast').booster_.feature_name()
        model = load_pickle(self.models_path, 'dam_forecast')
        lower = load_pickle(self.models_path, 'dam_lower')
        upper = load_pickle(self.models_path, 'dam_upper')
        return best_features, model, lower, upper

    def create_forecast(self, data, forecast_date):
        """
        Create forecast using a LightGBM model.

        Args:
            data (pd.DataFrame): Input DataFrame with necessary features.
            forecast_date (str): Date for which the forecast needs to be created (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: Forecasted values along with lower and upper bounds.
        """
        features = ['datetime'] + self.best_features
        data_for_training = data.reset_index()[features].copy()

        test_cutoff = datetime.strptime(forecast_date, '%Y-%m-%d') - timedelta(days=1)
        X = data_for_training.copy()
        X_test = X[(X['datetime'] >= test_cutoff) & (X['datetime'] < test_cutoff + timedelta(days=1))].iloc[:, 1:].copy()

        pred_test = self.model.predict(X_test)
        lower_pred = self.lower.predict(X_test)
        upper_pred = self.upper.predict(X_test)

        result = pd.DataFrame({
            'forecast': pred_test,
            'lower': lower_pred,
            'upper': upper_pred
        })

        result = np.round(result, 1)
        return result

    @staticmethod
    def forecasting_date(df, market_type):
        """
        Calculate the next date for forecasting based on the last datetime in the DataFrame.

        Args:
            market_type (str): Market type identifier ('dam' or 'rtm').

        Returns:
            str: Forecasting date in 'YYYY-MM-DD' format.
        """
        last_datetime = df['datetime'].max()
        days_to_add = 1 if market_type.lower() == 'dam' else 2
        forecasting_date = (last_datetime + pd.DateOffset(days=days_to_add)).strftime('%Y-%m-%d')
        return forecasting_date