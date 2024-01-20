'''
This script is used to perform forecasting using a trained model. 

Author: Aman Bhatt
'''
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

from src.utils import *
from config.paths import *

forecasting_logs = configure_logger(LOGS_PATH, 'forecasting.log')

class ModelForecaster:
    def __init__(self, models_path, market_type):
        """
        Initialize the LightGBMForecaster.

        Args:
            models_path (str): Path to the directory containing model files.
        """
        self.models_path = models_path
        if market_type == 'dam': 
            self.best_features, self.model, self.lower, self.upper = self.load_model(market_type)
        else:
            self.best_features, self.model = self.load_model(market_type) 

    def load_model(self, market_type):
        """
        Load LightGBM model and its best features.

        Returns:
            tuple: Tuple containing the loaded model and best features.
        """
        try:
            best_features = load_pickle(self.models_path, f'{market_type}_forecast').booster_.feature_name()
            if market_type == 'dam':
                model = load_pickle(self.models_path, f'{market_type}_forecast')
                lower = load_pickle(self.models_path, f'{market_type}_lower')
                upper = load_pickle(self.models_path, f'{market_type}_upper')
                return best_features, model, lower, upper
            else:
                model = load_pickle(self.models_path, f'{market_type}_forecast')
                return best_features, model 
        except Exception as e:
            print('Error while loading model: ', str(e))
            forecasting_logs.error('Error while loading model: %s', str(e))

    def create_forecast(self, data, forecast_date, market_type):
        """
        Create forecast using a LightGBM model.

        Args:
            data (pd.DataFrame): Input DataFrame with necessary features.
            forecast_date (str): Date for which the forecast needs to be created (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: Forecasted values along with lower and upper bounds.
        """
        try:
            features = ['datetime'] + self.best_features
            data_for_training = data.reset_index()[features].copy()

            if market_type == 'dam':
                n = 1
            elif market_type == 'rtm':
                n = 2
            else:
                print('chose either dam or rtm')
            test_cutoff = datetime.strptime(forecast_date, '%Y-%m-%d') - timedelta(days=n)
            X = data_for_training.copy()
            X_test = X[(X['datetime'] >= test_cutoff) & (X['datetime'] < test_cutoff + timedelta(days=1))].iloc[:, 1:].copy()
            pred_test = self.model.predict(X_test)

            if market_type == 'dam':
                lower_pred = self.lower.predict(X_test)
                upper_pred = self.upper.predict(X_test)
                result = pd.DataFrame({
                f'{market_type}_forecast': pred_test,
                'lower_bound': lower_pred,
                'upper_bound': upper_pred
                }) 
                result = self._create_daterange(forecast_date, result)
                result = self.modify_forecast(result, market_type) 
                result = np.round(result, 1)
                
                save_pickle(result, DAM_FORECAST_PATH, f'{market_type}_forecast_{forecast_date}')
                save_excel(result, DAM_FORECAST_PATH, f'{market_type}_forecast_{forecast_date}')
            elif market_type == 'rtm':
                pred_test = self.model.predict(X_test) 
                result = pd.DataFrame({
                f'{market_type}_forecast': pred_test,
                })

                result = self._create_daterange(forecast_date, result)
                result = self.modify_forecast(result, market_type) 
                result = np.round(result, 1)
                
                save_pickle(result, DIR_FORECAST_PATH, f'{market_type}_forecast_{forecast_date}')

            return result
        except Exception as e:
            print('Error while creating forecast: ', str(e))
            forecasting_logs.error('Error while creating forecast: %s', str(e))

    def forecasting_date(self, df, market_type):
        """
        Calculate the next date for forecasting based on the last datetime in the DataFrame.

        Args:
            market_type (str): Market type identifier ('dam' or 'rtm').

        Returns:
            str: Forecasting date in 'YYYY-MM-DD' format.
        """
        try:
            last_datetime = df['datetime'].max()
            days_to_add = 1 if market_type.lower() == 'dam' else 2
            forecasting_date = (last_datetime + pd.DateOffset(days=days_to_add)).strftime('%Y-%m-%d')
            return forecasting_date
        except Exception as e:
            print('Error while creating forecast date: ', str(e))
            forecasting_logs.error('Error while creating forecast date: %s', str(e)) 
    
    def modify_forecast(self, forecasts, market_type):
        """
        Modify forecast values and bounds.

        Args:
            forecast_df (pd.DataFrame): DataFrame with forecast values and bounds.
            forecast_date (str): Date for which the forecast needs to be modified (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: Modified forecast values along with lower and upper bounds.
        """
        try:
            if market_type == 'dam':
                # masking forecast values 
                forecasts[f'{market_type}_forecast'] = forecasts[f'{market_type}_forecast'].apply(lambda x: 10000 if x > 9000 else x)

                # making lower bound < forecast < upper_bound
                forecasts['lower_bound'] = forecasts.apply(lambda row: min(row['lower_bound'], row[f'{market_type}_forecast']), axis=1)
                forecasts['upper_bound'] = forecasts.apply(lambda row: max(row['upper_bound'], row[f'{market_type}_forecast']), axis=1)

                # masking upper bound values above 8500 to 10000
                forecasts['upper_bound'] = forecasts['upper_bound'].apply(lambda x: 10000 if x > 8500 else x)

                forecasts = forecasts[['datetime', f'{market_type}_forecast', 'lower_bound', 'upper_bound']]
            
            elif market_type == 'rtm':
                # masking forecast values
                forecasts[f'{market_type}_forecast'] = forecasts[f'{market_type}_forecast'].apply(lambda x: 10000 if x > 9000 else x)

            forecasts = forecasts.round(2)

            forecasts.set_index('datetime').plot()
            return forecasts
        except Exception as e:
            print('Error while modifying forecast values: ', str(e))
            forecasting_logs.error('Error while modifying forecast values: %s', str(e)) 

    def _create_daterange(self, forecast_date, forecast):
        """
        Create datetime for forecasted values.

        Args:
            forecast (pd.DataFrame): DataFrame with forecast values.
            forecast_date (str): Date for which the forecast needs to be modified (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: Modified forecast values along with lower and upper bounds.
        """
        try:
            q = pd.DataFrame(pd.date_range(start = forecast_date, periods = 96, freq = '15min'), columns = ['datetime'])
            forecast = pd.concat([q,forecast],axis = 1)
            return forecast
        except Exception as e:
            print('Error while creating daterange: ', str(e))
            forecasting_logs.error('Error while creating daterange: %s', str(e))