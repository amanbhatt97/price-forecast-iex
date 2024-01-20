'''
This script evaluates the accuracy of a trained model on a given dataset.

Author: Aman Bhatt
'''
import pandas as pd
import numpy as np
import os, sys
from sklearn.metrics import mean_absolute_percentage_error

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

from config.paths import LOGS_PATH
from src.utils import *

training_logs = configure_logger(LOGS_PATH, 'training.log')

from src.feature_engineering.build_features import FeatureEngineering
featured_data = FeatureEngineering(PROJECT_PATH) 

class ModelEvaluator:
    def __init__(self, model, best_features):
        """
        Initializes the ModelEvaluator object with a trained model and the best features.

        Args:
            model: Trained machine learning model.
            best_features (list): List of feature names considered as the best for the model.
        """
        self.model = model
        self.best_features = best_features

    def _process_results(self, predictions_df, market_type):
        """
        Processes the results of model predictions.

        Args:
            predictions_df (pd.DataFrame): DataFrame with target and prediction columns.
            market_type (str): Type of market data ('dam' or 'rtm').

        Returns:
            pd.DataFrame: Processed results DataFrame.
        """
        try:
            results = predictions_df.reset_index()
            results['prediction'] = results['prediction'].apply(lambda x: 10000 if x > 9000 else x)
            if market_type == 'dam':
                results = featured_data.shift_date(results, 1)
            elif market_type == 'rtm':
                results = featured_data.shift_date(results, 2) 
            else:
                training_logs.warning('Choose dam or rtm.')
            results['date'] = results['datetime'].dt.date
            results['mae'] = np.abs(results['target'] - results['prediction'])
            return results
        except Exception as e:
            print('Error while processing evaluation data: ', str(e))
            training_logs.error('Error while processing evaluation data: %s', str(e))

    def _plot_results(self, results, n):
        """
        Plots the target and prediction results for the last 'n' days.

        Args:
            results (pd.DataFrame): Processed results DataFrame.
            n (int): Number of days to plot.
        """
        results.tail(96 * n).set_index('datetime')[['target', 'prediction']].plot()

    def _calculate_mape(self, results, n):
        """
        Calculates the Mean Absolute Percentage Error (MAPE) for the last 'n' days and average MAPE.

        Args:
            results (pd.DataFrame): Processed results DataFrame.
            n (int): Number of days to calculate MAPE.

        Returns:
            list, float: List of daily MAPEs, Average MAPE.
        """
        try:
            mape_per_day = []
            for day in range(0, n):
                target_date = results['date'].max() - pd.Timedelta(days=day)
                day_results = results[results['date'] == target_date]
                daily_mape = mean_absolute_percentage_error(day_results['target'], day_results['prediction'])
                mape_per_day.append(daily_mape)
                print(f'  MAPE for {target_date}: {round(daily_mape * 100, 2)}')
                training_logs.info('  MAPE for %s: %s', target_date, round(daily_mape * 100, 2))

            avg_mape = round(mean_absolute_percentage_error(results['target'], results['prediction']) * 100, 2)
            print(f'  Average MAPE for the last {n} days: {avg_mape}')
            training_logs.info('  Average MAPE for the last %s days: %s', n, avg_mape) 
            return mape_per_day, avg_mape
        except Exception as e:
            print('Error while calculating MAPE: ', str(e))
            training_logs.error('Error while calculating MAPE: %s', str(e))

    def evaluate_on_data(self, X, y, n, market_type):
        """
        Evaluates the trained model on the provided dataset.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.DataFrame): Target values.
            n (int): Number of days to evaluate.
            market_type (str): Type of market data ('dam' or 'rtm').
        """
        try:
            X = X.tail(96*n)
            y = y.tail(96*n)
            
            # Predictions on the dataset
            predictions = self.model.predict(X[self.best_features])

            # DataFrame with target and predictions for the dataset
            predictions_df = pd.DataFrame()
            predictions_df['target'] = y['target']
            predictions_df['prediction'] = predictions

            # Process and evaluate results
            results = self._process_results(predictions_df, market_type)

            # Plot the results
            self._plot_results(results, n)

            # Calculate and print MAPE
            self._calculate_mape(results, n)
        except Exception as e:
            print('Error while evaluating model: ', str(e))
            training_logs.error('Error while evaluating model: %s', str(e))
