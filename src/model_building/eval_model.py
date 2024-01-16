import pandas as pd
import numpy as np
import os, sys
from sklearn.metrics import mean_absolute_percentage_error

PROJECT_PATH = os.getenv('PROJECT_DIR')
sys.path.append(PROJECT_PATH)

from src.feature_engineering.build_features import FeatureEngineering
featured_data = FeatureEngineering(PROJECT_PATH) 


class ModelEvaluator:
    def __init__(self, model, best_features):
        self.model = model
        self.best_features = best_features

    def _process_results(self, predictions_df, market_type):
        results = predictions_df.reset_index()
        results['prediction'] = results['prediction'].apply(lambda x: 10000 if x > 9000 else x)
        if market_type == 'dam':
            results = featured_data.shift_date(results, 1)
        elif market_type == 'rtm':
            results = featured_data.shift_date(results, 2) 
        else:
            print('chose dam or rtm')
        results['date'] = results['datetime'].dt.date
        results['mae'] = np.abs(results['target'] - results['prediction'])
        return results

    def _plot_results(self, results, n):
        results.tail(96 * n).set_index('datetime')[['target', 'prediction']].plot()

    def _calculate_mape(self, results, n):
        mape_per_day = []
        for day in range(0, n):
            target_date = results['date'].max() - pd.Timedelta(days=day)
            day_results = results[results['date'] == target_date]
            daily_mape = mean_absolute_percentage_error(day_results['target'], day_results['prediction'])
            mape_per_day.append(daily_mape)
            print(f'  MAPE for {target_date}: {round(daily_mape * 100, 2)}')

        avg_mape = round(mean_absolute_percentage_error(results['target'], results['prediction']) * 100, 2)
        print(f'  Average MAPE for last {n} days: {avg_mape}')

        return mape_per_day, avg_mape

    def evaluate_on_data(self, X, y, n, market_type):
        
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