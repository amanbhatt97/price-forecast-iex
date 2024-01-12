import pandas as pd
import numpy as np
import os, sys
from sklearn.metrics import mean_absolute_percentage_error

ROOT_PATH = os.getenv('ROOT_PATH')
sys.path.append(ROOT_PATH)

from config.utils import shift_date

class ModelEvaluator:
    def __init__(self, model, best_features):
        self.model = model
        self.best_features = best_features

    def _process_results(self, predictions_df):
        results = predictions_df.reset_index()
        results['prediction'] = results['prediction'].apply(lambda x: 10000 if x > 9000 else x)
        results = shift_date(results, 1)
        results['date'] = results['datetime'].dt.date
        results['mae'] = np.abs(results['target'] - results['prediction'])
        return results

    def _plot_results(self, results, n):
        results.tail(96 * n).set_index('datetime')[['target', 'prediction']].plot()

    def _calculate_mape(self, results):
        mape = round(mean_absolute_percentage_error(results['target'], results['prediction']), 4) * 100
        print(f'MAPE: {mape}')

    def evaluate_on_data(self, X, y, dataset_name, n):
        # Predictions on the dataset
        predictions = self.model.predict(X[self.best_features])

        # DataFrame with target and predictions for the dataset
        predictions_df = pd.DataFrame()
        predictions_df['target'] = y['target']
        predictions_df['prediction'] = predictions

        # Process and evaluate results
        results = self._process_results(predictions_df)

        # Plot the results
        print(f'Evaluating on {dataset_name} data:')
        self._plot_results(results, n)

        # Calculate and print MAPE
        self._calculate_mape(results)