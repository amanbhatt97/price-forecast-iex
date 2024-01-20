'''
This script is designed for the purpose of training a predictive model using the LightGBM algorithm 

Author: Aman Bhatt
'''
import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import warnings, os
from contextlib import redirect_stdout, redirect_stderr
import logging

# Suppress INFO messages from Optuna
optuna_logger = logging.getLogger('optuna')
optuna_logger.setLevel(logging.WARNING)

from config.paths import *
from src.utils import *

training_logs = configure_logger(LOGS_PATH, 'training.log')

class ModelTraining:

    def __init__(self, PROJECT_PATH):
        """
        Initialize the ModelTraining class.

        Args:
            PROJECT_PATH (str): Path to the project directory.
        """
        self.PROJECT_PATH = PROJECT_PATH

    def _split_data(self, data, training_cutoff, validation_cutoff):
        """
        Split the data into training, validation, and test sets.

        Args:
            data (pd.DataFrame): DataFrame containing the input data.
            training_cutoff (str): Date until which data is used for training.
            validation_cutoff (str): Date until which data is used for validation.

        Returns:
            tuple: Tuple containing training, validation, and test sets.
        """
        try:
            # Training set
            X_train = data[data['datetime'] < training_cutoff].drop('target', axis=1).set_index('datetime')
            y_train = data[data['datetime'] < training_cutoff][['datetime', 'target']].set_index('datetime')

            # Validation set
            X_valid = data[(data['datetime'] >= training_cutoff) & (data['datetime'] < validation_cutoff)].drop('target', axis=1).set_index('datetime')
            y_valid = data[(data['datetime'] >= training_cutoff) & (data['datetime'] < validation_cutoff)][['datetime', 'target']].set_index('datetime')

            # Test set
            X_test = data[data['datetime'] >= validation_cutoff].drop('target', axis=1).set_index('datetime')
            y_test = data[data['datetime'] >= validation_cutoff][['datetime', 'target']].set_index('datetime')

            return X_train, y_train, X_valid, y_valid, X_test, y_test
        except Exception as e:
            print('Error while splitting data: ', str(e))
            training_logs.error('Error while splitting data: %s', str(e))

    def _find_best_features(self, X_train, y_train, X_valid, y_valid, n_features=None):
        """
        Find the best features for training the model.

        Args:
            X_train (pd.DataFrame): Features of the training set.
            y_train (pd.DataFrame): Target variable of the training set.
            X_valid (pd.DataFrame): Features of the validation set.
            y_valid (pd.DataFrame): Target variable of the validation set.
            n_features (int): Number of top features to select.

        Returns:
            list: List of best features.
        """
        try:
            # defining parameters 
            params = {'task': 'train',   # for training
                    'boosting': 'gbdt',   # Gradient Boosting Decision Tree
                    'objective': 'regression',   # doing regression
                    'metric': 'mape',   # root mean squared error
                    'verbose': -1,    # no detailed logging will be displayed 
                    'categorical_feature': ''     # specify categorical features
                    }
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
            model = lgb.train(params, lgb_train, valid_sets=lgb_eval, num_boost_round=1000, early_stopping_rounds=10, verbose_eval=False)
            # creating a dataframe for feature importances
            imp_feat = pd.DataFrame({'features': model.feature_name(), 
                                'importance': model.feature_importance()})

            # sorting feature importances in descending order
            imp_feat = imp_feat.sort_values('importance', ascending=0).reset_index(drop=True)
            if n_features is None:
                best_features = imp_feat[imp_feat['importance'] >= 1]['features'].to_list()
            else:
                best_features = imp_feat['features'].to_list()[:n_features]
            return best_features
        except Exception as e:
            print('Error while finding features: ', str(e))
            training_logs.error('Error while finding features: %s', str(e))


    def _hyperparameter_tuning(self, X_train, y_train, X_valid, y_valid, n_trials, best_features):
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            X_train (pd.DataFrame): Features of the training set.
            y_train (pd.DataFrame): Target variable of the training set.
            X_valid (pd.DataFrame): Features of the validation set.
            y_valid (pd.DataFrame): Target variable of the validation set.
            n_trials (int): Number of hyperparameter tuning trials.
            best_features (list): List of best features.

        Returns:
            dict: Best hyperparameters found during tuning.
        """
        try:
            def objective(trial):
                param = {
                    "objective": "regression",
                    "metric": "mape",  
                    "boosting_type": "gbdt",
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0, 100, step=5),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0, 100, step=5),
                    "num_leaves": trial.suggest_int("num_leaves", 50, 10000, step=50),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
                    "max_bin": trial.suggest_int("max_bin", 200, 300),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0, step=0.1),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.3, 1.0, step=0.1),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                    "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
                    }

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)  
                    with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
                        model = lgb.LGBMRegressor(**param, verbose=-1)
                        model.fit(
                            X_train[best_features], y_train,
                            eval_set=[(X_valid[best_features], y_valid)],
                            early_stopping_rounds=10, eval_metric='mape', verbose=False
                        )
                preds = model.predict(X_valid[best_features])
                error = round(mean_absolute_percentage_error(y_valid, preds) * 100, 2)
            
                return error

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)  
                with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
                    study = optuna.create_study(direction='minimize')
                    study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            return best_params
        except Exception as e:
            print('Error during hyperpameters tuning: ', str(e))
            training_logs.error('Error during hyperpameters tuning: %s', str(e)) 


    def _features_n_params(self, training_data, n_trials, n_features):
        """
        Find the best features and hyperparameters for training the model.

        Args:
            training_data (pd.DataFrame): DataFrame containing the input data.
            n_trials (int): Number of hyperparameter tuning trials.
            n_features (int): Number of top features to select.

        Returns:
            tuple: Tuple containing best features and best hyperparameters.
        """
        # Split the data
        training_upto = training_data.iloc[int(training_data.shape[0]*0.7)]['datetime'].strftime('%Y-%m-%d')      
        validation_upto = training_data.iloc[int(training_data.shape[0]*0.85)]['datetime'].strftime('%Y-%m-%d')        
        X_train, y_train, X_valid, y_valid, _, _ = self._split_data(training_data, training_upto, validation_upto)
        
        best_features = self._find_best_features(X_train, y_train, X_valid, y_valid, n_features)
        best_params = self._hyperparameter_tuning(X_train, y_train, X_valid, y_valid, n_trials, best_features)
        return best_features, best_params


    def _train_model(self, X_train, y_train, best_params, best_features, objective, alpha=None):
        """
        Train the LightGBM model.

        Args:
            X_train (pd.DataFrame): Features of the training set.
            y_train (pd.DataFrame): Target variable of the training set.
            best_params (dict): Best hyperparameters for the model.
            best_features (list): List of best features.
            objective (str): Objective function for the model.
            alpha (float): Regularization parameter.

        Returns:
            lightgbm.LGBMRegressor: Trained LightGBM model.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)  
                with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
                    model = lgb.LGBMRegressor(objective=objective, **best_params, alpha=alpha)
                    model.fit(
                        X_train[best_features], y_train, 
                        verbose=-1
                    )
            return model
        except Exception as e:
            print('Error while training model: ', str(e))
            training_logs.error('Error while training model: %s', str(e)) 
