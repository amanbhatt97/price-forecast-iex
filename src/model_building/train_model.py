import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


class ModelTraining:

    def __init__(self, PROJECT_PATH):
        self.PROJECT_PATH = PROJECT_PATH

    def _split_data(self, data, training_cutoff, validation_cutoff):
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


    def _find_best_features(self, X_train, y_train, X_valid, y_valid, n_features=None):
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
        imp_feat = imp_feat.sort_values('importance', ascending = 0).reset_index(drop = True)
        if n_features == None:
            best_features = imp_feat[imp_feat['importance'] >= 1]['features'].to_list()
        else:
            best_features = imp_feat['features'].to_list()[:n_features]
        return best_features


    def _hyperparameter_tuning(self, X_train, y_train, X_valid, y_valid, n_trials, best_features):
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
        
            model = lgb.LGBMRegressor(**param, verbose=-1)
            model.fit(
                X_train[best_features], y_train,
                eval_set=[(X_valid[best_features], y_valid)],
                early_stopping_rounds=10, eval_metric='mape', verbose=False
            )
            preds = model.predict(X_valid[best_features])
            error = round(mean_absolute_percentage_error(y_valid, preds) * 100, 2)
        
            return error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print('\n\nBest MAPE achieved: ', study.best_trial.value) 
        return best_params


    def _features_n_params(self, training_data, n_trials, n_features):
        # Split the data
        training_upto = training_data.iloc[int(training_data.shape[0]*0.7)]['datetime'].strftime('%Y-%m-%d')      
        validation_upto = training_data.iloc[int(training_data.shape[0]*0.85)]['datetime'].strftime('%Y-%m-%d')        
        X_train, y_train, X_valid, y_valid, X_test, y_test = self._split_data(training_data, training_upto, validation_upto)
        
        best_features = self._find_best_features(X_train, y_train, X_valid, y_valid, n_features)
        best_params = self._hyperparameter_tuning(X_train, y_train, X_valid, y_valid, n_trials, best_features)
        return best_features, best_params


    def _train_model(self, X_train, y_train, best_params, best_features, objective, alpha = None):
        model = lgb.LGBMRegressor(objective = objective, **best_params, alpha = alpha)
        model.fit(X_train[best_features], y_train, 
                verbose = -1
                )
        return model