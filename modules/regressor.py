from xgboost import XGBRegressor

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score


class Regressor:

    def __init__(self, type: str='xgb', n_estimators: int=300):
        if type == 'xgb':
            self.regressor = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators)
        else: 
            raise ValueError("Undefined regressor")

    def display_scores(self, scores: np.ndarray):
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())
        
    def eval(self, X: pd.core.frame.DataFrame, Y: pd.core.frame.DataFrame, cv: int=10, parallel: bool=True):

        print('\033[3mEvaluation using Cross-Validation (',cv,')\033[0m')

        n_jobs = -1 if parallel else 1
        
        # RMSE - Cross Validation 
        print('\033[1m-  RMSE:\033[0m')
        scores = cross_val_score(estimator=self.regressor, 
                                 X=X, 
                                 y=Y, 
                                 scoring="neg_mean_squared_error", 
                                 cv=cv,
                                 n_jobs=n_jobs)
        self.display_scores(np.sqrt(-scores))
        
        # R2 - Cross Validation 
        print('\033[1m-  R2:\033[0m')
        scores = cross_val_score(estimator=self.regressor, 
                                 X=X, 
                                 y=Y, 
                                 scoring="r2", 
                                 cv=cv,
                                 n_jobs=n_jobs)
        self.display_scores(scores)
        
        # MAE - Cross Validation 
        print('\033[1m-  MAE:\033[0m')
        scores = cross_val_score(estimator=self.regressor, 
                                 X=X, 
                                 y=Y, 
                                 scoring="neg_mean_absolute_error", 
                                 cv=cv,
                                 n_jobs=n_jobs)
        self.display_scores(-1*scores)
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, test):
        return self.regressor.predict(test)