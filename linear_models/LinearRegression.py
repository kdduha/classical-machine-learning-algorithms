import pandas as pd
import numpy as np
from typing import Optional, Union
import random


class MyLineReg():
    """
    LinearRegression
    
    Parametres:
    n_iter = the number of iterations during training
    learning_rate = the step of training
    reg = the type of the gradiant regularization (None, l1, l2, elasticnet)
    l1_coef = the coef for the Lasso (l1) and ElasticNet
    l2_coef = the coef for the Rigde (l2) and ElasticNet
    sdg_sample = the number of rows in batches for the stochastic gradient
    random_state = the seed for the stochastic gradient
    """
    
    def __init__(self, n_iter: int = 100,
                 learning_rate: Union[int, float] = 0.1, 
                 metric: Optional[int] = None, 
                 reg: Optional[int] = None, 
                 l1_coef: Union[int, float] = 0, 
                 l2_coef: Union[int, float] = 0, 
                 sgd_sample: Union[int, float, None] = None, 
                 random_state: int = 42):
        self.__n_iter = n_iter
        self.__learning_rate = learning_rate
        self.__metric, self.__metric_func = self.__check_metrics(metric)
        self.__reg = reg
        self.__l1, self.__l2 = l1_coef, l2_coef
        self.__sgd_sample = sgd_sample
        self.__random_state = random_state
        self.__weights = None
        self.X, self.y = None, None
    
    # training
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False) -> np.ndarray:
        # random seed for the stochastic gradient
        random.seed(self.__random_state)
        
        # copies for the func 'get_best_score'
        self.X, self.y = X, y
        
        # adding ones to X and W for the w_0
        objects_number, features_number = X.shape
        
        self.__weights = np.ones(features_number + 1)
        X, y = np.c_[X.values, np.ones(objects_number)], y.values
        
        # iter steps
        for i in range(self.__n_iter):

            # defining batches for the stochastic gradient
            X_batch, y_batch = self.__train_batches(X, y, self.__sgd_sample)
            
            # new iter prediction
            pred_y = X_batch @ self.__weights
            
            # gradient based on the type of regularization
            grad = self.__gradient(X_batch, y_batch, pred_y, self.__reg, self.__l1, self.__l2)
            
            # minimization step (could be dynamic)
            k = self.__learning_rate if type(self.__learning_rate) in (int, float) else self.__learning_rate(i+1)
            
            # updating weights
            self.__weights -= k * grad
            
            # logging loss func each 'verbose' step
            self.__logging_loss(i, verbose, y, pred_y)
    
    # getting coefs without w_0
    def get_coef(self) -> np.ndarray:
        return self.__weights[:-1]
    
    # getting prediction
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # adding ones in the end of X matrix for the w_0
        X = np.c_[X.values, np.ones(X.shape[0])]
        return X @ self.__weights
    
    # getting the score by the chosen metric on the last iter step
    def get_best_score(self):
        # if X and Y are not None
        return self.__metric_func(self.y, self.predict(self.X))
    
    # computing MAE
    def _mae(self,  y: np.ndarray, pred_y: np.ndarray) -> float:
        y = self.__check_y_dimension(y)
        return np.mean(abs(y - pred_y))
    
    # computing MSE
    def _mse(self, y: np.ndarray, pred_y: np.ndarray) -> float:
        y = self.__check_y_dimension(y)
        return np.mean((y - pred_y)**2)
    
    # computing RMSE
    def _rmse(self, y: np.ndarray, pred_y: np.ndarray) -> float:
        y = self.__check_y_dimension(y)
        return np.mean((y - pred_y)**2)**0.5
    
    # computing MAPE
    def _mape(self, y: np.ndarray, pred_y: np.ndarray) -> float:
        y = self.__check_y_dimension(y)
        return 100 * np.mean(abs((y - pred_y) / y))
    
    # computing R2
    def _r2(self, y: np.ndarray, pred_y: np.ndarray) -> float: 
        y = self.__check_y_dimension(y)    
        return 1 - (np.mean((y - pred_y)**2))/(np.mean((y - np.mean(y))**2))

    # defining batches for the stochastic gradient
    def __train_batches(self, X: np.ndarray, y: np.ndarray, sgd: Union[int, float, None]) -> tuple:
            if isinstance(sgd, float):
                s = round(X.shape[0] * sgd)
                sample_rows_idx = random.sample(range(X.shape[0]), s)
                X_batch, y_batch = X[sample_rows_idx, :], y[sample_rows_idx]
            elif isinstance(sgd, int):
                sample_rows_idx = random.sample(range(X.shape[0]), sgd)
                X_batch, y_batch = X[sample_rows_idx, :], y[sample_rows_idx]
            else:
                X_batch, y_batch = X, y
            return X_batch, y_batch
    
    # logging loss func on the chosen iter step
    def __logging_loss(self, i: int, verbose: int, y: np.ndarray, pred_y: np.ndarray) -> None:
        # if this is a necessary step
        if verbose and (i+1) % verbose == 0:
                error = self.__metric_func(y, pred_y)
                print(f"{i if not i else 'start'} | {self.__metric} loss: {error}")
    
    # computing gradient
    def __gradient(self, X: np.ndarray, y: np.ndarray, pred_y: np.ndarray, 
                   reg: Optional[int], l1: Union[int, float], l2: Union[int, float]) -> np.ndarray:
         
        # checking regularization type
        reg_types = (None, 'l1', 'l2', 'elasticnet')
        if reg not in reg_types:
            return f'There is no such regularization. You can choose from {reg_types}'
        
        # computing deffault gradient
        grad = 2/X.shape[0] * (pred_y - y.T) @ X
        if reg is reg_types[0]:
            return grad
        # lasso MAE
        elif reg == reg_types[1]:
            return grad + l1 * np.sign(self.__weights)
        # ridge MAE
        elif reg == reg_types[2]:
            return grad + l2 * 2 * self.__weights
        # elasticnet
        return grad + l1 * np.sign(self.__weights) + l2 * 2 * self.__weights
    
    # checking given metrics and returning the metric function
    def __check_metrics(self, metric: Optional[str]):
        # MAE by deffault
        if metric is None:
            return 'mae', self._mae
        metrics = {'mae': self._mae, 'mse': self._mse, 'rmse': self._rmse, 
                        'mape': self._mape, 'r2': self._r2}
        if metric in metrics:
            return metric, dict_metrics[metric]
        return 'mae', self._mae
    
    # checking Y dimension for being (n,) not (1, n)
    def __check_y_dimension(self, y) -> bool:
        return y.T if len(y.shape) != 1 else y
    
    # representing object class
    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items() if not key.startswith('_')]
        return f"MyLineReg class: {', '.join(params)}"
    
    # str representing
    def __str__(self):
        return self.__repr__()
