import pandas as pd
import numpy as np
from typing import Optional, Union
import random


class MyLogReg():
    """
    LinearRegression
    
    Parametres:
    n_iter = the number of iterations during training
    learing_rate = the step of training
    reg = the type of gradiant regularization (None, l1, l2, elasticnet)
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
                 sgd_sample: Optional[int] = None, 
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
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, verbose: int = False) -> np.ndarray:
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
            if isinstance(self.__sgd_sample, float):
                s = round(X.shape[0] * self.__sgd_sample)
                sample_rows_idx = random.sample(range(X.shape[0]), s)
                X_batch, y_batch = X[sample_rows_idx, :], y[sample_rows_idx]
            elif isinstance(self.__sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.__sgd_sample)
                X_batch, y_batch = X[sample_rows_idx, :], y[sample_rows_idx]
            else:
                X_batch, y_batch = X, y
            
            # new iter prediction
            pred_y = X_batch @ self.__weights
            proba = self._sigmoid(pred_y)
          
            # gradient based on the type of regularization
            grad = grad = 1/X_batch.shape[0] * (proba - y.T) @ X_batch
          
            # minimization step (could be dynamic)
            k = self.__learning_rate if type(self.__learning_rate) in (int, float) else self.__learning_rate(i+1)
            
            # updating weights
            self.__weights -= k * grad
            
            # logging loss func each 'verbose' step
            self.__logging_loss(i, verbose, y, pred_y, proba)

  
     def _sigmoid(self, y):
        return 1 / (1 + np.exp(-y))
        
        
    # getting coefs without w_0
    def get_coef(self) -> np.ndarray:
        return self.__weights[:-1]


    # getting prediction
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X) > 0.5


    # getting prediction
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # adding ones in the end of X matrix for the w_0
        X = np.c_[X.values, np.ones(X.shape[0])]
        return self._sigmoid(X @ self.__weights)

    
    # getting the score by the chosen metric on the last iter step
    def get_best_score(self):
        # if X and Y are not None
        return self.__metric_func(self.predict(self.X), self.y)
    
    

   def _confusion_matrix(self, y, pred_y):
        # [[TN, FN], [FP, TP]]
        # initialize the confusion matrix
        classes = np.unique(y)
        conf_matrix = np.zeros((len(classes), len(classes)))

        for i in range(len(classes)):
            for j in range(len(classes)):
                # count the number of instances in each combination of actual / predicted classes
                conf_matrix[i, j] = np.sum((y == classes[i]) & (pred_y == classes[j]))
              
        return cconf_matrix
    
    def _accuracy(self, y, pred_y):
        conf_matrix = self._confusion_matrix(y, pred_y)
        return (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
    
    
    def _precision(self, y, pred_y):
        conf_matrix = self._confusion_matrix(pred_y, y)
        return conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

    def _recall(self, y, pred_y):
        conf_matrix = self._confusion_matrix(pred_y, y)
        return conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

    def _f1(self, y, pred_y):
        conf_matrix = self._confusion_matrix(pred_y, y)
        prec = self._precision(y, pred_y)
        rec = self._recall(y, pred_y)
        return 2 * (prec * rec) / (prec + rec)

    def _rocauc(self, y, y_proba):
        y_proba = np.sort(y_proba)[::-1]
        pred_y = y_proba > 0.5

        score_sum = []
        for i in range(1, len(pred_y)):
            positive_classes = np.sum(pred_y[:i])
            same_score = np.sum(pred_y[:i] == y_proba[i])/2
            score_sum.append(positive_classes + same_score)
        return sum(score_sum)
    
    # logging loss func on the chosen iter step
    def __logging_loss(self, i: int, verbose: int, y: np.ndarray, pred_y: np.ndarray, y_proba) -> None:
        # if this is a necessary step
        if verbose and (i+1) % verbose == 0:
                if self.__metric == 'rocauc':
                    error = self.__metric_func(y, proba_y)
                else:
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
        # ACCURACY by deffault
        if metric is None:
            return 'accuracy', self._accuracy

        dict_metrics = {'accuracy': self._accuracy, 'precision': self._precision, 
                        'recall': self._recall, 'f1': self._f1, 'rocauc': self._rocauc}
        try:
            return metric, dict_metrics[f'{metric}']
        # if there is no such metric choose the default one
        except KeyError:
            f'Sorry, there is no such metric. You can choose from {dict_metrics.values}'
            return 'accuracy', self._accuracy
    
    # checking Y dimension for being (n,) not (1, n)
    def __check_y_dimension(self, y) -> bool:
        return y.T if len(y.shape) != 1 else y
    
    # representing pbject class
    def __repr__(self):
        return f'MyLineReg class: n_iter={self.__n_iter}, learning_rate={self.__learning_rate}'
    
    # str representing
    def __str__(self):
        return self.__repr__()
