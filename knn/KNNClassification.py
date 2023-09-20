import pandas as pd
import numpy as np

class MyKNNClf():
    """
    KNearestNeighbours classification
    
    Parametres:
    k = the number of neighbours
    metric = the type of distance between vectors (euclidean, chebyshev, manhattan, cosine)
    weight = the type of weights for k-neighbors (uniform, rank, distance)
    """
    def __init__(self, k: int = 1, 
                 metric: str = 'euclidean', 
                 weight: str = 'uniform'):
        self.k = k
        self.train_size = None
        self.X_train, self.y_train = None, None
        self.metric, self.__distance = self.__define_distance(metric)
        self.weight, self.__compute_weight = self.__define_weight(weight)
        
    # training
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # saving datasets
        self.X_train, self.y_train = X, y
        self.train_size = X.shape
    
    # getting classes' predicton
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        prediction_vector = []
        # comparing each X-train-vector with each X-vector
        for i in range(X.shape[0]):
            current_point = X.iloc[i]
            distance_list = []
            
            for j in range(self.train_size[0]):
                next_point = self.X_train.iloc[j]
                
                # computing distance between two points
                distance_list.append(self.__distance(current_point, next_point))
            
            # sorting and choosing top-k nearest points
            distance_list = np.array(distance_list)
            close_points = np.argsort(distance_list)[:self.k]
            predictions = self.y_train.iloc[close_points].to_numpy().ravel()
            
            # computing predictions taking into account weights
            pred_classes = self.__compute_weight(predictions, distance_list[close_points]) 
            
            # choosing class with the largest total weight
            max_class = max(pred_classes.items(), key=lambda x: x[::-1])[0]
            prediction_vector.append(max_class)
            
        return np.array(prediction_vector)

    # getting proba prediction
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        
        prediction_vector = []
        # comparing each X-train-vector with each X-vector
        for i in range(X.shape[0]):
            
            current_point = X.iloc[i]
            distance_list = []
            
            for j in range(self.train_size[0]):
                next_point = self.X_train.iloc[j]
                
                 # computing distance between two points
                distance_list.append(self.__distance(current_point, next_point))
            
            # sorting and choosing top-k nearest points
            distance_list = np.array(distance_list)
            close_points = np.argsort(distance_list)[:self.k]
            predictions = self.y_train.iloc[close_points].to_numpy().ravel()
            
            # computing predictions taking into account weights
            pred_classes = self.__compute_weight(predictions, distance_list[close_points])  
            #max_class = max(res.items(), key=lambda x: x[::-1])[0]
            
            # choosing class with the largest total weight and returning its weight (proba)
            max_class = max(res.items(), key=lambda x: x[::-1])[0]
            prediction_vector.append(pred_classes[max_class])
            
        return np.array(prediction_vector)
    
    # euclidean distance
    def _euclidean(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return ((v2 - v1)**2).sum()**0.5
    
    # chebyshev distance
    def _chebyshev(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return abs(v2 - v1).max()
    
    # manhattan distance
    def _manhattan(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return abs(v2 - v1).sum()
    
    # cosine distance
    def _cosine(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return 1 - (v1*v2).sum() / ((v1**2).sum() * (v2**2).sum())**0.5
    
    # no weight, using mode
    def _uniform_weight(self, ranks: np.ndarray, score: np.ndarray) -> dict:
        classes = {}
        for c in np.unique(ranks):
            classes[c] = (ranks == c).sum() / len(ranks)
        return classes
    
    # rank weight
    def _rank_weight(self, ranks: np.ndarray, score: np.ndarray) -> dict:
        classes = dict()
        d = sum([1/(ind+1) for ind in range(len(ranks))])

        for c in np.unique(ranks):
            n = sum([1/(ind+1) for ind in range(len(ranks)) if ranks[ind] == c])
            classes[c] = n / d
        return classes
    
    # distance weight
    def _distance_weight(self, ranks: np.ndarray, score: np.ndarray) -> dict:
        classes = dict()
        d = sum([1/score[ind] for ind in range(len(ranks))])

        for c in np.unique(ranks):
            n = sum([1/score[ind] for ind in range(len(ranks)) if ranks[ind] == c])
            classes[c] = n / d
        return classes
        
    # choosing the type of metric/distance
    def __define_distance(self, metric: str):
        metrics = {'euclidean': self._euclidean, 'chebyshev': self._chebyshev, 
                   'manhattan': self._manhattan, 'cosine': self._cosine}
        if metric in metrics:
            return metric, metrics[metric]
        return 'euclidean', self._euclidean

    # choosing the type of weights
    def __define_weight(self, weight: str):
        weights = {'uniform': self._uniform_weight, 'rank': self._rank_weight, 
                   'distance': self._distance_weight}
        if weight in weights:
            return weight, weights[weight]
        return 'uniform', self._uniform_weight
        
    # representing object class
    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items() if not key.startswith('_')]
        return f"MyKNNClf class: {', '.join(params)}"
    
    # str representing
    def __str__(self):
        return self.__repr__()

