# classical-machine-learning-algorithms

<div id="stack badges">
    <a href="https://www.python.org">
        <img src="https://img.shields.io/badge/python-6a6a6a?style=flat&logo=python&logoColor=white" alt="python badge"/>
    </a>
    <a href="https://numpy.org">
        <img src="https://img.shields.io/badge/numpy-07607e?style=flat&logo=numpy&logoColor=white" alt="numpy badge"/>
    </a>
    <a href="https://pandas.pydata.org">
        <img src="https://img.shields.io/badge/pandas-7140ff?style=flat&logo=pandas&logoColor=white" alt="pandas badge"/>
</div>

Here I am trying to reproduce classic machine learning algorithms using **numpy**. The entire code is based on the materials of the course ["Machine Learning Algorithms from scratch"](https://stepik.org/course/68260/syllabus) on the stepik platform and on the video lectures of the Yandex SHDA course ["Machine Learning"](https://youtube.com/playlist?list=PLJOzdkh8T5krxc4HsHbB8g8f0hu7973fK&si=XWhZcZknFBiVp_yp). However algorithms may not always be optimized since my goal is to understand the idea and its approximate implementation and not to get perfectly verified solutions at the sklearn level. All code contains comments and type annotations. A brief description of algorithms is below:

#### Linear Models
- **Linear Regression**
  - MSE loss function
  - available metrics: *MAE, MSE, RMSE, MAPE, R2*
  - available loss regularizations: *Lasso, Rigde, ElasticNet*
  - can be used a *stochastic gradient* with different batch's sizes
  - learning step can be computed dynamicly if you pass a counting function to the `learning_rate` parameter, for example `lambda iter: 0.5 * (0.85 ** iter)`
 <p> </p>

- **Binary Linear Regression**
    - Log loss function
    - available metrics: *Accuracy, Precision, Recall, F1, ROC AUC*
    - available loss regularizations: *Lasso, Rigde, ElasticNet*
    - can be used a *stochastic gradient* with different batch's sizes
    - learning step can be computed dynamicly if you pass a counting function to the `learning_rate` parameter, for example `lambda iter: 0.5 * (0.85 ** iter)`
<p> </p>

#### KNearestNeighbours
- **KNNClassification**
  - available metrics/distances: *euclidean, chebyshev, manhattan, cosine*
  - available weights of the nearest neighbours: *uniform (default calculation of classes' mode), rank, distance*
  - also you can use `predict_proba` wich returns the degree of reliability of the prediction, based on the chosen method of calculating neighbours' weights
<p> </p>

- **KNNRegression**
  - available metrics/distances: *euclidean, chebyshev, manhattan, cosine*
  - available weights of the nearest neighbours: *uniform (default calculation of targets' mean value), rank, distance*
<p> </p>


