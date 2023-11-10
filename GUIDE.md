# How to use

This is a simple file that describes how to use **Generalized Model Selection Module** *(GMS for short)*. This module was created to help ML-engineers to evaluate their models AND/OR pipelines
and get verbose description of each model evaluation. This tool could be helpful for everyone that is looking for best model scores and quick evaluation.


## How to intsall?

Installation is very simple:

- This is the PyPI website with the GMS package: https://pypi.org/project/gms/
- You can install GMS Module for Python3 ONLY. There is no support for Python2 or lower

```python
pip3 install gms
```

- If you are using *Kaggle*, install it using

```python
!pip install gms
```

- If Google Collab:

```python
%pip3 install gms
```


## How to import?

To import this Module, use this command:

```python
from gms.GMSModule import GMSModule
```

Because, Module is located in: `./gms (directory) -> GMSModule.py -> GMSModule (class)`


## Start working

To initialize a GMSModule object use something like this:

```python
GMSPipe = GMSModule(mode="classification",
	pivot='f1-score',
	metrics=['accuracy', 'f1-score'],
	include=[LogisticRegression(), RandomForestClassifier()],
	data=[X_train, X_test, y_train, y_test])
```


# Supported Metrics

```python
# For classification
'accuracy'
'precision'
'recall'
'f1-score'
'roc-auc'
            
# For regression
'mae': mean_absolute_error,
'mape': mean_absolute_percentage_error,
'mse': mean_squared_error,
'rmse': lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
'r2-score': r2_score
```

# Class Arguments

- `mode`: "classification" or "regression". Choose a task user trying to achieve.
- `metrics`: A list of metrics to evaluate on. Check supported metrics above.
- `include`: A list of models OR Pipelines, created by user.
- `data`: A list of data variables. Provide them in a list in this order: `[X_train, X_validation, y_train, y_validation]`.
- `pivot` [Optional]: A metric *that is in `metrics` provided [!!]* to focus on when making a decision on best model/Pipeline.


# Main functions

## Best Model

Return two values, that should be unpacked: object of the best model evaluated by module and predictions made by this model.
Model scores evaluation is based on sum of metrics if `pivot` is not provided or `pivot` metric itself.   

`GMSModule.best_model(self, print_info: bool = False)`

- **Args**:
	- `print_info`: print information about retuned values / return value itself. `True` OR `False`
- **Returns**:
	- best model object as a first value / predictions of best model as second value
- **Usage**:
 	```python
  	myBestModel, bestModelPredictions = myPipe.best_model()
  	myPipe.best_model(print_info=True)
   	```

## Create Ranking

Return a dictionary which represents ranking of each model based on `pivot` or sum of scores. Ranking is made automatically and each
model could be accessed by key value. Like: `ranking[1]` will return best model given, `ranking[2]` will return second best model ranked.

`GMSModule.create_ranking(self)`

- **Returns**:
	- a Python dict. with keys as numbers from 1 to N and Models from `self.include` as values for each key. All models are sorted by pivot provided or 
- **Usage**:
 	```python
  	ranking = myPipe.create_ranking()
   	```
Example of returned value:

```python
{1: LinearRegression(), 2: SVR(), 3: LGBMRegressor()}
```
