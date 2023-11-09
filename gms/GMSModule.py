"""New Vers. 0.4.0"""

"""    === General Model Selection Module (GMSModule) ===    """

"""Brief desc.: A simple yet efficient module made
to get crucial info about evaluation of each model passed into the GMSModule object.
User can pass any or all wanted models/pipelines and metrics into the object and use
vast variety of methods that this module provides.

GMSModule automates the process of "blind" model selection of models/pipelines
and shrinks your model selection process into 2-3 lines of code!
This module provides verbose and elegant description for model selection"""

"""QUICKSTART:
- Create a GMSModule with all your wanted data:

>> GMS = GMSModule(mode="classification", metrics=['accuracy', 'f1-score'],
                   pivot='f1-score',
                   include=[LogisticRegression(), RandomForestClassifier()],
                   data=[X_train, X_test, y_train, y_test])

- Then, use one of the object's methods:

>> model, _ = GMS.best_model()
>> model
(out) RandomForestClassifier()

Voila!
"""


"""
CREDITS
___
Made by:

- Full name: Zhamkov Nikita Dmitr.
- Country, city: Russia, Saint-Petersburg
- Project's GitHub repo: https://github.com/plugg1N/gms-module
- Telegram: https://t.me/@jeberkarawita
- Phone: +79119109210
- Email: nikitazhamkov@gmail.com
___
"""

import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from tabulate import tabulate
from tqdm import tqdm
import pandas as pd
                            

class GMSModule:
    def __init__(self, *, mode: str, include: list, data: list, metrics: list, pivot: str = None) -> None:
        """
        Initiate variables for work / catch errors created by users while creating an object
        
        Args:
            - mode: A 'string' argument, that tells program which task is user solving (see 'valid_modes' for valid values)
                e.g.: mode='classification'
                
            - include: A 'list' arg. that contains a list of models/pipelines
                e.g.: include=[LinearRegression()]
                
            - data: A 'list' arg. that contains training and validation subsets
                e.g.: data=[X_train, X_test, y_train, y_val]
                
            - metrics: A 'list' arg. that contains strings of names of metrics (see 'valid_*_metrics' for valid values)
                e.g.: metrics=['accuracy', 'f1-score']
                
            - pivot: A 'str' argument that contains the name of the metric that the user considers important for solving their task
            (if pivot=None -> the sum of the metrics would be the judge of the best model to consider)
                e.g.: pivot='f1-score'
                
        Returns:
            Nothing.
        """

        self.mode = mode
        self.metrics = metrics
        self.include = include
        self.pivot = pivot
        self.X_train, self.X_test, self.y_train, self.y_test = data
        
        
        # Eval. functions to consider
        self.__evaluation_functions = {
            # For classification
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
            'f1-score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            'roc-auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovr'),
            
            
            # For regression
            'mae': mean_absolute_error,
            'mape': mean_absolute_percentage_error,
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
            'r2-score': r2_score
        }
        
        
        
        # Check for evaluation results (to avoid uneeded re-evaluation)
        self.model_results = None
        
        
        valid_modes = ['classification', 'regression']
        valid_data_types = ['list', 'tuple']
        valid_pivot_vals = list(self.__evaluation_functions.keys()) + [None]
        
        
        # Catch 'mode' error
        if self.mode not in valid_modes:
            raise ValueError(f"'{self.mode}' mode is not valid. Valid modes are in: {valid_modes}")
            
        # Catch 'metrics' error (for metrics, that are strictly added by author)
        if any(metric not in self.__evaluation_functions.keys() for metric in self.metrics):
            raise ValueError(f"{self.metrics} | One or more classification metrics are not valid. Valid metrics for classification are in: {self.evaluation_functions.keys()}")
            
        # Catch 'pivot' error
        if self.pivot not in valid_pivot_vals:
            raise ValueError(f"'{self.pivot}' name is not valid. Valid pivot names are in: {valid_pivot_vals}")
            
        # Catch 'pivot is not in metrics' error
        if self.pivot not in self.metrics + [None]:
            raise ValueError(f"'{self.pivot}' is not in given metrics: {self.metrics}")
            

            
    def _evaluate_models(self) -> (list, list):
        """
        The Backbone of this module. Evaluate each model on different subsets using metrics given by the user
        
        Args:
            - self: all self.variables from __init__ 
                
        Returns:
            A 'list' of 'tuple' values. Each tuple contains (*model_name, *dict_of_each_metric_evaluation). For both subsets: train and test
        """
        
        # Init. dicts to inflate with model_names as keys and predictions as values
        self.model_predictions_dict_test = dict()
        self.model_predictions_dict_train = dict()

        
        
        # Check if evaluation was already done
        if self.model_results is not None:
            return self.model_results

        else:
            results_test = []
            results_train = []

            for model in tqdm(self.include, desc="Evaluating Each Model", unit=" eval."):
                model = model.fit(self.X_train, self.y_train)
                
                
                y_pred_test = model.predict(self.X_test)
                y_pred_train = model.predict(self.X_train)
                
                self.model_predictions_dict_test[str(model)] = y_pred_test
                self.model_predictions_dict_train[str(model)] = y_pred_train

                
                # Keep keys as metric names and evals as values
                scores_test = {}
                scores_train = {}
                    
                # Evaluate each model on each metric given
                for metric in self.metrics:
                    evaluation_function = self.__evaluation_functions.get(metric)
                    
                    scores_test[metric] = evaluation_function(self.y_test, y_pred_test)
                    scores_train[metric] = evaluation_function(self.y_train, y_pred_train)
                
                

                results_test.append((model, scores_test))
                results_train.append((model, scores_train))

            self.model_results = (results_train, results_test)
            
            return results_train, results_test
            
            
            
            

    def best_model(self, print_info: bool = False) -> (str, dict):
        """
        Get name and scores of the best model accroding to sum of the scores
        
        Args:
            - self: variables from __init__ function
                
            - print_info: A 'bool' variable that tells function to print info or to return it for further usage
                
        Returns:
            - if print_info=True: Nothing. Prints model name and score based on pivot
            - else: Best model as object[0] / Predictions on test dataset[1]
        """
        # Make scores global to output scores to user
        global model_scores_dict_test
        
        result_train, result_test = self._evaluate_models()
        

        model_scores_dict_train = {}
        model_scores_dict_test = {}

        
        # If self.pivot == None -> calculate sum of all scores
        if self.pivot == None: 
            for model, scores in result_train:
                score_sum = sum(scores.values())
                model_scores_dict_train[model] = score_sum

            # Iterate through the results and calculate the sum of scores | TEST
            for model, scores in result_test:
                score_sum = sum(scores.values())
                model_scores_dict_test[model] = score_sum
        
        # If self.pivot != None -> get only self.pivot values
        else:
            for model, scores in result_train:
                pivot_score = scores[self.pivot]
                model_scores_dict_train[model] = pivot_score

            # Iterate through the results and calculate the sum of scores | TEST
            for model, scores in result_test:
                pivot_score = scores[self.pivot]
                model_scores_dict_test[model] = pivot_score

        # Find the key with the maximum value in the dictionary
        best_model = max(model_scores_dict_test, key=model_scores_dict_test.get)

        # Print the key of the model with the highest score
        if print_info:
            print(f"Model with the highest score: {best_model} \nWith scores: {model_scores_dict_test[best_model]}")
        else:
            best_model_preds = best_model.predict(self.X_test)
            return best_model, best_model_preds
        
        
        
        
    def create_ranking(self) -> dict:
        """
        Receive a dictionary of each model ranked by its pivot
        
        Args:
            - self: variables from __init__ function
                
        Returns:
            'dict' that consists of "Keys: 1-len(self.include), Values: models"
        """
        _, model_results = self._evaluate_models()

        # Create a ranking based on the sum of values in the dictionaries
        ranking = {}
        
        # Rank each model by its pivot metric
        if self.pivot is not None:
            for idx, (model, scores) in enumerate(
                    sorted(model_results, key=lambda x: x[1][self.pivot], reverse=True) ):
                ranking[idx + 1] = model
        
        # Rank each metric based on its sum of scores
        else:
            for idx, (model, scores) in enumerate(
                    sorted(model_results, key=lambda x: -sum(score for score in x[1].values())) ):
                ranking[idx + 1] = model

        return ranking
    
    
    
    
    def to_df(self, subset: str = "test") -> pd.DataFrame:
        """
        Get a Dataframe of subset scores
        
        Args:
            - subset: A string of two possible subsets: "train" or "test"
            
        Returns:
            'pd.Dataframe' with model name and scores provided by user
        """
        if subset not in ['train', 'test']:
            raise ValueError(f"Provided subset: '{subset}' is not in possible subsets: ['train', 'test']")
        
        
        results_train, results_test = self._evaluate_models()
    
        
        # Model Names would always be as a column 
        columns = ['Model Name']
        
        
        # For each metric provided create a column with its name
        for i in self.metrics:
            columns.append(i.title())
            
        data = []
        
        if subset == "test":
            for i in range(len(self.include)):
                # take its score predictions and create a list of [model_name, *scores] 
                temp_data = [str(self.include[i])] + list(results_test[i][1].values())
                data.append(temp_data)
                
        elif subset == "train":
            # the same as above ...
            for i in range(len(self.include)):
                temp_data = [str(self.include[i])] + list(results_train[i][1].values())
                data.append(temp_data)
            
            
        
        return pd.DataFrame(data=data, columns=columns)
    
    
    
    
    def get_predictions(self, model, subset: str = "test") -> list:
        """
        Get predictions of a wanted model from the list of models provided.
        
        Args:
            - model: An object of a model to get predictions from.
            - subset: A string of a name of subset to get predictions from.
                (default = "test")
                
        Returns:
            List of predictions made by a model provided.
        """
        # Receive predictions
        self._evaluate_models()
        
        # Check if model given is not evaluated by Module
        if str(model) not in list(map(str, self.include)):
            raise ValueError(f"Model name provided: '{model}' is not in evaluated models list: {self.include} ")
        

        if subset not in ['test', 'train']:
            raise ValueError(f"Subset provided: '{subset}' is not 'train' or 'test'")
            

        if subset == "train":
            return list(self.model_predictions_dict_train[str(model)])
        
        elif subset == "test":
            return list(self.model_predictions_dict_test[str(model)])
        
        
        
        
      

        
    def describe(self) -> None:
        """
        Print (log) verbose information about each variable passed into the object and all possible returns.
        Needed for user's self analysis and visual control
        
        Args:
            - self: variables from __init__ function
                
        Returns:
            Nothing. Prints description
        """
        result_train, result_test = self._evaluate_models()
        best_model_name, _ = self.best_model()
        ranking = self.create_ranking()
        
        
        print("==== DESCRIPTION ====\n")
        
        # Show inputed info
        print(f"Evaluation mode: {self.mode}\n")
        print(f"From models selected: {self.include}\n")
        print(f"Pivot selected: {self.pivot}\n" if self.pivot != None else f"Pivot is not selected. Counting sums of scores!: pivot: {self.pivot}\n")
        print(f"Metrics for evaluation: {self.metrics}\n\n")
        
        # Tables of evaluations for each model
        print("# 1. Evaluation metrics for each model passed into the object for | TRAIN / TEST:\n")
        print(tabulate(result_train, headers=["Models (train)", "Metrics (train)"]), "\n\n")
        print(tabulate(result_test, headers=["Models (test)", "Metrics (test)"]), "\n\n")
        
        # Name and scores of the best model
        print("# 2. Best model name and scores:\n")
        print(f"Best model name: {best_model_name}\nBest model score: {max(model_scores_dict_test.values())}\n\n")
        
        # Ranking of each model based on its pivot
        print("# 3. Ranking of each model:\n")
        for rank, model in ranking.items():
            print(f"{rank}: {model}")
            
        
