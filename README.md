![](https://github.com/plugg1N/gms-module/blob/main/images/chart1.png?raw=true)
*Chart 1: Basic GMS Workflow*

-  [Этот README.md на русском языке](https://github.com/plugg1N/gms-module/blob/main/README_Russian.md) ❤️
-  [Verbose Usage Guide](https://github.com/plugg1N/gms-module/blob/main/GUIDE.md)

# Brief Description

**<ins>General Model Selection Module</ins>** *(next: GMS-Module)* is a simple yet neat model selection tool that would help machine learning developers to get their hands on the most efficient model/pipeline for their specific task. *This
project has brought me 5 points additionally for IT General State Exam (ЕГЭ по информатике)* 😌

User only needs to pass:
- *Models AND/OR Pipelines* of their choice
- *Metrics* For evaluation
- *Pivot* if certain metric is more important than the others
- *Data* to train and evaluate on

Module would automatically make evaluations, store them and give verbose description of each model's performance!

# Installation

To install GMSModule ensure that python3 and pip are installed. In terminal simply type:
`pip install gms` OR `pip3 install gms`

```python
pip3 install gms
```

# How to use?

1. Make sure that all the variables are prepared to be used by GMS:
	- `mode`: A string of your ML task: `'regression' OR 'classification'`
	- `include`: A list of model-obj. of your choice: `[LinearRegression(), SVR()]`
	- `metrics`: A list of strings to evaluate on: 
		- `classification = ['accuracy', 'f1-score', 'precision', 'recall', 'roc-auc']`
		- `regression = ['mae', 'mape', 'mse', 'rmse', 'r2-score']`
	- `data`: A list of your data to train/validate on: 
		 `[X_train, X_test, y_train, y_test]`
	- `pivot`: *if necessary*: A string of one of metrics provided: `'accuracy'` (pivot is a metric that is most important for evaluation)

2. Import GMSModule into your project:

```python
from gms.GMSModule import GMSModule
```


2. Create a **GMSModule** object with your data:
```python
GMSPipe = GMSModule(mode="classification",
	pivot='f1-score',
	metrics=['accuracy', 'f1-score'],
	include=[LogisticRegression(), RandomForestClassifier()],
	data=[X_train, X_test, y_train, y_test])
```

3. Use any of methods provided:
```python
best_model, _ = GMSPipe.best_model()
print(best_model)
```

```python
RandomForestClassifier()
```


# Why this module?

Every Machine Learning developer, especially after extensive data analysis, has to pick **the most precise Machine Learning model**. Some engineers already know which model would fit perfectly, due to the ease of task given or due to the fact that ML model is evident.

> **But some engineers might struggle with the BLIND choice between dozens if not HUNDREDS of ML models / pipelines that they have built. That's where GMS Module could help!**

User doesn't have to build a custom function that would evaluate each model one by one on their metrics. **User just has to pass in each model and name metrics of their choice** and *voila!* 

Then, user is able to look at the `GMSModule.description()` and get verbose information about models' evaluations and see which models are better than the others.

Users can also get their data into variables for further usage, like in this <ins>example</ins>:

```python
# Get predictions of the best model from list
_, preds = GMSModule.best_model()

# DataFrame data
data = {
	'id': range(25000),
	'value': preds
}

# Create a DataFrame and pass information into it
df = pd.DataFrame(data)
df.to_csv('submit.csv', index=False)
```

# Project History


This project was created as a **fun side project for me** to experiment with scikit-learn tools. Project has helped me to become more focused on programming overall and taught me how to write m*y own PYPI module for others to use!*

> The idea was born on `16.10.2023` and the first draft of the project was so inefficient, so that I had to rewrite almost everything

Module used to re-evaluate each time I've tried to get evaluations of each model. Evaluations used 'if-statements' which looked hideous and unprofessional.

With the 5-th version done on `20.10.2023` everything has been changed. Re-evaluation problem was fixed, module could catch the most obvious exceptions caused by user and 'if-statements' were replaced with neat dictionaries.

As if `22.10.2023`, I am creating the first version of this Markdown (README.md) file. Project is polished. All I need is to:

- Create a module file (`*.py`)
- Write a bunch of documentation: this doc in Russian, code run-through, basic use-cases and much more!
- Get a license
- Post this module on PYPI

`21:54. 22.10.2023` I've already posted my project to PYPI. Everything seems to work fine.

`23:28. 07.11.2023` I've created a new 0.3.0 version that fixed some bugs I've encountered. Now module has less bugs. New feature added: `GMSModule.to_df()` :)

`11:02. 09.11.2023` New version 0.4.0. Now, you get get predictions of each model provided! Most of the comments were cleared due to the fact that they were
unnecessary.


# TO DO:

- Create a Markdown file for Usage description and examples
- *Maybe* cross_validation support

# Fixed:

- Fixed issue with `pivot = None` error
- Fixed issue with non-binary classification support 
- Added: `GMSModule.get_predictions()` function. Now you can evaluate each model provided!



# My Socials

- Full Name:  Nikita Zhamkov (Dmitrievich)
- Country, city:  Russia, Saint-Petersburg
- Phone number: +79119109210
- Email: nikitazhamkov@gmail.com
- GitHub: https://github.com/plugg1N
- Telegram: https://t.me/jeberkarawita
