# Bayesmark hyperparameter optimisation tasks

## Setup
```shell
pip install bayesmark==0.0.8
```

## Available tasks
A task is made of: a `model`, a `metric`, and a `database`
### Classification tasks
- Models: DT, MLP-adam, MLP-sgd, RF, SVM, ada, kNN, lasso, linear
- Metrics: nll, acc
- Database: breast, digits, iris, wine


### Regression tasks
- Models: DT, MLP-adam, MLP-sgd, RF, SVM, ada, kNN, lasso, linear
- Metrics: mae, mse
- Database: boston, diabetes

## Tasks involving mixed space

Tasks involving continuous and discrete dims:
- DT regression / classification
- MLP-adam regression / classification
- MLP-sgd regression / classification
- RF regression  / classification
- ada regression / classification
- kNN regression / classification

Tasks involving nominal dimensions:
- lasso regression
- linear regression