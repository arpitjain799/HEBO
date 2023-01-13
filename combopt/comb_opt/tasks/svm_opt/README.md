# SVM hyperparameter tuning

Given `california housing` dataset loaded from `sklearn` library, the task consist in optimising hyperparameters of
`NuSVR` regressor:

- kernel: {linear, poly, RBF, sigmoid}
- gamma: {scale, auto}
- shrinking: {on, off}
- C: \[1e-4, 10\]  (search in `pow`)
- tol: \[1e-6, 1\]  (search in `pow`)
- nu: \[1e-6, 1\]  (search in `pow`)

This is a mixed search space with 3 nominal and 3 numerical (power scale) dimensions.

## Setup

```shell
pip install sklearn
```

## Acknowledgements

This task has been considered in
[Mixed Variable Bayesian Optimization with Frequency Modulated Kernels
](https://proceedings.mlr.press/v161/oh21a/oh21a.pdf)
, Oh et al. 2020. Nevertheless, we changed the dataset from `boston` to `california` due to [ethical issues with 
the former dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#:~:text=The%20Boston%20housing%20prices%20dataset%20has%20an%20ethical%20problem%3A%20as,on%20house%20prices%20%5B2%5D.).
