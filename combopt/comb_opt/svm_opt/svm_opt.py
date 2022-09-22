from typing import Union, List

import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error

from comb_opt.tasks import TaskBase


class SVMOptTask(TaskBase):
    """ Tuning of sklearn SVM regression learner hyperparamters on the diabetes regression task """

    @property
    def name(self) -> str:
        return 'SVM Opt'

    def __init__(self, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.x = None
        self.y = None

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        """ Transform entry x into a RNA sequence and evaluate it's fitness given as the Hamming distance
            between the folded RNA sequence and the target
        """

        if self.x is None or self.y is None:
            diabetes = datasets.load_diabetes()
            self.x, self.y = diabetes['data'], diabetes['target']

        evaluations = []
        for i in range(len(x)):
            svm_hyp = x.iloc[i]

            scores = []

            for seed in range(5):
                x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(self.x, self.y,
                                                                                            test_size=.3,
                                                                                            random_state=seed)

                learner = NuSVR(kernel=svm_hyp.kernel, gamma=svm_hyp.gamma, shrinking=svm_hyp.shrinking,
                                C=svm_hyp.C, tol=svm_hyp.tol, nu=svm_hyp.nu)
                learner.fit(x_train, y_train)
                y_pred = learner.predict(x_test)
                scores.append(mean_squared_error(y_test, y_pred))
            evaluations.append(np.mean(scores))

        return np.array(evaluations).reshape(-1, 1)

    @staticmethod
    def get_search_space_params():
        """ Return search space params associated to this task """
        params = [
            {'name': 'kernel', 'type': 'nominal', 'categories': ['linear', 'poly', 'rbf', 'sigmoid']},
            {'name': 'gamma', 'type': 'nominal', 'categories': ['scale', 'auto']},
            {'name': 'shrinking', 'type': 'nominal', 'categories': [1, 0]},
            {'name': 'C', 'type': 'pow', 'lb': 1e-4, 'ub': 10},
            {'name': 'tol', 'type': 'pow', 'lb': 1e-6, 'ub': 1},
            {'name': 'nu', 'type': 'pow', 'lb': 1e-6, 'ub': 1},
        ]

        return params
