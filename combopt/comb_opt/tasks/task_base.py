# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd


class TaskBase(ABC):
    """ Abstract class to define optimisation (** MINIMISATION **) tasks """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._n_bb_evals = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Remember to include the @property decorator to this function
        :return:
        """
        return 'Task Name'

    @abstractmethod
    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        """
        Function to compute the problem specific black-box function.

        :param x: 2D numpy array containing the solutions at which the black-box should be evaluated.
        Shape: (batch_size, num_dims), where num_dims is the dimensionality of the problem and batch_size is the batch
        size. dtype: float32.
        :return: 2D numpy array containing evaluated black-box values at the input x. Shape: (batch_size, 1).
        dtype: float32
        """
        pass

    @property
    def num_func_evals(self):
        return self._n_bb_evals

    def restart(self):
        self._n_bb_evals = 0

    def __call__(self, x: pd.DataFrame) -> np.ndarray:
        self._n_bb_evals += len(x)
        return self.evaluate(x.copy())
