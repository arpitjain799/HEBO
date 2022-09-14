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
import torch

from comb_opt.search_space import SearchSpace
from comb_opt.utils.data_buffer import DataBuffer


class OptimizerBase(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        return 'Optimizer_name'

    def __init__(self,
                 search_space: SearchSpace,
                 dtype: torch.dtype = torch.float32
                 ):
        assert dtype in [torch.float32, torch.float64]

        self.dtype = dtype
        self.search_space = search_space

        self._best_x = None
        self.best_y = None

        self.data_buffer = DataBuffer(self.search_space, 1, self.dtype)

    @abstractmethod
    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        """
        Function used to suggest next query points.
        Should return a pandas dataframe with shape (n_suggestions, D) where
        D is the dimensionality of the problem. The column dtype may mismatch expected dtype (e.g. float for int)

        :param n_suggestions: number of suggestions
        :return: a DataFrame of suggestions
        """

    pass

    def suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        """
        Function used to suggest next query points. Should return a pandas dataframe with shape (n_suggestions, D) where
        D is the dimensionality of the problem.

        :param n_suggestions:
        :return:
        """
        suggestions = self.method_suggest(n_suggestions)
        # Convert the dtype of each column to proper dtype
        sample = self.search_space.sample(1)
        return suggestions.astype({column_name: sample.dtypes[column_name] for column_name in sample})

    @abstractmethod
    def observe(self, x: pd.DataFrame, y: np.ndarray):
        """
        Function used to store observations and to conduct any algorithm-specific computation.

        :param x:
        :param y:
        :return:
        """
        pass

    @abstractmethod
    def restart(self):
        """
        Function used to restart the internal state of the optimizer between different runs on the same task.
        :return:
        """
        pass

    @abstractmethod
    def set_x_init(self, x: pd.DataFrame):
        """
        Function to set query points that should be suggested during random exploration

        :param x:
        :return:
        """
        pass

    @abstractmethod
    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        """
        Function used to initialise an optimizer with a dataset of observations
        :param x:
        :param y:
        :return:
        """
        pass

    @property
    def best_x(self):
        if self.best_y is not None:
            return self.search_space.inverse_transform(self._best_x)

    def _restart(self):
        self._best_x = None
        self.best_y = None

        self.data_buffer.restart()
