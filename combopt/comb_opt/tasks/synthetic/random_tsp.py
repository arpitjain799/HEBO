# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import random

import numpy as np
import pandas as pd

from typing import Optional

from ..task_base import TaskBase


class RandomTSP(TaskBase):

    @property
    def name(self) -> str:
        return 'Random TSP'

    def __init__(self, num_dims: int, min_dist=1, max_dist=1000, seed: Optional[int] = None):
        super(RandomTSP, self).__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_dims = num_dims
        distance_matrix = np.random.uniform(low=min_dist, high=max_dist, size=(num_dims, num_dims))
        self.distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Ensure the matrix is symmetrical

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.num_dims

        x = x.to_numpy().astype(int)

        ind1 = x
        ind2 = np.concatenate((x[:, 1:], x[:, :1]), axis=1)

        return self.distance_matrix[ind1, ind2].sum(axis=1).reshape(-1, 1)
