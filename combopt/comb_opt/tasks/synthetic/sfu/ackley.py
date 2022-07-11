# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd

from comb_opt.tasks.task_base import TaskBase


class Ackley(TaskBase):
    """
    The Ackley Function. See https://www.sfu.ca/~ssurjano/ackley.html for details.
    """

    @property
    def name(self) -> str:
        return 'Ackley Function'

    def __init__(self, num_dims: int = 10, lb: float = -32.768, ub: float = 32.768, a: float = 20, b: float = 0.2,
                 c: float = 2 * np.pi):
        assert isinstance(num_dims, int)
        assert isinstance(lb, int) or isinstance(lb, float)
        assert isinstance(ub, int) or isinstance(ub, float)
        assert isinstance(a, int) or isinstance(a, float)
        assert isinstance(b, int) or isinstance(b, float)
        assert isinstance(c, int) or isinstance(c, float)

        super(Ackley, self).__init__()

        self.num_dims = num_dims
        self.lb = lb
        self.ub = ub
        self.a = a
        self.b = b
        self.c = c
        self.x_star = pd.DataFrame([[0 for _ in range(num_dims)]])  # Global minimiser
        if (self.x_star <= self.ub).all().all() and (self.x_star >= self.lb).all().all():
            self.global_optimum = self.evaluate(self.x_star)[0, 0]  # Global optimum

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.num_dims
        assert (x <= self.ub).all().all()
        assert (x >= self.lb).all().all()

        x = x.to_numpy().astype(float)

        sum1 = (x ** 2).sum(axis=1)
        sum2 = (np.cos(self.c * x)).sum(axis=1)

        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / self.num_dims))
        term2 = - np.exp(sum2 / self.num_dims)

        return (term1 + term2 + self.a + np.exp(1)).reshape(-1, 1)
