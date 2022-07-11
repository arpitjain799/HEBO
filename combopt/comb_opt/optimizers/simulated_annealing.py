# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd
import torch

from comb_opt.optimizers.optimizer_base import OptimizerBase
from comb_opt.search_space import SearchSpace


class SimulatedAnnealing(OptimizerBase):

    @property
    def name(self) -> str:
        return 'Simulated Annealing'

    def __init__(self,
                 search_space: SearchSpace,
                 init_temp: float = 100.,
                 tolerance: int = 1000,
                 store_observations: bool = True,
                 allow_repeating_suggestions: bool = False,
                 dtype: torch.dtype = torch.float32,
                 ):

        #TODO: add trust region manager
        assert search_space.num_nominal == search_space.num_params, \
            'Simulated Annealing is currently implemented for nominal variables only'

        super(SimulatedAnnealing, self).__init__(search_space, dtype)

        self.init_temp = init_temp
        self.tolerance = tolerance
        self.store_observations = store_observations
        self.allow_repeating_suggestions = allow_repeating_suggestions

        self.temp = self.init_temp
        self.x_init = self.search_space.sample(1)

        self._current_x = None
        self._current_y = None

        # For stability
        self.MAX_EXPONENT = 0  # 0 As probability can't be larger than 1
        self.MIN_EXPONENT = -12  # exp(-12) ~ 6e-6 < self.MIN_PROB
        self.MAX_PROB = 1.
        self.MIN_PROB = 1e-5

    def set_x_init(self, x: pd.DataFrame):
        self.x_init = x

    def restart(self):
        self._restart()

        self._current_x = None
        self._current_y = None
        self.temp = self.init_temp
        self.x_init = self.search_space.sample(1)

    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        assert y.ndim == 2
        assert y.shape[1] == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.search_space.num_dims

        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        # Add data to all previously observed data
        if self.store_observations or (not self.allow_repeating_suggestions):
            self.data_buffer.append(x.clone(), y.clone())

        # update best fx
        best_idx = y.flatten().argmin()
        best_y = y[best_idx, 0].item()

        if self.best_y is None or best_y < self.best_y:
            self.best_y = best_y
            self._best_x = x[best_idx: best_idx + 1]

        if self._current_x is None or self._current_y is None:
            self._current_y = best_y
            self._current_x = x[best_idx: best_idx + 1]

    def suggest(self, n_suggestions):

        idx = 0
        n_remaining = n_suggestions
        x_next = pd.DataFrame(index=range(n_suggestions), columns=self.search_space.df_col_names, dtype=float)

        if n_remaining and len(self.x_init):
            n = min(n_remaining, len(self.x_init))
            x_next.iloc[idx: idx + n] = self.x_init.iloc[idx: idx + n]
            self.x_init = self.x_init.drop([i for i in range(idx, idx + n)]).reset_index(drop=True)

            idx += n
            n_remaining -= n

        if n_remaining and self._current_x is None:
            x_next.iloc[idx: idx + n_remaining] = self.search_space.sample(n_remaining)  #TODO: sample within TR
            idx += n_remaining
            n_remaining -= n_remaining

        if n_remaining:
            current_x = self._current_x.clone() * torch.ones((n_remaining, self._current_x.shape[1])).to(
                self._current_x)

            x_next.iloc[idx: idx + n_remaining] = self.search_space.inverse_transform(
                self.sample_unseen_nominal_neighbour(current_x))

        return x_next

    def observe(self, x, y):

        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        assert len(x) == len(y)

        # Add data to all previously observed data
        if self.store_observations or (not self.allow_repeating_suggestions):
            self.data_buffer.append(x.clone(), y.clone())

        # update best fx
        if self.best_y is None:
            idx = y.flatten().argmin()
            self._current_x = x[idx: idx + 1]
            self._current_y = y[idx, 0].item()

            self._best_x = x[idx: idx + 1]
            self.best_y = y[idx, 0].item()

        else:
            self.temp *= 0.8
            idx = y.flatten().argmin()
            y_ = y[idx, 0].item()

            if y_ < self.best_y:
                self._current_x = x[idx: idx + 1]
                self._current_y = y_

                self._best_x = x[idx: idx + 1]
                self.best_y = y_

            else:
                exponent = np.clip(- (y_ - self._current_y) / self.temp, self.MIN_EXPONENT, self.MAX_EXPONENT)
                p = np.clip(np.exp(exponent), self.MIN_PROB, self.MAX_PROB)
                z = np.random.rand()

                if z < p:
                    self._current_x = x[idx: idx + 1]
                    self._current_y = y_

    def sample_unseen_nominal_neighbour(self, x_nominal: torch.Tensor):

        if not self.allow_repeating_suggestions:
            x_observed = self.data_buffer.x

        single_sample = False

        if x_nominal.ndim == 1:
            x_nominal = x_nominal.view(1, -1)
            single_sample = True

        x_nominal_neighbour = x_nominal.clone()

        for idx in range(len(x_nominal)):
            done = False
            tol = self.tolerance
            while not done:
                x = x_nominal[idx]
                # randomly choose a nominal variable #TODO: keeping it in TR
                var_idx = np.random.randint(low=0, high=self.search_space.num_nominal)
                choices = [j for j in range(int(self.search_space.nominal_lb[var_idx]),
                                            int(self.search_space.nominal_ub[var_idx]) + 1) if
                           j != x_nominal[idx, var_idx]]

                x[var_idx] = np.random.choice(choices)

                tol -= 1

                if self.allow_repeating_suggestions:
                    done = True
                elif not (x.unsqueeze(0) == x_observed).all(1).any():
                    done = True
                elif tol == 0:
                    x = self.search_space.transform(self.search_space.sample(1))[0]  #TODO: sample within TR

            x_nominal_neighbour[idx] = x

        if single_sample:
            x_nominal_neighbour = x_nominal_neighbour.view(-1)

        return x_nominal_neighbour
