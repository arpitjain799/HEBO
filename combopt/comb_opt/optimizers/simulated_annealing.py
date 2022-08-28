# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
from typing import Optional

import numpy as np
import pandas as pd
import torch

from comb_opt.optimizers.optimizer_base import OptimizerBase
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region import TrManagerBase
from comb_opt.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from comb_opt.utils.discrete_vars_utils import get_discrete_choices
from comb_opt.utils.distance_metrics import hamming_distance


class SimulatedAnnealing(OptimizerBase):

    @property
    def name(self) -> str:
        return 'Simulated Annealing'

    def __init__(self,
                 search_space: SearchSpace,
                 fixed_tr_manager: Optional[TrManagerBase] = None,
                 init_temp: float = 100.,
                 tolerance: int = 1000,
                 store_observations: bool = True,
                 allow_repeating_suggestions: bool = False,
                 max_n_perturb_num: int = 20,
                 neighbourhood_ball_transformed_radius: int = .1,
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        :param: fixed_tr_manager: the SA will evolve within the TR defined by the fixed_tr_manager
        :param: neighbourhood_ball_normalised_radius: in the transformed space, numerical dims are mutated by sampling
                                                      a Gaussian perturbation with std this value
        """
        assert search_space.num_permutation == 0, \
            'Simulated Annealing is currently not implemented for permutation variables'

        self.fixed_tr_manager = fixed_tr_manager
        super(SimulatedAnnealing, self).__init__(search_space, dtype)

        self.is_numeric = True if search_space.num_cont > 0 or search_space.num_disc > 0 else False
        self.is_nominal = True if search_space.num_nominal > 0 else False
        self.is_mixed = True if self.is_numeric and self.is_nominal else False
        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims
        self.discrete_choices = get_discrete_choices(search_space)
        self.max_n_perturb_num = max_n_perturb_num
        self.neighbourhood_ball_transformed_radius = neighbourhood_ball_transformed_radius

        self.init_temp = init_temp
        self.tolerance = tolerance
        self.store_observations = store_observations
        self.allow_repeating_suggestions = allow_repeating_suggestions

        self.temp = self.init_temp
        if self.fixed_tr_manager:
            self.x_init = self.search_space.inverse_transform(self.fixed_tr_manager.center.unsqueeze(0))
        else:
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
        if self.fixed_tr_manager:
            self.x_init = self.search_space.inverse_transform(self.fixed_tr_manager.center)
        else:
            self.x_init = self.search_space.sample(1)

    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        assert y.ndim == 2
        assert y.shape[1] == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.search_space.num_dims
        assert self.fixed_tr_manager is None, "Cannot initialize if a fixed tr_manager is set"

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

    def suggest(self, n_suggestions: int = 1) -> pd.DataFrame:

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
            if self.fixed_tr_manager:  # sample within TR
                transf_samples = sample_numeric_and_nominal_within_tr(x_centre=self.fixed_tr_manager.center,
                                                                      search_space=self.search_space,
                                                                      tr_manager=self.fixed_tr_manager,
                                                                      n_points=n_remaining,
                                                                      is_numeric=self.is_numeric,
                                                                      is_mixed=self.is_mixed,
                                                                      numeric_dims=self.numeric_dims,
                                                                      discrete_choices=self.discrete_choices,
                                                                      max_n_perturb_num=self.max_n_perturb_num,
                                                                      model=None,
                                                                      return_numeric_bounds=False)
                new_samples = self.search_space.inverse_transform(transf_samples)
            else:
                new_samples = self.search_space.sample(n_remaining)
            x_next.iloc[idx: idx + n_remaining] = new_samples
            idx += n_remaining
            n_remaining -= n_remaining

        if n_remaining:
            assert self._current_x is not None
            current_x = self._current_x.clone() * torch.ones((n_remaining, self._current_x.shape[1])).to(
                self._current_x)

            # create tensor with the good shape
            neighbors = self.search_space.transform(self.search_space.sample(n_remaining))

            # sample neighbor for nominal dims
            neighbors[:, self.search_space.nominal_dims] = self.sample_unseen_nominal_neighbour(
                current_x[:, self.search_space.nominal_dims])

            # TODO: check this  --> do we make sure everything is in [0, 1] in transformed space?
            # sample neighbor for numeric dims
            dim_arrays = [self.search_space.disc_dims, self.search_space.cont_dims]
            for dim_array in dim_arrays:
                if len(dim_array) > 0:
                    noise = torch.randn((n_remaining, len(dim_array))).to(
                        self._current_x) * self.neighbourhood_ball_transformed_radius
                    # project back to the state space
                    clip_lb = 0
                    clip_ub = 1
                    if self.fixed_tr_manager:  # make sure neighbor is in TR
                        clip_lb = max(0, self.fixed_tr_manager.center[dim_array] - self.fixed_tr_manager.radii['numeric'])
                        clip_ub = min(1, self.fixed_tr_manager.center[dim_array] + self.fixed_tr_manager.radii['numeric'])
                    neighbors[:, dim_array] = torch.clip(current_x[:, dim_array] + noise, clip_lb, clip_ub)

            x_next.iloc[idx: idx + n_remaining] = self.search_space.inverse_transform(neighbors)

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
            x_observed = self.data_buffer.x[:, self.search_space.nominal_dims]

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
                # randomly choose a nominal variable
                if self.fixed_tr_manager and hamming_distance(
                        self.fixed_tr_manager.center[self.search_space.nominal_dims].to(x), x,
                        normalize=False) >= self.fixed_tr_manager.radii['nominal']:
                    # choose a dim that won't suggest a neighbor out of the TR
                    var_idx = np.random.choice(
                        [d for d in self.search_space.nominal_dims if x[d] != self.fixed_tr_manager.center[d]])
                else:
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
                    if self.fixed_tr_manager:
                        x = sample_numeric_and_nominal_within_tr(x_centre=self.fixed_tr_manager.center,
                                                                 search_space=self.search_space,
                                                                 tr_manager=self.fixed_tr_manager,
                                                                 n_points=1,
                                                                 is_numeric=self.is_numeric,
                                                                 is_mixed=self.is_mixed,
                                                                 numeric_dims=self.numeric_dims,
                                                                 discrete_choices=self.discrete_choices,
                                                                 max_n_perturb_num=self.max_n_perturb_num,
                                                                 model=None,
                                                                 return_numeric_bounds=False)
                    else:
                        x = self.search_space.transform(self.search_space.sample(1))[0]
                    done = True
                    x = x[self.search_space.nominal_dims]

            x_nominal_neighbour[idx] = x

        if single_sample:
            x_nominal_neighbour = x_nominal_neighbour.view(-1)

        return x_nominal_neighbour
