# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
from abc import ABC
from typing import Optional

import numpy as np
import pandas as pd
import torch

from comb_opt.acq_funcs import AcqBase
from comb_opt.acq_optimizers import AcqOptimizerBase
from comb_opt.models import ModelBase
from comb_opt.optimizers.optimizer_base import OptimizerBase
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region.tr_manager_base import TrManagerBase
from comb_opt.utils.model_utils import move_model_to_device


class BoBase(OptimizerBase, ABC):

    def __init__(self,
                 search_space: SearchSpace,
                 n_init: int,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_optim: AcqOptimizerBase,
                 tr_manager: Optional[TrManagerBase] = None,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')
                 ):

        super(BoBase, self).__init__(search_space, dtype)

        assert isinstance(n_init, int) and n_init >= 0
        assert isinstance(search_space, SearchSpace)
        assert isinstance(model, ModelBase)
        assert isinstance(acq_func, AcqBase)
        assert isinstance(acq_optim, AcqOptimizerBase)
        assert isinstance(tr_manager, TrManagerBase) or tr_manager is None
        assert isinstance(dtype, torch.dtype) and dtype in [torch.float32, torch.float64]
        assert isinstance(device, torch.device)

        self.device = device

        self._init_model = copy.deepcopy(model)
        self._init_acq_optimizer = copy.deepcopy(acq_optim)

        self.model = model
        self.acq_func = acq_func
        self.acq_optimizer = acq_optim
        self.tr_manager = tr_manager

        self.n_init = n_init
        self.x_init = self.search_space.sample(self.n_init)

    def restart(self):
        self._restart()
        self.x_init = self.search_space.sample(self.n_init)
        self.model = copy.deepcopy(self._init_model)
        self.acq_optimizer = copy.deepcopy(self._init_acq_optimizer)
        if self.tr_manager is not None:
            self.tr_manager.restart()

    def set_x_init(self, x: pd.DataFrame):
        assert x.ndim == 2
        assert x.shape[1] == self.search_space.num_dims
        self.x_init = x

    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        assert y.ndim == 2
        assert x.ndim == 2
        assert y.shape[1] == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.search_space.num_dims

        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        # Add data to all previously observed data
        self.data_buffer.append(x, y)

        if self.tr_manager is not None:
            self.tr_manager.append(x, y)

        # update best fx
        best_idx = y.flatten().argmin()
        best_y = y[best_idx, 0].item()

        if self.best_y is None or best_y < self.best_y:
            self.best_y = best_y
            self._best_x = x[best_idx: best_idx + 1]

    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:

        if self.tr_manager is not None:
            trigger_tr_reset = False
            for variable_type in self.tr_manager.variable_types:
                if self.tr_manager.radii[variable_type] < self.tr_manager.min_radii[variable_type]:
                    trigger_tr_reset = True
                    break

            if trigger_tr_reset:
                self.x_init = self.tr_manager.suggest_new_tr(self.n_init,
                                                             self.data_buffer,
                                                             self.data_buffer.y_min)

        # Create a Dataframe that will store the candidates
        idx = 0
        n_remaining = n_suggestions
        x_next = pd.DataFrame(index=range(n_suggestions), columns=self.search_space.df_col_names, dtype=float)

        # Return as many points from initialisation as possible
        if len(self.x_init) and n_remaining:
            n = min(n_suggestions, len(self.x_init))
            x_next.iloc[idx: idx + n] = self.x_init.iloc[[i for i in range(0, n)]]
            self.x_init = self.x_init.drop([i for i in range(0, n)], inplace=False).reset_index(drop=True)

            idx += n
            n_remaining -= n

        # Sanity check
        if n_remaining and len(self.data_buffer) == 0:
            raise Exception('n_suggestion is larger than n_init and there is no data to fit a surrogate model to')

        # Get remaining points using standard BO loop
        if n_remaining:

            torch.cuda.empty_cache()  # Clear cached memory

            if self.tr_manager is not None:
                data_buffer = self.tr_manager.data_buffer
            else:
                data_buffer = self.data_buffer

            move_model_to_device(self.model, data_buffer, self.device)

            # Used to conduct and pre-fitting operations, such as creating a new model
            self.model.pre_fit_method(data_buffer.x, data_buffer.y)

            # Fit the model
            _ = self.model.fit(data_buffer.x, data_buffer.y)

            # Grab the current best x and y for acquisition evaluation and optimisation
            best_x, best_y = self.get_best_x_and_y()
            acq_evaluate_kwargs = {'best_y': best_y}

            torch.cuda.empty_cache()  # Clear cached memory

            # Optimise the acquisition function
            x_remaining = self.acq_optimizer.optimize(best_x, n_remaining, self.data_buffer.x, self.model,
                                                      self.acq_func, acq_evaluate_kwargs=acq_evaluate_kwargs,
                                                      tr_manager=self.tr_manager)

            x_next[idx: idx + n_remaining] = self.search_space.inverse_transform(x_remaining)

        return x_next

    def observe(self, x: pd.DataFrame, y: np.ndarray):

        num_nan = np.isnan(y).sum()
        if num_nan > 0:
            raise ValueError(f"Got {num_nan} / {len(y)} NaN observations.\n"
                             f"X:\n"
                             f"    {x}\n"
                             f"Y:\n"
                             f"    {y}")
        # Transform x and y to torch tensors
        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        assert len(x) == len(y)

        # Add data to all previously observed data and to the trust region manager
        self.data_buffer.append(x, y)

        if self.tr_manager is not None:
            if len(self.tr_manager.data_buffer) > self.n_init:
                self.tr_manager.adjust_tr_radii(y)
            self.tr_manager.append(x, y)
            self.tr_manager.adjust_tr_center()

        # update best x and y
        if self.best_y is None:
            idx = y.flatten().argmin()
            self.best_y = y[idx, 0].item()
            self._best_x = x[idx: idx + 1]

        else:
            idx = y.flatten().argmin()
            y_ = y[idx, 0].item()

            if y_ < self.best_y:
                self.best_y = y_
                self._best_x = x[idx: idx + 1]

        # Used to update internal state of the optimizer if needed
        self.acq_optimizer.post_observe_method(x, y, self.data_buffer, self.n_init)

    def get_best_x_and_y(self):
        """
        :return: Returns best x and best y used for acquisition optimisation.
        """
        if self.tr_manager is None:
            x, y = self.data_buffer.x, self.data_buffer.y

        else:
            x, y = self.tr_manager.data_buffer.x, self.tr_manager.data_buffer.y

        idx = y.argmin()
        best_x = x[idx]
        best_y = y[idx]

        return best_x, best_y

    @property
    def is_numeric(self) -> bool:
        return True if (self.search_space.num_cont > 0 or self.search_space.num_disc > 0) else False

    @property
    def is_nominal(self) -> bool:
        return True if self.search_space.num_nominal > 0 else False

    @property
    def is_mixed(self) -> bool:
        return self.is_nominal and self.is_numeric
