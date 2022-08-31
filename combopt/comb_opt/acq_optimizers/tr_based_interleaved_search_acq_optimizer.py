# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
from typing import Optional

import numpy as np
import torch

from comb_opt.acq_funcs.acq_base import AcqBase
from comb_opt.acq_optimizers.acq_optimizer_base import AcqOptimizerBase
from comb_opt.models import ModelBase
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region import TrManagerBase
from comb_opt.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from comb_opt.utils.discrete_vars_utils import get_discrete_choices
from comb_opt.utils.discrete_vars_utils import round_discrete_vars
from comb_opt.utils.distance_metrics import hamming_distance
from comb_opt.utils.model_utils import add_hallucinations_and_retrain_model


class TrBasedInterleavedSearch(AcqOptimizerBase):

    def __init__(self,
                 search_space: SearchSpace,
                 tr_manager: Optional[TrManagerBase] = None,
                 n_iter: int = 50,
                 n_restarts: int = 3,
                 max_n_perturb_num: int = 20,
                 num_optimizer: str = 'sgd',
                 num_lr: Optional[float] = None,
                 nominal_tol: int = 100,
                 dtype: torch.dtype = torch.float32
                 ):
        super(TrBasedInterleavedSearch, self).__init__(search_space, dtype)
        # if TR manager is None: we set TR to be the entire space
        if tr_manager is None:
            tr_manager = TrManagerBase(search_space=search_space, dtype=search_space.dtype)
            tr_manager.register_radius('numeric', 0, 1, 1)
            tr_manager.register_radius('nominal', min_radius=0, max_radius=search_space.num_nominal + 1,
                                       init_radius=search_space.num_nominal + 1)
            tr_manager.set_center(center=search_space.transform(search_space.sample()))

        assert search_space.num_cont + search_space.num_disc + search_space.num_nominal == search_space.num_dims, \
            'Interleaved Search only supports continuous, discrete and nominal variables'

        assert tr_manager.get_nominal_radius() > 0, 'Trust Region Manager needs to have a categorical radius'

        self.is_numeric = True if search_space.num_cont > 0 or search_space.num_disc > 0 else False
        self.is_nominal = True if search_space.num_nominal > 0 else False
        self.is_mixed = True if self.is_numeric and self.is_nominal else False

        if self.is_numeric:
            assert 'numeric' in tr_manager.radii, 'Trust Region Manager needs to have a numeric radius'

        self.tr_manager = tr_manager
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.max_n_perturb_num = max_n_perturb_num
        self.num_optimiser = num_optimizer
        self.nominal_tol = nominal_tol
        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims

        # Dimensions of discrete variables in tensors containing only numeric variables
        self.disc_dims_in_numeric = [i + len(self.search_space.cont_dims) for i in
                                     range(len(self.search_space.disc_dims))]

        self.discrete_choices = get_discrete_choices(search_space)

        self.inverse_mapping = [(self.numeric_dims + self.search_space.nominal_dims).index(i) for i in
                                range(self.search_space.num_dims)]

        # Determine the learning rate used to optimise numeric variables if needed
        if len(self.numeric_dims) > 0:
            if num_lr is None:
                if self.search_space.num_disc > 0:
                    num_lr = 1 / (len(self.discrete_choices[0]) - 1)
                else:
                    num_lr = 0.1
            else:
                assert 0 < num_lr < 1, \
                    'Numeric variables are normalised in the range [0, 1]. The learning rate should not exceed 1'
            self.num_lr = num_lr

    def optimize(self,
                 x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_evaluate_kwargs: dict,
                 **kwargs
                 ) -> torch.Tensor:

        if n_suggestions == 1:
            return self._optimize(x, 1, x_observed, model, acq_func, acq_evaluate_kwargs)
        else:
            x_next = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)
            model = copy.deepcopy(model)  # create a local copy of the model to be able to retrain it
            x_observed = x_observed.clone()

            for i in range(n_suggestions):
                x_ = self._optimize(x, 1, x_observed, model, acq_func, acq_evaluate_kwargs)
                x_next = torch.cat((x_next, x_), dim=0)

                # No need to add hallucinations during last iteration as the model will not be used
                if i < n_suggestions - 1:
                    x_observed = torch.cat((x_observed, x_), dim=0)
                    add_hallucinations_and_retrain_model(model, x_[0])

            return x_next

    def _optimize(self,
                  x: torch.Tensor,
                  n_suggestions: int,
                  x_observed: torch.Tensor,
                  model: ModelBase,
                  acq_func: AcqBase,
                  acq_evaluate_kwargs: dict,
                  ) -> torch.Tensor:

        if n_suggestions > 1:
            import warnings
            warnings.warn('Interleaved search acquisition optimizer was not extended to batched setting')

        dtype, device = model.dtype, model.device

        # Sample initialisation
        x_centre = x.clone()
        x0, numeric_lb, numeric_ub = sample_numeric_and_nominal_within_tr(x_centre=x_centre,
                                                                          search_space=self.search_space,
                                                                          tr_manager=self.tr_manager,
                                                                          n_points=self.n_restarts,
                                                                          is_numeric=self.is_numeric,
                                                                          is_mixed=self.is_mixed,
                                                                          numeric_dims=self.numeric_dims,
                                                                          discrete_choices=self.discrete_choices,
                                                                          max_n_perturb_num=self.max_n_perturb_num,
                                                                          model=model,
                                                                          return_numeric_bounds=True)

        x, acq = [], []

        for x_ in x0:

            x_numeric, x_nominal = x_[self.numeric_dims], x_[self.search_space.nominal_dims]

            for _ in range(self.n_iter):

                if self.search_space.num_numeric > 0:

                    # Optimise numeric variables
                    x_numeric.requires_grad_(True)
                    # magnitude of grad descent update in each dimension == learning rate
                    if self.num_optimiser == 'adam':
                        optimizer = torch.optim.Adam([{"params": x_numeric}], lr=self.num_lr)
                    elif self.num_optimiser == 'sgd':
                        optimizer = torch.optim.SGD([{"params": x_numeric}], lr=self.num_lr)
                    else:
                        raise NotImplementedError(f'optimiser {self.num_optimiser} is not implemented.')

                    optimizer.zero_grad()
                    x_cand = self._reconstruct_x(x_numeric, x_nominal)
                    acq_x = acq_func(x_cand.to(device, dtype), model, **acq_evaluate_kwargs)

                    try:
                        acq_x.backward()
                        optimizer.step()
                    except RuntimeError:
                        print('Exception occurred during backpropagation. NaN encountered?')
                        pass
                    with torch.no_grad():
                        x_numeric.data = round_discrete_vars(x_numeric, self.disc_dims_in_numeric,
                                                             self.discrete_choices)
                        x_numeric.data = torch.clip(x_numeric, min=numeric_lb, max=numeric_ub)

                    x_numeric.requires_grad_(False)

                if self.search_space.num_nominal > 0:

                    if self.search_space.num_numeric == 0:
                        with torch.no_grad():
                            acq_x = acq_func(self._reconstruct_x(x_numeric, x_nominal).to(device, dtype), model,
                                             **acq_evaluate_kwargs)

                    is_valid = False
                    tol_ = self.nominal_tol
                    while not is_valid:

                        neighbour_nominal = self._mutate_nominal(x_nominal)
                        if 0 <= hamming_distance(x_centre[self.search_space.nominal_dims], neighbour_nominal,
                                                 normalize=False) <= self.tr_manager.get_nominal_radius():
                            is_valid = True
                        else:
                            tol_ -= 1
                    if tol_ < 0:
                        break

                    with torch.no_grad():
                        x_cand = self._reconstruct_x(x_numeric, neighbour_nominal)
                        acq_neighbour = acq_func(x_cand.to(device, dtype), model, **acq_evaluate_kwargs)

                    if acq_neighbour < acq_x:
                        x_nominal = neighbour_nominal.clone()
                        acq_x = acq_neighbour

            x_ = self._reconstruct_x(x_numeric, x_nominal)
            with torch.no_grad():
                acq_x = acq_func(x_.to(device, dtype), model, **acq_evaluate_kwargs)

            x.append(x_)
            acq.append(acq_x.item())

        indices = np.argsort(acq)

        valid = False
        for idx in indices:
            x_ = x[idx]
            if torch.logical_not((x_.unsqueeze(0) == x_observed).all(axis=1)).all():
                valid = True
                break

        if not valid:
            x_ = self.search_space.transform(self.search_space.sample(1))
        else:
            x_ = x_.unsqueeze(0)

        return x_

    def _reconstruct_x(self, x_numeric: torch.FloatTensor, x_nominal: torch.FloatTensor) -> torch.FloatTensor:

        return torch.cat((x_numeric, x_nominal))[self.inverse_mapping]

    def _mutate_nominal(self, x_nominal):
        x_nominal_ = x_nominal.clone()
        # randomly choose an ordinal variable
        idx = np.random.randint(low=0, high=self.search_space.num_nominal)
        choices = [i for i in range(int(self.search_space.nominal_lb[idx]), int(self.search_space.nominal_ub[idx]) + 1)
                   if i != x_nominal[idx]]
        x_nominal_[idx] = np.random.choice(choices)

        return x_nominal_
