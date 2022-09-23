# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Optional

import pandas as pd
import torch

from comb_opt.acq_funcs import AcqBase
from comb_opt.acq_optimizers import AcqOptimizerBase
from comb_opt.models import ModelBase
from comb_opt.optimizers.genetic_algorithm import GeneticAlgorithm
from comb_opt.optimizers.genetic_algorithm import PymooGeneticAlgorithm
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region.tr_manager_base import TrManagerBase
from comb_opt.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from comb_opt.utils.data_buffer import DataBuffer
from comb_opt.utils.discrete_vars_utils import get_discrete_choices


class PymooGeneticAlgoAcqOptimizer(AcqOptimizerBase):

    def __init__(self,
                 search_space: SearchSpace,
                 ga_num_iter: int = 500,
                 ga_pop_size: int = 100,
                 ga_num_offsprings: Optional[int] = None,
                 dtype: torch.dtype = torch.float32,
                 ):

        super(PymooGeneticAlgoAcqOptimizer, self).__init__(search_space, dtype)

        self.ga_num_iter = ga_num_iter
        self.ga_pop_size = ga_pop_size
        self.ga_num_offsprings = ga_num_offsprings

        self.nominal_dims = self.search_space.nominal_dims
        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims
        # Dimensions of discrete variables in tensors containing only numeric variables
        self.disc_dims_in_numeric = [i + len(self.search_space.cont_dims) for i in
                                     range(len(self.search_space.disc_dims))]

        self.discrete_choices = get_discrete_choices(search_space)

    def optimize(self, x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_evaluate_kwargs: dict,
                 tr_manager: Optional[TrManagerBase] = None,
                 **kwargs
                 ) -> torch.Tensor:

        assert n_suggestions == 1, 'Genetic Algorithm acquisition optimizer does not support suggesting batches of data'

        ga = PymooGeneticAlgorithm(search_space=self.search_space,
                                   pop_size=self.ga_pop_size,
                                   n_offsprings=self.ga_num_offsprings,
                                   fixed_tr_manager=tr_manager,
                                   store_all=True,
                                   dtype=self.dtype)

        x_init = pd.DataFrame(index=range(self.ga_pop_size), columns=self.search_space.df_col_names, dtype=float)

        if tr_manager is None:
            x_init.iloc[0:1] = self.search_space.inverse_transform(x.unsqueeze(0))
            x_init.iloc[1:self.ga_pop_size] = self.search_space.sample(self.ga_pop_size - 1)
        else:
            x_init.iloc[0:1] = self.search_space.inverse_transform(tr_manager.center.unsqueeze(0))
            # Sample the remaining points
            x_init.iloc[1:self.ga_pop_size] = self.search_space.inverse_transform(
                sample_numeric_and_nominal_within_tr(x_centre=tr_manager.center,
                                                     search_space=self.search_space,
                                                     tr_manager=tr_manager,
                                                     n_points=self.ga_pop_size - 1,
                                                     numeric_dims=self.numeric_dims,
                                                     discrete_choices=self.discrete_choices,
                                                     model=None,
                                                     return_numeric_bounds=False))

        ga.set_x_init(x_init)

        with torch.no_grad():
            for _ in range(int(round(self.ga_num_iter / self.ga_pop_size))):
                x_next = ga.suggest(self.ga_pop_size)
                y_next = acq_func(self.search_space.transform(x_next), model,
                                  **acq_evaluate_kwargs).view(-1, 1).detach().cpu()
                ga.observe(x_next, y_next)

        valid = False
        # Iterate through all observed samples from the GA and return the one with the best black-box value
        for idx in torch.argsort(ga.data_buffer.y.flatten()):
            best_x = ga.data_buffer.x[idx].unsqueeze(0)
            if torch.logical_not((best_x == x_observed).all(axis=1)).all():
                valid = True
                break

        # If a valid sample was not found, suggest a random sample
        if not valid:
            if tr_manager is None:
                best_x = self.search_space.transform(self.search_space.sample(1))
            else:
                best_x = sample_numeric_and_nominal_within_tr(x_centre=tr_manager.center,
                                                              search_space=self.search_space,
                                                              tr_manager=tr_manager,
                                                              n_points=1,
                                                              numeric_dims=self.numeric_dims,
                                                              discrete_choices=self.discrete_choices,
                                                              model=None,
                                                              return_numeric_bounds=False)

        return best_x


class CategoricalGeneticAlgoAcqOptimizer(AcqOptimizerBase):

    def __init__(self,
                 search_space: SearchSpace,
                 ga_num_iter: int = 500,
                 ga_pop_size: int = 100,
                 ga_num_parents: int = 20,
                 ga_num_elite: int = 10,
                 ga_store_x: bool = True,
                 ga_allow_repeating_x: bool = False,
                 dtype: torch.dtype = torch.float32,
                 ):

        super(CategoricalGeneticAlgoAcqOptimizer, self).__init__(search_space, dtype)

        self.ga_num_iter = ga_num_iter
        self.ga_pop_size = ga_pop_size
        self.ga_num_parents = ga_num_parents
        self.ga_num_elite = ga_num_elite
        self.ga_store_x = ga_store_x
        self.ga_allow_repeating_x = ga_allow_repeating_x

        assert self.search_space.num_nominal + self.search_space.num_ordinal == self.search_space.num_dims, \
            'The Categorical GA acq optimiser currently only supports nominal and ordinal variables'

    def optimize(self, x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_evaluate_kwargs: dict,
                 tr_manager: Optional[TrManagerBase],
                 **kwargs
                 ) -> torch.Tensor:

        assert n_suggestions == 1, 'Genetic Algorithm acquisition optimizer does not support suggesting batches of data'

        ga = GeneticAlgorithm(search_space=self.search_space,
                              pop_size=self.ga_pop_size,
                              cat_ga_num_parents=self.ga_num_parents,
                              cat_ga_num_elite=self.ga_num_elite,
                              store_observations=self.ga_store_x,
                              cat_ga_allow_repeating_suggestions=self.ga_allow_repeating_x,
                              fixed_tr_manager=tr_manager,
                              dtype=self.dtype)

        ga.backend_ga.x_queue.iloc[0:1] = self.search_space.inverse_transform(x.unsqueeze(0))

        with torch.no_grad():
            for _ in range(int(round(self.ga_num_iter / self.ga_pop_size))):
                x_next = ga.suggest(self.ga_pop_size)
                y_next = acq_func(self.search_space.transform(x_next), model,
                                  **acq_evaluate_kwargs).view(-1, 1).detach().cpu()
                ga.observe(x_next, y_next)

        # Check if any of the elite samples was previous unobserved
        valid = False
        for x in ga.backend_ga.x_elite:
            if torch.logical_not((x.unsqueeze(0) == x_observed).all(axis=1)).all():
                valid = True
                break

        # If possible, check if any of the samples suggested by GA were unobserved
        if not valid and len(ga.data_buffer) > 0:
            ga_x, ga_y = ga.data_buffer.x, ga.data_buffer.y
            for idx in ga_y.flatten().argsort():
                x = ga_x[idx]
                if torch.logical_not((x.unsqueeze(0) == x_observed).all(axis=1)).all():
                    valid = True
                    break

        # If a valid sample was still not found, suggest a random sample
        if not valid:
            x = self.search_space.transform(self.search_space.sample(1))
        else:
            x = x.unsqueeze(0)

        return x


class GeneticAlgoAcqOptimizer(AcqOptimizerBase):

    def __init__(self,
                 search_space: SearchSpace,
                 ga_num_iter: int = 500,
                 ga_pop_size: int = 100,
                 pymoo_ga_num_offsprings: Optional[int] = None,
                 cat_ga_num_parents: int = 20,
                 cat_ga_num_elite: int = 10,
                 cat_ga_store_x: bool = False,
                 cat_ga_allow_repeating_x: bool = True,
                 dtype: torch.dtype = torch.float32,
                 ):

        super(GeneticAlgoAcqOptimizer, self).__init__(search_space, dtype)

        if self.search_space.num_nominal + self.search_space.num_ordinal == self.search_space.num_dims:
            self.ga_acq_optim = CategoricalGeneticAlgoAcqOptimizer(search_space=self.search_space,
                                                                   ga_num_iter=ga_num_iter,
                                                                   ga_pop_size=ga_pop_size,
                                                                   ga_num_parents=cat_ga_num_parents,
                                                                   ga_num_elite=cat_ga_num_elite,
                                                                   ga_store_x=cat_ga_store_x,
                                                                   ga_allow_repeating_x=cat_ga_allow_repeating_x,
                                                                   dtype=self.dtype)
        else:
            self.ga_acq_optim = PymooGeneticAlgoAcqOptimizer(search_space=self.search_space,
                                                             ga_num_iter=ga_num_iter,
                                                             ga_pop_size=ga_pop_size,
                                                             ga_num_offsprings=pymoo_ga_num_offsprings,
                                                             dtype=self.dtype)

    def optimize(self,
                 x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_evaluate_kwargs: dict,
                 tr_manager: Optional[TrManagerBase],
                 **kwargs
                 ) -> torch.Tensor:
        return self.ga_acq_optim.optimize(x=x, n_suggestions=n_suggestions, x_observed=x_observed, model=model,
                                          acq_func=acq_func, acq_evaluate_kwargs=acq_evaluate_kwargs,
                                          tr_manager=tr_manager, **kwargs)

    def post_observe_method(self, x: torch.Tensor, y: torch.Tensor, data_buffer: DataBuffer, n_init: int, **kwargs):
        self.ga_acq_optim.post_observe_method(x=x, y=y, data_buffer=data_buffer, n_init=n_init, **kwargs)
