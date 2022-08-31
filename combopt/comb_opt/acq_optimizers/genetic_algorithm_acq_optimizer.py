# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Optional

import torch

from comb_opt.acq_funcs import AcqBase
from comb_opt.acq_optimizers import AcqOptimizerBase
from comb_opt.models import ModelBase
from comb_opt.optimizers.genetic_algorithm import GeneticAlgorithm
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region.tr_manager_base import TrManagerBase


class GeneticAlgoAcqOptimizer(AcqOptimizerBase):

    def __init__(self,
                 search_space: SearchSpace,
                 ga_num_iter: int = 500,
                 ga_pop_size: int = 100,
                 ga_num_parents: int = 20,
                 ga_num_elite: int = 10,
                 ga_store_x: bool = False,
                 ga_allow_repeating_x: bool = True,
                 dtype: torch.dtype = torch.float32,
                 ):

        super(GeneticAlgoAcqOptimizer, self).__init__(search_space, dtype)

        self.ga_num_iter = ga_num_iter
        self.ga_pop_size = ga_pop_size
        self.ga_num_parents = ga_num_parents
        self.ga_num_elite = ga_num_elite
        self.ga_store_x = ga_store_x
        self.ga_allow_repeating_x = ga_allow_repeating_x

        assert self.search_space.num_nominal + self.search_space.num_ordinal == self.search_space.num_dims, \
            'Genetic Algorithm currently supports only nominal and ordinal variables'

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

        ga = GeneticAlgorithm(self.search_space,
                              self.ga_pop_size,
                              self.ga_num_parents,
                              self.ga_num_elite,
                              self.ga_store_x,
                              self.ga_allow_repeating_x,
                              tr_manager=tr_manager,
                              dtype=self.dtype)

        ga.x_queue.iloc[0:1] = self.search_space.inverse_transform(x.unsqueeze(0))

        with torch.no_grad():
            for _ in range(int(round(self.ga_num_iter / self.ga_pop_size))):
                x_next = ga.suggest(self.ga_pop_size)
                y_next = acq_func(self.search_space.transform(x_next), model,
                                  **acq_evaluate_kwargs).view(-1, 1).detach().cpu()
                ga.observe(x_next, y_next)

        # Check if any of the elite samples was previous unobserved
        valid = False
        for x in ga.x_elite:
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
