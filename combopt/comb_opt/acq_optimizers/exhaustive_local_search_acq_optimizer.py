# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
import os
from typing import List

import numpy as np
import torch

from comb_opt.acq_funcs.acq_base import AcqBase
from comb_opt.acq_optimizers.acq_optimizer_base import AcqOptimizerBase
from comb_opt.models import ModelBase
from comb_opt.search_space import SearchSpace
from comb_opt.utils.graph_utils import cartesian_neighbors
from comb_opt.utils.model_utils import add_hallucinations_and_retrain_model


class ExhaustiveLsAcqOptimizer(AcqOptimizerBase):
    def __init__(self,
                 search_space: SearchSpace,
                 adjacency_mat_list: List[torch.FloatTensor],
                 n_vertices: np.array,
                 n_random_vertices: int = 20000,
                 n_greedy_ascent_init: int = 20,
                 n_spray: int = 10,
                 max_n_ascent: float = float('inf'),
                 dtype: torch.dtype = torch.float32,
                 ):

        #TODO: add TR manager, sample initial points within the TR, during the greedy search we filter the points outside TR

        assert search_space.num_nominal + search_space.num_ordinal == search_space.num_params, \
            'The greedy descent acquisition optimizer only supports nominal and ordinal variables.'

        super(ExhaustiveLsAcqOptimizer, self).__init__(search_space, dtype)

        self.n_spray = n_spray
        self.n_random_vertices = n_random_vertices
        self.n_greedy_ascent_init = n_greedy_ascent_init
        self.max_n_descent = max_n_ascent

        if self.n_greedy_ascent_init % 2 == 1:
            self.n_greedy_ascent_init += 1

        self.n_vertices = n_vertices
        self.adjacency_mat_list = adjacency_mat_list

        self.n_cpu = os.cpu_count()
        self.n_available_cores = min(10, self.n_cpu)

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
                  **kwargs
                  ) -> torch.Tensor:

        assert n_suggestions == 1, 'Greedy Ascent acquisition optimisation does not support n_suggestions > 1'

        device, dtype = model.device, model.dtype

        # Initial point selection
        x_random = self.search_space.transform(self.search_space.sample(self.n_random_vertices)).to(dtype)

        x_neighbours = cartesian_neighbors(x.long(), self.adjacency_mat_list).to(dtype)
        shuffled_ind = list(range(x_neighbours.size(0)))
        np.random.shuffle(shuffled_ind)
        x_init_candidates = torch.cat(tuple([x_neighbours[shuffled_ind[:self.n_spray]], x_random]), dim=0)
        with torch.no_grad():
            acq_values = acq_func(x_init_candidates, model, **acq_evaluate_kwargs)

        non_nan_ind = ~torch.isnan(acq_values)
        x_init_candidates = x_init_candidates[non_nan_ind]
        acq_values = acq_values[non_nan_ind]

        acq_sorted, acq_sort_ind = torch.sort(acq_values, descending=False)
        x_init_candidates = x_init_candidates[acq_sort_ind]

        x_inits, acq_inits = x_init_candidates[:self.n_greedy_ascent_init], acq_sorted[:self.n_greedy_ascent_init]

        # Greedy Descent
        exhaustive_ls_return_values = [self._exhaustive_ls(x_inits[i], acq_func, model, acq_evaluate_kwargs) for i in
                                       range(self.n_greedy_ascent_init)]

        x_greedy_ascent, acq_greedy_ascent = zip(*exhaustive_ls_return_values)

        # Grab a previously unseen point
        x_greedy_ascent = torch.stack(x_greedy_ascent).cpu()
        acq_greedy_ascent = torch.tensor(acq_greedy_ascent)

        indices = acq_greedy_ascent.argsort()
        x_next = None
        for idx in indices:  # Attempt to grab a point from the suggested points
            if not torch.all(x_greedy_ascent[idx] == x_observed, dim=1).any():
                x_next = x_greedy_ascent[idx:idx + 1]
                break

        if x_next is None:  # Attempt to grab a neighbour of the suggested points
            for idx in indices:
                neighbours = cartesian_neighbors(x_greedy_ascent[idx].long(), self.adjacency_mat_list)
                for j in range(neighbours.size(0)):
                    if not torch.all(neighbours[j] == x_observed, dim=1).any():
                        x_next = neighbours[j: j + 1]
                        break
                if x_next is not None:
                    break

        if x_next is None:  # Else, suggest a random point
            x_next = self.search_space.transform(self.search_space.sample(1))

        return x_next

    def _exhaustive_ls(self, x_init: torch.FloatTensor, acq_func: AcqBase, model: ModelBase,
                       acq_evaluate_kwargs: dict):
        """
        In order to find local minima of an acquisition function, at each vertex,
        it follows the most decreasing edge starting from an initial point
        if self.max_descent is infinity, this method tries to find local maximum, otherwise,
        it may stop at a noncritical vertex (this option is for a computational reason)
        """
        dtype = model.dtype

        n_ascent = 0
        x = x_init
        min_acq = acq_func(x, model, **acq_evaluate_kwargs)

        while n_ascent < self.max_n_descent:
            x_neighbours = cartesian_neighbors(x.long(), self.adjacency_mat_list).to(dtype)
            with torch.no_grad():
                acq_neighbours = acq_func(x_neighbours, model, **acq_evaluate_kwargs)

            min_neighbour_index = acq_neighbours.argmin()
            min_neighbour_acq = acq_neighbours[min_neighbour_index]

            if min_neighbour_acq < min_acq:
                min_acq = min_neighbour_acq
                x = x_neighbours[min_neighbour_index.item()]
                n_ascent += 1
            else:
                break
        return x, min_acq.item()
