# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch

from comb_opt.acq_funcs import acq_factory
from comb_opt.acq_optimizers.exhaustive_local_search_acq_optimizer import ExhaustiveLsAcqOptimizer
from comb_opt.models import LinRegModel
from comb_opt.optimizers import BoBase
from comb_opt.search_space import SearchSpace
from comb_opt.utils.graph_utils import laplacian_eigen_decomposition


class LrSparseHsExhaustiveLsAcqOptim(BoBase):
    @property
    def name(self) -> str:
        return 'LR (Sparse HS) - LS acq optim'

    def __init__(self,
                 search_space: SearchSpace,
                 n_init: int,
                 model_order: int = 2,
                 model_estimator: str = 'sparse_horseshoe',
                 model_a_prior: float = 2.,
                 model_b_prior: float = 1.,
                 model_sparse_horseshoe_threshold: float = 0.1,
                 model_n_gibbs: int = int(1e3),
                 acq_optim_n_random_vertices: int = 20000,
                 acq_optim_n_greedy_ascent_init: int = 20,
                 acq_optim_n_spray: int = 10,
                 acq_optim_max_n_ascent: float = float('inf'),
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')
                 ):
        assert search_space.num_nominal + search_space.num_ordinal == search_space.num_params, \
            'This Optimiser only supports nominal and ordinal variables.'

        # Eigen decomposition of the graph laplacian
        n_vertices, adjacency_mat_list, fourier_freq_list, fourier_basis_list = laplacian_eigen_decomposition(
            search_space)

        model = LinRegModel(search_space=search_space,
                            order=model_order,
                            estimator=model_estimator,
                            a_prior=model_a_prior,
                            b_prior=model_b_prior,
                            sparse_horseshoe_threshold=model_sparse_horseshoe_threshold,
                            n_gibbs=model_n_gibbs,
                            dtype=dtype,
                            device=device)

        acq_func = acq_factory('thompson')

        acq_optim = ExhaustiveLsAcqOptimizer(search_space=search_space,
                                             adjacency_mat_list=adjacency_mat_list,
                                             n_vertices=n_vertices,
                                             n_random_vertices=acq_optim_n_random_vertices,
                                             n_greedy_ascent_init=acq_optim_n_greedy_ascent_init,
                                             n_spray=acq_optim_n_spray,
                                             max_n_ascent=acq_optim_max_n_ascent,
                                             dtype=dtype)

        tr_manager = None

        super(LrSparseHsExhaustiveLsAcqOptim, self).__init__(search_space, n_init, model, acq_func, acq_optim, tr_manager,
                                                             dtype, device)
