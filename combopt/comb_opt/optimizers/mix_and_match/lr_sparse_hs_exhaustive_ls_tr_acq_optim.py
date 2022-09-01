# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
import copy
import math
from typing import Optional, Union

import torch

from comb_opt.acq_funcs import acq_factory
from comb_opt.acq_optimizers.exhaustive_local_search_acq_optimizer import ExhaustiveLsAcqOptimizer
from comb_opt.models import LinRegModel
from comb_opt.optimizers import BoBase
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region.casmo_tr_manager import CasmopolitanTrManager
from comb_opt.utils.graph_utils import laplacian_eigen_decomposition


class LrSparseHsExhaustiveLsTRAcqOptim(BoBase):
    @property
    def name(self) -> str:
        return 'LR (Sparse HS) - LS-TR acq optim'

    def __init__(self,
                 search_space: SearchSpace,
                 n_init: int,
                 model_order: int = 2,
                 model_estimator: str = 'sparse_horseshoe',
                 model_a_prior: float = 2.,
                 model_b_prior: float = 1.,
                 model_sparse_horseshoe_threshold: float = 0.1,
                 model_n_gibbs: int = int(1e3),
                 restart_acq_name: str = 'lcb',
                 restart_n_cand: Optional[int] = None,
                 acq_optim_n_random_vertices: int = 20000,
                 acq_optim_n_greedy_ascent_init: int = 20,
                 acq_optim_n_spray: int = 10,
                 acq_optim_max_n_ascent: float = float('inf'),
                 tr_min_num_radius: Optional[Union[int, float]] = None,
                 tr_max_num_radius: Optional[Union[int, float]] = None,
                 tr_init_num_radius: Optional[Union[int, float]] = None,
                 tr_min_nominal_radius: Optional[Union[int, float]] = None,
                 tr_max_nominal_radius: Optional[Union[int, float]] = None,
                 tr_init_nominal_radius: Optional[Union[int, float]] = None,
                 tr_radius_multiplier: Optional[float] = None,
                 tr_succ_tol: Optional[int] = None,
                 tr_fail_tol: Optional[int] = None,
                 tr_verbose: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')
                 ):
        assert search_space.num_nominal + search_space.num_ordinal == search_space.num_params, \
            'This Optimiser only supports nominal and ordinal variables.'

        # Eigen decomposition of the graph laplacian
        n_vertices, adjacency_mat_list, fourier_freq_list, fourier_basis_list = laplacian_eigen_decomposition(
            search_space)

        # Trust region for numeric variables
        if search_space.num_numeric > 0:
            if tr_min_num_radius is None:
                tr_min_num_radius = 2 ** -5
            else:
                assert 0 < tr_min_num_radius <= 1, \
                    'Numeric variables are normalised to the interval [0, 1]. Please specify appropriate Trust Region Bounds'

            if tr_max_num_radius is None:
                tr_max_num_radius = 1
            else:
                assert 0 < tr_max_num_radius <= 1, \
                    'Numeric variables are normalised to the interval [0, 1]. Please specify appropriate Trust Region Bounds'

            if tr_init_num_radius is None:
                tr_init_num_radius = 0.8 * tr_max_num_radius
            else:
                assert tr_min_num_radius < tr_init_num_radius <= tr_max_num_radius

            assert tr_min_num_radius < tr_init_num_radius <= tr_max_num_radius
        else:
            tr_min_num_radius = tr_init_num_radius = tr_max_num_radius = None

        if search_space.num_nominal > 1:
            if tr_min_nominal_radius is None:
                tr_min_nominal_radius = 1
            else:
                assert 1 <= tr_min_nominal_radius <= search_space.num_nominal

            if tr_max_nominal_radius is None:
                tr_max_nominal_radius = search_space.num_nominal
            else:
                assert 1 <= tr_max_nominal_radius <= search_space.num_nominal

            if tr_init_nominal_radius is None:
                tr_init_nominal_radius = math.ceil(0.8 * tr_max_nominal_radius)
            else:
                assert tr_min_nominal_radius <= tr_init_nominal_radius <= tr_max_nominal_radius

            assert tr_min_nominal_radius < tr_init_nominal_radius <= tr_max_nominal_radius
        else:
            tr_min_nominal_radius = tr_init_nominal_radius = tr_max_nominal_radius = None

        if tr_radius_multiplier is None:
            tr_radius_multiplier = 1.5

        if tr_succ_tol is None:
            tr_succ_tol = 3

        if tr_fail_tol is None:
            tr_fail_tol = 40

        model = LinRegModel(search_space=search_space,
                            order=model_order,
                            estimator=model_estimator,
                            a_prior=model_a_prior,
                            b_prior=model_b_prior,
                            sparse_horseshoe_threshold=model_sparse_horseshoe_threshold,
                            n_gibbs=model_n_gibbs,
                            dtype=dtype,
                            device=device)

        # Initialise the trust region manager
        tr_model = copy.deepcopy(model)

        tr_acq_func = acq_factory(acq_func_name=restart_acq_name)

        tr_manager = CasmopolitanTrManager(search_space=search_space,
                                           model=tr_model,
                                           acq_func=tr_acq_func,
                                           n_init=n_init,
                                           min_num_radius=tr_min_num_radius,
                                           max_num_radius=tr_max_num_radius,
                                           init_num_radius=tr_init_num_radius,
                                           min_nominal_radius=tr_min_nominal_radius,
                                           max_nominal_radius=tr_max_nominal_radius,
                                           init_nominal_radius=tr_init_nominal_radius,
                                           radius_multiplier=tr_radius_multiplier,
                                           succ_tol=tr_succ_tol,
                                           fail_tol=tr_fail_tol,
                                           restart_n_cand=restart_n_cand,
                                           verbose=tr_verbose,
                                           dtype=dtype,
                                           device=device)

        acq_func = acq_factory('thompson')

        acq_optim = ExhaustiveLsAcqOptimizer(search_space=search_space,
                                             tr_manager=tr_manager,
                                             adjacency_mat_list=adjacency_mat_list,
                                             n_vertices=n_vertices,
                                             n_random_vertices=acq_optim_n_random_vertices,
                                             n_greedy_ascent_init=acq_optim_n_greedy_ascent_init,
                                             n_spray=acq_optim_n_spray,
                                             max_n_ascent=acq_optim_max_n_ascent,
                                             dtype=dtype)

        super(LrSparseHsExhaustiveLsTRAcqOptim, self).__init__(search_space, n_init, model, acq_func, acq_optim, tr_manager,
                                                             dtype, device)
