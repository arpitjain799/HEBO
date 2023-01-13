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
from gpytorch.constraints import Interval
from gpytorch.priors import Prior

from comb_opt.acq_funcs import acq_factory
from comb_opt.acq_optimizers.local_search_acq_optimizer import LsAcqOptimizer
from comb_opt.models.gp import ExactGPModel
from comb_opt.models.gp.kernel_factory import mixture_kernel_factory
from comb_opt.optimizers import BoBase
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region.casmo_tr_manager import CasmopolitanTrManager
from comb_opt.utils.graph_utils import laplacian_eigen_decomposition


class GpToLsAcqOptim(BoBase):

    @property
    def name(self) -> str:
        if self.use_tr:
            name = f'GP (TO) - Tr-based LS acq optim'
        else:
            name = f'GP (TO) - LS acq optim'
        return name

    def __init__(self,
                 search_space: SearchSpace,
                 n_init: int,
                 model_num_kernel_ard: bool = True,
                 model_num_kernel_lengthscale_constr: Optional[Interval] = None,
                 model_cat_kernel_ard: bool = True,
                 model_cat_kernel_lengthscale_constr: Optional[Interval] = None,
                 model_noise_prior: Optional[Prior] = None,
                 model_noise_constr: Optional[Interval] = None,
                 model_noise_lb: float = 1e-5,
                 model_pred_likelihood: bool = True,
                 model_optimizer: str = 'adam',
                 model_lr: float = 3e-2,
                 model_num_epochs: int = 100,
                 model_max_cholesky_size: int = 2000,
                 model_max_training_dataset_size: int = 1000,
                 acq_name: str = 'ei',
                 acq_optim_n_random_vertices: int = 20000,
                 acq_optim_n_greedy_ascent_init: int = 20,
                 acq_optim_n_spray: int = 10,
                 acq_optim_max_n_ascent: float = float('inf'),
                 use_tr: bool = False,
                 tr_restart_acq_name: str = 'lcb',
                 tr_restart_n_cand: Optional[int] = None,
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
        assert search_space.num_cont + search_space.num_disc + search_space.num_nominal == search_space.num_dims, \
            'This Optimiser only supports nominal and ordinal variables.'

        if use_tr:
            if tr_restart_n_cand is None:
                tr_restart_n_cand = min(100 * search_space.num_dims, 5000)
            else:
                assert isinstance(tr_restart_n_cand, int)
                assert tr_restart_n_cand > 0

            # Trust region for numeric variables (only if needed)
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

            # Trust region for nominal variables (only if needed)
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

                assert tr_min_nominal_radius < tr_init_nominal_radius <= tr_max_nominal_radius, (
                    tr_min_nominal_radius, tr_init_nominal_radius, tr_max_nominal_radius)
            else:
                tr_min_nominal_radius = tr_init_nominal_radius = tr_max_nominal_radius = None

            if tr_radius_multiplier is None:
                tr_radius_multiplier = 1.5

            if tr_succ_tol is None:
                tr_succ_tol = 3

            if tr_fail_tol is None:
                tr_fail_tol = 40

        # Eigen decomposition of the graph laplacian
        n_vertices, adjacency_mat_list, fourier_freq_list, fourier_basis_list = laplacian_eigen_decomposition(
            search_space)

        # Initialise the model
        kernel = mixture_kernel_factory(search_space=search_space,
                                        numeric_kernel_name='mat52',
                                        numeric_kernel_use_ard=model_num_kernel_ard,
                                        numeric_lengthscale_constraint=model_num_kernel_lengthscale_constr,
                                        nominal_kernel_name='transformed_overlap',
                                        nominal_kernel_use_ard=model_cat_kernel_ard,
                                        nominal_lengthscale_constraint=model_cat_kernel_lengthscale_constr)

        model = ExactGPModel(search_space=search_space,
                             num_out=1,
                             kernel=kernel,
                             noise_prior=model_noise_prior,
                             noise_constr=model_noise_constr,
                             noise_lb=model_noise_lb,
                             pred_likelihood=model_pred_likelihood,
                             lr=model_lr,
                             num_epochs=model_num_epochs,
                             optimizer=model_optimizer,
                             max_cholesky_size=model_max_cholesky_size,
                             max_training_dataset_size=model_max_training_dataset_size,
                             dtype=dtype,
                             device=device)

        # Initialise the acquisition function
        acq_func = acq_factory(acq_func_name=acq_name)

        acq_optim = LsAcqOptimizer(search_space=search_space,
                                   adjacency_mat_list=adjacency_mat_list,
                                   n_vertices=n_vertices,
                                   n_random_vertices=acq_optim_n_random_vertices,
                                   n_greedy_ascent_init=acq_optim_n_greedy_ascent_init,
                                   n_spray=acq_optim_n_spray,
                                   max_n_ascent=acq_optim_max_n_ascent,
                                   dtype=dtype)

        if use_tr:
            # Initialise the trust region manager
            tr_model = copy.deepcopy(model)

            tr_acq_func = acq_factory(acq_func_name=tr_restart_acq_name)

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
                                               restart_n_cand=tr_restart_n_cand,
                                               verbose=tr_verbose,
                                               dtype=dtype,
                                               device=device)
        else:
            tr_manager = None

        self.use_tr = use_tr

        super(GpToLsAcqOptim, self).__init__(search_space, n_init, model, acq_func, acq_optim, tr_manager,
                                             dtype, device)
