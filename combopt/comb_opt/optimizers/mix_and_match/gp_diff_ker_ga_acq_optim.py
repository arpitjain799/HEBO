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

from comb_opt.acq_funcs import acq_factory, SingleObjAcqExpectation
from comb_opt.acq_optimizers.genetic_algorithm_acq_optimizer import GeneticAlgoAcqOptimizer
from comb_opt.models.gp.combo_gp import ComboEnsembleGPModel
from comb_opt.optimizers import BoBase
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region.casmo_tr_manager import CasmopolitanTrManager
from comb_opt.utils.graph_utils import laplacian_eigen_decomposition


class GpDiffusionGaAcqOptim(BoBase):

    @property
    def name(self) -> str:
        if self.use_tr:
            name = f'GP (Diffusion) - Tr-based GA acq optim'
        else:
            name = f'GP (Diffusion) - GA acq optim'
        return name

    def __init__(self,
                 search_space: SearchSpace,
                 n_init: int,
                 n_models: int = 10,
                 model_noise_lb: float = 1e-6,
                 model_n_burn: int = 0,
                 model_n_burn_init: int = 100,
                 model_max_training_dataset_size: int = 1000,
                 model_verbose: bool = False,
                 acq_name: str = 'ei',
                 acq_optim_ga_num_iter: int = 500,
                 acq_optim_ga_pop_size: int = 100,
                 acq_optim_ga_num_parents: int = 20,
                 acq_optim_ga_num_elite: int = 10,
                 acq_optim_ga_store_x: bool = False,
                 acq_optim_ga_allow_repeating_x: bool = True,
                 use_tr: bool = False,
                 tr_restart_acq_name: str = 'lcb',
                 tr_restart_n_cand: Optional[int] = None,
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

        if use_tr:

            assert search_space.num_ordinal == 0, 'The Casmopolitan trust region manager does not support ordinal variables'

            if tr_restart_n_cand is None:
                tr_restart_n_cand = min(100 * search_space.num_dims, 5000)
            else:
                assert isinstance(tr_restart_n_cand, int)
                assert tr_restart_n_cand > 0

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

        # Eigen decomposition of the graph laplacian
        n_vertices, adjacency_mat_list, fourier_freq_list, fourier_basis_list = laplacian_eigen_decomposition(
            search_space)

        model = ComboEnsembleGPModel(search_space=search_space,
                                     fourier_freq_list=fourier_freq_list,
                                     fourier_basis_list=fourier_basis_list,
                                     n_vertices=n_vertices,
                                     adjacency_mat_list=adjacency_mat_list,
                                     n_models=n_models,
                                     n_lb=model_noise_lb,
                                     n_burn=model_n_burn,
                                     n_burn_init=model_n_burn_init,
                                     max_training_dataset_size=model_max_training_dataset_size,
                                     verbose=model_verbose,
                                     dtype=dtype,
                                     device=device,
                                     )

        # Initialise the acquisition function
        acq_func = SingleObjAcqExpectation(acq_factory(acq_func_name=acq_name))

        # Initialise the acquisition optimizer
        acq_optim = GeneticAlgoAcqOptimizer(search_space=search_space,
                                            ga_num_iter=acq_optim_ga_num_iter,
                                            ga_pop_size=acq_optim_ga_pop_size,
                                            cat_ga_num_parents=acq_optim_ga_num_parents,
                                            cat_ga_num_elite=acq_optim_ga_num_elite,
                                            cat_ga_store_x=acq_optim_ga_store_x,
                                            cat_ga_allow_repeating_x=acq_optim_ga_allow_repeating_x,
                                            dtype=dtype)

        if use_tr:
            # Initialise the trust region manager
            tr_model = copy.deepcopy(model)

            tr_acq_func = acq_factory(acq_func_name=tr_restart_acq_name)

            tr_manager = CasmopolitanTrManager(search_space=search_space,
                                               model=tr_model,
                                               acq_func=tr_acq_func,
                                               n_init=n_init,
                                               min_num_radius=0,  # predefined as not relevant
                                               max_num_radius=1,  # predefined as not relevant
                                               init_num_radius=0.8,  # predefined as not relevant
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

        super(GpDiffusionGaAcqOptim, self).__init__(search_space, n_init, model, acq_func, acq_optim, tr_manager, dtype,
                                                    device)
