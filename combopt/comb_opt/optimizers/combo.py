# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch

from comb_opt.acq_funcs import acq_factory, SingleObjAcqExpectation
from comb_opt.acq_optimizers.exhaustive_local_search_acq_optimizer import ExhaustiveLsAcqOptimizer
from comb_opt.models import ComboEnsembleGPModel
from comb_opt.optimizers import BoBase
from comb_opt.search_space import SearchSpace
from comb_opt.utils.graph_utils import laplacian_eigen_decomposition


class COMBO(BoBase):

    @property
    def name(self) -> str:
        return 'COMBO'

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
                 acq_optim_n_random_vertices: int = 20000,
                 acq_optim_n_greedy_ascent_init: int = 20,
                 acq_optim_n_spray: int = 10,
                 acq_optim_max_n_ascent: float = float('inf'),
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu'),
                 ):
        assert search_space.num_nominal + search_space.num_ordinal == search_space.num_params, \
            'COMBO only supports nominal and ordinal variables.'

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

        acq_optim = ExhaustiveLsAcqOptimizer(search_space=search_space,
                                             adjacency_mat_list=adjacency_mat_list,
                                             n_vertices=n_vertices,
                                             n_random_vertices=acq_optim_n_random_vertices,
                                             n_greedy_ascent_init=acq_optim_n_greedy_ascent_init,
                                             n_spray=acq_optim_n_spray,
                                             max_n_ascent=acq_optim_max_n_ascent,
                                             dtype=dtype)

        tr_manager = None

        super(COMBO, self).__init__(search_space, n_init, model, acq_func, acq_optim, tr_manager, dtype, device)
