# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch

from comb_opt.acq_funcs import acq_factory, SingleObjAcqExpectation
from comb_opt.acq_optimizers.genetic_algorithm_acq_optimizer import GeneticAlgoAcqOptimizer
from comb_opt.models.gp.combo_gp import ComboEnsembleGPModel
from comb_opt.optimizers import BoBase
from comb_opt.search_space import SearchSpace
from comb_opt.utils.graph_utils import laplacian_eigen_decomposition


class GpDiffusionGaAcqOptim(BoBase):

    @property
    def name(self) -> str:
        return 'GP (Diffusion) - GA acq optim'

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
                 acq_optim_ga_num_parents: int = 40,
                 acq_optim_ga_num_elite: int = 20,
                 acq_optim_ga_store_x: bool = True,
                 acq_optim_allow_repeating_x: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')

                 ):
        assert search_space.num_nominal + search_space.num_ordinal == search_space.num_params, \
            'This Optimiser only supports nominal and ordinal variables.'

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

        acq_optim = GeneticAlgoAcqOptimizer(search_space=search_space,
                                            ga_num_iter=acq_optim_ga_num_iter,
                                            ga_pop_size=acq_optim_ga_pop_size,
                                            ga_num_parents=acq_optim_ga_num_parents,
                                            ga_num_elite=acq_optim_ga_num_elite,
                                            ga_store_x=acq_optim_ga_store_x,
                                            ga_allow_repeating_x=acq_optim_allow_repeating_x,
                                            dtype=dtype)

        super(GpDiffusionGaAcqOptim, self).__init__(search_space, n_init, model, acq_func, acq_optim, None, dtype,
                                                    device)
