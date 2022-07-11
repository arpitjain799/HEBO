# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Optional

import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel
from gpytorch.priors import Prior

from comb_opt.acq_funcs.factory import acq_factory
from comb_opt.acq_optimizers.genetic_algorithm_acq_optimizer import GeneticAlgoAcqOptimizer
from comb_opt.models import ExactGPModel
from comb_opt.models.gp.kernels import SubStringKernel
from comb_opt.optimizers import BoBase
from comb_opt.search_space import SearchSpace


class BOSS(BoBase):

    @property
    def name(self) -> str:
        return 'BOSS'

    def __init__(self,
                 search_space: SearchSpace,
                 n_init: int,
                 model_max_subsequence_length: int = 3,
                 model_noise_prior: Optional[Prior] = None,
                 model_noise_constr: Optional[Interval] = None,
                 model_noise_lb: float = 1e-5,
                 model_pred_likelihood: bool = True,
                 model_optimiser: str = 'adam',
                 model_lr: float = 3e-2,
                 model_num_epochs: int = 100,
                 model_max_cholesky_size: int = 2000,
                 model_max_training_dataset_size: int = 1000,
                 model_max_batch_size: int = 5000,
                 model_normalize_kernel_output: bool = True,
                 acq_name: str = 'ei',
                 acq_optim_ga_num_iter: int = 500,
                 acq_optim_ga_pop_size: int = 100,
                 acq_optim_ga_num_parents: int = 40,
                 acq_optim_ga_num_elite: int = 20,
                 acq_optim_ga_store_x: bool = True,
                 acq_optim_allow_repeating_x: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu'),
                 ):
        assert search_space.num_nominal == search_space.num_params, \
            'BOSS only supports nominal variables. To use BOSS, define the search space such that the entire string' + \
            'is composed of only nominal variables. The alphabet for each position in the string is defined via the' + \
            ' \'categories\' argument of that parameter.'

        alphabet = search_space.params[search_space.param_names[0]].categories
        for param in search_space.params:
            assert search_space.params[param].categories == alphabet, \
                '\'categories\' must be the same for each of the nominal variables.'

        alphabet_size = len(alphabet)

        kernel = ScaleKernel(SubStringKernel(seq_length=search_space.num_dims,
                                             alphabet_size=alphabet_size,
                                             max_subsequence_length=model_max_subsequence_length,
                                             normalize=model_normalize_kernel_output,
                                             device=device))

        model = ExactGPModel(search_space=search_space,
                             num_out=1,
                             kernel=kernel,
                             noise_prior=model_noise_prior,
                             noise_constr=model_noise_constr,
                             noise_lb=model_noise_lb,
                             pred_likelihood=model_pred_likelihood,
                             lr=model_lr,
                             num_epochs=model_num_epochs,
                             optimizer=model_optimiser,
                             max_cholesky_size=model_max_cholesky_size,
                             max_training_dataset_size=model_max_training_dataset_size,
                             max_batch_size=model_max_batch_size,
                             dtype=dtype,
                             device=device)

        # Initialise the acquisition function
        acq_func = acq_factory(acq_func_name=acq_name)

        # Initialise the acquisition optimizer
        acq_optim = GeneticAlgoAcqOptimizer(search_space=search_space,
                                            ga_num_iter=acq_optim_ga_num_iter,
                                            ga_pop_size=acq_optim_ga_pop_size,
                                            ga_num_parents=acq_optim_ga_num_parents,
                                            ga_num_elite=acq_optim_ga_num_elite,
                                            ga_store_x=acq_optim_ga_store_x,
                                            ga_allow_repeating_x=acq_optim_allow_repeating_x,
                                            dtype=dtype)

        tr_manager = None

        super(BOSS, self).__init__(search_space, n_init, model, acq_func, acq_optim, tr_manager, dtype, device)
