# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import warnings

from comb_opt.acq_funcs import acq_factory
from comb_opt.acq_optimizers.simulated_annealing_acq_optimizer import SimulatedAnnealingAcqOptimizer
from comb_opt.models import LinRegModel
from comb_opt.optimizers import BoBase
from comb_opt.search_space import SearchSpace
from comb_opt.search_space.params.bool_param import BoolPara


class BOCS(BoBase):
    @property
    def name(self) -> str:
        return 'BOCS'

    def __init__(self,
                 search_space: SearchSpace,
                 n_init: int,
                 model_order: int = 2,
                 model_estimator: str = 'sparse_horseshoe',
                 model_a_prior: float = 2.,
                 model_b_prior: float = 1.,
                 model_sparse_horseshoe_threshold: float = 0.1,
                 model_n_gibbs: int = int(1e3),
                 acq_optim_num_iter: int = 200,
                 acq_optim_init_temp: int = 1,
                 acq_optim_tolerance: int = 100,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')
                 ):
        assert search_space.num_nominal == search_space.num_params, 'BOCS only supports nominal  variables.'

        # Check if the problem is purely binary
        binary_problem = True
        for param_name in search_space.params:
            binary_problem = binary_problem and isinstance(search_space.params[param_name], BoolPara)
            if not binary_problem:
                break

        if binary_problem:
            warning_message = 'This is the general form implementation of BOCS (see Appendix A of ' + \
                              'https://arxiv.org/abs/1806.08838), which differs from the standard implementation ' +\
                              'for purely binary problems. The differences are: (1) binary variables are ' +\
                              'represented by their one hot encoding, and (2) SA is used to optimise the acquisition' +\
                              'in place of SDP.'
            warnings.warn(warning_message, category=UserWarning)

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

        acq_optim = SimulatedAnnealingAcqOptimizer(search_space=search_space,
                                                   sa_num_iter=acq_optim_num_iter,
                                                   sa_init_temp=acq_optim_init_temp,
                                                   sa_tolerance=acq_optim_tolerance,
                                                   dtype=dtype)

        tr_manager = None

        super(BOCS, self).__init__(search_space, n_init, model, acq_func, acq_optim, tr_manager, dtype, device)
