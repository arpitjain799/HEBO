# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Optional

import torch
from gpytorch.constraints import Interval
from gpytorch.priors import Prior

from comb_opt.acq_funcs.factory import acq_factory
from comb_opt.acq_optimizers.mab_acq_optimizer import MabAcqOptimizer
from comb_opt.models import ExactGPModel
from comb_opt.models.gp.kernel_factory import mixture_kernel_factory
from comb_opt.optimizers import BoBase
from comb_opt.search_space import SearchSpace


class CoCaBO(BoBase):

    @property
    def name(self) -> str:
        return 'CoCaBO'

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
                 model_max_batch_size: int = 5000,
                 acq_name: str = 'ei',
                 acq_optim_batch_size: int = 1,
                 acq_optim_max_n_iter: int = 200,
                 acq_optim_mab_resample_tol: int = 500,
                 acq_optim_n_cand: Optional[int] = None,
                 acq_optim_n_restarts: int = 5,
                 acq_optim_cont_optimizer: str = 'sgd',
                 acq_optim_cont_lr: float = 3e-3,
                 acq_optim_cont_n_iter: int = 100,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')
                 ):
        assert search_space.num_dims == search_space.num_cont + search_space.num_disc + search_space.num_nominal, \
            'CoCaBO only supports continuous, discrete and nominal variables'

        if acq_optim_n_cand is None:
            acq_optim_n_cand = min(100 * search_space.num_dims, 5000)

        # Determine problem type
        is_numeric = True if search_space.num_cont > 0 or search_space.num_disc > 0 else False
        is_nominal = True if search_space.num_nominal > 0 else False
        is_mixed = True if is_numeric and is_nominal else False

        # Initialise the model
        kernel = mixture_kernel_factory(search_space=search_space,
                                        is_mixed=is_mixed,
                                        is_numeric=is_numeric,
                                        is_nominal=is_nominal,
                                        numeric_kernel_name='mat52',
                                        numeric_kernel_use_ard=model_num_kernel_ard,
                                        numeric_lengthscale_constraint=model_num_kernel_lengthscale_constr,
                                        nominal_kernel_name='overlap',
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
                             max_batch_size=model_max_batch_size,
                             dtype=dtype,
                             device=device)

        # Initialise the acquisition function
        acq_func = acq_factory(acq_func_name=acq_name)

        # Initialise the acquisition optimizer
        acq_optim = MabAcqOptimizer(search_space=search_space,
                                    acq_func=acq_func,
                                    batch_size=acq_optim_batch_size,
                                    max_n_iter=acq_optim_max_n_iter,
                                    mab_resample_tol=acq_optim_mab_resample_tol,
                                    n_cand=acq_optim_n_cand,
                                    n_restarts=acq_optim_n_restarts,
                                    cont_optimizer=acq_optim_cont_optimizer,
                                    cont_lr=acq_optim_cont_lr,
                                    cont_n_iter=acq_optim_cont_n_iter,
                                    dtype=dtype)

        tr_manager = None

        super(CoCaBO, self).__init__(search_space, n_init, model, acq_func, acq_optim, tr_manager, dtype, device)
