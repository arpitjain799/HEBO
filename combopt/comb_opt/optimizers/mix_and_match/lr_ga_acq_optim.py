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
from comb_opt.acq_optimizers.genetic_algorithm_acq_optimizer import GeneticAlgoAcqOptimizer
from comb_opt.models import LinRegModel
from comb_opt.optimizers import BoBase
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region.casmo_tr_manager import CasmopolitanTrManager


class LrGaAcqOptim(BoBase):

    @property
    def name(self) -> str:
        if self.use_tr:
            name = f'LR ({self.model_estimator}) - Tr-Based GA acq optim'
        else:
            name = f'LR ({self.model_estimator}) - GA acq optim'
        return name

    def __init__(self,
                 search_space: SearchSpace,
                 n_init: int,
                 model_order: int = 2,
                 model_estimator: str = 'sparse_horseshoe',
                 model_a_prior: float = 2.,
                 model_b_prior: float = 1.,
                 model_sparse_horseshoe_threshold: float = 0.1,
                 model_n_gibbs: int = int(1e3),
                 acq_optim_ga_num_iter: int = 500,
                 acq_optim_ga_pop_size: int = 100,
                 acq_optim_ga_num_parents: int = 20,
                 acq_optim_ga_num_elite: int = 10,
                 acq_optim_ga_store_x: bool = False,
                 acq_optim_ga_allow_repeating_x: bool = True,
                 use_tr: bool = False,
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

        assert search_space.num_nominal == search_space.num_params, 'This Optimiser only supports nominal variables.'

        if use_tr:

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

        assert model_estimator in ['mle', 'bayes', 'horseshoe', 'sparse_horseshoe']
        self.model_estimator = model_estimator
        self.use_tr = use_tr

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

            tr_acq_func = acq_factory('thompson')

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

        super(LrGaAcqOptim, self).__init__(search_space, n_init, model, acq_func, acq_optim, tr_manager, dtype, device)
