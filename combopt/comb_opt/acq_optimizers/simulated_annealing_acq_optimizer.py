# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch

from comb_opt.acq_funcs import AcqBase
from comb_opt.acq_optimizers import AcqOptimizerBase
from comb_opt.models import ModelBase
from comb_opt.optimizers.simulated_annealing import SimulatedAnnealing
from comb_opt.search_space import SearchSpace
from comb_opt.utils.data_buffer import DataBuffer


class SimulatedAnnealingAcqOptimizer(AcqOptimizerBase):

    def __init__(self,
                 search_space: SearchSpace,
                 sa_num_iter: int = 500,
                 sa_init_temp: float = 1.,
                 sa_tolerance: int = 100,
                 dtype: torch.dtype = torch.float32,
                 ):

        self.sa_num_iter = sa_num_iter
        self.sa_init_temp = sa_init_temp
        self.sa_tolerance = sa_tolerance

        super(SimulatedAnnealingAcqOptimizer, self).__init__(search_space, dtype)

        assert search_space.num_nominal == search_space.num_params, \
            'Simulated Annealing is currently implemented for nominal variables only'

    def optimize(self,
                 x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_evaluate_kwargs: dict,
                 ) -> torch.Tensor:

        assert n_suggestions == 1, 'Simulated annealing acquisition optimizer does not support suggesting batches of data'

        dtype = model.dtype
        device = model.device

        sa = SimulatedAnnealing(search_space=self.search_space,
                                init_temp=self.sa_init_temp,
                                tolerance=self.sa_tolerance,
                                store_observations=True,
                                allow_repeating_suggestions=False,
                                dtype=dtype)

        sa.x_init.iloc[0:1] = self.search_space.inverse_transform(x.unsqueeze(0))

        with torch.no_grad():
            for _ in range(self.sa_num_iter):
                x_next = sa.suggest(1)
                y_next = acq_func(self.search_space.transform(x_next).to(dtype), model,
                                  **acq_evaluate_kwargs).view(-1, 1).detach().cpu()
                sa.observe(x_next, y_next)

        # Check if any of the elite samples was previous unobserved
        valid = False
        x_sa, y_sa = sa.data_buffer.x, sa.data_buffer.y
        indices = y_sa.flatten().argsort()
        for idx in indices:
            x = x_sa[idx]
            if torch.logical_not((x.unsqueeze(0) == x_observed).all(axis=1)).all():
                valid = True
                break

        # If a valid sample was still not found, suggest a random sample
        if not valid:
            x = self.search_space.transform(self.search_space.sample(1))
        else:
            x = x.unsqueeze(0)

        return x

    def post_observe_method(self, x: torch.Tensor, y: torch.Tensor, data_buffer: DataBuffer, n_init: int, **kwargs):
        """
        This function is used to set the initial temperature parameter of simulated annealing based on the observed
        data.

        :param x:
        :param y:
        :param data_buffer:
        :param n_init:
        :param kwargs:
        :return:
        """
        if len(data_buffer) == 1:
            self.sa_init_temp = data_buffer.y[0].item()
        else:
            y = data_buffer.y
            init_temp = (y.max() - y.min()).item()
            init_temp = init_temp if init_temp != 0 else 1.
            self.sa_init_temp = init_temp
