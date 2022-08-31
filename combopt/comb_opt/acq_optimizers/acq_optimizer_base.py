# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC
from abc import abstractmethod
from typing import Optional

import torch

from comb_opt.acq_funcs import AcqBase
from comb_opt.models import ModelBase
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region import TrManagerBase
from comb_opt.utils.data_buffer import DataBuffer


class AcqOptimizerBase(ABC):

    def __init__(self,
                 search_space: SearchSpace,
                 dtype: torch.dtype,
                 **kwargs
                 ):
        self.search_space = search_space
        self.dtype = dtype
        self.kwargs = kwargs

    @abstractmethod
    def optimize(self,
                 x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_evaluate_kwargs: dict,
                 tr_manager: Optional[TrManagerBase],
                 **kwargs
                 ) -> torch.Tensor:
        """
        Function used to optimise the acquisition function. Should return a 2D tensor with shape
        (n_suggestions, n_dims), where n_dims is the dimensionality of x.

        If an optimiser does not support return batches of data, this can be handled by imposing with "assert
        n_suggestions == 1"


        :param x:
        :param n_suggestions:
        :param x_observed:
        :param model:
        :param acq_func:
        :param acq_evaluate_kwargs:
        :param tr_manager: a trust region within which to perform the optimization
        :param kwargs:
        :return:
        """
        pass

    def post_observe_method(self, x: torch.Tensor, y: torch.Tensor, data_buffer: DataBuffer, n_init: int, **kwargs):
        """
        Function called at the end of observe method. Can be used to update the internal state of the acquisition
        optimizer based on the observed x and y values. Use cases may include updating the weights of a multi-armed
        bandit based on previously suggested nominal variables and the observed black-box function value.

        :param x:
        :param y:
        :param data_buffer:
        :param n_init:
        :param kwargs:
        :return:
        """

        pass
