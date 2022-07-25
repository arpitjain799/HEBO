# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC, abstractmethod
from typing import Union, Optional

import pandas as pd
import torch

from comb_opt.search_space import SearchSpace
from comb_opt.utils.data_buffer import DataBuffer


class TrManagerBase(ABC):

    def __init__(self,
                 search_space: SearchSpace,
                 dtype: torch.dtype = torch.float32,
                 **kwargs
                 ):
        self.radii = {}
        self.min_radii = {}
        self.max_radii = {}
        self.init_radii = {}
        self.variable_types = []
        self._center = None

        self.search_space = search_space
        self.data_buffer = DataBuffer(search_space, 1, dtype)

    def set_center(self, center: Optional[torch.Tensor]):
        if center is None:
            self._center = None
        else:
            self._center = center.to(self.search_space.dtype)

    @property
    def center(self) -> Optional[torch.Tensor]:
        return self._center.clone()

    def register_radius(self,
                        variable_type: str,
                        min_radius: Union[int, float],
                        max_radius: Union[int, float],
                        init_radius: Union[int, float]
                        ):
        assert min_radius < init_radius <= max_radius

        self.variable_types.append(variable_type)

        self.radii[variable_type] = init_radius
        self.init_radii[variable_type] = init_radius
        self.min_radii[variable_type] = min_radius
        self.max_radii[variable_type] = max_radius

    def append(self, x: torch.Tensor, y: torch.Tensor):
        self.data_buffer.append(x, y)

    def restart_tr(self):
        self.data_buffer.restart()

        for var_type in self.variable_types:
            self.radii[var_type] = self.init_radii[var_type]

        self.set_center(None)

    @abstractmethod
    def restart(self):
        pass

    @abstractmethod
    def adjust_tr_radii(self, y: torch.Tensor, **kwargs):
        """
        Function used to update each radius stored in self.radii
        :return:
        """
        pass

    def adjust_tr_center(self, **kwargs):
        """
        Function used to update the TR center
        :return:
        """
        self.set_center(self.data_buffer.x_min)

    @abstractmethod
    def guided_restart(self, n_init: int, x_init: pd.DataFrame, observed_data_buffer: DataBuffer,
                       best_y: Optional[Union[float, torch.Tensor]] = None, **kwargs) -> pd.DataFrame:
        """
        Function used to conduct a guided restart of the trust region and to return a initialisation for the new trust
        region. This function is called during every suggest step. When the guided restart is not triggered, it should
        return x_init.

        :param n_init:
        :param x_init:
        :param observed_data_buffer: Data buffer containing all previously observed points
        :param best_y: Used for evaluating some acquisition functions such as the Expected Improvement acquisition
        :param kwargs:
        :return:
        """

        pass
