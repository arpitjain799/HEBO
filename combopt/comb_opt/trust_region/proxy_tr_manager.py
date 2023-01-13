# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Union, Optional

import pandas as pd
import torch

from comb_opt.search_space import SearchSpace
from comb_opt.trust_region import TrManagerBase
from comb_opt.utils.data_buffer import DataBuffer


class ProxyTrManager(TrManagerBase):

    def __init__(self,
                 search_space: SearchSpace,
                 dtype: torch.dtype = torch.float32,
                 ):
        super(ProxyTrManager, self).__init__(search_space, dtype)

    def restart(self):
        pass

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

    def suggest_new_tr(self, n_init: int, observed_data_buffer: DataBuffer, **kwargs) -> pd.DataFrame:
        """
        Function used to suggest a new trust region centre and neighbouring points

        :param n_init:
        :param observed_data_buffer: Data buffer containing all previously observed points
        :param best_y: Used for evaluating some acquisition functions such as the Expected Improvement acquisition
        :param kwargs:
        :return:
        """

        pass
