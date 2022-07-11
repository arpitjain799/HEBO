# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch

from comb_opt.acq_funcs.acq_base import SingleObjAcqBase
from comb_opt.models import ModelBase


class LCB(SingleObjAcqBase):
    def __init__(self, beta: float = 1.96):
        super(LCB, self).__init__()
        self.beta = beta

    def evaluate(self,
                 x: torch.Tensor,
                 model: ModelBase,
                 **kwargs
                 ) -> torch.Tensor:
        mean, var = model.predict(x)
        mean = mean.flatten()
        std = var.clamp_min(1e-9).sqrt().flatten()

        return mean - self.beta * std
