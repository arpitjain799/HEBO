# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC
from typing import Optional

import torch

from comb_opt.search_space import SearchSpace


class DataBuffer(ABC):

    def __init__(self, search_space: SearchSpace, num_out: int, dtype: torch.dtype):
        super(DataBuffer, self).__init__()
        self.num_dims = search_space.num_dims
        self.num_out = num_out
        self.dtype = dtype

        self._x = torch.zeros((0, self.num_dims), dtype=self.dtype)
        self._y = torch.zeros((0, self.num_out), dtype=self.dtype)

    def append(self, x: torch.Tensor, y: torch.Tensor):
        assert x.ndim == 2
        assert y.ndim == 2, y
        assert len(x) == len(y)
        assert y.shape[1] == self.num_out

        self._x = torch.cat((self._x, x.clone()), axis=0)
        self._y = torch.cat((self._y, y.clone()), axis=0)

    @property
    def x(self) -> torch.Tensor:
        return self._x.clone()

    @property
    def y(self) -> torch.Tensor:
        return self._y.clone()

    @property
    def y_min(self) -> torch.Tensor:
        if len(self._y) > 0:
            return self._y.flatten().min().clone()
        else:
            return torch.tensor(0., dtype=self.dtype)

    @property
    def y_argmin(self) -> int:
        if len(self._y) > 0:
            return self._y.flatten().argmin().item()
        else:
            return 0

    @property
    def x_min(self) -> Optional[torch.Tensor]:
        if len(self._y) > 0:
            return self._x[self.y_argmin]
        else:
            return None

    def __len__(self) -> int:
        return len(self._y)

    def restart(self):
        self._x = torch.zeros((0, self.num_dims), dtype=self.dtype)
        self._y = torch.zeros((0, self.num_out), dtype=self.dtype)
