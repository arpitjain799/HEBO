# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch


def tsp_repair_func(x: np.array) -> np.array:
    is_np_array = isinstance(x, np.ndarray)
    is_torch_tensor = isinstance(x, torch.Tensor)
    if is_np_array:
        x_corrected = np.zeros_like(x)
        for i in range(len(x)):
            # Find index of 0
            idx = np.argmax(x[i] == 0)
            x_corrected[i] = np.concatenate((x[i, idx:], x[i, :idx]))
    elif is_torch_tensor:
        x_corrected = torch.zeros_like(x)
        for i in range(len(x)):
            # Find index of 0
            idx = np.argmax(x[i] == 0)
            x_corrected[i] = torch.cat((x[i, idx:], x[i, :idx]))
    return x_corrected