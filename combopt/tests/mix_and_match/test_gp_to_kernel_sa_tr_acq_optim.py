# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

from comb_opt.optimizers.mix_and_match.lr_sparse_hs_exhaustive_ls_tr_acq_optim import LrSparseHsExhaustiveLsTRAcqOptim

import torch

from comb_opt.utils.plotting_utils import plot_convergence_curve

if __name__ == '__main__':
    from comb_opt.factory import task_factory

    dtype = torch.float32

    task, search_space = task_factory('levy', dtype, num_dims=5, variable_type='nominal', num_categories=5)

    optimiser = LrSparseHsExhaustiveLsTRAcqOptim(search_space, n_init=20, dtype=dtype, device=torch.device('cpu'))

    for i in range(100):
        x_next = optimiser.suggest(1)
        y_next = task(x_next)
        optimiser.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) {optimiser.best_y:.3f}')

    plot_convergence_curve(optimiser, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimiser.name}_test.png'), plot_per_iter=True)
