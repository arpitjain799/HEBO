# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved. Redistribution and use in source and binary
# forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
# following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from pathlib import Path

from comb_opt.optimizers import BOiLS

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT

import torch


if __name__ == '__main__':
    from comb_opt.factory import task_factory

    n_init = 10
    operator_space_id = "basic"
    seq_operators_pattern_id = "basic"

    task, search_space = task_factory(
        task_name='aig_optimization',
        dtype=torch.float32,
        designs_group_id="i2c",
        operator_space_id=operator_space_id,
        seq_operators_pattern_id=seq_operators_pattern_id,
        n_parallel=1
    )

    optimiser = BOiLS(
        search_space=search_space,
        n_init=n_init,
        device=torch.device('cpu'),
        tr_succ_tol=2,
        tr_fail_tol=20,
        model_max_training_dataset_size=500,
        tr_min_num_radius=0.5 ** 5,
        tr_max_num_radius=1,
        ls_acq_name='ei',
        restart_acq_name='lcb'
    )

    for i in range(8):
        x_next = optimiser.suggest(5)
        y_next = task(x_next)
        optimiser.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) {optimiser.best_y:.3f}')
    #
    # plot_convergence_curve(optimiser, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
    #                                                      f'{optimiser.name}_test.png'), plot_per_iter=True)
