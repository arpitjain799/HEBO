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

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT

from comb_opt.factory import task_factory
from comb_opt.optimizers.multi_armed_bandit import MultiArmedBandit
from comb_opt.trust_region.random_restart_tr_manager import RandomRestartTrManager
from comb_opt.utils.distance_metrics import hamming_distance

if __name__ == '__main__':
    task, search_space = task_factory('ackley', num_dims=[2, 1, 2, 1],
                                      variable_type=['nominal', 'nominal', 'nominal', 'nominal'],
                                      num_categories=[3, 3, 4, 5])

    tr_manager = RandomRestartTrManager(search_space,
                                        min_num_radius=2 ** -5,
                                        max_num_radius=1.,
                                        init_num_radius=0.8,
                                        min_nominal_radius=1,
                                        max_nominal_radius=10,
                                        init_nominal_radius=8,
                                        fail_tol=5,
                                        succ_tol=2,
                                        verbose=True)
    center = search_space.transform(search_space.sample(1))[0]
    tr_manager.set_center(center)
    tr_manager.radii['nominal'] = 4

    optimizer = MultiArmedBandit(search_space, fixed_tr_manager=tr_manager)

    n = 200

    for i in range(n):
        x_next = optimizer.suggest(5)

        dist = hamming_distance(search_space.transform(x_next), center.unsqueeze(0), normalize=False)
        print(f'All suggestions are within the trust region: {(dist <= tr_manager.get_nominal_radius()).all().item()}')
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f"Iteration {i + 1:03d}/{n} Current value: {y_next[0, 0]:.2f} - best value: {optimizer.best_y:.2f}")
