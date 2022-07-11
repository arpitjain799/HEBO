import os
from pathlib import Path

import torch

from comb_opt.optimizers.mix_and_match.gp_ssk_kernel_exhaustive_ls_acq_optim import GpSskExhaustiveLsAcqOptim
from comb_opt.utils.plotting_utils import plot_convergence_curve

if __name__ == '__main__':
    from comb_opt.factory import task_factory

    dtype = torch.float32

    task, search_space = task_factory('levy', dtype, num_dims=5, variable_type='nominal', num_categories=5)

    optimiser = GpSskExhaustiveLsAcqOptim(search_space, n_init=20, dtype=dtype, device=torch.device('cpu'))

    for i in range(100):
        x_next = optimiser.suggest(1)
        y_next = task(x_next)
        optimiser.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) {optimiser.best_y:.3f}')

    plot_convergence_curve(optimiser, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimiser.name}_test.png'), plot_per_iter=True)
