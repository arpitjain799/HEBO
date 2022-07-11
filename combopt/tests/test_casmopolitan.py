import os.path
from pathlib import Path

import torch

from comb_opt.optimizers.casmopolitan import Casmopolitan
from comb_opt.utils.plotting_utils import plot_convergence_curve

if __name__ == '__main__':
    from comb_opt.factory import task_factory

    # task, search_space = task_factory('levy', torch.float32, num_dims=[1, 4, 1],
    #                                   variable_type=['int', 'nominal', 'num'],
    #                                   num_categories=[None, 21, None])
    task, search_space = task_factory('levy', torch.float32, num_dims=10, variable_type='nominal', num_categories=21)

    optimiser = Casmopolitan(search_space, n_init=10, tr_fail_tol=4, model_num_epochs=10, device=torch.device('cuda:1'))

    for i in range(200):
        x_next = optimiser.suggest(2)
        y_next = task(x_next)
        optimiser.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) {optimiser.best_y:.3f}')

    plot_convergence_curve(optimiser, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimiser.name}_test.png'), plot_per_iter=True)
