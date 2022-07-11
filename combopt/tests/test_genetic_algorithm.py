import os
from pathlib import Path

import torch

from comb_opt.optimizers.genetic_algorithm import GeneticAlgorithm
from comb_opt.utils.plotting_utils import plot_convergence_curve

if __name__ == '__main__':
    from comb_opt.factory import task_factory

    task, search_space = task_factory('levy', torch.float32, num_dims=5, variable_type='nominal', num_categories=21)

    optimiser = GeneticAlgorithm(search_space, allow_repeating_suggestions=False)

    for i in range(500):
        x_next = optimiser.suggest(1)
        y_next = task(x_next)
        optimiser.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) {optimiser.best_y:.3f}')

    plot_convergence_curve(optimiser, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimiser.name}_test.png'), plot_per_iter=True)
