import torch

from comb_opt.factory import task_factory
from comb_opt.optimizers import Casmopolitan

if __name__ == '__main__':
    task, search_space = task_factory(task_name='antibody_design', dtype=torch.float32)
    optimizer = Casmopolitan(search_space, n_init=20, dtype=torch.float32, device=torch.device('cuda'))

    for i in range(100):
        x = optimizer.suggest(1)
        y = task(x)
        optimizer.observe(x, y)
        print(f'Iteration {i + 1:3d}/{100:3d} - f(x) = {y:.3f} - f(x*) = {optimizer.best_y:.3f}')