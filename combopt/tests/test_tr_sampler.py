from comb_opt.factory import task_factory
from comb_opt.optimizers import Casmopolitan

if __name__ == '__main__':
    task, search_space = task_factory('ackley', num_dims=[5, 5], variable_type=['num', 'nominal'], num_categories=[None, 5])

    optimiser = Casmopolitan(search_space, n_init=50, model_num_kernel_ard=False)
    n = 100
    for i in range(n):
        x_next = optimiser.suggest()
        y_next = task(x_next)
        optimiser.observe(x_next, y_next)
        print(f'Iteration {i + 1:03d}/{n} Current value: {y_next[0, 0]:.2f} - best value: {optimiser.best_y:.2f}')
