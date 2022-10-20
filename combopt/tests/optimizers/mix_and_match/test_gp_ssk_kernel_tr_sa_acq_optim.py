from comb_opt.factory import task_factory
from comb_opt.optimizers.mix_and_match import GpSskSaAcqOptim

if __name__ == '__main__':
    task, search_space = task_factory('ackley', num_dims=20, variable_type='nominal', num_categories=5)

    optimizer = GpSskSaAcqOptim(search_space, 10, use_tr=True, tr_verbose=True)

    n = 2000
    for i in range(n):
        x_next = optimizer.suggest()
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f"Iteration {i + 1:03d}/{n} Current value: {y_next[0, 0]:.2f} - best value: {optimizer.best_y:.2f}")

