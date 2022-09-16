import torch

from comb_opt.factory import task_factory
from comb_opt.optimizers.mix_and_match.gp_diff_ker_ga_acq_optim import GpDiffusionGaAcqOptim

if __name__ == '__main__':
    task, search_space = task_factory('ackley', num_dims=20, variable_type='nominal', num_categories=5)

    optimizer = GpDiffusionGaAcqOptim(search_space,
                                      n_init=10,
                                      use_tr=True,
                                      model_n_burn_init=10,
                                      tr_succ_tol=20,
                                      tr_fail_tol=1,
                                      tr_verbose=True,
                                      device=torch.device('cpu')
                                      )
    n = 50
    for i in range(n):
        x_next = optimizer.suggest()
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f"Iteration {i + 1:03d}/{n} Current value: {y_next[0, 0]:.2f} - best value: {optimizer.best_y:.2f}")
