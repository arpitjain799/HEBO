from comb_opt.factory import task_factory
from comb_opt.optimizers import GeneticAlgorithm
from comb_opt.trust_region.random_restart_tr_manager import RandomRestartTrManager
from comb_opt.utils.distance_metrics import hamming_distance

if __name__ == '__main__':
    task, search_space = task_factory('ackley', num_dims=20, variable_type='nominal', num_categories=5)

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
    tr_manager.radii['nominal'] = 10

    optimiser = GeneticAlgorithm(search_space, tr_manager=tr_manager)
    n = 2000
    for i in range(n):
        x_next = optimiser.suggest()
        y_next = task(x_next)
        optimiser.observe(x_next, y_next)
        dist = hamming_distance(search_space.transform(x_next)[0:1], center.unsqueeze(0), False)[0]
        print(f"Iteration {i + 1:03d}/{n} Current value: {y_next[0, 0]:.2f} - best value: {optimiser.best_y:.2f} - Hamming distance {dist} / {tr_manager.radii['nominal']}")
        if dist > tr_manager.radii['nominal']:
            raise Exception('\n\nWarning! Last sample was outside of the trust region!\n\n')
