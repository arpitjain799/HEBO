import matplotlib

matplotlib.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt

from comb_opt.utils.experiment_utils import load_results, filter_results
from comb_opt import RESULTS_DIR
import os

if __name__ == '__main__':

    # Setting this to true will result in a plotting time ~60 s. Set to false when experimenting with the aesthetics of the plot
    final_plot = False
    save_dir = RESULTS_DIR

    task_names = ['Ackley Function', 'Pest Control', 'RNA Inverse Folding', '1ADQ_A Antibody Design']

    method_names = ['BOCS',
                    'COMBO',
                    'BOSS',
                    'Casmopolitan',
                    'BOiLS',
                    'CoCaBO',
                    'Random Search',
                    'Genetic Algorithm',
                    'Local Search',
                    'Simulated Annealing',
                    ]

    results = filter_results(load_results(task_names), method_names)
    sns.set_style("whitegrid")
    g = sns.FacetGrid(results, col='Task', hue='Optimizer', sharey=False, )

    if final_plot:
        g.map(sns.lineplot, "Eval Num", "f(x*)")
    else:
        g.map(sns.lineplot, "Eval Num", "f(x*)", ci=None)

    # Shrink current axis's height by 10% on the bottom
    for ax in g.axes.flatten():
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                         box.width, box.height * 0.80])

    # Put a legend below current axis
    g.axes.flatten()[2].legend(loc='upper center', bbox_to_anchor=(-0.2, -0.25),
                               fancybox=True, shadow=True, ncol=5)

    plt.savefig(os.path.join(save_dir, 'benchmarking_baselines.png'))
    plt.savefig(os.path.join(save_dir, 'benchmarking_baselines.pdf'))
    plt.close()
