import argparse
import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT

import matplotlib

matplotlib.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt

from comb_opt.utils.experiment_utils import load_results, filter_results
from comb_opt import RESULTS_DIR
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True,
                                     description='Plot combinatorial optimisation results.')

    parser.add_argument("--final_plot", "-f", action="store_true",
                        help="Make the final plot style, Setting this to true will result in a plotting time ~60 s."
                             " Set to false when experimenting with the aesthetics of the plot")
    parser.add_argument("--start_step", "-s", type=int, default=0, help="Plot performance of optimizers from step `start_step`")

    args = parser.parse_args()

    final_plot = args.final_plot
    start_step = args.start_step
    save_dir = RESULTS_DIR

    task_names = [
        'Ackley Function',
        'Pest Control',
        'RNA Inverse Folding',
        '1ADQ_A Antibody Design',
        'EDA Sequence Optimization - Design sin - Ops basic - Pattern basic - Obj both',
        # 'Bayesmark task | Mod-lasso | DB diabetes | Metr-mae',
        # 'Bayesmark task | Mod-lasso | DB boston | Metr-mae',
        # 'Bayesmark task | Mod-linear | DB diabetes | Metr-mae',
        'Bayesmark task | Mod-linear | DB boston | Metr-mae',
    ]

    method_names = [
        'BOCS',
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
    g = sns.FacetGrid(results[results["Eval Num"] >= start_step], col='Task', hue='Optimizer', sharey=False, )

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
