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

from comb_opt import RESULTS_DIR
from comb_opt.utils.experiment_utils import load_results, filter_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Plot combinatorial optimisation results.')

    parser.add_argument("--final_plot", "-f", action="store_true",
                        help="Make the final plot style, Setting this to true will result in a plotting time ~60 s."
                             " Set to false when experimenting with the aesthetics of the plot")

    args = parser.parse_args()

    final_plot = args.final_plot

    save_dir = RESULTS_DIR

    task_names = [
        'Ackley Function',
        'Pest Control',
        'RNA Inverse Folding',
        '1ADQ_A Antibody Design',
        'EDA Sequence Optimization - Design sin - Ops basic - Pattern basic - Obj both',
    ]

    method_names = [
        'BOCS',
        'LR (Sparse HS) - GA acq optim',
        'LR (Sparse HS) - LS acq optim',
        'LR (Sparse HS) - Tr-based LS acq optim'
        'COMBO',
        'GP (Diffusion) - GA acq optim',
        'GP (Diffusion) - SA acq optim',
        'GP (Diffusion) - Tr-based LS acq optim'
        'Casmopolitan',
        'GP (TO) - GA acq optim',
        'GP (TO) - SA acq optim',
        'GP (TO) - LS acq optim',
        'BOSS',
        'BOiLS',
        'GP (SSK) - SA acq optim',
        'GP (SSK) - LS acq optim',
    ]

    results = filter_results(load_results(task_names), method_names)
    sns.set_style("whitegrid")
    g = sns.FacetGrid(results, row='Model', col='Task', hue='Optimizer', sharey=False, )

    if final_plot:
        g.map(sns.lineplot, "Eval Num", "f(x*)")
    else:
        g.map(sns.lineplot, "Eval Num", "f(x*)", ci=None)

    # TODO couldn't format this that well and didn't manage to get all legends into the plot
    # Shrink current axis's height by 10% on the bottom
    for ax in g.axes.flatten():
        box = ax.get_position()
        ax.set_position([box.x0, box.y0,
                         box.width * 0.80, box.height])

    g.axes[0, 3].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                        fancybox=True, shadow=True, ncol=2)
    g.axes[1, 3].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                        fancybox=True, shadow=True, ncol=2)
    g.axes[2, 3].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                        fancybox=True, shadow=True, ncol=2)
    g.axes[3, 3].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                        fancybox=True, shadow=True, ncol=2)

    plt.savefig(os.path.join(RESULTS_DIR, 'benchmarking_mix_and_matching.png'))
    plt.savefig(os.path.join(RESULTS_DIR, 'benchmarking_mix_and_matching.pdf'))
    plt.close()
